import gc
import json
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------
# LIBELLA ROOT RESOLUTION & ENVIRONMENT INITIALIZATION
# ---------------------------------------------------------------------
LIBELLA_ROOT = Path(__file__).resolve().parent
if str(LIBELLA_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBELLA_ROOT))

from libella.config import cfg, init_env, paths
from libella.data import make_meta_batches, pad_mps_shapes
from libella.model import LibellaGNN
from libella.utils import PhaseTracker, get_device

# Lock runtime flags and MPS watermark (Non-blocking async mode)
init_env()


# =====================================================================
# 1. ASYNC THREAD-SAFE MEMORY AUDITOR
# =====================================================================
class MemoryAuditor:
    """Tracks OS RSS and Device VRAM without blocking GPU pipelines."""
    records: List[Dict[str, Any]] = []
    _lock = threading.Lock()
    
    def __init__(self, step_name: str, meta_info: str = "", is_worker_thread: bool = False):
        self.step_name = step_name
        self.meta_info = meta_info
        self.is_worker = is_worker_thread
        self.process = psutil.Process(os.getpid())
        self.device = get_device()
        
    def _get_vram(self) -> float:
        # Never query MPS backend from background threads to avoid Metal lockups
        if self.is_worker:
            return 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        elif self.device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        return 0.0

    def __enter__(self):
        self.start_rss = self.process.memory_info().rss / (1024 ** 2)
        self.start_vram = self._get_vram()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.end_rss = self.process.memory_info().rss / (1024 ** 2)
        self.end_vram = self._get_vram()
        
        duration = (self.end_time - self.start_time) * 1000  # ms
        delta_rss = self.end_rss - self.start_rss
        delta_vram = self.end_vram - self.start_vram
        
        record = {
            "step": self.step_name,
            "info": self.meta_info,
            "start_rss_mb": self.start_rss,
            "end_rss_mb": self.end_rss,
            "delta_rss_mb": delta_rss,
            "start_vram_mb": self.start_vram,
            "end_vram_mb": self.end_vram,
            "delta_vram_mb": delta_vram,
            "duration_ms": duration,
        }
        
        with MemoryAuditor._lock:
            self.records.append(record)
        
        # Report spikes > 50 MB
        if abs(delta_rss) > 50 or (not self.is_worker and abs(delta_vram) > 50):
            vram_str = f"VRAM: {self.end_vram:7.1f} MB (Δ {delta_vram:+6.1f} MB)" if not self.is_worker else "VRAM: [Worker CPU]"
            print(
                f"  ⚠️ [SPIKE] {self.step_name:<32} | "
                f"RAM: {self.end_rss:7.1f} MB (Δ {delta_rss:+6.1f} MB) | "
                f"{vram_str} | "
                f"Time: {duration:6.1f}ms"
            )

    @classmethod
    def print_diagnostic_table(cls):
        print("\n" + "=" * 110)
        print(" " * 40 + "📊 SUBPROCESS MEMORY AUDIT 📊")
        print("=" * 110)
        
        if not cls.records:
            print("No profiling events recorded.")
            return

        with cls._lock:
            df = pd.DataFrame(cls.records)
        
        top_spikes = df.sort_values(by="delta_rss_mb", ascending=False).head(15)
        
        display_df = pd.DataFrame({
            "Subprocess / Step": top_spikes["step"],
            "Context / Chunk": top_spikes["info"],
            "Start RAM": top_spikes["start_rss_mb"].map(lambda x: f"{x:.1f} MB"),
            "End RAM": top_spikes["end_rss_mb"].map(lambda x: f"{x:.1f} MB"),
            "Δ RAM": top_spikes["delta_rss_mb"].map(lambda x: f"{x:+.1f} MB"),
            "Δ VRAM": top_spikes["delta_vram_mb"].map(lambda x: f"{x:+.1f} MB"),
            "Duration": top_spikes["duration_ms"].map(lambda x: f"{x:.1f} ms"),
        })
        
        print("\nTop 15 RAM Spikes (by Delta Expansion):")
        print(display_df.to_string(index=False))
        
        print("\n" + "-" * 110)
        print("Aggregated Summary by Subprocess Type:")
        print("-" * 110)
        agg_df = df.groupby("step").agg({
            "delta_rss_mb": ["sum", "max"],
            "delta_vram_mb": ["sum", "max"],
            "duration_ms": ["sum", "count"]
        })
        agg_df.columns = ["Sum ΔRAM (MB)", "Max ΔRAM (MB)", "Sum ΔVRAM (MB)", "Max ΔVRAM (MB)", "Total Time (ms)", "Calls"]
        agg_df = agg_df.sort_values(by="Max ΔRAM (MB)", ascending=False)
        print(agg_df.to_string())
        print("=" * 110 + "\n")


# =====================================================================
# 2. MULTI-FORMAT PRIOR LOADER
# =====================================================================
def safe_load_priors(priors_path: Path) -> Tuple[np.ndarray | None, int]:
    raw_obj = None

    try:
        import joblib
        raw_obj = joblib.load(priors_path)
    except Exception:
        pass

    if raw_obj is None:
        try:
            raw_obj = torch.load(priors_path, map_location="cpu", weights_only=False)
        except Exception:
            pass

    if raw_obj is None:
        try:
            raw_obj = np.load(priors_path, allow_pickle=True)
            if hasattr(raw_obj, "files"):
                raw_obj = {k: raw_obj[k] for k in raw_obj.files}
        except Exception:
            pass

    if raw_obj is None:
        try:
            with open(priors_path, "rb") as f:
                raw_obj = pickle.load(f)
        except Exception:
            pass

    init_components = None
    optimal_k = cfg.n_prior_lineages

    if isinstance(raw_obj, dict):
        for key in ["init_components", "priors", "components", "basis", "H", "W"]:
            if key in raw_obj:
                init_components = raw_obj[key]
                break
        optimal_k = raw_obj.get("optimal_k", raw_obj.get("k", init_components.shape[0] if init_components is not None else optimal_k))
    elif isinstance(raw_obj, (tuple, list)):
        init_components = raw_obj[0]
        optimal_k = raw_obj[1] if len(raw_obj) > 1 and isinstance(raw_obj[1], int) else init_components.shape[0]
    elif isinstance(raw_obj, (np.ndarray, torch.Tensor)):
        init_components = raw_obj
        optimal_k = init_components.shape[0]

    if isinstance(init_components, torch.Tensor):
        init_components = init_components.detach().cpu().numpy()

    n_extra_slots = getattr(cfg, "extra_topics", 0)
    if n_extra_slots > 0 and init_components is not None:
        extra_slots = np.zeros((n_extra_slots, init_components.shape[1]), dtype=init_components.dtype)
        init_components = np.vstack([init_components, extra_slots])
        optimal_k += n_extra_slots

    return init_components, int(optimal_k)


# =====================================================================
# 3. DEADLOCK-PROOF ASYNC PREFETCHER
# =====================================================================
def prefetch_batches_async(
    meta_batches: List[List[Dict[str, Any]]]
) -> Iterator[Tuple[List[Dict[str, Any]], Iterator[Any]]]:
    """Prefetches strictly 1 chunk asynchronously without locking Metal or queues."""
    if not meta_batches:
        return

    for meta_meta in meta_batches:
        chunk_queue = queue.Queue(maxsize=1)
        stop_event = Event()

        def safe_put(item: Any) -> bool:
            while not stop_event.is_set():
                try:
                    chunk_queue.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def worker():
            try:
                for b in meta_meta:
                    if stop_event.is_set():
                        break
                    with MemoryAuditor("0. SSD Chunk Read (Worker)", b['chunk_file'].name, is_worker_thread=True):
                        chunk = torch.load(b['chunk_file'], map_location='cpu', weights_only=False)
                    if not safe_put(chunk):
                        break
            except Exception as e:
                safe_put(e)
            finally:
                safe_put(None)

        t = Thread(target=worker, daemon=True)
        t.start()

        def chunk_iterator():
            try:
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk
            finally:
                stop_event.set()
                # Drain any remaining queue items to release the worker
                while not chunk_queue.empty():
                    try:
                        chunk_queue.get_nowait()
                    except queue.Empty:
                        break

        yield meta_meta, chunk_iterator()


# =====================================================================
# 4. SINGLE-EPOCH 1-1 SIMULATOR
# =====================================================================
def run_single_epoch_diagnostic(
    genes_path: Path,
    priors_path: Path,
    chunks_dir: Path,
    epoch_idx: int = 0
):
    device = get_device()
    print(f"\n[i] Compute Device: {device}")
    
    # -------------------------------------------------------------
    # Step A: Load Genes & Priors
    # -------------------------------------------------------------
    with MemoryAuditor("Load common_genes.json"):
        with open(genes_path, "r") as f:
            common_genes = json.load(f)
        print(f"  ↳ Loaded {len(common_genes)} consensus genes.")

    with MemoryAuditor("Load global_cnmf_priors.pkl"):
        init_components, optimal_k = safe_load_priors(priors_path)
        comp_shape = init_components.shape if init_components is not None else None
        print(f"  ↳ Loaded cNMF Prior Components shape: {comp_shape} (Optimal K={optimal_k})")

    # -------------------------------------------------------------
    # Step B: Build Training Cache from SSD Chunks
    # -------------------------------------------------------------
    with MemoryAuditor("Index temp_training_chunks"):
        chunk_files = sorted(list(chunks_dir.glob("*.pt")))
        if not chunk_files:
            raise FileNotFoundError(f"No .pt chunk files found in {chunks_dir}")
            
        training_cache = []
        for chunk_file in chunk_files:
            patient_name = chunk_file.stem.split("_chunk_")[0]
            training_cache.append({
                "patient_name": patient_name,
                "chunk_file": chunk_file
            })
        print(f"  ↳ Indexed {len(training_cache)} SSD sub-graph chunks across patients.")

    # -------------------------------------------------------------
    # Step C: Model & Optimizer Initialization
    # -------------------------------------------------------------
    with MemoryAuditor("Instantiate LibellaGNN"):
        model = LibellaGNN(
            in_channels=len(common_genes),
            n_metaprograms=optimal_k,
            init_components=init_components
        ).to(device)

    with MemoryAuditor("Initialize AdamW & Schedulers"):
        base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
        anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
            {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )

    # -------------------------------------------------------------
    # Step D: Pipeline Execution Loop (100% Async)
    # -------------------------------------------------------------
    accumulation_steps = getattr(cfg, "meta_batch_size", 5)
    tracker = PhaseTracker()
    ema_mean = None
    
    print(f"\n[➤] STARTING 1-1 SIMULATOR EPOCH (Batch Accumulation: {accumulation_steps})...\n")
    
    model.train()
    train_loss, val_loss = 0.0, 0.0
    train_steps, val_steps = 0, 0
    nan_detected = False

    epoch_telemetry = {
        'ent': 0.0, 'col_r': 0.0, 'kl_w': 0.0, 'g_w': 0.0, 'p_w': 0.0, 
        'l_rec': 0.0, 'l_anc': 0.0, 'l_ort': 0.0
    }
    epoch_p_mean_sum = torch.zeros(optimal_k, dtype=torch.float32)
    train_chunk_count = 0
    
    with MemoryAuditor("make_meta_batches"):
        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)

    total_steps_per_epoch = len(meta_batches)
    alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9))

    for step_idx, (meta_meta, chunk_iter) in enumerate(tqdm(prefetch_batches_async(meta_batches), total=total_steps_per_epoch, desc="Executing Simulator Epoch")):
        with MemoryAuditor("optimizer.zero_grad", f"Step {step_idx}"):
            optimizer.zero_grad(set_to_none=True)

        for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):
            tag = f"S{step_idx}_C{chunk_idx}_{batch_ref['chunk_file'].name}"

            # 1. SciPy CSR to Dense Matrix
            with MemoryAuditor("1. batch['x'].toarray() (Dense)", tag):
                x_dense_np = batch["x"].toarray()

            # 2. Dense Tensor Transfer to Device
            with MemoryAuditor("2. x.to(device)", tag):
                x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)
                del x_dense_np

            # 3. COO Adjacency Unpacking & Edge Dropout
            with MemoryAuditor("3. Adjacency COO -> Device", tag):
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32)
                del adj_coo

                if model.training:
                    keep_mask = torch.rand(src.size(0)) > cfg.edge_dropout
                    src = src[keep_mask]
                    dst = dst[keep_mask]
                    weights = weights[keep_mask]
                    del keep_mask

                src = src.to(device)
                dst = dst.to(device)
                weights = weights.to(device)

            # 4. MPS Bucket Padding & Index Type Conformance
            with MemoryAuditor("4. pad_mps_shapes", tag):
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != "mps":
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

            # 5. GNN Forward Pass & Dynamic Parameter Updates
            with MemoryAuditor("5. model.forward (GNN)", tag):
                progress = tracker.get_progress()
                model.current_progress = progress
                model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * progress)
                model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * progress)
                model.current_temp = cfg.temp_start - ((cfg.temp_start - cfg.temp_end) * progress)

                fracs, pure_anchors = model(x, src, dst, weights)

            # 6. Core Node Extraction & Target Math
            with MemoryAuditor("6. Calculate Targets & Dynamic KL", tag):
                local_core = batch["local_core_idx"]
                core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
                t_mask_np = batch["train_mask"][local_core]

            if t_mask_np.sum() > 0:
                with MemoryAuditor("7. Train Slice & Reconstruction", tag):
                    t_mask_gpu = torch.from_numpy(t_mask_np).to(dtype=torch.bool, device=device)
                    train_idx = core_gpu[t_mask_gpu]

                    f_train = fracs[train_idx]
                    x_train = x[train_idx]

                    p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                    current_p_mean = p_train.mean(dim=0)

                    uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
                    if ema_mean is None:
                        ema_mean = current_p_mean.detach()
                    else:
                        ema_mean = alpha_ema * current_p_mean.detach() + (1 - alpha_ema) * ema_mean

                    ideal_c = uniform_prior * 2.0 - ema_mean
                    ideal_c = torch.clamp(ideal_c, min=1e-5)
                    target_f_dist = ideal_c / ideal_c.sum()

                    ema_entropy = -torch.sum(ema_mean * torch.log(ema_mean + 1e-9))
                    max_entropy = np.log(pure_anchors.shape[0])
                    collapse_ratio = torch.clamp(1.0 - (ema_entropy / max_entropy), min=0.0, max=1.0).item()

                    peak_p = ema_mean.max().item()
                    hub_multiplier = max(0.0, (peak_p / cfg.hub_threshold) - 1.0) * 10.0
                    dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + hub_multiplier

                    recon = f_train @ pure_anchors

                with MemoryAuditor("8. model.calc_loss", tag):
                    true_batch_loss, base_recon_val = model.calc_loss(
                        recon, x_train, pure_anchors, None, epoch_idx, cfg.epochs,
                        f_train=f_train, target_f_dist=target_f_dist, kl_weight=dynamic_kl_w
                    )

                    if torch.isnan(true_batch_loss) or torch.isinf(true_batch_loss):
                        nan_detected = True
                        print(f"  ↳ [!] NaN gradient encountered at step {step_idx}.")
                        break

                with MemoryAuditor("9. loss.backward()", tag):
                    (true_batch_loss / len(meta_meta)).backward()
                    train_loss += true_batch_loss.item()
                    train_steps += 1

                # Update Telemetry Metrics
                g_w_val = pure_anchors.max(dim=1).values.mean().item() * 100.0
                p_w_val = p_train.max(dim=1).values.mean().item() * 100.0
                ent_val = ema_entropy.detach().cpu().item() if isinstance(ema_entropy, torch.Tensor) else ema_entropy

                epoch_telemetry['ent'] += ent_val
                epoch_telemetry['col_r'] += collapse_ratio
                epoch_telemetry['kl_w'] += dynamic_kl_w
                epoch_telemetry['g_w'] += g_w_val
                epoch_telemetry['p_w'] += p_w_val

                chunk_losses = getattr(model, '_last_losses', {})
                epoch_telemetry['l_rec'] += float(chunk_losses.get('rec', 0.0))
                epoch_telemetry['l_anc'] += float(chunk_losses.get('anc', 0.0))
                epoch_telemetry['l_ort'] += float(chunk_losses.get('ort', 0.0))

                epoch_p_mean_sum += current_p_mean.detach().cpu()
                train_chunk_count += 1

                del t_mask_gpu, train_idx, f_train, x_train, p_train, current_p_mean, uniform_prior, target_f_dist, recon, true_batch_loss, base_recon_val

            # Validation Mask Evaluation
            v_mask_np = batch["val_mask"][local_core]
            if v_mask_np.sum() > 0:
                with MemoryAuditor("10. Val Evaluation (no_grad)", tag):
                    v_mask_gpu = torch.from_numpy(v_mask_np).to(dtype=torch.bool, device=device)
                    val_idx = core_gpu[v_mask_gpu]

                    with torch.no_grad():
                        f_val = fracs[val_idx]
                        x_val = x[val_idx]
                        val_recon = f_val @ pure_anchors

                        is_non_zero_val = (x_val > 0)
                        w_mat = torch.where(is_non_zero_val, model.dynamic_w_ema, 1.0)
                        zero_expectation_mask = torch.where(is_non_zero_val, 1.0, cfg.zero_mask_rate).to(x_val.dtype)
                        masked_w_mat_val = w_mat * zero_expectation_mask

                        raw_delta_val = val_recon - x_val
                        asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * 2.0) * (raw_delta_val < 0).to(x_val.dtype)
                        scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-cfg.delta_clamp, max=cfg.delta_clamp)

                        val_loss_sum = torch.sum(masked_w_mat_val * torch.log(torch.cosh(scaled_delta_val + 1e-6)))
                        N_cells_val = torch.clamp(torch.tensor(x_val.shape[0], dtype=torch.float32, device=device), min=1.0)
                        val_log_cosh = val_loss_sum / N_cells_val

                        val_loss += val_log_cosh.item()
                        val_steps += 1

                    del v_mask_gpu, val_idx, f_val, x_val, val_recon, w_mat, raw_delta_val, asym_val, scaled_delta_val, val_loss_sum, val_log_cosh

            with MemoryAuditor("11. Subgraph Memory Cleanup", tag):
                del batch, src, dst, weights, x, fracs, pure_anchors, core_gpu, local_core, t_mask_np, v_mask_np
                model.current_f_prob = None

        if nan_detected:
            optimizer.zero_grad(set_to_none=True)
            break

        # Accumulation Step Execution
        with MemoryAuditor("12. clip_grad & optimizer.step", f"Step {step_idx}"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

    with MemoryAuditor("scheduler.step"):
        scheduler.step()

    # Step PhaseTracker and aggregate telemetry
    if train_chunk_count > 0:
        for k in epoch_telemetry:
            epoch_telemetry[k] /= train_chunk_count

        epoch_p_mean = epoch_p_mean_sum / train_chunk_count
        top_topic_val, top_topic_idx = epoch_p_mean.max(dim=0)
        epoch_telemetry['top_t_pct'] = top_topic_val.item() * 100.0
        epoch_telemetry['top_t_id'] = top_topic_idx.item()

    is_done = tracker.step(epoch_telemetry, epoch_idx)

    # Telemetry Line Output
    mean_train_loss = train_loss / (train_steps + 1e-9)
    mean_val_loss = val_loss / (val_steps + 1e-9)

    print("\n" + "-" * 110)
    print(
        f"[Ep {(epoch_idx+1):03d}] Pure_Rec:{epoch_telemetry.get('l_rec', 0.0):<5.3f} "
        f"V_Loss:{mean_val_loss:<5.3f} (Tot_Loss:{mean_train_loss:<5.3f}) | "
        f"G_W:{epoch_telemetry.get('g_w', 0.0):<4.1f}% P_W:{epoch_telemetry.get('p_w', 0.0):<4.1f}% "
        f"TopT:{epoch_telemetry.get('top_t_id', 0)}({epoch_telemetry.get('top_t_pct', 0.0):<4.1f}%) "
        f"Ent:{epoch_telemetry.get('ent', 0.0):<4.2f} | "
        f"KL_W:{epoch_telemetry.get('kl_w', 0.0):<4.2f} L_Anc:{epoch_telemetry.get('l_anc', 0.0):<4.2f} "
        f"L_Ort:{epoch_telemetry.get('l_ort', 0.0):<4.2f}"
    )
    print("-" * 110)
    print(f"[✓] 1-1 Diagnostic Epoch Simulation Finished. PhaseTracker Complete: {is_done}\n")


# =====================================================================
# 5. SCRIPT ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    out_dirs = paths.make_dirs(cfg.suffix)
    
    GENES_FILE = out_dirs["genes"]
    PRIORS_FILE = out_dirs["cnmf_priors"]
    CHUNKS_DIR = out_dirs["out"] / "temp_training_chunks"

    if not GENES_FILE.exists():
        GENES_FILE = Path("/Users/Hemato/project_3/libella/src/libella/libella_output/run/common_genes.json")
    if not PRIORS_FILE.exists():
        PRIORS_FILE = Path("/Users/Hemato/project_3/libella/src/libella/libella_output/run/global_cnmf_priors.pkl")
    if not CHUNKS_DIR.exists():
        CHUNKS_DIR = Path("/Users/Hemato/project_3/libella/src/libella/libella_output/run/temp_training_chunks")

    try:
        run_single_epoch_diagnostic(
            genes_path=GENES_FILE,
            priors_path=PRIORS_FILE,
            chunks_dir=CHUNKS_DIR,
            epoch_idx=0
        )
    finally:
        MemoryAuditor.print_diagnostic_table()