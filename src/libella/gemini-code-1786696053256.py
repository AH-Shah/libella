import ctypes
import ctypes.util
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

# Cached OS & Process Singletons
PROCESS = psutil.Process(os.getpid())
DEVICE = get_device()


# =====================================================================
# 1. DARWIN MACH KERNEL TASK_VM_INFO (EXACT /usr/bin/time -l METRIC)
# =====================================================================
class TaskVMInfo(ctypes.Structure):
    """Structure matching Darwin mach/task_info.h task_vm_info_data_t."""
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_uint32),
        ("page_size", ctypes.c_int32),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("device_peak", ctypes.c_uint64),
        ("internal", ctypes.c_uint64),
        ("internal_peak", ctypes.c_uint64),
        ("external", ctypes.c_uint64),
        ("external_peak", ctypes.c_uint64),
        ("reusable", ctypes.c_uint64),
        ("reusable_peak", ctypes.c_uint64),
        ("purgeable_volatile_pmap", ctypes.c_uint64),
        ("purgeable_volatile_resident", ctypes.c_uint64),
        ("purgeable_volatile_virtual", ctypes.c_uint64),
        ("compressed", ctypes.c_uint64),
        ("compressed_peak", ctypes.c_uint64),
        ("compressed_lifetime", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),  # <--- True Unified Memory Footprint
        ("min_address", ctypes.c_uint64),
        ("max_address", ctypes.c_uint64),
    ]

_LIBC = ctypes.CDLL(ctypes.util.find_library("c")) if sys.platform == "darwin" else None
_TASK_VM_INFO = 22
_TASK_VM_INFO_COUNT = ctypes.sizeof(TaskVMInfo) // ctypes.sizeof(ctypes.c_uint32)

def get_phys_footprint_mb() -> float:
    """Returns exact physical kernel footprint (CPU Dirty + Metal/GPU Unified Pools) in MB."""
    if _LIBC is None:
        return PROCESS.memory_info().rss / (1024 ** 2)
    try:
        info = TaskVMInfo()
        count = ctypes.c_uint32(_TASK_VM_INFO_COUNT)
        ret = _LIBC.task_info(_LIBC.mach_task_self(), _TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count))
        if ret == 0:
            return info.phys_footprint / (1024 ** 2)
    except Exception:
        pass
    return PROCESS.memory_info().rss / (1024 ** 2)


# =====================================================================
# 2. ZERO-OVERHEAD FAST MEMORY AUDITOR
# =====================================================================
class MemoryAuditor:
    """Ultra-low overhead snapshot recorder for CPU, VRAM, and Mach Unified Footprint."""
    records: List[Dict[str, Any]] = []
    
    def __init__(self, step_name: str, meta_info: str = ""):
        self.step_name = step_name
        self.meta_info = meta_info
        
    def _get_active_vram(self) -> float:
        if DEVICE.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        elif DEVICE.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        return 0.0

    def __enter__(self):
        self.start_footprint = get_phys_footprint_mb()
        self.start_rss = PROCESS.memory_info().rss / (1024 ** 2)
        self.start_vram = self._get_active_vram()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.end_footprint = get_phys_footprint_mb()
        self.end_rss = PROCESS.memory_info().rss / (1024 ** 2)
        self.end_vram = self._get_active_vram()
        
        duration = (self.end_time - self.start_time) * 1000  # ms
        delta_footprint = self.end_footprint - self.start_footprint
        delta_rss = self.end_rss - self.start_rss
        delta_vram = self.end_vram - self.start_vram
        
        self.records.append({
            "step": self.step_name,
            "info": self.meta_info,
            "end_footprint_mb": self.end_footprint,
            "delta_footprint_mb": delta_footprint,
            "end_rss_mb": self.end_rss,
            "delta_rss_mb": delta_rss,
            "end_vram_mb": self.end_vram,
            "delta_vram_mb": delta_vram,
            "duration_ms": duration,
        })
        
        # Display spikes expanding unified memory footprint by > 100 MB
        if abs(delta_footprint) > 100 or abs(delta_rss) > 50:
            print(
                f"  ⚡ [SPIKE] {self.step_name:<28} | "
                f"Phys Footprint: {self.end_footprint:7.1f} MB (Δ {delta_footprint:+6.1f} MB) | "
                f"RSS: {self.end_rss:6.1f} MB | VRAM: {self.end_vram:6.1f} MB | "
                f"{duration:5.1f}ms"
            )

    @classmethod
    def print_diagnostic_table(cls):
        print("\n" + "=" * 120)
        print(" " * 42 + "📊 COMPLETE PIPELINE MEMORY AUDIT 📊")
        print("=" * 120)
        
        if not cls.records:
            print("No profiling events recorded.")
            return

        df = pd.DataFrame(cls.records)
        top_spikes = df.sort_values(by="delta_footprint_mb", ascending=False).head(15)
        
        display_df = pd.DataFrame({
            "Subprocess / Step": top_spikes["step"],
            "Context / Chunk": top_spikes["info"],
            "Phys Footprint": top_spikes["end_footprint_mb"].map(lambda x: f"{x:.1f} MB"),
            "Δ Footprint": top_spikes["delta_footprint_mb"].map(lambda x: f"{x:+.1f} MB"),
            "CPU RSS": top_spikes["end_rss_mb"].map(lambda x: f"{x:.1f} MB"),
            "Active VRAM": top_spikes["end_vram_mb"].map(lambda x: f"{x:.1f} MB"),
            "Duration": top_spikes["duration_ms"].map(lambda x: f"{x:.1f} ms"),
        })
        
        print("\nTop 15 Physical Unified Memory Spikes (by Δ Kernel Footprint):")
        print(display_df.to_string(index=False))
        
        print("\n" + "-" * 120)
        print("Aggregated Summary by Step Type:")
        print("-" * 120)
        agg_df = df.groupby("step").agg({
            "delta_footprint_mb": ["sum", "max"],
            "delta_rss_mb": ["sum", "max"],
            "delta_vram_mb": ["sum", "max"],
            "duration_ms": ["sum", "count"]
        })
        agg_df.columns = ["Sum ΔFootprint", "Max ΔFootprint", "Sum ΔRSS", "Max ΔRSS", "Sum ΔVRAM", "Max ΔVRAM", "Total Time (ms)", "Calls"]
        agg_df = agg_df.sort_values(by="Max ΔFootprint", ascending=False)
        print(agg_df.to_string())
        print("=" * 120 + "\n")


# =====================================================================
# 3. PRIOR LOADER & NATIVE PREFETCHER
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

def prefetch_batches_native(
    meta_batches: List[List[Dict[str, Any]]]
) -> Iterator[Tuple[List[Dict[str, Any]], Iterator[Any], Event, Thread]]:
    """Native bounded prefetcher without thread profiling overhead."""
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
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        yield meta_meta, chunk_iterator(), stop_event, t


# =====================================================================
# 4. HIGH-FIDELITY PIPELINE SIMULATOR
# =====================================================================
def run_simulation(
    genes_path: Path,
    priors_path: Path,
    chunks_dir: Path,
    epochs: int = 2
):
    print(f"\n[i] Compute Device: {DEVICE}")
    print(f"[i] Initial Mach Physical Footprint: {get_phys_footprint_mb():.1f} MB | Initial CPU RSS: {PROCESS.memory_info().rss / (1024**2):.1f} MB")
    
    # -------------------------------------------------------------
    # Step A: Load Genes & Priors
    # -------------------------------------------------------------
    with MemoryAuditor("Load Genes & Priors"):
        with open(genes_path, "r") as f:
            common_genes = json.load(f)
        init_components, optimal_k = safe_load_priors(priors_path)
        print(f"  ↳ Loaded {len(common_genes)} genes | K={optimal_k} topics")

    # -------------------------------------------------------------
    # Step B: Build Training Cache from SSD Chunks
    # -------------------------------------------------------------
    with MemoryAuditor("Index SSD Subgraphs"):
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
        print(f"  ↳ Indexed {len(training_cache)} SSD chunks across cohort.")

    # -------------------------------------------------------------
    # Step C: Model & Optimizer Initialization
    # -------------------------------------------------------------
    with MemoryAuditor("Instantiate Model & AdamW"):
        model = LibellaGNN(
            in_channels=len(common_genes),
            n_metaprograms=optimal_k,
            init_components=init_components
        ).to(DEVICE)

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
    # Step D: Training Loop Simulation
    # -------------------------------------------------------------
    accumulation_steps = getattr(cfg, "meta_batch_size", 5)
    tracker = PhaseTracker()
    ema_mean = None
    
    print(f"\n[➤] EXECUTING REAL-SPEED SIMULATION ({epochs} Epochs, Batch Accumulation: {accumulation_steps})...\n")

    for epoch in range(epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        train_steps, val_steps = 0, 0
        train_chunk_count = 0
        epoch_telemetry = {'rec': 0.0, 'anc': 0.0, 'ort': 0.0, 'ent': 0.0, 'kl_w': 0.0, 'g_w': 0.0, 'p_w': 0.0}
        epoch_p_mean_sum = torch.zeros(optimal_k, dtype=torch.float32)

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps_per_epoch = len(meta_batches)
        alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9))

        pbar = tqdm(
            prefetch_batches_native(meta_batches), 
            total=total_steps_per_epoch, 
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True
        )

        for step_idx, (meta_meta, chunk_iter, stop_ev, worker_thread) in enumerate(pbar):
            with MemoryAuditor("optimizer.zero_grad", f"Ep{epoch}_S{step_idx}"):
                optimizer.zero_grad(set_to_none=True)

            for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):
                tag = f"E{epoch}_S{step_idx}_C{chunk_idx}"

                # 1. CSR to Dense
                with MemoryAuditor("1. Dense Array Conversion", tag):
                    x_dense_np = batch["x"].toarray()

                # 2. Transfer X to Device
                with MemoryAuditor("2. x.to(device)", tag):
                    x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=DEVICE)
                    del x_dense_np

                # 3. Adjacency Extraction & Dropout
                with MemoryAuditor("3. Adjacency Unpack & EdgeDrop", tag):
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

                    src = src.to(DEVICE)
                    dst = dst.to(DEVICE)
                    weights = weights.to(DEVICE)

                # 4. MPS Bucket Padding
                with MemoryAuditor("4. pad_mps_shapes", tag):
                    x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                    if DEVICE.type != "mps":
                        src = src.to(torch.int64)
                        dst = dst.to(torch.int64)

                # 5. GNN Forward Pass
                with MemoryAuditor("5. GNN Forward Pass", tag):
                    progress = tracker.get_progress()
                    model.current_progress = progress
                    model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * progress)
                    model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * progress)
                    model.current_temp = cfg.temp_start - ((cfg.temp_start - cfg.temp_end) * progress)

                    fracs, pure_anchors = model(x, src, dst, weights)

                # 6. Train Slicing & Loss Calculation
                local_core = batch["local_core_idx"]
                core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=DEVICE)
                t_mask_np = batch["train_mask"][local_core]

                if t_mask_np.sum() > 0:
                    with MemoryAuditor("6. Reconstruction & Math", tag):
                        t_mask_gpu = torch.from_numpy(t_mask_np).to(dtype=torch.bool, device=DEVICE)
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

                    with MemoryAuditor("7. model.calc_loss", tag):
                        true_batch_loss, _ = model.calc_loss(
                            recon, x_train, pure_anchors, None, epoch, cfg.epochs,
                            f_train=f_train, target_f_dist=target_f_dist, kl_weight=dynamic_kl_w
                        )

                    with MemoryAuditor("8. loss.backward()", tag):
                        (true_batch_loss / len(meta_meta)).backward()
                        train_loss += true_batch_loss.item()
                        train_steps += 1

                    # Telemetry Tracking
                    epoch_telemetry['g_w'] += pure_anchors.max(dim=1).values.mean().item() * 100.0
                    epoch_telemetry['p_w'] += p_train.max(dim=1).values.mean().item() * 100.0
                    epoch_telemetry['ent'] += (ema_entropy.item() if isinstance(ema_entropy, torch.Tensor) else ema_entropy)
                    epoch_telemetry['kl_w'] += dynamic_kl_w
                    
                    chunk_losses = getattr(model, '_last_losses', {})
                    epoch_telemetry['rec'] += float(chunk_losses.get('rec', 0.0))
                    epoch_telemetry['anc'] += float(chunk_losses.get('anc', 0.0))
                    epoch_telemetry['ort'] += float(chunk_losses.get('ort', 0.0))

                    epoch_p_mean_sum += current_p_mean.detach().cpu()
                    train_chunk_count += 1

                    del t_mask_gpu, train_idx, f_train, x_train, p_train, current_p_mean, uniform_prior, target_f_dist, recon, true_batch_loss

                # Validation Evaluation
                v_mask_np = batch["val_mask"][local_core]
                if v_mask_np.sum() > 0:
                    with MemoryAuditor("9. Val Evaluation", tag):
                        v_mask_gpu = torch.from_numpy(v_mask_np).to(dtype=torch.bool, device=DEVICE)
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
                            N_cells_val = torch.clamp(torch.tensor(x_val.shape[0], dtype=torch.float32, device=DEVICE), min=1.0)
                            val_log_cosh = val_loss_sum / N_cells_val

                            val_loss += val_log_cosh.item()
                            val_steps += 1

                        del v_mask_gpu, val_idx, f_val, x_val, val_recon, w_mat, raw_delta_val, asym_val, scaled_delta_val, val_loss_sum, val_log_cosh

                with MemoryAuditor("10. Subgraph Cleanup", tag):
                    del batch, src, dst, weights, x, fracs, pure_anchors, core_gpu, local_core, t_mask_np, v_mask_np
                    model.current_f_prob = None

            stop_ev.set()
            worker_thread.join(timeout=0.2)

            with MemoryAuditor("11. clip_grad & optimizer.step", f"Ep{epoch}_S{step_idx}"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()

        with MemoryAuditor("scheduler.step"):
            scheduler.step()

        # Telemetry Aggregation
        if train_chunk_count > 0:
            for k in epoch_telemetry:
                epoch_telemetry[k] /= train_chunk_count
            top_t_val, top_t_idx = (epoch_p_mean_sum / train_chunk_count).max(dim=0)
            epoch_telemetry['top_t_pct'] = top_t_val.item() * 100.0
            epoch_telemetry['top_t_id'] = top_t_idx.item()

        tracker.step(epoch_telemetry, epoch)

        current_phys = get_phys_footprint_mb()
        print(
            f" [Ep {(epoch+1):03d}] Pure_Rec:{epoch_telemetry.get('rec', 0.0):<5.3f} "
            f"V_Loss:{val_loss / (val_steps + 1e-9):<5.3f} | "
            f"Phys Footprint: {current_phys:7.1f} MB (Peak Unified) | "
            f"CPU RSS: {PROCESS.memory_info().rss / (1024**2):6.1f} MB | "
            f"Active VRAM: {model.dict_temp.device.type and (torch.mps.current_allocated_memory()/(1024**2) if DEVICE.type=='mps' else 0.0):5.1f} MB"
        )


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
        run_simulation(
            genes_path=GENES_FILE,
            priors_path=PRIORS_FILE,
            chunks_dir=CHUNKS_DIR,
            epochs=2
        )
    finally:
        MemoryAuditor.print_diagnostic_table()