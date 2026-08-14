import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
from tqdm import tqdm

# Ensure libella source directory is in python path
LIBELLA_ROOT = Path(__file__).resolve().parent
if str(LIBELLA_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBELLA_ROOT))

from libella.config import cfg, paths
from libella.data import make_meta_batches, pad_mps_shapes
from libella.model import LibellaGNN
from libella.utils import PhaseTracker, get_device


# =====================================================================
# 1. GRANULAR STEP MEMORY PROFILER
# =====================================================================
class MemoryAuditor:
    """Tracks OS Resident Set Size (RSS) and Device VRAM across pipeline sub-steps."""
    records: List[Dict[str, Any]] = []
    
    def __init__(self, step_name: str, meta_info: str = ""):
        self.step_name = step_name
        self.meta_info = meta_info
        self.process = psutil.Process(os.getpid())
        self.device = get_device()
        
    def _get_vram(self) -> float:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        elif self.device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        return 0.0

    def __enter__(self):
        self.start_rss = self.process.memory_info().rss / (1024 ** 2)  # MB
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
        
        self.records.append({
            "step": self.step_name,
            "info": self.meta_info,
            "start_rss_mb": self.start_rss,
            "end_rss_mb": self.end_rss,
            "delta_rss_mb": delta_rss,
            "start_vram_mb": self.start_vram,
            "end_vram_mb": self.end_vram,
            "delta_vram_mb": delta_vram,
            "duration_ms": duration,
        })
        
        # Log any step expanding RAM or VRAM by > 50 MB
        if abs(delta_rss) > 50 or abs(delta_vram) > 50:
            print(
                f"  ⚠️ [SPIKE] {self.step_name:<32} | "
                f"RAM: {self.end_rss:7.1f} MB (Δ {delta_rss:+6.1f} MB) | "
                f"VRAM: {self.end_vram:7.1f} MB (Δ {delta_vram:+6.1f} MB) | "
                f"Time: {duration:6.1f}ms"
            )

    @classmethod
    def print_diagnostic_table(cls):
        print("\n" + "=" * 105)
        print(" " * 35 + "📊 SUBPROCESS MEMORY AUDIT 📊")
        print("=" * 105)
        
        if not cls.records:
            print("No profiling events recorded.")
            return

        df = pd.DataFrame(cls.records)
        
        # Sort by peak RSS delta
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
        
        # Aggregated stats by operation type
        print("\n" + "-" * 105)
        print("Aggregated Summary by Subprocess Type:")
        print("-" * 105)
        agg_df = df.groupby("step").agg({
            "delta_rss_mb": ["sum", "max"],
            "delta_vram_mb": ["sum", "max"],
            "duration_ms": ["sum", "count"]
        })
        agg_df.columns = ["Sum ΔRAM (MB)", "Max ΔRAM (MB)", "Sum ΔVRAM (MB)", "Max ΔVRAM (MB)", "Total Time (ms)", "Calls"]
        agg_df = agg_df.sort_values(by="Max ΔRAM (MB)", ascending=False)
        print(agg_df.to_string())
        print("=" * 105 + "\n")


# =====================================================================
# 2. ROBUST MULTI-FORMAT PRIOR LOADER
# =====================================================================
def safe_load_priors(priors_path: Path) -> tuple[np.ndarray | None, int]:
    """Loads cNMF priors serialized with torch, joblib, numpy, or pickle."""
    raw_obj = None

    # 1. Try torch.load (primary cause of \x10 header)
    try:
        raw_obj = torch.load(priors_path, map_location="cpu", weights_only=False)
    except Exception:
        pass

    # 2. Try joblib
    if raw_obj is None:
        try:
            import joblib
            raw_obj = joblib.load(priors_path)
        except Exception:
            pass

    # 3. Try numpy
    if raw_obj is None:
        try:
            raw_obj = np.load(priors_path, allow_pickle=True)
            if hasattr(raw_obj, "files"):  # npz file
                raw_obj = {k: raw_obj[k] for k in raw_obj.files}
        except Exception:
            pass

    # 4. Try standard pickle
    if raw_obj is None:
        with open(priors_path, "rb") as f:
            raw_obj = pickle.load(f)

    # Extract components and K from loaded payload
    init_components = None
    optimal_k = 30  # default fallback

    if isinstance(raw_obj, dict):
        for key in ["init_components", "priors", "components", "basis", "H", "W"]:
            if key in raw_obj:
                init_components = raw_obj[key]
                break
        optimal_k = raw_obj.get("optimal_k", raw_obj.get("k", init_components.shape[0] if init_components is not None else 30))
    elif isinstance(raw_obj, (tuple, list)):
        init_components = raw_obj[0]
        optimal_k = raw_obj[1] if len(raw_obj) > 1 and isinstance(raw_obj[1], int) else init_components.shape[0]
    elif isinstance(raw_obj, (np.ndarray, torch.Tensor)):
        init_components = raw_obj
        optimal_k = init_components.shape[0]

    if isinstance(init_components, torch.Tensor):
        init_components = init_components.detach().cpu().numpy()

    return init_components, int(optimal_k)


# =====================================================================
# 3. DIAGNOSTIC SINGLE-EPOCH RUNNER
# =====================================================================
def run_single_epoch_diagnostic(
    genes_path: Path,
    priors_path: Path,
    chunks_dir: Path
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
    # Step D: Single Epoch Execution Loop
    # -------------------------------------------------------------
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    tracker = PhaseTracker()
    ema_mean = None
    
    print(f"\n[➤] STARTING INSTRUMENTED EPOCH (Batch Accumulation: {accumulation_steps})...\n")
    
    model.train()
    train_loss, val_loss = 0.0, 0.0
    train_steps, val_steps = 0, 0
    
    with MemoryAuditor("make_meta_batches"):
        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)

    total_steps_per_epoch = len(meta_batches)
    alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9))

    for step_idx, meta_batch in enumerate(tqdm(meta_batches, desc="Executing Single Epoch", leave=True)):
        with MemoryAuditor("optimizer.zero_grad", f"Batch {step_idx}"):
            optimizer.zero_grad(set_to_none=True)

        for chunk_idx, batch_ref in enumerate(meta_batch):
            c_file: Path = batch_ref["chunk_file"]
            tag = f"B{step_idx}_C{chunk_idx}_{c_file.name}"

            # 1. SSD Read
            with MemoryAuditor("1. torch.load (SSD)", tag):
                batch = torch.load(c_file, map_location="cpu", weights_only=False)

            # 2. SciPy CSR to Dense Matrix
            with MemoryAuditor("2. batch['x'].toarray() (Dense)", tag):
                x_dense_np = batch["x"].toarray()

            # 3. Dense Tensor Transfer to Device
            with MemoryAuditor("3. x.to(device)", tag):
                x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)
                del x_dense_np

            # 4. COO Adjacency Unpacking
            with MemoryAuditor("4. Adjacency COO -> Device", tag):
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32)
                del adj_coo

                keep_mask = torch.rand(src.size(0)) > 0.40
                src = src[keep_mask].to(device)
                dst = dst[keep_mask].to(device)
                weights = weights[keep_mask].to(device)
                del keep_mask

            # 5. MPS Bucket Padding
            with MemoryAuditor("5. pad_mps_shapes", tag):
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != "mps":
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

            # 6. GNN Forward Pass
            with MemoryAuditor("6. model.forward (GNN)", tag):
                progress = tracker.get_progress()
                model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * progress)
                model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * progress)
                model.current_temp = cfg.temp_start - ((cfg.temp_start - cfg.temp_end) * progress)

                fracs, pure_anchors = model(x, src, dst, weights)

            # 7. Core Node Extraction & Target Math
            with MemoryAuditor("7. Calculate Targets & Priors", tag):
                local_core = batch["local_core_idx"]
                core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
                t_mask_np = batch["train_mask"][local_core]

            if t_mask_np.sum() > 0:
                with MemoryAuditor("8. Train Slice & Reconstruction", tag):
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
                    hub_multiplier = max(0.0, (peak_p / 0.15) - 1.0) * 10.0
                    dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + hub_multiplier

                    recon = f_train @ pure_anchors

                with MemoryAuditor("9. model.calc_loss", tag):
                    true_batch_loss, base_recon_val = model.calc_loss(
                        recon, x_train, pure_anchors, None, 0, cfg.epochs,
                        f_train=f_train, target_f_dist=target_f_dist, kl_weight=dynamic_kl_w
                    )

                with MemoryAuditor("10. loss.backward()", tag):
                    (true_batch_loss / len(meta_batch)).backward()
                    train_loss += true_batch_loss.item()
                    train_steps += 1

                del t_mask_gpu, train_idx, f_train, x_train, p_train, current_p_mean, uniform_prior, target_f_dist, recon, true_batch_loss

            # Validation Mask Evaluation
            v_mask_np = batch["val_mask"][local_core]
            if v_mask_np.sum() > 0:
                with MemoryAuditor("11. Val Evaluation (no_grad)", tag):
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

            with MemoryAuditor("12. Subgraph Cleanup", tag):
                del batch, src, dst, weights, x, fracs, pure_anchors, core_gpu, local_core, t_mask_np, v_mask_np
                model.current_f_prob = None

        with MemoryAuditor("13. clip_grad & optimizer.step", f"Batch {step_idx}"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

        with MemoryAuditor("14. gc.collect()", f"Batch {step_idx}"):
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    with MemoryAuditor("scheduler.step"):
        scheduler.step()

    print("\n[✓] Diagnostic Epoch Complete.")
    print(f"    Train Loss: {train_loss / (train_steps + 1e-9):.4f} | Val Loss: {val_loss / (val_steps + 1e-9):.4f}")


# =====================================================================
# 4. SCRIPT ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    GENES_FILE = Path("/Users/Hemato/project_3/libella/src/libella/libella_output/run/common_genes.json")
    PRIORS_FILE = Path("/Users/Hemato/project_3/libella/src/libella/libella_output/run/global_cnmf_priors.pkl")
    CHUNKS_DIR = Path("/Users/Hemato/project_3/libella/src/libella/libella_output/run/temp_training_chunks")

    try:
        run_single_epoch_diagnostic(
            genes_path=GENES_FILE,
            priors_path=PRIORS_FILE,
            chunks_dir=CHUNKS_DIR
        )
    finally:
        MemoryAuditor.print_diagnostic_table()