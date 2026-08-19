"""Utility functions for Libella pipeline operations."""

import ast
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm
import psutil
import torch
import gc

from collections.abc import Iterator
import scipy.sparse as sp

from .config import NOISE_REGEX, cfg

def get_device() -> torch.device:
    """Get optimal compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_whitelist(csv_path: Path) -> set[str]:
    """Get pruned target genes from CSV."""
    if not csv_path.exists():
        return set()

    raw_genes: set[str] = set()
    df = pd.read_csv(csv_path)
    
    for gene_str in df["Genes"].dropna():
        try:
            gene_list = ast.literal_eval(gene_str)
            raw_genes.update(str(g).strip() for g in gene_list)
        except (ValueError, SyntaxError):
            continue

    clean_genes: set[str] = {g for g in raw_genes if not NOISE_REGEX.match(g)}
    return clean_genes

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Project logits to probability simplex."""
    sorted_logits, _ = torch.sort(logits, descending=True, dim=dim)
    z = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    bound = 1 + k * sorted_logits > z
    rho = torch.sum(bound.to(logits.dtype), dim=dim, keepdim=True)
    tau = (torch.gather(z, dim, (rho - 1).long()) - 1) / rho
    return torch.clamp(logits - tau, min=0.0)

def scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Fast scatter softmax without CPU sync."""
    src_safe = torch.clamp(src, min=-60.0, max=60.0)

    exp_val = torch.exp(src_safe)
    sum_val = torch.zeros(num_nodes, dtype=src.dtype, device=src.device).scatter_add(0, index, exp_val)
    return exp_val / (sum_val[index] + 1e-9)



"""Adaptive Elastic Phase Controller with Dual-Horizon Oscillation Filtering."""

import math
from typing import Any
import numpy as np
from .config import cfg


class PhaseTracker:
    """Adaptive training governor with PID-like pressure modulation and automated early stopping."""

    def __init__(
        self,
        total_epochs: int = 50,
        rel_tolerance: float = 0.08,
    ) -> None:
        self.phase: int = 1
        self.total_epochs: int = max(1, total_epochs)
        self.rel_tolerance: float = rel_tolerance

        # --- Phase 1 Horizons ---
        self.min_p1_epochs: int = max(4, int(self.total_epochs * 0.10))
        self.max_p1_epochs: int = max(self.min_p1_epochs + 2, int(self.total_epochs * 0.25))
        self.p1_plateau_patience: int = 3
        self.p1_plateau_count: int = 0

        # --- Dynamic Window Sizing ---
        self.cycle_window: int = max(3, int(self.total_epochs * 0.08))
        self.patience_epochs: int = max(4, int(self.total_epochs * 0.10))

        # --- Metric Histories ---
        self.rec_history: list[float] = []
        self.val_history: list[float] = []
        self.corr_history: list[float] = []

        self.best_rec_loss: float = float("inf")
        self.best_val_loss: float = float("inf")
        self.p1_baseline_rec: float = 0.0
        self.p2_start_epoch: int = 0

        # --- Closed-Loop Governor State ---
        self.pressure: float = 0.0
        self.no_improve_count: int = 0
        self._progress: float = 0.0
        self.spatial_warmup: float = 0.10

    @staticmethod
    def _fit_ols(series: list[float]) -> tuple[float, float, float]:
        n = len(series)
        if n < 3:
            return 0.0, float(series[-1]) if series else 0.0, 0.01

        t = np.arange(n, dtype=np.float64)
        y = np.array(series, dtype=np.float64)
        t_mean = (n - 1) / 2.0
        y_mean = float(np.mean(y))

        num = float(np.sum((t - t_mean) * (y - y_mean)))
        den = float(np.sum((t - t_mean) ** 2))
        slope = num / max(1e-9, den)
        intercept = y_mean - slope * t_mean

        res_sq = np.sum((y - (slope * t + intercept)) ** 2)
        residual_std = float(np.sqrt(res_sq / max(1, n - 2)))
        return slope, y_mean, residual_std

    def get_progress(self) -> float:
        if self.phase == 1:
            return 0.0
        return float(0.5 * (1.0 - math.cos(math.pi * min(1.0, max(0.0, self.pressure)))))

    def _update_schedules(self, epoch: int) -> None:
        self._progress = self.get_progress()
        if self.phase == 1:
            p1_fraction = min(1.0, float(epoch + 1) / max(1, self.min_p1_epochs))
            self.spatial_warmup = 0.10 + 0.40 * p1_fraction
        else:
            self.spatial_warmup = 0.50 + 0.50 * self._progress

    def step(
        self,
        epoch_telemetry: dict[str, float],
        epoch: int,
        val_loss: float | None = None,
    ) -> bool:
        current_rec = float(epoch_telemetry.get("l_rec", 0.0))
        current_val = float(val_loss if val_loss is not None else current_rec)
        max_corr = float(epoch_telemetry.get("dict/max_cross_corr", epoch_telemetry.get("d_max_cross_corr", 0.0)))

        self.rec_history.append(current_rec)
        self.val_history.append(current_val)
        self.corr_history.append(max_corr)

        if self.phase == 2 and current_rec < self.best_rec_loss:
            self.best_rec_loss = current_rec

        self._update_schedules(epoch)

        if len(self.rec_history) < self.cycle_window:
            return False

        window_rec = self.rec_history[-self.cycle_window:]
        rec_slope, rec_mu, rec_sigma = self._fit_ols(window_rec)

        # =============================================================
        # PHASE 1: Manifold Alignment
        # =============================================================
        if self.phase == 1:
            relative_drop_rate = (-rec_slope * self.cycle_window) / max(1e-5, rec_mu)
            if relative_drop_rate < 0.015:  # Flat exploration detected
                self.p1_plateau_count += 1
            else:
                self.p1_plateau_count = 0

            can_exit_min = epoch >= (self.min_p1_epochs - 1)
            hit_max_budget = epoch >= (self.max_p1_epochs - 1)
            sustained_plateau = self.p1_plateau_count >= self.p1_plateau_patience

            if hit_max_budget or (can_exit_min and sustained_plateau):
                self.force_phase2(epoch, rec_mu)
            return False

        # =============================================================
        # PHASE 2: Dynamic Closed-Loop Proportional Tuning
        # =============================================================
        remaining_epochs = max(1, self.total_epochs - self.p2_start_epoch)
        base_step = 1.0 / remaining_epochs

        # 1. Loss Budget & Overshoot Scaling
        dynamic_budget = max(self.best_rec_loss * self.rel_tolerance, 2.0 * rec_sigma)
        loss_overshoot = max(0.0, (current_rec - (self.best_rec_loss + dynamic_budget)) / max(1e-5, dynamic_budget))

        # 2. Dynamic Correlation Shock Tracking (Relative to local mean)
        corr_window = self.corr_history[-min(len(self.corr_history), 10):]
        corr_baseline = float(np.mean(corr_window[:-1])) if len(corr_window) > 1 else max_corr
        corr_spike = max(0.0, (max_corr - corr_baseline) / max(1e-3, corr_baseline))

        # 3. Smooth Proportional Adjustment (No binary throttling)
        panic_factor = max(0.0, (max_corr - 0.60) * 5.0)
        stress_factor = (loss_overshoot * 1.5) + (corr_spike * 2.0) + panic_factor

        if stress_factor > 0.05:
            # Scale down pressure smoothly relative to stress intensity
            pressure_delta = -base_step * min(2.0, stress_factor)
        else:
            # Healthy training: Boost progression if loss is dropping rapidly
            velocity_boost = min(1.5, max(1.0, -rec_slope * 5.0))
            pressure_delta = base_step * velocity_boost

        self.pressure = min(1.0, max(0.0, self.pressure + pressure_delta))
        self._update_schedules(epoch)

        # =============================================================
        # ADAPTIVE EARLY STOPPING
        # =============================================================
        # Dynamically lower progress gate if validation plateau is strongly established
        val_slope, val_mu, _ = self._fit_ols(self.val_history[-self.cycle_window:])
        val_flat = abs(val_slope * self.cycle_window) / max(1e-5, val_mu) < 0.005

        active_gate = 0.50 if val_flat else 0.75

        if self._progress >= active_gate:
            rel_improvement = (self.best_val_loss - current_val) / max(1e-5, self.best_val_loss)
            if rel_improvement > 1e-3:
                self.best_val_loss = current_val
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            if self.no_improve_count >= self.patience_epochs:
                return True
        else:
            if current_val < self.best_val_loss:
                self.best_val_loss = current_val
            self.no_improve_count = 0

        return False

    def force_phase2(self, epoch: int, current_baseline: float) -> None:
        if self.phase == 1:
            self.phase = 2
            self.p2_start_epoch = epoch
            self.p1_baseline_rec = float(current_baseline)
            self.best_rec_loss = float(current_baseline)
            self.best_val_loss = float("inf")
            self.pressure = 0.0
            self.no_improve_count = 0
            self._update_schedules(epoch)

class UnifiedLogger:
    """Zero-overhead logger for Gradients, Trajectory, and Hardware Memory."""
    def __init__(self, backend: str, run_name: str, log_dir: str):
        self.backend = backend.lower()
        self.writer = None

        if self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            import os
            tb_dir = os.path.join(log_dir, "tb_logs", run_name)
            self.writer = SummaryWriter(log_dir=tb_dir)
            print(f"[*] TensorBoard Logger initialized at: {tb_dir}")

    def log_metrics(self, step: int, metrics: dict[str, float]):
        """Logs a dictionary of scalar numbers to TensorBoard."""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, global_step=step)

    def log_model_telemetry(self, step: int, model: torch.nn.Module, log_histograms: bool = False):
        """Pulls the deep telemetry you built into the model and logs it."""
        if not self.writer:
            return

        # 1. Pull live deep telemetry from model (SVD, Effective Rank, Gram, Ambient, Gradients)
        if hasattr(model, 'get_deep_telemetry'):
            stats = model.get_deep_telemetry()
            self.log_metrics(step, stats)

        # 2. Add visual histograms of the weights/gradients if requested
        if log_histograms:
            for name, param in model.named_parameters():
                p_clean = name.replace('.', '/')
                if param.requires_grad:
                    self.writer.add_histogram(f"weights/{p_clean}", param.data, global_step=step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"grads/{p_clean}", param.grad, global_step=step)

    def log_checkpoint_autopsy(self, step: int, ckpt_path: str) -> dict[str, float]:
        """Loads a saved checkpoint file, performs deep dictionary autopsy, and logs metrics."""
        import torch
        import torch.nn.functional as F

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        metrics = {}

        if "decoder_weight" in state:
            w = F.normalize(state["decoder_weight"], p=2, dim=-1)
            s = torch.linalg.svdvals(w)
            eff_rank = (s.sum() ** 2) / torch.clamp((s ** 2).sum(), min=1e-9)

            gram = torch.mm(w, w.t())
            off_diag_mask = ~torch.eye(w.size(0), dtype=torch.bool)
            off_diag_vals = gram.masked_select(off_diag_mask)

            metrics["autopsy/effective_rank"] = eff_rank.item()
            metrics["autopsy/svd_sigma_1"] = s[0].item()
            metrics["autopsy/svd_sigma_2"] = s[1].item() if s.numel() > 1 else 0.0
            metrics["autopsy/max_cross_corr"] = off_diag_vals.max().item() if off_diag_vals.numel() > 0 else 0.0
            metrics["autopsy/mean_cross_corr"] = off_diag_vals.abs().mean().item() if off_diag_vals.numel() > 0 else 0.0

        if "ambient_scale" in state:
            amb_val = state["ambient_scale"].item()
            amb_pct = (1.0 / (1.0 + 2.718281828459045 ** (-amb_val * 5.0))) * 0.40 * 100.0
            metrics["autopsy/ambient_absorption_pct"] = amb_pct

        if self.writer:
            self.log_metrics(step, metrics)

        return metrics

    @staticmethod
    def get_memory_metrics(device: torch.device) -> dict[str, float]:
        """Hardware-aware memory extraction for RAM and VRAM."""
        mem = {}
        mem["memory/ram_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)

        if device.type == "cuda":
            mem["memory/cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
        elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            mem["memory/mps_allocated_gb"] = torch.mps.current_allocated_memory() / (1024 ** 3)

        return mem

    def close(self):
        """Safely shuts down the logger."""
        if self.writer:
            self.writer.flush()
            self.writer.close()



def export_latents_from_graphs(
    model: torch.nn.Module,
    graph_paths: List[Union[str, Path]],
    out_dir: Union[str, Path],
    device: torch.device,
    batch_size: int = 4096,
    k_hops: int = 2,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """
    Extracts and exports sparse CSR representations directly from graph.pt files.
    
    Guarantees:
      1. Zero halo-node duplication (only core nodes are extracted).
      2. Exact 1:1 cell index and barcode alignment with graph.pt.
      3. Embeds barcodes directly into libella_latent.npz for automated benchmark subsetting.
    """
    import warnings
    from .data import SpatialBatcher, pad_mps_shapes, pt_to_scipy_csr

    out_dir = Path(out_dir).resolve()
    sample_out_dir = out_dir / "sample_latents"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n-> Extracting & Exporting Libella Latent Representations from graph.pt...")
    model.eval()

    all_patient_csrs = []
    all_cell_metadata = []
    all_barcodes_master = []

    with torch.no_grad():
        for g_path in graph_paths:
            g_path = Path(g_path)
            data = torch.load(g_path, map_location="cpu", weights_only=False)

            X_sp = pt_to_scipy_csr(data, "x_in")
            N_cells = X_sp.shape[0]

            # Reconstruct adjacency
            e_attr = data.edge_attr.numpy()
            e_row = data.edge_index[0].numpy()
            e_col = data.edge_index[1].numpy()
            adj_sp = sp.csr_matrix((e_attr, (e_row, e_col)), shape=(N_cells, N_cells))

            patient_name = getattr(data, "patient_name", g_path.stem.replace("_graph", ""))
            
            # --- 1. Robust barcode discovery ---
            barcodes_raw = None
            for attr in ["obs_names", "barcodes", "cell_ids", "cell_names"]:
                if hasattr(data, attr) and getattr(data, attr) is not None:
                    barcodes_raw = getattr(data, attr)
                    break

            if barcodes_raw is None:
                warnings.warn(f"[!] No cell barcodes found in {g_path.name}. Using synthetic IDs.")
                barcodes = [f"{patient_name}_cell_{i}" for i in range(N_cells)]
            elif isinstance(barcodes_raw, torch.Tensor):
                barcodes = barcodes_raw.cpu().numpy().astype(str).tolist()
            elif isinstance(barcodes_raw, np.ndarray):
                barcodes = barcodes_raw.astype(str).tolist()
            else:
                barcodes = [str(b) for b in barcodes_raw]

            # Validation check
            if len(barcodes) != N_cells:
                raise ValueError(
                    f"Barcode count mismatch in {g_path.name}: {len(barcodes)} barcodes for {N_cells} cells."
                )

            # SpatialBatcher over ALL cells (shuffle=False keeps chunking deterministic)
            dummy_mask = np.ones(N_cells, dtype=bool)
            pos_coords = data.pos.numpy() if hasattr(data, "pos") and data.pos is not None else np.zeros((N_cells, 2), dtype=np.float32)
            
            batcher = SpatialBatcher(
                X=X_sp,
                adj=adj_sp,
                coords=pos_coords,
                train_mask=dummy_mask,
                val_mask=dummy_mask,
                batch_size=batch_size,
                k_hops=k_hops,
                shuffle=False,
            )

            patient_z_chunks = []
            patient_cell_indices = []

            for chunk_idx, core_idx in enumerate(batcher.chunks):
                chunk = batcher.get_chunk(chunk_idx)
                
                if hasattr(chunk["x"], "toarray"):
                    chunk_x = torch.from_numpy(chunk["x"].toarray()).to(torch.float32)
                elif not isinstance(chunk["x"], torch.Tensor):
                    chunk_x = torch.tensor(chunk["x"], dtype=torch.float32)
                else:
                    chunk_x = chunk["x"].to(torch.float32)

                adj_coo = chunk["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32)

                chunk_x = chunk_x.to(device)
                src = src.to(device)
                dst = dst.to(device)
                weights = weights.to(device)

                chunk_x, src, dst, weights = pad_mps_shapes(chunk_x, src, dst, weights)
                if device.type != "mps":
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

                # Forward inference
                forward_eval = model(chunk_x, src, dst, weights)
                z = forward_eval[1]

                # Slice ONLY core nodes (drops halo receptive field)
                local_core = chunk["local_core_idx"]
                z_core = z[local_core].detach().cpu().numpy()

                patient_z_chunks.append(sp.csr_matrix(z_core))
                patient_cell_indices.extend(core_idx)

            # Reconstruct original graph order (0..N_cells-1)
            patient_csr_unordered = sp.vstack(patient_z_chunks)
            order_map = np.argsort(patient_cell_indices)
            patient_csr = patient_csr_unordered[order_map]

            # Save Patient-Specific Artifacts
            p_latent_file = sample_out_dir / f"{patient_name}_latent.npz"
            p_meta_file = sample_out_dir / f"{patient_name}_cells.csv.gz"

            np.savez_compressed(
                p_latent_file,
                data=patient_csr.data,
                indices=patient_csr.indices,
                indptr=patient_csr.indptr,
                shape=patient_csr.shape,
                barcodes=np.array(barcodes, dtype=str),
            )
            pd.DataFrame({"cell_id": barcodes, "patient_name": patient_name}).to_csv(
                p_meta_file, index=False, compression="gzip"
            )
            print(f"  ↳ Saved patient {patient_name} -> {p_latent_file.name} ({patient_csr.shape[0]} cells)")

            all_patient_csrs.append(patient_csr)
            all_barcodes_master.extend(barcodes)
            for cid in barcodes:
                all_cell_metadata.append({"cell_id": cid, "patient_name": patient_name})

            del data, X_sp, adj_sp, batcher
            gc.collect()

    # --- 2. Save Master CSR along with Barcodes into single NPZ archive ---
    master_csr = sp.vstack(all_patient_csrs)
    master_barcodes_arr = np.array(all_barcodes_master, dtype=str)
    
    assert master_csr.shape[0] == len(master_barcodes_arr), "Master CSR and Barcode count mismatch!"
    
    master_latent_path = out_dir / "libella_latent.npz"
    np.savez_compressed(
        master_latent_path,
        data=master_csr.data,
        indices=master_csr.indices,
        indptr=master_csr.indptr,
        shape=master_csr.shape,
        barcodes=master_barcodes_arr,
    )

    master_meta_df = pd.DataFrame(all_cell_metadata)
    master_meta_path = out_dir / "libella_cell_metadata.csv.gz"
    master_meta_df.to_csv(master_meta_path, index=False, compression="gzip")

    print(f"\n[✓] Master export complete -> {master_latent_path}")
    print(f"    Shape: {master_csr.shape[0]:,} cells × {master_csr.shape[1]:,} latents (Barcodes embedded)")

    return master_csr, master_meta_df

