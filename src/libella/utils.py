"""Utility functions for Libella pipeline operations."""

import ast
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class PhaseTracker:
    """
    High-velocity training governor that ramps to 100% squeeze rapidly,
    then holds the network at max pressure for fine-tuning.
    
    Termination only occurs if:
      1. Val loss diverges terribly (>15% above best), OR
      2. The model has completed a burn-in at 100% squeeze AND budget loss has flatlined.
    """

    def __init__(
        self,
        total_epochs: int = 50,
        surge_tolerance: float = 0.50,        # Allow up to 50% loss fluctuation during pressure changes
        divergence_threshold: float = 0.25,   # Soak early stop: val loss degrades >25% above hard baseline
        ramp_divergence_slack: float = 1.00,  # Ramp early stop: allow up to 100% surge during compression
    ) -> None:
        self.phase: int = 1
        self.total_epochs: int = max(1, total_epochs)
        self.surge_tolerance: float = surge_tolerance
        self.divergence_threshold: float = divergence_threshold
        self.ramp_divergence_slack: float = ramp_divergence_slack

        # --- Phase 1 Horizons ---
        self.min_p1_epochs: int = max(2, int(self.total_epochs * 0.05))
        self.max_p1_epochs: int = max(self.min_p1_epochs + 1, int(self.total_epochs * 0.08))
        self.p1_plateau_patience: int = 2
        self.p1_plateau_count: int = 0

        # --- Dynamic Window Sizing & Max-Squeeze Horizons ---
        self.cycle_window: int = max(3, int(self.total_epochs * 0.06))
        # Soak at 100% squeeze for at least 25% of training before convergence stopping unlocks
        self.min_max_squeeze_epochs: int = max(8, int(self.total_epochs * 0.25))
        self.max_squeeze_epochs_count: int = 0
        self.patience_epochs: int = max(6, int(self.total_epochs * 0.15))

        # --- Metric Histories ---
        self.rec_history: list[float] = []
        self.val_history: list[float] = []
        self.budget_history: list[float] = []
        self.corr_history: list[float] = []
        self.align_history: list[float] = []

        self.best_rec_loss: float = float("inf")
        self.best_val_loss: float = float("inf")
        self.hard_baseline_val_loss: float = float("inf")
        self.hard_baseline_established: bool = False
        self.p1_baseline_rec: float = 0.0
        self.p2_start_epoch: int = 0

        # --- Governor State ---
        self.pressure: float = 0.0
        self.no_improve_count: int = 0
        self._progress: float = 0.0
        self.spatial_warmup: float = 0.10

    @staticmethod
    def _fit_ols(series: list[float]) -> tuple[float, float, float]:
        n = len(series)
        if n < 2:
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
        residual_std = float(np.sqrt(res_sq / max(1, n - 2))) if n > 2 else 0.01
        return slope, y_mean, residual_std

    def get_global_progress(self, epoch: int, step_fraction: float = 0.0) -> float:
        return min(1.0, max(0.0, (epoch + step_fraction) / float(self.total_epochs)))

    def get_squeeze_progress(self) -> float:
        if self.phase == 1:
            return 0.0
        return float(min(1.0, max(0.0, self.pressure)))

    def get_schedules(self, epoch: int, step_fraction: float = 0.0) -> dict[str, float]:
        global_prog = self.get_global_progress(epoch, step_fraction)
        squeeze_prog = self.get_squeeze_progress()

        # Spatial Warm-up: 0.0 for first 10% of run, then linear ramp to 1.0 from [0.10 -> 0.50]
        if global_prog < 0.10:
            spatial_prog = 0.0
        else:
            spatial_prog = min(1.0, (global_prog - 0.10) / 0.40)

        if self.phase == 1:
            gamma_prog = 0.56
        else:
            # Cosine-smoothed hardness ramp to prevent sudden cliff at gamma=0.99
            smooth_squeeze = 0.5 * (1.0 - math.cos(math.pi * squeeze_prog))
            gamma_prog = 0.56 + 0.44 * smooth_squeeze

        return {
            "global_progress": global_prog,
            "squeeze_progress": squeeze_prog,
            "spatial_progress": spatial_prog,
            "gamma_progress": gamma_prog,
        }

    def _update_schedules(self, epoch: int) -> None:
        schedules = self.get_schedules(epoch)
        self._progress = schedules["squeeze_progress"]
        self.spatial_warmup = schedules["spatial_progress"]

    def step(
        self,
        epoch_telemetry: dict[str, float],
        epoch: int,
        val_loss: float | None = None,
    ) -> bool:
        current_rec = float(
            epoch_telemetry.get("l_rec", epoch_telemetry.get("l_recon_x", epoch_telemetry.get("l_recon", 0.0)))
        )
        current_val = float(val_loss if val_loss is not None else current_rec)
        current_budget = float(
            epoch_telemetry.get("l_budget", epoch_telemetry.get("loss/budget", epoch_telemetry.get("l_sparse", 0.0)))
        )

        max_corr = float(epoch_telemetry.get("dict/max_cross_corr", epoch_telemetry.get("d_max_cross_corr", 0.0)))
        min_align = float(epoch_telemetry.get("dict/alignment_min", epoch_telemetry.get("d_alignment_min", 1.0)))

        self.rec_history.append(current_rec)
        self.val_history.append(current_val)
        self.budget_history.append(current_budget)
        self.corr_history.append(max_corr)
        self.align_history.append(min_align)

        if self.phase == 2 and current_rec < self.best_rec_loss:
            self.best_rec_loss = current_rec

        if current_val < self.best_val_loss:
            self.best_val_loss = current_val

        self._update_schedules(epoch)

        if len(self.rec_history) < self.cycle_window:
            return False

        window_rec = self.rec_history[-self.cycle_window:]
        rec_slope, rec_mu, _ = self._fit_ols(window_rec)

        # =============================================================
        # PHASE 1: Rapid Manifold Alignment
        # =============================================================
        if self.phase == 1:
            relative_drop_rate = (-rec_slope * self.cycle_window) / max(1e-5, rec_mu)
            if relative_drop_rate < 0.02:
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
        # PHASE 2: Fast-Track to Max Pressure (1.0)
        # =============================================================
        target_ramp_epochs = max(1, int((self.total_epochs - self.p2_start_epoch) * 0.50))
        base_step = 1.0 / target_ramp_epochs

        # Surge brake: only pause ramp if reconstruction loss jumps >25%
        surge_threshold = self.best_rec_loss * (1.0 + self.surge_tolerance)
        if current_rec <= surge_threshold:
            self.pressure = min(1.0, self.pressure + base_step)

        self._update_schedules(epoch)

        if self._progress >= 1.0:
            self.max_squeeze_epochs_count += 1

        # =============================================================
        # SOAK MODE TERMINATION GOVERNOR
        # =============================================================

        # 1. Recalibrate baseline once entering full hard-sparsity regime (pressure == 1.0)
        if self._progress >= 1.0:
            if not self.hard_baseline_established:
                self.hard_baseline_val_loss = current_val
                self.best_val_loss = current_val
                self.best_rec_loss = current_rec
                self.hard_baseline_established = True
            elif current_val < self.hard_baseline_val_loss:
                self.hard_baseline_val_loss = current_val

        # 2. Catastrophic Divergence Guard
        if self._progress < 1.0:
            # During compression ramp: allow wider margin to absorb discretization shock
            if current_val > self.best_val_loss * (1.0 + self.ramp_divergence_slack):
                return True
        else:
            # During max squeeze soak: evaluate divergence against hard baseline only
            if current_val > self.hard_baseline_val_loss * (1.0 + self.divergence_threshold):
                return True

        # 3. Budget Flatline Gate: Only evaluate convergence when 100% squeeze has soaked
        if self._progress >= 1.0 and self.max_squeeze_epochs_count >= self.min_max_squeeze_epochs:
            window_budget = self.budget_history[-self.cycle_window:]
            b_slope, b_mu, _ = self._fit_ols(window_budget)
            budget_relative_velocity = abs(b_slope * self.cycle_window) / max(1e-5, b_mu)
            
            # Budget loss is stationary (< 0.5% change over window)
            budget_flat = budget_relative_velocity < 0.005

            rel_val_improvement = (self.hard_baseline_val_loss - current_val) / max(1e-5, self.hard_baseline_val_loss)
            val_stagnant = rel_val_improvement <= 1e-3

            if budget_flat and val_stagnant:
                self.no_improve_count += 1
            else:
                self.no_improve_count = 0

            if self.no_improve_count >= self.patience_epochs:
                return True

        return False

    def force_phase2(self, epoch: int, current_baseline: float) -> None:
        if self.phase == 1:
            self.phase = 2
            self.p2_start_epoch = epoch
            self.p1_baseline_rec = float(current_baseline)
            self.best_rec_loss = float(current_baseline)
            self.best_val_loss = float("inf")
            self.hard_baseline_val_loss = float("inf")
            self.hard_baseline_established = False
            self.pressure = 0.0
            self.max_squeeze_epochs_count = 0
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
    import gc
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

                # Direct encoder inference (bypasses decoder reconstruction FLOPs)
                z_contextual, *_ = model.encode(chunk_x, src, dst, weights)

                # Slice ONLY core nodes (drops halo receptive field)
                local_core = chunk["local_core_idx"]
                z_core = z_contextual[local_core].detach().cpu().numpy()

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


class SafePadeActivation(nn.Module):
    """
    Direct Safe-Padé rational activation unit (Yin & Yu, 2026).
    Inputs are softly bounded to [-1, 1] to prevent runaway polynomial oscillation (Runge's Phenomenon).
    """
    def __init__(self, p_deg: int = 3, q_deg: int = 2) -> None:
        super().__init__()
        # Initialize near Identity function (f(x) = x)
        p_init = torch.zeros(p_deg + 1)
        p_init[1] = 1.0  
        
        self.p_coeffs = nn.Parameter(p_init + torch.randn(p_deg + 1) * 0.01)
        self.q_coeffs = nn.Parameter(torch.abs(torch.randn(q_deg) * 0.01))
        
        # RSAE C_in scale to map raw scores into the safe [-1, 1] polynomial design space
        self.c_in = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softly bind inputs into [-1, 1] to prevent x^3 from exploding/inverting
        # This forces Padé to behave purely as a smooth shape inside its stable domain
        x_safe = torch.tanh(x / F.softplus(self.c_in))
        
        p_val = self.p_coeffs[0]
        for i in range(1, len(self.p_coeffs)):
            p_val = p_val + self.p_coeffs[i] * (x_safe ** i)
            
        q_val = 1.0
        abs_x = torch.abs(x_safe)
        for i in range(len(self.q_coeffs)):
            q_val = q_val + torch.abs(self.q_coeffs[i]) * (abs_x ** (i + 1))
            
        return p_val / q_val

class ExactLaPruneFunction(torch.autograd.Function):
    """
    Exact-budget differentiable Top-K layer with normalized second-moment hardness control (Antczak et al., 2026).
    Executes a 2D Newton-Raphson root solve in forward and an exact 2x2 IFT VJP in backward.
    """
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        k_target: torch.Tensor,
        gamma: torch.Tensor | float,
        max_iters: int = 25,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        B, N = scores.shape
        device = scores.device
        dtype = scores.dtype

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, device=device, dtype=dtype)
        if gamma.dim() == 0:
            gamma = gamma.view(1, 1).expand(B, 1)

        a = k_target / float(N)
        beta = a + (1.0 - a) * gamma
        target_m1 = k_target
        target_m2 = beta * k_target

        mean_s = scores.mean(dim=-1, keepdim=True)
        std_s = scores.std(dim=-1, keepdim=True).clamp(min=1e-4)

        b = mean_s + std_s * torch.clamp(1.0 - 2.0 * a, min=-2.5, max=2.5)
        tau = torch.zeros((B, 1), device=device, dtype=dtype)

        for _ in range(max_iters):
            t = torch.exp(tau).clamp(min=1e-5, max=100.0)
            u = (scores - b) / t
            p = torch.where(u <= 0.0, 0.5 * torch.exp(u), 1.0 - 0.5 * torch.exp(-u))
            q = 0.5 * torch.exp(-torch.abs(u))

            F1 = p.sum(dim=-1, keepdim=True) - target_m1
            F2 = (p ** 2).sum(dim=-1, keepdim=True) - target_m2

            if torch.max(F1.abs()) < tol and torch.max(F2.abs()) < tol:
                break

            inv_t = 1.0 / t
            J11 = -inv_t * q.sum(dim=-1, keepdim=True)
            J12 = -(u * q).sum(dim=-1, keepdim=True)
            J21 = -2.0 * inv_t * (p * q).sum(dim=-1, keepdim=True)
            J22 = -2.0 * (u * p * q).sum(dim=-1, keepdim=True)

            det = J11 * J22 - J12 * J21
            det_stable = torch.where(det.abs() < 1e-7, torch.sign(det + 1e-7) * 1e-7, det)

            delta_b = -(J22 * F1 - J12 * F2) / det_stable
            delta_tau = -(-J21 * F1 + J11 * F2) / det_stable

            b = b + delta_b.clamp(min=-5.0, max=5.0)
            tau = (tau + delta_tau.clamp(min=-2.0, max=2.0)).clamp(min=-10.0, max=5.0)

        t_final = torch.exp(tau).clamp(min=1e-5, max=100.0)
        u_final = (scores - b) / t_final
        p_final = torch.where(u_final <= 0.0, 0.5 * torch.exp(u_final), 1.0 - 0.5 * torch.exp(-u_final))

        ctx.save_for_backward(p_final, t_final, scores, b, k_target, gamma)
        return p_final

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        p, t, scores, b, k_target, gamma = ctx.saved_tensors
        B, N = scores.shape

        u = (scores - b) / t
        q = 0.5 * torch.exp(-torch.abs(u))
        inv_t = 1.0 / t

        v_b = -inv_t * (grad_output * q).sum(dim=-1, keepdim=True)
        v_tau = -(grad_output * u * q).sum(dim=-1, keepdim=True)

        J11 = -inv_t * q.sum(dim=-1, keepdim=True)
        J12 = -(u * q).sum(dim=-1, keepdim=True)
        J21 = -2.0 * inv_t * (p * q).sum(dim=-1, keepdim=True)
        J22 = -2.0 * (u * p * q).sum(dim=-1, keepdim=True)

        det = J11 * J22 - J12 * J21
        det_stable = torch.where(det.abs() < 1e-7, torch.sign(det + 1e-7) * 1e-7, det)

        lambda_1 = (v_b * J22 - v_tau * J21) / det_stable
        lambda_2 = (-v_b * J12 + v_tau * J11) / det_stable

        grad_scores = inv_t * q * (grad_output - lambda_1 - 2.0 * lambda_2 * p)

        a = k_target / float(N)
        c_a = 2.0 * a + (1.0 - 2.0 * a) * gamma
        grad_k = lambda_1 + c_a * lambda_2

        grad_gamma = lambda_2 * (1.0 - a) * k_target

        if ctx.needs_input_grad[2]:
            return grad_scores, grad_k, grad_gamma, None, None
        return grad_scores, grad_k, None, None, None