"""Utility functions for Libella pipeline operations."""

import ast
import re
from pathlib import Path

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
    """Coordinates training horizons, spatial warmup, and Pareto early stopping."""

    def __init__(
        self,
        total_epochs: int = 50,
        rel_tolerance: float = 0.08,
    ) -> None:
        self.phase: int = 1
        self.total_epochs: int = max(1, total_epochs)
        self.rel_tolerance: float = rel_tolerance

        # --- Phase 1 Horizons ---
        self.min_p1_epochs: int = max(
            getattr(cfg, "min_p1_epochs", 6),
            int(self.total_epochs * getattr(cfg, "p1_min_ratio", 0.12)),
        )
        self.max_p1_epochs: int = max(
            self.min_p1_epochs + 2,
            int(self.total_epochs * getattr(cfg, "p1_max_ratio", 0.25)),
        )
        self.p1_plateau_patience: int = 3
        self.p1_plateau_count: int = 0

        # --- Window & Horizon Sizing ---
        self.cycle_window: int = max(
            3, int(self.total_epochs * getattr(cfg, "tracker_window_ratio", 0.08))
        )
        self.patience_epochs: int = max(
            5, int(self.total_epochs * getattr(cfg, "patience_ratio", 0.12))
        )

        # --- Histories & Baselines ---
        self.rec_history: list[float] = []
        self.val_history: list[float] = []

        self.best_rec_loss: float = float("inf")
        self.best_val_loss: float = float("inf")
        self.p1_baseline_rec: float = 0.0
        self.p2_start_epoch: int = 0

        # --- Dynamic Governor State ---
        self.pressure: float = 0.0
        self.breathing_cooldown: int = 0
        self.no_improve_count: int = 0

        # --- Schedule State ---
        self._progress: float = 0.0
        self.spatial_warmup: float = 0.10

    @staticmethod
    def _fit_ols(series: list[float]) -> tuple[float, float, float]:
        """Calculates trend slope, mean, and residual standard deviation."""
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
        """Returns smooth Cosine S-curve progress in [0.0, 1.0]."""
        if self.phase == 1:
            return 0.0
        return float(0.5 * (1.0 - math.cos(math.pi * min(1.0, max(0.0, self.pressure)))))

    def _update_schedules(self, epoch: int) -> None:
        """Centralized schedule generator for spatial warmup and regularization."""
        self._progress = self.get_progress()

        if self.phase == 1:
            # Linear exploration ramp [0.10 -> 0.50]
            p1_fraction = min(1.0, float(epoch + 1) / max(1, self.min_p1_epochs))
            self.spatial_warmup = 0.10 + 0.40 * p1_fraction
        else:
            # S-curve consolidation ramp [0.50 -> 1.00]
            self.spatial_warmup = 0.50 + 0.50 * self._progress

    def step(
        self,
        epoch_telemetry: dict[str, float],
        epoch: int,
        val_loss: float | None = None,
    ) -> bool:
        """Evaluates epoch telemetry, updates governor pressure, and checks early stopping."""
        current_rec = float(epoch_telemetry.get("l_rec", 0.0))
        current_val = float(val_loss if val_loss is not None else current_rec)
        
        # Use active telemetry keys for orthogonality/loss pressure
        l_ort = float(epoch_telemetry.get("l_ort", 0.0))

        self.rec_history.append(current_rec)
        self.val_history.append(current_val)

        if self.phase == 2 and current_rec < self.best_rec_loss:
            self.best_rec_loss = current_rec

        self._update_schedules(epoch)

        if len(self.rec_history) < self.cycle_window:
            return False

        window_rec = self.rec_history[-self.cycle_window:]
        rec_slope, rec_mu, rec_sigma = self._fit_ols(window_rec)

        # -------------------------------------------------------------
        # PHASE 1: Manifold Alignment & Plateau Discovery
        # -------------------------------------------------------------
        if self.phase == 1:
            drop_tol = getattr(cfg, "p1_drop_tol", 0.010)
            relative_drop_rate = (-rec_slope * self.cycle_window) / max(1e-5, rec_mu)

            if relative_drop_rate < drop_tol:
                self.p1_plateau_count += 1
            else:
                self.p1_plateau_count = 0

            can_exit_min = epoch >= (self.min_p1_epochs - 1)
            hit_max_budget = epoch >= (self.max_p1_epochs - 1)
            sustained_plateau = self.p1_plateau_count >= self.p1_plateau_patience

            if hit_max_budget or (can_exit_min and sustained_plateau):
                self.force_phase2(epoch, rec_mu)

            return False

        # -------------------------------------------------------------
        # PHASE 2: Elastic Governor & Regularization Consolidation
        # -------------------------------------------------------------
        remaining_epochs = max(1, self.total_epochs - self.p2_start_epoch)
        base_step = 1.0 / remaining_epochs

        dynamic_budget = max(self.best_rec_loss * self.rel_tolerance, 2.0 * rec_sigma)
        loss_ceiling = self.best_rec_loss + dynamic_budget
        overshoot = current_rec - loss_ceiling

        # Manifold stress detection via reconstruction overshoot or orthogonality spikes
        ortho_stress = l_ort > getattr(cfg, "ortho_stress_threshold", 5.0)

        if overshoot > 0.0 or ortho_stress:
            severity = 1.5 if ortho_stress else min(2.0, overshoot / max(1e-5, dynamic_budget))
            self.pressure = max(0.0, self.pressure - (base_step * 1.0 * severity))
            self.breathing_cooldown = max(1, int(self.cycle_window * 0.5))
        else:
            if self.breathing_cooldown > 0:
                self.breathing_cooldown -= 1
            else:
                self.pressure = min(1.0, self.pressure + base_step)

        self._update_schedules(epoch)

        # -------------------------------------------------------------
        # SCALE-INVARIANT PARETO EARLY STOPPING
        # -------------------------------------------------------------
        min_progress = getattr(cfg, "min_stop_progress", 0.85)
        rel_tol = getattr(cfg, "early_stop_rel_tol", 1e-3)

        if self._progress >= min_progress:
            improvement = (self.best_val_loss - current_val) / max(1e-5, self.best_val_loss)
            if improvement > rel_tol:
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
        """Transitions tracker from Discovery (Phase 1) to Consolidation (Phase 2)."""
        if self.phase == 1:
            self.phase = 2
            self.p2_start_epoch = epoch
            self.p1_baseline_rec = float(current_baseline)
            self.best_rec_loss = float(current_baseline)
            self.best_val_loss = float("inf")
            self.pressure = 0.0
            self.breathing_cooldown = 0
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
      3. Outputs master and per-patient .npz matrices and .csv.gz metadata.
    """
    from .data import SpatialBatcher, pad_mps_shapes, pt_to_scipy_csr

    out_dir = Path(out_dir).resolve()
    sample_out_dir = out_dir / "sample_latents"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n-> Extracting & Exporting Libella Latent Representations from graph.pt...")
    model.eval()

    all_patient_csrs = []
    all_cell_metadata = []

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
            barcodes_raw = getattr(data, "barcodes", getattr(data, "cell_ids", None))
            if barcodes_raw is None:
                barcodes = [f"{patient_name}_cell_{i}" for i in range(N_cells)]
            elif isinstance(barcodes_raw, np.ndarray):
                barcodes = barcodes_raw.tolist()
            else:
                barcodes = list(barcodes_raw)

            # SpatialBatcher over ALL cells (shuffle=False to keep order deterministic)
            dummy_mask = np.ones(N_cells, dtype=bool)
            batcher = SpatialBatcher(
                X=X_sp,
                adj=adj_sp,
                coords=data.pos.numpy() if hasattr(data, "pos") else np.zeros((N_cells, 2)),
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
                
                # Tensorize input expression
                if hasattr(chunk["x"], "toarray"):
                    chunk_x = torch.from_numpy(chunk["x"].toarray()).to(torch.float32)
                elif not isinstance(chunk["x"], torch.Tensor):
                    chunk_x = torch.tensor(chunk["x"], dtype=torch.float32)
                else:
                    chunk_x = chunk["x"].to(torch.float32)

                # Tensorize edges
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

                # CRITICAL: Slice ONLY local core nodes (drops 2-hop halo neighbors)
                local_core = chunk["local_core_idx"]
                z_core = z[local_core].detach().cpu().numpy()

                patient_z_chunks.append(sp.csr_matrix(z_core))
                patient_cell_indices.extend(core_idx)

            # Assemble patient CSR in exact original graph order (0..N_cells-1)
            patient_csr_unordered = sp.vstack(patient_z_chunks)
            order_map = np.argsort(patient_cell_indices)
            patient_csr = patient_csr_unordered[order_map]

            # Save Patient-Specific Artifacts
            p_latent_file = sample_out_dir / f"{patient_name}_latent.npz"
            p_meta_file = sample_out_dir / f"{patient_name}_cells.csv.gz"

            sp.save_npz(p_latent_file, patient_csr)
            pd.DataFrame({"cell_id": barcodes, "patient_name": patient_name}).to_csv(
                p_meta_file, index=False, compression="gzip"
            )
            print(f"  ↳ Saved patient {patient_name} -> {p_latent_file.name} ({patient_csr.shape[0]} cells)")

            all_patient_csrs.append(patient_csr)
            for cid in barcodes:
                all_cell_metadata.append({"cell_id": cid, "patient_name": patient_name})

            del data, X_sp, adj_sp, batcher
            gc.collect()

    # Save Master Cohort Artifacts
    master_csr = sp.vstack(all_patient_csrs)
    master_latent_path = out_dir / "libella_latent.npz"
    sp.save_npz(master_latent_path, master_csr)

    master_meta_df = pd.DataFrame(all_cell_metadata)
    master_meta_path = out_dir / "libella_cell_metadata.csv.gz"
    master_meta_df.to_csv(master_meta_path, index=False, compression="gzip")

    print(f"\n[✓] Master export complete -> {master_latent_path}")
    print(f"    Shape: {master_csr.shape[0]:,} cells × {master_csr.shape[1]:,} latents")

    return master_csr, master_meta_df

