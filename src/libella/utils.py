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

from .config import NOISE_REGEX

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
from typing import Dict, List, Optional, Tuple


class PhaseTracker:
    def __init__(
        self,
        cycle_window: int = 8,        # Cut window from 12 -> 6 (responds 2x faster)
        target_pw: float = 70.0,
        rel_tolerance: float = 0.1,  # 6% reconstruction budget ceiling
        max_p1_epochs: int = 20,      
    ) -> None:
        self.phase: int = 1
        self.cycle_window: int = cycle_window
        self.target_pw: float = target_pw
        self.rel_tolerance: float = rel_tolerance
        self.max_p1_epochs: int = max_p1_epochs
        
        self.rec_history: list[float] = []
        self.pw_history: list[float] = []
        
        self.best_rec_loss: float = float("inf")
        self.p1_baseline_rec: float | None = None
        
        # Faster ramp dynamics
        self.pressure: float = 0.0
        self.squeeze_momentum: float = 0.04   # Step size 0.05 (ramps in ~20 epochs instead of 80)
        self.breathing_cooldown: int = 0
        
        self.saturation_streak: int = 0
        self.required_saturation_streak: int = 4

    @staticmethod
    def _fit_ols(series: List[float]) -> Tuple[float, float, float]:
        """Calculates slope, mean, and residual standard deviation."""
        n = len(series)
        if n < 3:
            return 0.0, float(series[-1]) if series else 0.0, 0.01

        t_mean = (n - 1) / 2.0
        y_mean = sum(series) / n

        num = sum((i - t_mean) * (series[i] - y_mean) for i in range(n))
        den = sum((i - t_mean) ** 2 for i in range(n))
        slope = num / max(1e-9, den)
        intercept = y_mean - slope * t_mean

        res_sq = sum((series[i] - (slope * i + intercept)) ** 2 for i in range(n))
        residual_std = math.sqrt(res_sq / max(1, n - 2))

        return slope, y_mean, residual_std

    def get_progress(self) -> float:
        """Returns smooth Cosine S-curve progress in [0.0, 1.0]."""
        if self.phase == 1:
            return 0.0
        # S-curve smooths out micro-adjustments
        return 0.5 * (1.0 - math.cos(math.pi * self.pressure))

    def step(self, epoch_telemetry: Dict[str, float], epoch: int) -> bool:
        """Evaluates epoch telemetry, adjusting squeeze pressure elastically."""
        current_rec = float(epoch_telemetry.get("l_rec", 0.0))
        current_pw = float(epoch_telemetry.get("p_w", 0.0))

        self.rec_history.append(current_rec)
        self.pw_history.append(current_pw)

        # Track absolute lowest reconstruction loss achieved
        if current_rec < self.best_rec_loss:
            self.best_rec_loss = current_rec

        if len(self.rec_history) < self.cycle_window:
            return False

        window_rec = self.rec_history[-self.cycle_window:]
        window_pw = self.pw_history[-self.cycle_window:]

        rec_slope, rec_mu, rec_sigma = self._fit_ols(window_rec)
        pw_slope, pw_mu, _ = self._fit_ols(window_pw)

        # -----------------------------------------------------------------
        # PHASE 1: Manifold Discovery & Plateau Detection
        # -----------------------------------------------------------------
        if self.phase == 1:
            # 1. Hard cutoff: Force Phase 2 at max_p1_epochs regardless
            if epoch >= self.max_p1_epochs:
                self.force_phase2(epoch, rec_mu)
                return False
                
            # 2. Faster slope trigger (transition when relative drop < 0.8% per epoch)
            relative_drop_rate = (-rec_slope * self.cycle_window) / max(1e-5, rec_mu)
            if relative_drop_rate < 0.008:
                self.force_phase2(epoch, rec_mu)
            return False

        # -----------------------------------------------------------------
        # PHASE 2: Elastic Squeeze & Breathe Dynamic Governor
        # -----------------------------------------------------------------
        if self.phase == 2:
            # Dynamic Variance Ceiling: minimum 6% budget or 2.5x oscillation noise
            dynamic_budget = max(self.best_rec_loss * self.rel_tolerance, 2.5 * rec_sigma)
            loss_ceiling = self.best_rec_loss + dynamic_budget

            overshoot = current_rec - loss_ceiling

            # SENSE FRAGILITY & BREATHE: Loss exceeded tolerance band
            if overshoot > 0.0:
                # Severity ratio of the loss spike
                severity = min(2.0, overshoot / max(1e-5, dynamic_budget))
                
                # Proportional elastic release: drops pressure rapidly to relieve strain
                release_amount = 0.04 * severity
                self.pressure = max(0.10, self.pressure - release_amount)
                self.squeeze_momentum = 0.008  # Reset momentum to cautious
                self.breathing_cooldown = 3    # Hold pressure for 2 epochs to recover
                self.saturation_streak = 0
            
            # SAFE TO SQUEEZE: Loss is healthy inside the manifold envelope
            else:
                if self.breathing_cooldown > 0:
                    self.breathing_cooldown -= 1
                else:
                    # Gradually accelerate squeeze momentum when stable
                    self.squeeze_momentum = min(0.035, self.squeeze_momentum + 0.002)
                    self.pressure = min(1.0, self.pressure + self.squeeze_momentum)

            # -------------------------------------------------------------
            # STRICT TERMINATION AUDIT (Prevents premature exit at < 70% P_W)
            # -------------------------------------------------------------
            # Only consider stopping if:
            # 1. Full pressure is deployed (pressure >= 0.95)
            # 2. P_W reached the target regime (>= 72.0%)
            # 3. P_W slope is completely flat (< +0.05% / epoch over 12 epochs)
            # 4. Sustained over a full 8-epoch oscillation streak
            if self.pressure >= 0.95 and current_pw >= (self.target_pw - 3.0):
                if pw_slope < 0.05:
                    self.saturation_streak += 1
                else:
                    self.saturation_streak = max(0, self.saturation_streak - 1)

                if self.saturation_streak >= self.required_saturation_streak:
                    return True
            else:
                self.saturation_streak = 0

        return False

    def force_phase2(self, epoch: int, current_baseline: float) -> None:
        """Transitions tracker to Phase 2."""
        if self.phase == 1:
            self.phase = 2
            self.p1_baseline_rec = current_baseline
            if self.best_rec_loss == float("inf") or current_baseline < self.best_rec_loss:
                self.best_rec_loss = current_baseline




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

        # 1. Pull the custom telemetry dictionary you built in model.py
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

