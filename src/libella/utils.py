"""Utility functions for Libella pipeline operations."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import math
from typing import Dict, List, Optional, Tuple

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



class PhaseTracker:
    """Fast, Variance-Aware Statistical Controller for Spatial GNN Distillation.
    
    Uses Online Ordinary Least Squares (OLS) Signal-to-Noise Ratio (SNR) and
    dynamic variance bands (2*sigma) to detect convergence precisely amidst
    multi-epoch loss oscillations.
    """

    def __init__(self, window_size: int = 8) -> None:
        self.phase: int = 1
        self.window_size: int = window_size
        
        # Raw historical telemetry buffers
        self.raw_rec_history: List[float] = []
        self.raw_pw_history: List[float] = []
        
        # Historical baseline tracking
        self.best_rec_loss: float = float("inf")
        self.p1_baseline_rec: Optional[float] = None
        self.internal_progress: float = 0.0
        
        # Phase 2 Governor Dynamics
        self.step_size: float = 0.04  # ~25 epochs to full pressure
        
        # Termination Stability Counter
        self.termination_streak: int = 0
        self.required_term_streak: int = 2

    @staticmethod
    def _compute_ols_stats(series: List[float]) -> Tuple[float, float, float]:
        """Fits OLS linear regression y = m*t + c and computes residual variance.
        
        Returns:
            slope (m): Rate of change per step.
            mean (mu): Average value in window.
            residual_std (sigma): Standard deviation of deviations from regression line.
        """
        n = len(series)
        if n < 3:
            return 0.0, float(series[-1]) if series else 0.0, 1.0

        # Fast vector math
        t = list(range(n))
        t_mean = (n - 1) / 2.0
        y_mean = sum(series) / n

        num = sum((t[i] - t_mean) * (series[i] - y_mean) for i in range(n))
        den = sum((t[i] - t_mean) ** 2 for i in range(n))
        
        slope = num / max(1e-9, den)
        intercept = y_mean - slope * t_mean

        # Compute residual standard deviation (noise floor after removing trend)
        res_sq_sum = sum((series[i] - (slope * t[i] + intercept)) ** 2 for i in range(n))
        residual_std = math.sqrt(res_sq_sum / max(1, n - 2))

        return slope, y_mean, residual_std

    def get_progress(self) -> float:
        """Returns smooth Cosine S-curve progress in [0.0, 1.0]."""
        if self.phase == 1:
            return 0.0
        return 0.5 * (1.0 - math.cos(math.pi * self.internal_progress))

    def step(self, epoch_telemetry: Dict[str, float], epoch: int) -> bool:
        """Evaluates training telemetry each epoch with variance-adjusted statistics."""
        current_rec = float(epoch_telemetry.get("l_rec", 0.0))
        current_pw = float(epoch_telemetry.get("p_w", 0.0))

        self.raw_rec_history.append(current_rec)
        self.raw_pw_history.append(current_pw)

        if current_rec < self.best_rec_loss:
            self.best_rec_loss = current_rec

        # Need at least window_size points to compute reliable statistics
        if len(self.raw_rec_history) < self.window_size:
            return False

        # Extract active rolling window
        rec_window = self.raw_rec_history[-self.window_size:]
        pw_window = self.raw_pw_history[-self.window_size:]

        # Fit OLS trends to both loss and topic purity
        rec_slope, rec_mu, rec_noise_std = self._compute_ols_stats(rec_window)
        pw_slope, pw_mu, _ = self._compute_ols_stats(pw_window)

        # Total expected loss drop over the window
        expected_drop = -rec_slope * self.window_size
        
        # Descent Signal-to-Noise Ratio (SNR)
        # Ratio of systematic downward progress to random noise amplitude
        descent_snr = expected_drop / max(1e-5, rec_noise_std)

        # -----------------------------------------------------------------
        # PHASE 1: Variance-Adjusted Manifold Discovery
        # -----------------------------------------------------------------
        if self.phase == 1:
            # Transition condition:
            # 1. Downward velocity has dropped below 40% of the noise amplitude (SNR < 0.40)
            # 2. Or absolute relative drop over entire window is < 0.5%
            relative_drop = expected_drop / max(1.0, rec_mu)
            
            if descent_snr < 0.40 or relative_drop < 0.005:
                self.force_phase2(epoch, rec_mu)
            return False

        # -----------------------------------------------------------------
        # PHASE 2: Dynamic Variance-Gated Squeezing
        # -----------------------------------------------------------------
        if self.phase == 2:
            # Dynamic Upper Noise Ceiling: Best recorded loss + 2 * sigma noise floor
            # Automatically expands or contracts with the current epoch's variance
            dynamic_tolerance = max(self.best_rec_loss * 0.02, 2.0 * rec_noise_std)
            loss_ceiling = self.best_rec_loss + dynamic_tolerance

            # If loss breaks through the 2*sigma boundary, ease pressure
            if current_rec > loss_ceiling:
                self.internal_progress = max(0.0, self.internal_progress - (self.step_size * 0.5))
            else:
                # Progress healthy: Advance sharpening pressure
                self.internal_progress = min(1.0, self.internal_progress + self.step_size)

            # -------------------------------------------------------------
            # TERMINATION AUDIT (Active only at full pressure)
            # -------------------------------------------------------------
            if self.internal_progress >= 1.0:
                # 1. Topic Sharpness (P_W) slope has flattened (< +0.03% gain per epoch)
                pw_saturated = (pw_slope < 0.03)
                
                # 2. Reconstruction slope has flattened (|drop| is negligible relative to noise)
                rec_saturated = (abs(rec_slope * self.window_size) < max(1.0, rec_noise_std * 0.5))

                if pw_saturated and rec_saturated:
                    self.termination_streak += 1
                else:
                    self.termination_streak = max(0, self.termination_streak - 1)

                if self.termination_streak >= self.required_term_streak:
                    return True

        return False

    def force_phase2(self, epoch: int, current_baseline: float) -> None:
        """Transitions tracker to Phase 2."""
        if self.phase == 1:
            self.phase = 2
            self.p1_baseline_rec = current_baseline
