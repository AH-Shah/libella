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
        self.step_size: float = 0.1
        
        # Termination Stability Counter
        self.termination_streak: int = 0
        self.required_term_streak: int = 2

    @staticmethod
    def _compute_ols_stats(series: List[float]) -> Tuple[float, float, float]:
        """Fits OLS linear regression y = m*t + c and computes residual variance."""
        n = len(series)
        if n < 3:
            return 0.0, float(series[-1]) if series else 0.0, 1.0

        t_mean = (n - 1) / 2.0
        y_mean = sum(series) / n

        num = sum((i - t_mean) * (series[i] - y_mean) for i in range(n))
        den = sum((i - t_mean) ** 2 for i in range(n))
        
        slope = num / max(1e-9, den)
        intercept = y_mean - slope * t_mean

        # Residual standard deviation (noise floor after trend removal)
        res_sq_sum = sum((series[i] - (slope * i + intercept)) ** 2 for i in range(n))
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

        if len(self.raw_rec_history) < self.window_size:
            return False

        rec_window = self.raw_rec_history[-self.window_size:]
        pw_window = self.raw_pw_history[-self.window_size:]

        # Fit OLS trends
        rec_slope, rec_mu, rec_noise_std = self._compute_ols_stats(rec_window)
        pw_slope, pw_mu, _ = self._compute_ols_stats(pw_window)

        # Anchor budget to statistical mean of the window (Trough-Lock Fix)
        if rec_mu < self.best_rec_loss:
            self.best_rec_loss = rec_mu

        expected_drop = -rec_slope * self.window_size
        descent_snr = expected_drop / max(1e-5, rec_noise_std)

        # -----------------------------------------------------------------
        # PHASE 1: Variance-Adjusted Manifold Discovery
        # -----------------------------------------------------------------
        if self.phase == 1:
            relative_drop = expected_drop / max(1.0, rec_mu)
            
            # Transition when descent slope flattens into noise floor
            # (guarding with rec_slope > -0.5 to prevent trigger on runaway divergence)
            if (0.0 <= descent_snr < 0.40) or (abs(relative_drop) < 0.005 and rec_slope >= 0.0):
                self.force_phase2(epoch, rec_mu)
            return False

        # -----------------------------------------------------------------
        # PHASE 2: Dynamic Variance-Gated Squeezing
        # -----------------------------------------------------------------
        if self.phase == 2:
            # Dynamic Ceiling: min 2.5% budget, expanding up to 2*sigma noise floor
            dynamic_tolerance = max(self.best_rec_loss * 0.05, 2.5 * rec_noise_std)
            loss_ceiling = self.best_rec_loss + dynamic_tolerance

            # Smooth mean evaluated against ceiling prevents single-batch outlier braking
            if rec_mu > loss_ceiling:
                self.internal_progress = max(0.0, self.internal_progress - (self.step_size * 0.5))
            else:
                self.internal_progress = min(1.0, self.internal_progress + self.step_size)

            # -------------------------------------------------------------
            # TERMINATION AUDIT (Evaluated once at full pressure)
            # -------------------------------------------------------------
            if self.internal_progress >= 1.0:
                # 1. P_W slope flattened (< +0.03% gain per epoch)
                pw_saturated = (pw_slope < 0.1)
                
                # 2. Rec loss change is smaller than half a standard deviation of noise
                rec_saturated = (abs(expected_drop) < max(1.0, rec_noise_std * 0.5))

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
            if self.best_rec_loss == float("inf") or current_baseline < self.best_rec_loss:
                self.best_rec_loss = current_baseline
