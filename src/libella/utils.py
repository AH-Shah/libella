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



"""Adaptive Elastic Phase Controller with Dual-Horizon Oscillation Filtering."""

import math
from typing import Dict, List, Optional, Tuple


class PhaseTracker:
    def __init__(
        self,
        cycle_window: int = 6,        # Cut window from 12 -> 6 (responds 2x faster)
        target_pw: float = 70.0,
        rel_tolerance: float = 0.06,  # 6% reconstruction budget ceiling
        max_p1_epochs: int = 15,      # Hard limit: Phase 1 CANNOT exceed 15 epochs
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
        self.squeeze_momentum: float = 0.05   # Step size 0.05 (ramps in ~20 epochs instead of 80)
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
                self.breathing_cooldown = 2    # Hold pressure for 2 epochs to recover
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
