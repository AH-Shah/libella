"""Utility functions for Libella pipeline operations."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import math

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
    """
    Closed-Loop Adaptive Scheduler. 
    Progress is gated by the health of the reconstruction manifold.
    """
    def __init__(self) -> None:
        self.history_rec = []
        self.history_pw = []
        
        self.phase = 1
        self.p1_baseline_rec = None
        self.p1_epochs = 0
        
        self.internal_progress = 0.0
        self.epochs_at_max = 0

    def get_progress(self) -> float:
        """Apply an S-curve easing to our adaptive linear progress."""
        if self.phase == 1:
            return 0.0
        
        # Smooth interpolation: prevents sudden shocks even if internal_progress jumps
        return 0.5 * (1.0 - math.cos(math.pi * self.internal_progress))

    def _get_trend(self, history: list, window: int = 3) -> float:
        """Calculate the relative change between two recent windows."""
        if len(history) < (window * 2):
            return 1.0 
            
        recent = sum(history[-window:]) / window
        previous = sum(history[-(window * 2):-window]) / window
        
        if previous == 0: return 0.0
        return (previous - recent) / previous

    def step(self, epoch_telemetry: dict, epoch: int) -> bool:
        """Update controller state; return True if training is optimally complete."""
        current_rec = epoch_telemetry.get('l_rec', 0.0)
        current_pw = epoch_telemetry.get('p_w', 0.0)
        
        self.history_rec.append(current_rec)
        self.history_pw.append(current_pw)

        # ---------------------------------------------------------
        # PHASE 1: Manifold Discovery (Wait for Plateau)
        # ---------------------------------------------------------
        if self.phase == 1:
            if epoch >= 6:
                rec_improvement = self._get_trend(self.history_rec, window=3)
                
                # If Pure_Rec improves by < 0.2%, manifold is formed.
                if rec_improvement < 0.002:
                    self.phase = 2
                    # Lock in a stable baseline average
                    self.p1_baseline_rec = sum(self.history_rec[-3:]) / 3 
                    self.p1_epochs = epoch
                    print(f"\n[↳] Manifold saturated at Epoch {epoch}. Engaging Adaptive Sparsification...")
            return False

        # ---------------------------------------------------------
        # PHASE 2: Loss-Gated Sparsification
        # ---------------------------------------------------------
        if self.phase == 2:
            # The speed proxy: If it took 35 eps to learn the manifold, 
            # the safest time to deform it is ~35 eps. (Max speed limit 10 eps)
            base_step = 1.0 / max(10, self.p1_epochs)
            
            # Check manifold health
            rec_ratio = current_rec / self.p1_baseline_rec
            
            if rec_ratio <= 1.02:
                # Safe: Manifold is intact, accelerate sparsity
                self.internal_progress = min(1.0, self.internal_progress + base_step)
            elif rec_ratio <= 1.05:
                # Caution: Manifold under stress, slow down
                self.internal_progress = min(1.0, self.internal_progress + (base_step * 0.33))
            else:
                # Danger: Manifold tearing. Relieve pressure to let optimizer recover.
                self.internal_progress = max(0.0, self.internal_progress - (base_step * 0.5))

            # ---------------------------------------------------------
            # PHASE 3: Polish & Dynamic Termination
            # ---------------------------------------------------------
            if self.internal_progress >= 1.0:
                self.epochs_at_max += 1
                
                # Wait for at least two windows (6 epochs) of full pressure to settle
                if self.epochs_at_max >= 6:
                    recent_pw = sum(self.history_pw[-3:]) / 3
                    prev_pw = sum(self.history_pw[-6:-3]) / 3
                    
                    pw_absolute_gain = recent_pw - prev_pw
                    
                    # If P_W fails to grow by at least 0.5% over the window, we are saturated.
                    if pw_absolute_gain < 0.5:
                        print(f"\n[✓] Topic Sharpness (P_W) saturated at {current_pw:.2f}%. Terminating gracefully.")
                        return True
                        
        return False