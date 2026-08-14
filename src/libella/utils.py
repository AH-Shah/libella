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
    EMA-Smoothed Closed-Loop Adaptive Scheduler. 
    Filters out batch noise to make deterministic decisions on manifold health.
    """
    def __init__(self) -> None:
        self.phase = 1
        
        # Exponential Moving Averages (The Low-Pass Filter)
        self.ema_rec = None
        self.ema_pw = None
        self.history_ema_rec = []
        self.history_ema_pw = []
        
        self.p1_baseline_rec = None
        self.p1_epochs = 0
        
        self.internal_progress = 0.0
        self.epochs_at_max = 0

    def get_progress(self) -> float:
        if self.phase == 1:
            return 0.0
        # Smooth interpolation: prevents sudden shocks
        return 0.5 * (1.0 - math.cos(math.pi * self.internal_progress))

    def step(self, epoch_telemetry: dict, epoch: int) -> bool:
        current_rec = epoch_telemetry.get('l_rec', 0.0)
        current_pw = epoch_telemetry.get('p_w', 0.0)
        
        # 1. Update EMAs (Alpha=0.4: 40% today, 60% history)
        if self.ema_rec is None:
            self.ema_rec = current_rec
            self.ema_pw = current_pw
        else:
            self.ema_rec = 0.4 * current_rec + 0.6 * self.ema_rec
            self.ema_pw = 0.4 * current_pw + 0.6 * self.ema_pw
            
        self.history_ema_rec.append(self.ema_rec)
        self.history_ema_pw.append(self.ema_pw)

        # ---------------------------------------------------------
        # PHASE 1: Manifold Discovery (Wait for EMA Plateau)
        # ---------------------------------------------------------
        if self.phase == 1:
            # Require at least 8 epochs to establish a reliable smoothed curve
            if len(self.history_ema_rec) >= 8:
                # Compare today's smoothed loss against 5 epochs ago
                past_ema = self.history_ema_rec[-6]
                current_ema = self.history_ema_rec[-1]
                
                # Formula: (Past - Current) / Past
                improvement = (past_ema - current_ema) / past_ema
                
                # If the smoothed curve improved by < 0.5% over 5 epochs, we are flat.
                if improvement < 0.005:
                    self.force_phase2(epoch, current_ema)
                    
            return False

        # ---------------------------------------------------------
        # PHASE 2: Loss-Gated Sparsification
        # ---------------------------------------------------------
        if self.phase == 2:
            base_step = 1.0 / max(45, self.p1_epochs * 3)
            
            # Check manifold health using SMOOTHED loss (Immune to 1-epoch noise spikes)
            rec_ratio = self.ema_rec / max(1e-9, self.p1_baseline_rec)
            
            if rec_ratio <= 1.02:
                # Safe: Manifold is intact, accelerate sparsity
                self.internal_progress = min(1.0, self.internal_progress + base_step)
            elif rec_ratio <= 1.05:
                # Caution: Stress, slow down
                self.internal_progress = min(1.0, self.internal_progress + (base_step * 0.33))
            else:
                # Danger: Tearing. Relieve pressure.
                self.internal_progress = max(0.0, self.internal_progress - (base_step * 0.5))

            # ---------------------------------------------------------
            # PHASE 3: Polish & Dynamic Termination
            # ---------------------------------------------------------
            if self.internal_progress >= 1.0:
                self.epochs_at_max += 1
                
                # Wait for 8 epochs to let the network settle at max parameters
                if self.epochs_at_max >= 15:
                    past_pw = self.history_ema_pw[-6]
                    current_pw_ema = self.history_ema_pw[-1]
                    
                    # Absolute gain of P_W over 5 epochs
                    pw_gain = current_pw_ema - past_pw
                    
                    # If smoothed sharpness grew by < 0.25% over 5 epochs, we are squeezed dry.
                    if pw_gain < 0.25:
                        return True
                        
        return False
        
    def force_phase2(self, epoch: int, current_ema: float) -> None:
        """Triggered either naturally by loss plateaus, or forcefully by epoch limits."""
        if self.phase == 1:
            self.phase = 2
            self.p1_baseline_rec = current_ema 
            self.p1_epochs = epoch