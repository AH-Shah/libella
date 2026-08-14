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
    def __init__(self) -> None:
        self.phase = 1
        
        # EMAs to filter out all batch/epoch noise
        self.ema_rec = None
        self.ema_pw = None
        self.history_ema_rec = []
        self.history_ema_pw = []
        
        self.p1_baseline_rec = None
        self.internal_progress = 0.0
        
        # Smooth pressure changes (2% increments)
        self.step_size = 0.02 

    def get_progress(self) -> float:
        if self.phase == 1:
            return 0.0
        # Smooth curve so parameters don't jump violently
        return 0.5 * (1.0 - math.cos(math.pi * self.internal_progress))

    def step(self, epoch_telemetry: dict, epoch: int) -> bool:
        current_rec = float(epoch_telemetry.get('l_rec', 0.0))
        current_pw = float(epoch_telemetry.get('p_w', 0.0))
        
        # 1. Update EMAs (Crushes noise)
        if self.ema_rec is None:
            self.ema_rec = current_rec
            self.ema_pw = current_pw
        else:
            self.ema_rec = 0.3 * current_rec + 0.7 * self.ema_rec
            self.ema_pw = 0.3 * current_pw + 0.7 * self.ema_pw
            
        self.history_ema_rec.append(self.ema_rec)
        self.history_ema_pw.append(self.ema_pw)

        # 2. Measure Velocities (Look back 10 epochs)
        rec_improvement = 1.0
        pw_gain = 1.0
        if len(self.history_ema_rec) >= 11:
            past_rec = self.history_ema_rec[-11]
            past_pw = self.history_ema_pw[-11]
            
            rec_improvement = (past_rec - self.ema_rec) / past_rec
            pw_gain = self.ema_pw - past_pw

        if self.phase == 1:
            if len(self.history_ema_rec) >= 8:
                # If rec_loss improvement is tiny (< 0.5%), it stopped.
                if rec_improvement < 0.005:
                    self.force_phase2(epoch, self.ema_rec)
            return False

        if self.phase == 2:
            # Did the smoothed loss get 2% worse than our baseline?
            if self.ema_rec > (self.p1_baseline_rec * 1.02):
                # GET BACK (Relieve pressure)
                self.internal_progress = max(0.0, self.internal_progress - self.step_size)
            else:
                # KEEP SHARPENING
                self.internal_progress = min(1.0, self.internal_progress + self.step_size)
                
                # Dynamic Baseline: If sharpening actually IMPROVED the loss, 
                # save this new low as the standard to protect!
                if self.ema_rec < self.p1_baseline_rec:
                    self.p1_baseline_rec = self.ema_rec

            if self.internal_progress >= 1.0:
                # rec_flat: Improvement is < 0.2%
                rec_flat = (rec_improvement < 0.002) 
                
                # pw_flat: Grew by < 0.25% absolute
                pw_flat = (pw_gain < 0.25) 
                
                # Only terminate if BOTH metrics are exhausted
                if rec_flat and pw_flat:
                    return True
                        
        return False
        
    def force_phase2(self, epoch: int, current_ema: float) -> None:
        if self.phase == 1:
            self.phase = 2
            self.p1_baseline_rec = current_ema
