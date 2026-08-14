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
    """Two-phase dynamic scheduler for decoupled representation and sparsification."""
    def __init__(self, num_cells: int, num_samples: int) -> None:
        size_ratio = max(1.0, num_cells / 100_000.0)
        patient_bump = 1.0 + 0.15 * max(0.0, num_samples - 1)
        
        # Combined complexity scalar
        comp = math.sqrt(size_ratio) * patient_bump
        
        # Phase 1: Representation (Strict minimums)
        self.min_p1 = int(25 * comp)
        self.window = max(4, int(6 * comp))
        
        # Phase 2: Gradual Sparsification
        self.p2_duration = max(20, int(35 * comp))
        
        self.min_delta = 0.001
        self.history = []
        self.phase = 1
        self.p2_step = 0

    def get_progress(self) -> float:
        """Fetch current non-linear progress multiplier."""
        if self.phase == 1:
            return 0.0
            
        linear_p = self.p2_step / max(1, self.p2_duration - 1)
        return 0.5 * (1.0 - math.cos(math.pi * linear_p))

    def step(self, loss: float, epoch: int) -> bool:
        """Update internal state; return True if training is complete."""
        if self.phase == 2:
            self.p2_step += 1
            return self.p2_step >= self.p2_duration

        self.history.append(loss)
        
        if epoch < self.min_p1 or len(self.history) < (self.window * 2):
            return False

        recent = sum(self.history[-self.window:]) / self.window
        previous = sum(self.history[-(self.window * 2):-self.window]) / self.window
        
        if ((previous - recent) / previous) < self.min_delta:
            self.force_phase2()
            
        return False

    def force_phase2(self) -> None:
        if self.phase == 1:
            self.phase = 2
            self.p2_step = 0