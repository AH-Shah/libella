"""Utility functions for Libella pipeline operations."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    with torch.no_grad():
        max_val = torch.zeros(num_nodes, dtype=src.dtype, device=src.device).scatter_reduce(0, index, src, reduce="amax")
    exp_val = torch.exp(src - max_val[index])
    sum_val = torch.zeros(num_nodes, dtype=src.dtype, device=src.device).scatter_add(0, index, exp_val)
    return exp_val / (sum_val[index] + 1e-9)


def get_pt_id(sample_name: str | Path) -> str:
    """Extract patient ID from filename."""
    name = str(sample_name).replace(".pt", "").replace(".h5ad", "")
    name = re.sub(r"_graph$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(_chunk_\d+|_fov\d+|_core\d+|_roi\d+)$", "", name, flags=re.IGNORECASE)
    name = name.replace("GSE280314_", "").replace("GSE303070_", "")
    
    patterns = [r"^(\d+[TN])_", r"^(GSM\d+)_slide", r"^(GSM\d+)_Tumor", r"^(pt\d+)"]
    for pat in patterns:
        if match := re.match(pat, name):
            return match.group(1)
            
    return name

def get_ds_id(file_path: str | Path) -> str:
    """Extract cohort ID from filename."""
    name = Path(file_path).stem
    if match := re.match(r"^(GSE\d+)", name):
        return match.group(1)
    if "Ajou" in name:
        return "GSE226997"
    return "Unknown_Cohort"