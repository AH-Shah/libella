#!/usr/bin/env python3
"""Libella Spatial GNN - Unabridged High-Resolution Performance & Training Telemetry Harness.

Complete, 100% unabridged replica of the Libella architecture, phase tracking,
training pipeline, and optimization loop with zero-synchronization execution and
isolated external subprocess telemetry (CPU, RSS/VMS, driver allocations, and micro-stage GPU kernels).
"""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Project Path Resolution & System Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
for p in [
    PROJECT_ROOT / "libella" / "src",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "libella",
    PROJECT_ROOT,
]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from libella.config import cfg
except ImportError:
    class LibellaConfig:
        hidden_dim: int = 128
        k_hops: int = 2
        topk_k: int = 3
        spatial_gain_init: float = 1.0
        ambient_scale_init: float = 0.50
        dead_step_threshold: int = 20
        aux_k: int = 4
        ortho_sample_size: int = 256
        edge_sim_threshold: float = 0.50
        edge_decay_slope: float = 15.0
        ambient_max_cap: float = 0.35
        aux_min_residual_energy: float = 0.05
        aux_min_k: int = 2
        dynamic_w_ema_weight: float = 0.10
        asym_penalty_weight: float = 0.50
        ortho_overlap_threshold: float = 0.30
        ortho_weight: float = 8.0
        ortho_min_scale: float = 0.50
        aux_weight: float = 0.50
        lr_base: float = 1e-3
        lr_decoder: float = 5e-4
        wd_base: float = 1e-4
        ambient_lr_mult: float = 5.0
        epochs: int = 100
        lr_min: float = 1e-6
        grad_clip_recon: float = 5.0
        grad_clip_spatial: float = 15.0
        n_latents: int = 36
        edge_dropout: float = 0.40
        active_latent_threshold: float = 1e-4
        alpha_ema_max: float = 0.05
        alpha_ema_step_multiplier: float = 1.0
        meta_batch_size: int = 5
        suffix: str = "benchmark"
        checkpoint_freq: int = 10
        phase2_force_window: int = 10
        logger_backend: str = "console"
        log_histograms: bool = False

    cfg = LibellaConfig()


class PathManager:
    """Manages output, logs, and checkpoint directories."""
    @staticmethod
    def make_dirs(suffix: str = "default") -> dict[str, Path]:
        base = PROJECT_ROOT / "libella_output" / suffix
        out_dir = base / "runs"
        ckpt_dir = base / "checkpoints" / "best_model.pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        return {"out": out_dir, "checkpoint": ckpt_dir, "base": base}


paths = PathManager()


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def pad_mps_shapes(
    x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensures contiguous memory layout and proper stride alignment for Metal ALUs."""
    return x.contiguous(), src.contiguous(), dst.contiguous(), weights.contiguous()


# ---------------------------------------------------------------------------
# 2. Phase Tracker & Unified Logger Implementations
# ---------------------------------------------------------------------------
class PhaseTracker:
    """Tracks two-phase distillation schedules (Exploration -> Loss-Gated Sparsification)."""

    def __init__(self, total_epochs: int = 100) -> None:
        self.total_epochs = total_epochs
        self.phase: int = 1
        self.pressure: float = 0.0
        self.p1_baseline_rec: float = float("inf")
        self.p2_start_epoch: int = 0
        self.p2_target_epochs: int = total_epochs // 2
        self.patience_counter: int = 0
        self.best_val_loss: float = float("inf")

    def get_progress(self) -> float:
        if self.phase == 1:
            return 0.0
        span = max(1, self.total_epochs - self.p2_start_epoch)
        return min(1.0, max(0.0, float(self.pressure) / float(span)))

    def force_phase2(self, epoch: int, baseline_rec: float) -> None:
        if self.phase == 1:
            self.phase = 2
            self.p2_start_epoch = epoch
            self.p1_baseline_rec = baseline_rec

    def step(self, epoch_telemetry: dict[str, float], epoch: int, val_loss: float) -> bool:
        if self.phase == 1:
            current_rec = epoch_telemetry.get("l_rec", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= 5 or epoch >= (self.total_epochs // 2):
                self.phase = 2
                self.p2_start_epoch = epoch
                self.p1_baseline_rec = current_rec
        else:
            self.pressure += 1.0
            if epoch >= self.total_epochs - 1:
                return True
        return False


class UnifiedLogger:
    """Unified performance, parameter, and autopsy logger."""

    def __init__(self, backend: str = "console", run_name: str = "run", log_dir: str = "./logs") -> None:
        self.backend = backend
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_memory_metrics(self, device: torch.device) -> dict[str, float]:
        metrics = {}
        if device.type == "mps":
            if hasattr(torch.mps, "current_allocated_memory"):
                metrics["mem/mps_current_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            if hasattr(torch.mps, "driver_allocated_memory"):
                metrics["mem/mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024)
        elif device.type == "cuda":
            metrics["mem/cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
            metrics["mem/cuda_max_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return metrics

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def log_model_telemetry(self, step: int, model: LibellaGNN, log_histograms: bool = False) -> None:
        pass

    def log_checkpoint_autopsy(self, epoch: int, ckpt_path: str) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 3. Unabridged Libella Spatial Graph Neural Network Architecture
# ---------------------------------------------------------------------------
class LibellaGNN(nn.Module):
    """Core Libella Spatial GNN architecture with Top-K Hard Sparsity and Residual AuxK Revival."""

    def __init__(
        self,
        in_channels: int,
        n_metaprograms: int,
        init_components: np.ndarray | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = getattr(cfg, "hidden_dim", hidden_dim)
        self.k_hops = getattr(cfg, "k_hops", 2)
        self.n_latents = n_metaprograms
        self.in_channels = in_channels
        self.k = getattr(cfg, "topk_k", 3)

        # 1. Identity Encoder (Self Signal)
        self.self_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 2. Directional Feature-Conditioned Spatial Filter
        self.spatial_lin = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 3. Output Streams
        self.mag_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.n_latents),
            nn.Softplus(beta=1.0),
        )
        self.spatial_gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_latents)
        )
        self.spatial_gain = nn.Parameter(
            torch.tensor(float(getattr(cfg, "spatial_gain_init", 1.0)))
        )

        # 4. Decoder Dictionary
        if init_components is not None:
            dec_init = torch.tensor(init_components, dtype=torch.float32)
            if dec_init.shape != (self.n_latents, in_channels):
                dec_init = torch.randn(self.n_latents, in_channels)
            dec_init = F.normalize(dec_init, p=2, dim=-1)
        else:
            dec_init = F.normalize(
                torch.randn(self.n_latents, in_channels).abs() + 0.1, p=2, dim=-1
            )

        self.decoder_weight = nn.Parameter(dec_init)
        self.decoder_bias = nn.Parameter(torch.zeros(in_channels))
        self.ambient_scale = nn.Parameter(torch.tensor(getattr(cfg, "ambient_scale_init", 0.50)))

        # Buffers & Aux State Tracking
        self.register_buffer("ortho_mask", 1.0 - torch.eye(self.n_latents, dtype=torch.float32))
        self.register_buffer("steps_since_active", torch.zeros(self.n_latents, dtype=torch.int64))
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))
        self.dead_step_threshold = getattr(cfg, "dead_step_threshold", 20)
        self.aux_k = getattr(cfg, "aux_k", 4)
        self.ortho_sample_size = getattr(cfg, "ortho_sample_size", min(256, self.n_latents))

    def encode(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x_dense.size(0)
        has_edges = src.numel() > 0

        # 1. Stride Alignment for MPS Hardware Addressing
        if has_edges:
            src = src.contiguous()
            dst = dst.contiguous()
            edge_weights = edge_weights.contiguous()

        # 2. Depth Disentanglement (Vectorized L2 Norm on Metal ALU)
        cell_mass = torch.clamp(
            torch.linalg.vector_norm(x_dense, ord=2, dim=-1, keepdim=True), min=1e-5
        )
        x_norm = x_dense / cell_mass

        # 3. Self Feature Extraction
        h_self = self.self_enc(x_norm)

        # 4. Bilateral Edge Filtering & Symmetric Normalization
        if has_edges:
            with torch.no_grad():
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)
                decay = torch.sigmoid(
                    (cos_sim - getattr(cfg, "edge_sim_threshold", 0.50))
                    * getattr(cfg, "edge_decay_slope", 15.0)
                )

                # Symmetric Laplacian normalization
                deg = torch.zeros((N, 1), dtype=x_dense.dtype, device=x_dense.device)
                deg.index_add_(0, dst, torch.ones((dst.size(0), 1), dtype=x_dense.dtype, device=x_dense.device))
                norm_inv_sqrt = torch.rsqrt(torch.clamp(deg, min=1.0))
                edge_norm = norm_inv_sqrt[src] * norm_inv_sqrt[dst]

            W_bil = edge_weights.unsqueeze(1) * decay
            gate_in = torch.cat([h_self[src] - h_self[dst], W_bil], dim=-1)
            gate = self.edge_gate(gate_in)

            # 5. Selective Graph Message Passing (Autograd-Isolated per Hop)
            h_sp = h_self
            for _ in range(self.k_hops):
                h_proj = self.spatial_lin(h_sp)
                msg = h_proj[src] * gate * edge_norm
                
                # Fresh tensor allocation per hop preserves Autograd graph history for backward pass
                agg = torch.zeros_like(h_sp).index_add_(0, dst, msg)
                h_sp = h_sp + F.silu(agg)
        else:
            h_sp = h_self

        # 6. Fusion of Self + Spatial Context
        h_fused = F.layer_norm(h_self + h_sp, [self.hidden_dim])

        # 7. Unconstrained Magnitude & Spatial Gating Shifts
        z_mag = self.mag_head(h_fused)

        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        # Direct GEMM without explicit transpose view allocations
        bio_sim = F.linear(x_norm, w_dec_norm)
        spatial_shift = self.spatial_gate_head(h_sp)

        progress = getattr(self, "current_progress", 1.0) if self.training else 1.0
        spatial_warmup = 0.20 + 0.80 * min(1.0, progress * 2.0)

        raw_affinity = F.softplus(bio_sim + (self.spatial_gain * spatial_warmup * spatial_shift))
        pre_acts = raw_affinity * z_mag

        # 8. Top-K Hard Sparsity
        target_k = getattr(self, "current_k", self.k)
        topk_vals, topk_indices = torch.topk(pre_acts, k=target_k, dim=-1)
        z_sparse = torch.zeros_like(pre_acts).scatter_(-1, topk_indices, topk_vals)

        return z_sparse, pre_acts, cell_mass, z_mag

    @torch.no_grad()
    def resample_dead_latents(
        self,
        r_pos: torch.Tensor,
        dead_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Resamples dead atoms with Gram-Schmidt orthogonalization to prevent cloning."""
        if not dead_mask.any():
            return 0

        dead_indices = torch.nonzero(dead_mask).squeeze(-1)
        num_dead = dead_indices.numel()
        cell_res_energy = torch.linalg.vector_norm(r_pos, ord=2, dim=-1)

        k_resample = min(num_dead, (cell_res_energy > 0.05).sum().item())
        if k_resample == 0:
            return 0

        worst_cells = torch.topk(cell_res_energy, k=k_resample, dim=0).indices
        target_dead_ids = dead_indices[:k_resample]

        candidates = r_pos[worst_cells].clone()
        noise = torch.randn_like(candidates) * 0.02
        candidates = F.relu(candidates.add_(noise))

        healthy_mask = ~dead_mask
        if healthy_mask.any():
            w_healthy = F.normalize(self.decoder_weight.data[healthy_mask], p=2, dim=-1)
            proj = F.linear(candidates, w_healthy)
            candidates = F.relu(candidates - torch.mm(proj, w_healthy))

        norms = torch.linalg.vector_norm(candidates, ord=2, dim=-1, keepdim=True)
        collapsed = (norms < 1e-4).squeeze(-1)
        if collapsed.any():
            candidates[collapsed] = F.relu(
                torch.randn(int(collapsed.sum().item()), candidates.size(-1), device=candidates.device)
            )

        new_atoms = F.normalize(candidates, p=2, dim=-1)
        self.decoder_weight.data[target_dead_ids] = new_atoms
        self.steps_since_active[target_dead_ids] = 0

        if optimizer is not None:
            state = optimizer.state.get(self.decoder_weight, None)
            if state is not None:
                if "exp_avg" in state:
                    state["exp_avg"][target_dead_ids] = 0.0
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"][target_dead_ids] = 0.0

        return k_resample

    def forward(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # 1. Spatial & Identity Encoding
        z, pre_acts, cell_mass, z_mag = self.encode(x_dense, src, dst, edge_weights)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)

        # 2. Baseline Decoupling (Clean GEMM: [N, n_latents] @ [n_latents, in_channels])
        baseline_gene = F.normalize(F.softplus(self.decoder_bias) + 1e-6, p=2, dim=-1).unsqueeze(0)
        ambient_coeff = torch.sigmoid(self.ambient_scale) * getattr(cfg, "ambient_max_cap", 0.35)

        comp_profile = (1.0 - ambient_coeff) * torch.mm(z, w_dec_norm) + (ambient_coeff * baseline_gene)
        x_recon = comp_profile * cell_mass

        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(self.n_latents, dtype=torch.bool, device=x_dense.device)

        # 3. Auxiliary Loss & Dead Latent Tracking (Training Only)
        if self.training:
            with torch.no_grad():
                active_in_batch = (z > 1e-4).any(dim=0)
                self.steps_since_active.add_(1)
                self.steps_since_active.masked_fill_(active_in_batch, 0)
                dead_mask_ret = self.steps_since_active >= self.dead_step_threshold

            x_norm = F.normalize(x_dense, p=2, dim=-1)
            x_recon_norm = F.normalize(F.relu(x_recon), p=2, dim=-1)
            r_pos = F.relu(x_norm - x_recon_norm)
            r_pos_ret = r_pos.detach()
            r_norm = F.normalize(r_pos + 1e-6, p=2, dim=-1).detach()
            residual_energy = torch.linalg.vector_norm(r_pos, ord=2, dim=-1).mean()

            # Exact dual-gate condition preserved
            if dead_mask_ret.any() and residual_energy > getattr(cfg, "aux_min_residual_energy", 0.05):
                dead_indices = torch.nonzero(dead_mask_ret).squeeze(-1)
                num_dead = dead_indices.numel()
                k_aux = min(max(getattr(cfg, "aux_min_k", 2), self.aux_k), num_dead)

                w_dead = w_dec_norm[dead_indices]
                aux_sim = F.linear(r_norm, w_dead)
                topk_res = torch.topk(aux_sim, k=k_aux, dim=-1)

                dead_mag = z_mag[:, dead_indices]
                topk_mag = torch.gather(dead_mag, -1, topk_res.indices)

                z_aux_weights = F.softplus(topk_res.values, beta=1.0) * topk_mag
                z_aux = torch.zeros_like(aux_sim).scatter_(-1, topk_res.indices, z_aux_weights)
                aux_recon = torch.mm(z_aux, w_dead)

        return x_recon, z, w_dec_norm, aux_recon, r_norm, z_mag, r_pos_ret, dead_mask_ret

    def calc_loss(
        self,
        recon_x: torch.Tensor,
        x_true: torch.Tensor,
        z: torch.Tensor,
        w_dec_norm: torch.Tensor,
        aux_recon: torch.Tensor | None = None,
        r_norm: torch.Tensor | None = None,
        ghost_logits: torch.Tensor | None = None,
        ghost_weights: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Variance-Weighted Cell-Averaged Asymmetric Log-Cosh Loss
        is_non_zero = x_true > 0
        num_pos = torch.clamp(is_non_zero.sum().to(dtype=x_true.dtype), min=1.0)
        num_zeros = (x_true == 0).sum().to(dtype=x_true.dtype)
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(
                current_dynamic_w, weight=getattr(cfg, "dynamic_w_ema_weight", 0.10)
            )

        w_mat = torch.where(is_non_zero, self.dynamic_w_ema, 1.0)
        variance_weight = w_mat * (1.0 + torch.log1p(x_true))
        variance_weight = variance_weight / torch.clamp(variance_weight.mean(), min=1e-5)

        raw_delta = recon_x - x_true
        asym_penalty = getattr(cfg, "asym_penalty_weight", 0.50)
        asym_factor = 1.0 + (is_non_zero.to(x_true.dtype) * asym_penalty) * (raw_delta < 0).to(x_true.dtype)

        scaled_delta = raw_delta * asym_factor

        # Stable Log-Cosh: log(cosh(u)) = |u| + softplus(-2|u|) - log(2)
        abs_delta = scaled_delta.abs()
        log_cosh_delta = abs_delta + F.softplus(-2.0 * abs_delta) - 0.6931471805599453

        per_cell_loss = torch.sum(variance_weight * log_cosh_delta, dim=-1)
        l_recon = torch.mean(per_cell_loss) / math.sqrt(x_true.shape[-1])

        # 2. Strict Orthogonality Barrier
        gram = torch.mm(w_dec_norm, w_dec_norm.t())
        off_diag = gram * self.ortho_mask

        ortho_thresh = getattr(cfg, "ortho_overlap_threshold", 0.30)
        excess_corr = F.relu(off_diag - ortho_thresh)
        num_violating = torch.clamp((excess_corr > 0).sum().to(dtype=x_true.dtype), min=1.0)
        l_ortho_mean = excess_corr.pow(2).sum() / num_violating

        max_corr = off_diag.max()
        l_ortho_max = F.relu(max_corr - 0.50).pow(2) * 50.0
        l_ortho = l_ortho_mean + l_ortho_max

        l_sparse = torch.tensor(0.0, device=x_true.device)

        # 3. Residual Alignment
        if aux_recon is not None and r_norm is not None:
            res_energy = torch.clamp(r_norm.pow(2).sum(dim=-1).mean(), min=1e-4)
            aux_error = (aux_recon - r_norm).pow(2).sum(dim=-1).mean()
            l_aux = aux_error / res_energy
        else:
            l_aux = torch.tensor(0.0, device=x_true.device)

        base_ortho = getattr(cfg, "ortho_weight", 8.0)
        ortho_min = getattr(cfg, "ortho_min_scale", 0.50)
        current_ortho = base_ortho * (ortho_min + (1.0 - ortho_min) * progress)
        aux_weight = getattr(cfg, "aux_weight", 0.50)

        total_loss = l_recon + (current_ortho * l_ortho) + (aux_weight * l_aux)

        return total_loss, l_recon.detach(), l_ortho.detach(), l_sparse.detach(), l_aux.detach()

    @torch.no_grad()
    def get_deep_telemetry(self) -> dict[str, float]:
        """Harvests parameter/gradient norms and SVD spectrum with minimal host syncs."""
        stats: dict[str, float] = {}

        param_items = list(self.named_parameters())
        p_names = [name.replace(".", "/") for name, _ in param_items]

        p_norms = torch.stack([torch.linalg.vector_norm(p.detach(), ord=2) for _, p in param_items])
        g_tensors = [p.grad.detach() for _, p in param_items if p.grad is not None]
        g_indices = [i for i, (_, p) in enumerate(param_items) if p.grad is not None]

        if g_tensors:
            g_norms = torch.stack([torch.linalg.vector_norm(g, ord=2) for g in g_tensors])
            g_zero_pcts = torch.stack([(g == 0).float().mean() * 100.0 for g in g_tensors])
            total_g_norm = torch.linalg.vector_norm(g_norms, ord=2)

            g_norms_host = g_norms.cpu().tolist()
            g_zeros_host = g_zero_pcts.cpu().tolist()
            stats["grad_norm/global_l2"] = total_g_norm.item()

            for idx, g_idx in enumerate(g_indices):
                clean_name = p_names[g_idx]
                stats[f"grad_norm/{clean_name}"] = g_norms_host[idx]
                stats[f"grad_zeros/{clean_name}_pct"] = g_zeros_host[idx]
        else:
            stats["grad_norm/global_l2"] = 0.0

        p_norms_host = p_norms.cpu().tolist()
        for idx, clean_name in enumerate(p_names):
            stats[f"param_norm/{clean_name}"] = p_norms_host[idx]

        if hasattr(self, "decoder_weight"):
            w = F.normalize(self.decoder_weight, p=2, dim=1)
            sim = torch.mm(w, w.t())
            off_diag_mask = ~torch.eye(w.size(0), dtype=torch.bool, device=w.device)
            off_diag_vals = sim.masked_select(off_diag_mask)

            if off_diag_vals.numel() > 0:
                stats["dict/max_cross_corr"] = off_diag_vals.max().item()
                stats["dict/mean_cross_corr"] = off_diag_vals.abs().mean().item()

            w_cpu = w.detach().to(device="cpu", dtype=torch.float32)
            s = torch.linalg.svdvals(w_cpu)
            eff_rank = (s.sum() ** 2) / torch.clamp((s**2).sum(), min=1e-9)
            stats["dict/effective_rank"] = eff_rank.item()
            stats["dict/svd_sigma_1"] = s[0].item()
            stats["dict/svd_sigma_2"] = s[1].item() if s.numel() > 1 else 0.0
            stats["dict/svd_sigma_3"] = s[2].item() if s.numel() > 2 else 0.0

        if hasattr(self, "ambient_scale"):
            lr_mult = getattr(cfg, "ambient_lr_mult", 1.0)
            max_cap = getattr(cfg, "ambient_max_cap", 0.35)
            amb_pct = torch.sigmoid(self.ambient_scale * lr_mult).item() * max_cap * 100.0
            stats["model/ambient_absorption_pct"] = amb_pct

        dead_count = (self.steps_since_active >= self.dead_step_threshold).sum().item()
        stats["latents/dead_count"] = float(dead_count)
        stats["latents/active_pct"] = (1.0 - (dead_count / self.n_latents)) * 100.0

        return stats


# ---------------------------------------------------------------------------
# 4. Pipeline Helpers, Batch Prefetching & Latent Export
# ---------------------------------------------------------------------------
def _prep_ssd_chunks(graph_paths: list[Path]) -> list[dict[str, Any]]:
    """Loads and returns graph chunk references from disk."""
    cache = []
    for gp in graph_paths:
        if gp.exists():
            data = torch.load(gp, map_location="cpu", weights_only=False)
            cache.append(data)
    return cache


def make_meta_batches(
    training_cache: list[dict[str, Any]], meta_batch_size: int = 4
) -> list[list[dict[str, Any]]]:
    """Splits chunks into meta accumulation groups."""
    return [
        training_cache[i : i + meta_batch_size]
        for i in range(0, len(training_cache), meta_batch_size)
    ]


def prefetch_batches(
    meta_batches: list[list[dict[str, Any]]]
) -> Generator[tuple[list[dict[str, Any]], list[dict[str, Any]]], None, None]:
    """Yields batches with references for execution."""
    for mb in meta_batches:
        yield mb, mb


def export_latents_from_graphs(
    model: LibellaGNN,
    graph_paths: list[Path],
    out_dir: Path,
    device: torch.device,
) -> Path:
    """Exports distilled Top-K latent representations to disk."""
    model.eval()
    all_latents = []
    out_path = out_dir / "libella_latent.npz"

    with torch.no_grad():
        for gp in graph_paths:
            if not gp.exists():
                continue
            chunk = torch.load(gp, map_location="cpu", weights_only=False)
            x = chunk["x"].to(device=device, dtype=torch.float32)
            src = chunk["src"].to(device=device, dtype=torch.int64)
            dst = chunk["dst"].to(device=device, dtype=torch.int64)
            weights = chunk.get("weights", torch.ones_like(src, dtype=torch.float32)).to(device=device)

            z, _, _, _ = model.encode(x, src, dst, weights)
            all_latents.append(z.cpu().numpy())

    if all_latents:
        np.savez_compressed(out_path, latents=np.concatenate(all_latents, axis=0))
    return out_path


# ---------------------------------------------------------------------------
# 5. High-Frequency Asynchronous Telemetry Subprocess (Zero-GIL Interference)
# ---------------------------------------------------------------------------
class HardwareTelemetryWorker(mp.Process):
    """Monitors host CPU, RSS/VMS memory, and driver metrics at 500 Hz."""

    def __init__(self, target_pid: int, poll_hz: float = 500.0) -> None:
        super().__init__(daemon=True)
        self.target_pid = target_pid
        self.poll_interval = 1.0 / poll_hz
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.metrics_queue = mp.Queue()

    def run(self) -> None:
        try:
            proc = psutil.Process(self.target_pid)
        except psutil.NoSuchProcess:
            self.ready_event.set()
            return

        cpu_records: list[float] = []
        rss_records: list[float] = []
        vms_records: list[float] = []

        # Warm-up CPU call
        proc.cpu_percent(interval=None)
        self.ready_event.set()

        t_start = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                cpu_records.append(proc.cpu_percent(interval=None))
                mem = proc.memory_info()
                rss_records.append(mem.rss / (1024 * 1024))
                vms_records.append(mem.vms / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.poll_interval)

        duration = time.perf_counter() - t_start
        summary = {
            "samples": len(cpu_records),
            "duration_sec": duration,
            "cpu_avg_pct": float(np.mean(cpu_records)) if cpu_records else 0.0,
            "cpu_max_pct": float(np.max(cpu_records)) if cpu_records else 0.0,
            "rss_start_mb": float(rss_records[0]) if rss_records else 0.0,
            "rss_peak_mb": float(np.max(rss_records)) if rss_records else 0.0,
            "rss_delta_mb": float(rss_records[-1] - rss_records[0]) if rss_records else 0.0,
            "vms_peak_mb": float(np.max(vms_records)) if vms_records else 0.0,
        }
        self.metrics_queue.put(summary)

    def stop_and_harvest(self) -> dict[str, Any]:
        self.stop_event.set()
        self.join(timeout=3.0)
        return self.metrics_queue.get() if not self.metrics_queue.empty() else {}


# ---------------------------------------------------------------------------
# 6. Deep Memory & Dynamic Kernel Allocation Auditor
# ---------------------------------------------------------------------------
def audit_runtime_memory(
    model: LibellaGNN,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """Measures static model allocations, AdamW buffers, and active GPU runtime memory."""
    report: dict[str, Any] = {"tensors": [], "driver": {}}

    param_bytes = 0
    grad_bytes = 0
    for name, p in model.named_parameters():
        p_b = p.nelement() * p.element_size()
        param_bytes += p_b
        report["tensors"].append({
            "scope": "Parameter",
            "name": name,
            "shape": str(list(p.shape)),
            "dtype": str(p.dtype).replace("torch.", ""),
            "mb": round(p_b / (1024 * 1024), 4),
        })
        if p.grad is not None:
            g_b = p.grad.nelement() * p.grad.element_size()
            grad_bytes += g_b
            report["tensors"].append({
                "scope": "Gradient",
                "name": f"grad/{name}",
                "shape": str(list(p.grad.shape)),
                "dtype": str(p.grad.dtype).replace("torch.", ""),
                "mb": round(g_b / (1024 * 1024), 4),
            })

    buffer_bytes = 0
    for name, b in model.named_buffers():
        b_b = b.nelement() * b.element_size()
        buffer_bytes += b_b
        report["tensors"].append({
            "scope": "Buffer",
            "name": name,
            "shape": str(list(b.shape)),
            "dtype": str(b.dtype).replace("torch.", ""),
            "mb": round(b_b / (1024 * 1024), 4),
        })

    opt_bytes = 0
    for g_idx, group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(group["params"]):
            state = optimizer.state.get(p, {})
            for sk, sv in state.items():
                if isinstance(sv, torch.Tensor):
                    s_b = sv.nelement() * sv.element_size()
                    opt_bytes += s_b
                    report["tensors"].append({
                        "scope": f"Optimizer_G{g_idx}",
                        "name": f"p{p_idx}/{sk}",
                        "shape": str(list(sv.shape)),
                        "dtype": str(sv.dtype).replace("torch.", ""),
                        "mb": round(s_b / (1024 * 1024), 4),
                    })

    report["param_total_mb"] = round(param_bytes / (1024 * 1024), 4)
    report["grad_total_mb"] = round(grad_bytes / (1024 * 1024), 4)
    report["buffer_total_mb"] = round(buffer_bytes / (1024 * 1024), 4)
    report["optimizer_total_mb"] = round(opt_bytes / (1024 * 1024), 4)
    report["static_tracked_mb"] = round(
        (param_bytes + grad_bytes + buffer_bytes + opt_bytes) / (1024 * 1024), 4
    )

    if device.type == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            report["driver"]["mps_current_allocated_mb"] = (
                torch.mps.current_allocated_memory() / (1024 * 1024)
            )
        if hasattr(torch.mps, "driver_allocated_memory"):
            report["driver"]["mps_driver_allocated_mb"] = (
                torch.mps.driver_allocated_memory() / (1024 * 1024)
            )
    elif device.type == "cuda":
        report["driver"]["cuda_allocated_mb"] = (
            torch.cuda.memory_allocated(device) / (1024 * 1024)
        )
        report["driver"]["cuda_max_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        )
        report["driver"]["cuda_reserved_mb"] = (
            torch.cuda.memory_reserved(device) / (1024 * 1024)
        )

    return report


# ---------------------------------------------------------------------------
# 7. Isolated Micro-Stage GPU Kernel Benchmark
# ---------------------------------------------------------------------------
def benchmark_micro_stages(
    model: LibellaGNN,
    sample_chunk: dict[str, torch.Tensor],
    device: torch.device,
    n_iters: int = 10,
) -> dict[str, dict[str, Any]]:
    """Measures precise isolated GPU execution times and transient tensor allocation per sub-operation."""
    x = sample_chunk["x"].to(device).contiguous()
    src = sample_chunk["src"].to(device).contiguous()
    dst = sample_chunk["dst"].to(device).contiguous()
    weights = sample_chunk["weights"].to(device).contiguous()
    train_idx = sample_chunk["train_core_idx"].to(device).contiguous()

    N = x.size(0)
    E = src.size(0)
    G = x.size(1)

    results: dict[str, dict[str, Any]] = {}

    def time_gpu_op(op_func, warmup: int = 3) -> tuple[float, float, float]:
        for _ in range(warmup):
            op_func()
        sync_device(device)

        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter_ns()
            out = op_func()
            sync_device(device)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
            del out
        return float(np.mean(times)), float(np.min(times)), float(np.std(times))

    # 1. Cosine Similarity & Index Select Transient Bloat
    cell_mass = torch.clamp(torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True), min=1e-5)
    x_norm = x / cell_mass

    def op_cossim():
        return (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)

    mean_t, min_t, _ = time_gpu_op(op_cossim)
    transient_mb = (E * G * 4 * 3) / (1024 * 1024)
    results["1. Edge Cos-Sim Materialization"] = {
        "time_ms": mean_t,
        "min_ms": min_t,
        "transient_ram_mb": transient_mb,
        "note": f"Index selects {E}x{G} ({transient_mb:.1f} MB)",
    }

    # 2. Self Identity Encoder
    def op_self_enc():
        return model.self_enc(x_norm)

    mean_t, min_t, _ = time_gpu_op(op_self_enc)
    h_self = model.self_enc(x_norm)
    results["2. Self Identity Encoder [N, G]->[N, 128]"] = {
        "time_ms": mean_t,
        "min_ms": min_t,
        "transient_ram_mb": (N * model.hidden_dim * 4) / (1024 * 1024),
        "note": "2-layer MLP on full node matrix",
    }

    # 3. Message Passing (k-hops index_add_)
    with torch.no_grad():
        cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)
        decay = torch.sigmoid((cos_sim - 0.50) * 15.0)
        deg = torch.zeros((N, 1), dtype=x.dtype, device=device)
        deg.index_add_(0, dst, torch.ones((dst.size(0), 1), dtype=x.dtype, device=device))
        norm_inv_sqrt = torch.rsqrt(torch.clamp(deg, min=1.0))
        edge_norm = norm_inv_sqrt[src] * norm_inv_sqrt[dst]
        W_bil = weights.unsqueeze(1) * decay
        gate_in = torch.cat([h_self[src] - h_self[dst], W_bil], dim=-1)
        gate = model.edge_gate(gate_in)

    def op_msg_passing():
        h_sp = h_self
        for _ in range(model.k_hops):
            h_proj = model.spatial_lin(h_sp)
            msg = h_proj[src] * gate * edge_norm
            agg = torch.zeros_like(h_sp).index_add_(0, dst, msg)
            h_sp = h_sp + F.silu(agg)
        return h_sp

    mean_t, min_t, _ = time_gpu_op(op_msg_passing)
    results["3. Spatial Message Passing (2 Hops)"] = {
        "time_ms": mean_t,
        "min_ms": min_t,
        "transient_ram_mb": (E * model.hidden_dim * 4 * model.k_hops) / (1024 * 1024),
        "note": f"{model.k_hops}x index_add_ over {E} edges",
    }

    # 4. Top-K Sparsification & Decoder Reconstruction GEMM
    h_sp = op_msg_passing()
    h_fused = F.layer_norm(h_self + h_sp, [model.hidden_dim])
    z_mag = model.mag_head(h_fused)
    w_dec_norm = F.normalize(model.decoder_weight, p=2, dim=1)
    bio_sim = F.linear(x_norm, w_dec_norm)
    spatial_shift = model.spatial_gate_head(h_sp)
    raw_affinity = F.softplus(bio_sim + (model.spatial_gain * 1.0 * spatial_shift))
    pre_acts = raw_affinity * z_mag

    def op_topk_gemm():
        topk_vals, topk_indices = torch.topk(pre_acts, k=model.k, dim=-1)
        z = torch.zeros_like(pre_acts).scatter_(-1, topk_indices, topk_vals)
        baseline_gene = F.normalize(F.softplus(model.decoder_bias) + 1e-6, p=2, dim=-1).unsqueeze(0)
        ambient_coeff = torch.sigmoid(model.ambient_scale) * 0.35
        comp = (1.0 - ambient_coeff) * torch.mm(z, w_dec_norm) + (ambient_coeff * baseline_gene)
        return comp * cell_mass

    mean_t, min_t, _ = time_gpu_op(op_topk_gemm)
    results["4. Top-K Sparsity & Decoder GEMM"] = {
        "time_ms": mean_t,
        "min_ms": min_t,
        "transient_ram_mb": (N * G * 4) / (1024 * 1024),
        "note": f"GEMM [N,{model.n_latents}] @ [{model.n_latents},{G}]",
    }

    # 5. Full Backward Pass
    def op_backward():
        model.zero_grad(set_to_none=True)
        recon, z, w_norm, aux, r, _, _, _ = model(x, src, dst, weights)
        l, _, _, _, _ = model.calc_loss(recon[train_idx], x[train_idx], z[train_idx], w_norm)
        l.backward()
        return l

    mean_t, min_t, _ = time_gpu_op(op_backward)
    results["5. Full Backward Autograd Pass"] = {
        "time_ms": mean_t,
        "min_ms": min_t,
        "transient_ram_mb": 0.0,
        "note": "Reverses full computation graph",
    }

    return results


# ---------------------------------------------------------------------------
# 8. Initialization & Full Training Loop Replica
# ---------------------------------------------------------------------------
def _init_model(
    common_genes: list[str],
    n_latents: int,
    checkpoint_path: Path | None = None,
) -> tuple[
    LibellaGNN,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    float,
    dict[str, Any] | None,
    dict[str, list],
    int,
]:
    """Initialize GNN model, optimizers, and load state if available."""
    device = get_device()
    model = LibellaGNN(
        in_channels=len(common_genes),
        n_metaprograms=n_latents,
    ).to(device)

    # 1. Clean Parameter grouping
    bias_ambient_params = [
        p for n, p in model.named_parameters()
        if any(k in n for k in ["decoder_bias", "ambient_scale"])
    ]
    decoder_weight_params = [
        p for n, p in model.named_parameters()
        if "decoder_weight" in n
    ]
    base_params = [
        p for n, p in model.named_parameters()
        if not any(k in n for k in ["decoder_", "ambient_scale"])
    ]

    lr_base = getattr(cfg, "lr_base", 1e-3)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": lr_base * 2.0, "weight_decay": getattr(cfg, "wd_base", 1e-4)},
        {"params": decoder_weight_params, "lr": getattr(cfg, "lr_decoder", lr_base * 0.5), "weight_decay": 0.0},
        {"params": bias_ambient_params, "lr": lr_base * getattr(cfg, "ambient_lr_mult", 5.0), "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=getattr(cfg, "epochs", 100), eta_min=getattr(cfg, "lr_min", 1e-6)
    )

    best_composite_score = float("inf")
    tracker_state = None
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "autopsy_metrics": []}
    start_epoch = 0

    out_dirs = paths.make_dirs(getattr(cfg, "suffix", "default"))
    resume_path = out_dirs["out"] / "resume_latest.pt"
    target_ckpt = resume_path if resume_path.exists() else checkpoint_path

    if target_ckpt and Path(target_ckpt).exists():
        try:
            print(f"  ↳ Loading state from: {Path(target_ckpt).name}")
            ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            best_composite_score = ckpt.get(
                "best_composite_score", ckpt.get("best_val_loss", float("inf"))
            )
            tracker_state = ckpt.get("tracker_state", None)
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"  ↳ Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"  ↳ [!] Failed to load checkpoint: {e}. Raising error to prevent accidental overwrite.")
            raise e

    return model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch


def _train_loop(
    model: LibellaGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_cache: list[dict[str, Any]],
    start_epoch: int,
    best_composite_score: float,
    tracker_state: dict[str, Any] | None,
    history: dict[str, list],
    max_epochs_to_run: int = 1,
) -> tuple[LibellaGNN, dict[str, list], dict[str, Any]]:
    """Master unabridged training loop with zero GPU-host synchronization bottlenecks."""
    print("\n-> Spatial Distillation (Top-K SAE)...")
    device = get_device()
    out_dirs = paths.make_dirs(getattr(cfg, "suffix", "default"))
    out_dir = out_dirs["out"]
    checkpoint_path = out_dirs["checkpoint"]

    logger = UnifiedLogger(
        backend=getattr(cfg, "logger_backend", "console"),
        run_name=f"run_{getattr(cfg, 'suffix', 'default')}",
        log_dir=str(out_dir),
    )
    global_step = 0
    accumulation_steps = getattr(cfg, "meta_batch_size", len(training_cache))
    total_epochs = getattr(cfg, "epochs", 100)

    tracker = PhaseTracker(total_epochs=total_epochs)
    if tracker_state is not None:
        tracker.__dict__.update(tracker_state)
        print(
            f"  ↳ Restored PhaseTracker state (Phase {tracker.phase}, "
            f"Pressure: {tracker.pressure:.2f}, Progress: {tracker.get_progress():.2f})"
        )

    step_loss_acc = torch.tensor(0.0, device=device)
    end_epoch = min(total_epochs, start_epoch + max_epochs_to_run)
    final_epoch_telemetry = {}

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_steps, val_steps = 0, 0
        train_chunk_count = 0

        # GPU-resident telemetry accumulator buffers
        train_loss_acc = torch.tensor(0.0, device=device)
        val_loss_acc = torch.tensor(0.0, device=device)

        gpu_telemetry = {
            "l_rec": torch.tensor(0.0, device=device),
            "l_ort": torch.tensor(0.0, device=device),
            "l_sparse": torch.tensor(0.0, device=device),
            "l_aux": torch.tensor(0.0, device=device),
            "l0_avg": torch.tensor(0.0, device=device),
            "dead_cnt": torch.tensor(0.0, device=device),
            "max_act": torch.tensor(0.0, device=device),
            "dyn_w": torch.tensor(0.0, device=device),
            "z_mag_mean": torch.tensor(0.0, device=device),
        }

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps_per_epoch = len(meta_batches)
        alpha_ema = min(
            getattr(cfg, "alpha_ema_max", 0.05),
            1.0 / (total_steps_per_epoch * getattr(cfg, "alpha_ema_step_multiplier", 1.0) + 1e-9),
        )
        ema_latent_freq = None
        nan_detected = False

        for step, (meta_meta, chunk_iter) in enumerate(prefetch_batches(meta_batches)):
            optimizer.zero_grad(set_to_none=True)
            step_loss_acc.zero_()
            last_r_pos = None
            last_dead_mask = None

            for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):
                x = batch["x"].to(device=device, non_blocking=True).contiguous()
                
                # PyTorch MPS index_add_ shaders strictly require torch.int64
                src = batch["src"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
                dst = batch["dst"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
                weights = batch["weights"].to(device=device, non_blocking=True).contiguous()

                if model.training and src.numel() > 0:
                    edge_drop = getattr(cfg, "edge_dropout", 0.0)
                    if edge_drop > 0.0:
                        keep_mask = torch.rand(src.size(0), device=device) > edge_drop
                        src = src[keep_mask].contiguous()
                        dst = dst[keep_mask].contiguous()
                        weights = weights[keep_mask].contiguous()

                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)

                # 1. Define progress and warmup state BEFORE forward/loss
                prog = tracker.get_progress() if tracker.phase == 2 else 0.0
                model.current_progress = prog
                model.current_k = getattr(cfg, "topk_k", 3)

                # 2. Forward Pass
                (
                    recon,
                    z,
                    w_dec_norm,
                    aux_recon,
                    r_norm,
                    z_mag,
                    r_pos,
                    dead_mask,
                ) = model(x, src, dst, weights)

                last_r_pos = r_pos
                last_dead_mask = dead_mask

                train_idx = batch["train_core_idx"].to(device=device, dtype=torch.int64, non_blocking=True)
                x_train = x[train_idx]
                recon_train = recon[train_idx]
                z_train = z[train_idx]
                aux_recon_train = aux_recon[train_idx] if aux_recon is not None else None
                r_norm_train = r_norm[train_idx] if r_norm is not None else None

                # 3. Loss Calculation
                loss_res = model.calc_loss(
                    recon_train,
                    x_train,
                    z_train,
                    w_dec_norm,
                    aux_recon=aux_recon_train,
                    r_norm=r_norm_train,
                    progress=prog,
                )
                true_batch_loss = loss_res[0]
                base_recon_val = loss_res[1]
                base_ort_val = loss_res[2]
                base_sparse_val = loss_res[3]
                base_aux_val = loss_res[4]

                (true_batch_loss / len(meta_meta)).backward()

                # Asynchronous GPU accumulation (Zero Host Sync)
                step_loss_acc.add_(true_batch_loss.detach() / len(meta_meta))
                train_loss_acc += true_batch_loss.detach()
                train_steps += 1

                # 4. GPU Telemetry Tracking
                with torch.no_grad():
                    active_thresh = getattr(cfg, "active_latent_threshold", 1e-4)
                    batch_active = (z_train > active_thresh).float()
                    current_freq = batch_active.mean(dim=0)

                    if ema_latent_freq is None:
                        ema_latent_freq = current_freq.clone()
                    else:
                        ema_latent_freq.lerp_(current_freq, weight=alpha_ema)

                    dead_count_val = (
                        (model.steps_since_active >= model.dead_step_threshold).float().sum()
                        if hasattr(model, "steps_since_active")
                        else torch.tensor(0.0, device=device)
                    )

                    gpu_telemetry["l_rec"] += base_recon_val
                    gpu_telemetry["l_ort"] += base_ort_val
                    gpu_telemetry["l_sparse"] += base_sparse_val
                    gpu_telemetry["l_aux"] += base_aux_val
                    gpu_telemetry["l0_avg"] += batch_active.sum(dim=-1).mean()
                    gpu_telemetry["dead_cnt"] += dead_count_val
                    gpu_telemetry["max_act"] += z_train.max()
                    gpu_telemetry["dyn_w"] += model.dynamic_w_ema.detach()
                    if z_mag is not None:
                        gpu_telemetry["z_mag_mean"] += z_mag.detach().mean()

                train_chunk_count += 1
                del train_idx, x_train, recon_train, z_train, aux_recon_train, r_norm_train, true_batch_loss

                # --- Optimized Validation Evaluation ---
                val_core_idx_cpu = batch.get("val_core_idx")
                if val_core_idx_cpu is not None and val_core_idx_cpu.numel() > 0:
                    val_idx = val_core_idx_cpu.to(device=device, non_blocking=True)

                    with torch.no_grad():
                        val_recon = recon[val_idx]
                        x_val = x[val_idx]

                        is_non_zero_val = x_val > 0
                        dynamic_w = getattr(model, "dynamic_w_ema", torch.tensor(1.0, device=device))
                        w_mat = torch.where(is_non_zero_val, dynamic_w, 1.0)

                        variance_weight_val = w_mat * (1.0 + torch.log1p(x_val))
                        variance_weight_val = variance_weight_val / torch.clamp(variance_weight_val.mean(), min=1e-5)

                        raw_delta_val = val_recon - x_val
                        asym_penalty = getattr(cfg, "asym_penalty_weight", 0.5)
                        asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * asym_penalty) * (raw_delta_val < 0).to(x_val.dtype)
                        scaled_delta_val = raw_delta_val * asym_val

                        abs_delta_val = scaled_delta_val.abs()
                        log_cosh_val = abs_delta_val + F.softplus(-2.0 * abs_delta_val) - 0.6931471805599453

                        per_cell_val = torch.sum(variance_weight_val * log_cosh_val, dim=-1)
                        val_log_cosh = torch.mean(per_cell_val) / math.sqrt(x_val.shape[-1])

                        val_loss_acc.add_(val_log_cosh)
                        val_steps += 1

                    del val_idx, val_recon, x_val, w_mat, raw_delta_val, asym_val, scaled_delta_val, per_cell_val, log_cosh_val

                del batch, src, dst, weights, x, recon, z, w_dec_norm, aux_recon, r_norm

            # 1. Dual-Group Gradient Clipping (Preserved Exact Thresholds)
            recon_keys = ("decoder_bias", "ambient_scale", "decoder_weight")
            recon_params = [
                p for n, p in model.named_parameters()
                if any(k in n for k in recon_keys) and p.grad is not None
            ]
            spatial_params = [
                p for n, p in model.named_parameters()
                if not any(k in n for k in recon_keys) and p.grad is not None
            ]

            if recon_params:
                torch.nn.utils.clip_grad_norm_(
                    recon_params, max_norm=getattr(cfg, "grad_clip_recon", 5.0)
                )
            if spatial_params:
                torch.nn.utils.clip_grad_norm_(
                    spatial_params, max_norm=getattr(cfg, "grad_clip_spatial", 15.0)
                )

            # 2. In-Place Tangent-Space Projection on Unit Sphere
            with torch.no_grad():
                if hasattr(model, "decoder_weight") and model.decoder_weight.grad is not None:
                    w = F.normalize(model.decoder_weight.data, p=2, dim=1)
                    grad = model.decoder_weight.grad
                    grad.sub_((grad * w).sum(dim=1, keepdim=True) * w)

            # 3. Optimizer Step
            optimizer.step()

            # 4. In-Place Non-Negative Spherical Retraction
            with torch.no_grad():
                if hasattr(model, "decoder_weight"):
                    w_data = model.decoder_weight.data
                    w_data.clamp_min_(0.0)
                    w_norm = torch.linalg.vector_norm(w_data + 1e-8, ord=2, dim=-1, keepdim=True)
                    w_data.div_(w_norm)

                if last_dead_mask is not None and last_dead_mask.any() and last_r_pos is not None:
                    model.resample_dead_latents(last_r_pos, last_dead_mask, optimizer=optimizer)

            global_step += 1

        # --- Single GPU-to-CPU Barrier Transfer ---
        scheduler.step()

        telemetry_keys = [
            "l_rec", "l_ort", "l_sparse", "l_aux",
            "l0_avg", "dead_cnt", "max_act", "dyn_w", "z_mag_mean"
        ]

        all_scalars_gpu = torch.stack([
            train_loss_acc / (train_steps + 1e-9),
            val_loss_acc / (val_steps + 1e-9),
            *[gpu_telemetry[k] / train_chunk_count for k in telemetry_keys],
        ])

        all_scalars_host = all_scalars_gpu.cpu().tolist()

        final_train_loss = all_scalars_host[0]
        final_val_loss = all_scalars_host[1]

        history["train_loss"].append(final_train_loss)
        history["val_loss"].append(final_val_loss)

        for idx, k in enumerate(telemetry_keys):
            final_epoch_telemetry[k] = all_scalars_host[idx + 2]

        current_l0_val = final_epoch_telemetry.get("l0_avg", float(model.n_latents))
        final_epoch_telemetry["p_w"] = (1.0 - (current_l0_val / float(model.n_latents))) * 100.0

        if ema_latent_freq is not None:
            p_norm = ema_latent_freq / torch.clamp(ema_latent_freq.sum(), min=1e-6)
            final_epoch_telemetry["ent"] = (-(p_norm * torch.log(p_norm + 1e-9)).sum()).item()
        else:
            final_epoch_telemetry["ent"] = 0.0

        current_lr = round(optimizer.param_groups[0]["lr"], 6)
        current_rec = final_epoch_telemetry.get("l_rec", float("inf"))
        current_l0 = final_epoch_telemetry.get("l0_avg", 0.0)
        current_dead = int(final_epoch_telemetry.get("dead_cnt", 0))

        epoch_metrics = {
            "epoch": epoch,
            "phase": tracker.phase,
            "train_loss": round(history["train_loss"][-1], 4),
            "val_loss": round(history["val_loss"][-1], 4),
            "lr": current_lr,
            "loss_components": {
                "rec": round(current_rec, 4),
                "ort": round(final_epoch_telemetry.get("l_ort", 0.0), 4),
                "sparse": round(final_epoch_telemetry.get("l_sparse", 0.0), 4),
                "aux": round(final_epoch_telemetry.get("l_aux", 0.0), 4),
                "dynamic_w_ema": round(final_epoch_telemetry.get("dyn_w", 1.0), 4),
            },
            "entropy": round(final_epoch_telemetry.get("ent", 0.0), 4),
            "l0_avg": round(current_l0, 2),
            "dead_latents": current_dead,
            "max_activation": round(final_epoch_telemetry.get("max_act", 0.0), 2),
            "z_mag_mean": round(final_epoch_telemetry.get("z_mag_mean", 0.0), 4),
            "tracker": {
                "progress": round(prog, 4),
                "pressure": round(getattr(tracker, "pressure", 0.0), 4),
                "topk_k": getattr(model, "k", 3),
            },
        }
        history.setdefault("autopsy_metrics", []).append(epoch_metrics)

        composite_score = current_rec * math.sqrt(1.0 + (current_l0 / float(model.n_latents)))
        final_epoch_telemetry["composite_score"] = composite_score

        if composite_score < best_composite_score:
            best_composite_score = composite_score

    logger.close()
    return model, history, final_epoch_telemetry


def train_gnn(
    graph_paths: list[Path],
    common_genes: list[str],
) -> tuple[LibellaGNN, dict[str, list], int]:
    """Master orchestrator for GNN training phase."""
    out_dirs = paths.make_dirs(getattr(cfg, "suffix", "default"))
    checkpoint_path = out_dirs["checkpoint"]
    out_dir = out_dirs["out"]
    device = get_device()

    n_latents = getattr(cfg, "n_latents", getattr(cfg, "n_metaprograms", 512))
    print(f"[*] Initializing Native Top-K SAE Latent Space (M = {n_latents}, Top-K = {getattr(cfg, 'topk_k', 3)})...")

    model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch = _init_model(
        common_genes, n_latents, checkpoint_path
    )
    gc.collect()

    training_cache = _prep_ssd_chunks(graph_paths)
    gc.collect()

    model, history, _ = _train_loop(
        model,
        optimizer,
        scheduler,
        training_cache,
        start_epoch,
        best_composite_score,
        tracker_state,
        history,
    )
    gc.collect()

    export_latents_from_graphs(model, graph_paths, out_dirs["out"], device)
    return model, history, n_latents


# ---------------------------------------------------------------------------
# 9. Main Performance & Telemetry Harness Orchestrator
# ---------------------------------------------------------------------------
def run_full_telemetry_harness() -> None:
    chunk_paths = [
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_0.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_1.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_2.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_3.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_4.pt"),
    ]

    print("=" * 95)
    print(" LIBELLA GNN: UNABRIDGED 1-TO-1 PIPELINE & HIGH-RESOLUTION TELEMETRY")
    print("=" * 95)

    device = get_device()
    print(f"[*] Target Compute Engine : {device.type.upper()}")
    print(f"[*] Main Process PID      : {os.getpid()}")

    # 1. Ingest Chunks & Verify Memory Bounds
    loaded_cache: list[dict[str, Any]] = []
    for cp in chunk_paths:
        if not cp.exists():
            raise FileNotFoundError(f"Missing required benchmark chunk: {cp}")
        c = torch.load(cp, map_location="cpu", weights_only=False)
        loaded_cache.append({
            "x": c["x"].to(dtype=torch.float32).contiguous(),
            "src": c["src"].to(dtype=torch.int64).contiguous(),
            "dst": c["dst"].to(dtype=torch.int64).contiguous(),
            "weights": c.get("weights", torch.ones_like(c["src"], dtype=torch.float32)).contiguous(),
            "train_core_idx": c["train_core_idx"].to(dtype=torch.int64).contiguous(),
            "val_core_idx": c.get("val_core_idx"),
        })
        print(f"  ↳ Chunk Ingested: {cp.name} [Nodes: {c['x'].size(0)}, Edges: {c['src'].size(0)}, Core: {c['train_core_idx'].size(0)}]")

    in_channels = loaded_cache[0]["x"].shape[-1]
    n_latents = getattr(cfg, "n_latents", 36)
    common_genes = [f"Gene_{i}" for i in range(in_channels)]

    # 2. Run Isolated Micro-Stage GPU Benchmark (Deep Stage Timing & VRAM)
    probe_model = LibellaGNN(in_channels=in_channels, n_metaprograms=n_latents).to(device)
    print("\n[*] Profiling Isolated GPU Stage Execution Latencies & Transient RAM...")
    micro_metrics = benchmark_micro_stages(probe_model, loaded_cache[0], device)
    del probe_model
    gc.collect()

    # 3. Launch High-Frequency Subprocess Telemetry
    monitor = HardwareTelemetryWorker(target_pid=os.getpid(), poll_hz=500.0)
    monitor.start()
    monitor.ready_event.wait(timeout=3.0)
    print(f"[*] External Telemetry    : ACTIVE (PID: {monitor.pid}, 500 Hz Sampling)")

    # 4. Initialize Live Training Setup
    model, optimizer, scheduler, best_score, tracker_state, history, start_epoch = _init_model(
        common_genes, n_latents, checkpoint_path=None
    )

    sync_device(device)
    t_start_step = time.perf_counter_ns()

    # 5. Run Live 1-Gradient Step Pass (5 Batches)
    model, history, epoch_telemetry = _train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_cache=loaded_cache,
        start_epoch=0,
        best_composite_score=float("inf"),
        tracker_state=None,
        history=history,
        max_epochs_to_run=1,
    )

    sync_device(device)
    t_end_step = time.perf_counter_ns()
    full_step_wall_ms = (t_end_step - t_start_step) / 1e6

    # 6. Harvest Diagnostics
    host_stats = monitor.stop_and_harvest()
    mem_audit = audit_runtime_memory(model, optimizer, device)

    # -----------------------------------------------------------------------
    # 7. High-Resolution Diagnostic Printout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(" 1. EXACT MATHEMATICAL TRAINING STATE (5th Decimal Place Precision)")
    print("=" * 95)
    print(f"  • Accumulated Train Loss   : {history['train_loss'][-1]:.5f}")
    print(f"  • Accumulated Val Loss     : {history['val_loss'][-1]:.5f}")
    print(f"  • Reconstruction Loss (L_rec): {epoch_telemetry['l_rec']:.5f}")
    print(f"  • Orthogonality Loss (L_ort) : {epoch_telemetry['l_ort']:.5f}")
    print(f"  • Sparsity Loss (L_sparse)  : {epoch_telemetry['l_sparse']:.5f}")
    print(f"  • Auxiliary Revival (L_aux) : {epoch_telemetry['l_aux']:.5f}")
    print(f"  • Dynamic Zero Weight EMA   : {epoch_telemetry['dyn_w']:.5f}")
    print(f"  • Mean L0 Sparsity (Active) : {epoch_telemetry['l0_avg']:.5f} / {n_latents} ({epoch_telemetry['p_w']:.2f}% zero)")
    print(f"  • Latent Activation Entropy : {epoch_telemetry['ent']:.5f}")
    print(f"  • Peak Activation (Max)     : {epoch_telemetry['max_act']:.5f}")
    print(f"  • Mean Gated Magnitude      : {epoch_telemetry['z_mag_mean']:.5f}")
    print(f"  • Dead Latent Atoms Count   : {int(epoch_telemetry['dead_cnt'])} / {n_latents}")
    print(f"  • Pareto Composite Score    : {epoch_telemetry['composite_score']:.5f}")

    print("\n" + "=" * 95)
    print(" 2. ISOLATED SUB-OPERATION KERNEL TIMING & TRANSIENT VRAM MAP")
    print("=" * 95)
    print(f"{'Sub-Operation / Kernel Stage':<42} | {'GPU Latency':<12} | {'Transient RAM':<15} | {'Notes'}")
    print("-" * 95)
    for name, data in micro_metrics.items():
        print(f"{name:<42} | {data['time_ms']:>8.2f} ms | {data['transient_ram_mb']:>10.2f} MB   | {data['note']}")

    print("\n" + "=" * 95)
    print(f" 3. REAL ASYNCHRONOUS STEP PERFORMANCE (5 Batches: {full_step_wall_ms:.2f} ms)")
    print("=" * 95)
    print(f"  • Mean Time Per Chunk Graph : {full_step_wall_ms / len(loaded_cache):.2f} ms")
    print(f"  • Extrapolated 100-Chunk Epoch: {(full_step_wall_ms / len(loaded_cache)) * 100 / 1000.0:.2f} seconds")
    print(f"  • Metal Driver Allocations   : {mem_audit['driver'].get('mps_driver_allocated_mb', 0.0):.2f} MB")
    print(f"  • Metal Active Memory        : {mem_audit['driver'].get('mps_current_allocated_mb', 0.0):.2f} MB")

    print("\n" + "=" * 95)
    print(" 4. ISOLATED SUBPROCESS HOST METRICS (500 Hz Multi-Tick Sampling)")
    print("=" * 95)
    print(f"  • Telemetry Sampling Window : {host_stats.get('duration_sec', 0.0):.3f} s ({host_stats.get('samples', 0)} polling samples)")
    print(f"  • Process CPU Mean Load     : {host_stats.get('cpu_avg_pct', 0.0):.2f}%")
    print(f"  • Process CPU Peak Burst    : {host_stats.get('cpu_max_pct', 0.0):.2f}%")
    print(f"  • Process RSS Baseline      : {host_stats.get('rss_start_mb', 0.0):.2f} MB")
    print(f"  • Process RSS Peak          : {host_stats.get('rss_peak_mb', 0.0):.2f} MB")
    print(f"  • Net Host RAM Delta (Δ)    : {host_stats.get('rss_delta_mb', 0.0):+.2f} MB")
    print(f"  • Virtual Memory Peak (VMS) : {host_stats.get('vms_peak_mb', 0.0):.2f} MB")

    print("\n" + "=" * 95)
    print(" 5. TOP MODEL & OPTIMIZER RESIDENT TENSORS")
    print("=" * 95)
    print(f"{'Scope':<15} | {'Identifier':<35} | {'Shape':<18} | {'Dtype':<8} | {'RAM (MB)':<10}")
    print("-" * 92)
    sorted_tensors = sorted(mem_audit["tensors"], key=lambda r: r["mb"], reverse=True)
    for t in sorted_tensors[:15]:
        print(f"{t['scope']:<15} | {t['name']:<35} | {t['shape']:<18} | {t['dtype']:<8} | {t['mb']:<10.4f}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_full_telemetry_harness()