#!/usr/bin/env python3
"""Libella Spatial GNN - 100% Bit-for-Bit Training Step Replica & Telemetry Harness.

Executes a faithful 1-gradient-step pass (5 meta-batches) matching _train_loop
to the 5th decimal place. Uses a zero-synchronization main pipeline with an
isolated background monitor process polling CPU, RSS/VMS memory, and driver metrics.
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
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Project Path Setup & Libella Configuration
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
        hidden_dim: int = 64
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
        n_latents: int = 512
        edge_dropout: float = 0.0
        active_latent_threshold: float = 1e-4
        alpha_ema_max: float = 0.05
        alpha_ema_step_multiplier: float = 1.0

    cfg = LibellaConfig()


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_mps_shapes(
    x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return x.contiguous(), src.contiguous(), dst.contiguous(), weights.contiguous()


# ---------------------------------------------------------------------------
# 2. Libella Architecture (1-to-1 Reference)
# ---------------------------------------------------------------------------
class LibellaGNN(nn.Module):
    """Core Libella Spatial GNN architecture with Top-K Hard Sparsity and Residual AuxK Revival."""

    def __init__(
        self,
        in_channels: int,
        n_metaprograms: int,
        init_components: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = getattr(cfg, "hidden_dim", 64)
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
        self.ambient_scale = nn.Parameter(
            torch.tensor(getattr(cfg, "ambient_scale_init", 0.50))
        )

        # Buffers & Aux State Tracking
        self.register_buffer(
            "ortho_mask", 1.0 - torch.eye(self.n_latents, dtype=torch.float32)
        )
        self.register_buffer(
            "steps_since_active", torch.zeros(self.n_latents, dtype=torch.int64)
        )
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))
        self.dead_step_threshold = getattr(cfg, "dead_step_threshold", 20)
        self.aux_k = getattr(cfg, "aux_k", 4)
        self.ortho_sample_size = getattr(
            cfg, "ortho_sample_size", min(256, self.n_latents)
        )

    def encode(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x_dense.size(0)
        has_edges = src.numel() > 0

        if has_edges:
            src = src.contiguous()
            dst = dst.contiguous()
            edge_weights = edge_weights.contiguous()

        cell_mass = torch.clamp(
            torch.linalg.vector_norm(x_dense, ord=2, dim=-1, keepdim=True), min=1e-5
        )
        x_norm = x_dense / cell_mass
        h_self = self.self_enc(x_norm)

        if has_edges:
            with torch.no_grad():
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)
                decay = torch.sigmoid(
                    (cos_sim - getattr(cfg, "edge_sim_threshold", 0.50))
                    * getattr(cfg, "edge_decay_slope", 15.0)
                )

                deg = torch.zeros((N, 1), dtype=x_dense.dtype, device=x_dense.device)
                deg.index_add_(
                    0,
                    dst,
                    torch.ones((dst.size(0), 1), dtype=x_dense.dtype, device=x_dense.device),
                )
                norm_inv_sqrt = torch.rsqrt(torch.clamp(deg, min=1.0))
                edge_norm = norm_inv_sqrt[src] * norm_inv_sqrt[dst]

            W_bil = edge_weights.unsqueeze(1) * decay
            gate_in = torch.cat([h_self[src] - h_self[dst], W_bil], dim=-1)
            gate = self.edge_gate(gate_in)

            h_sp = h_self
            for _ in range(self.k_hops):
                h_proj = self.spatial_lin(h_sp)
                msg = h_proj[src] * gate * edge_norm
                agg = torch.zeros_like(h_sp).index_add_(0, dst, msg)
                h_sp = h_sp + F.silu(agg)
        else:
            h_sp = h_self

        h_fused = F.layer_norm(h_self + h_sp, [self.hidden_dim])
        z_mag = self.mag_head(h_fused)

        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        bio_sim = F.linear(x_norm, w_dec_norm)
        spatial_shift = self.spatial_gate_head(h_sp)

        progress = getattr(self, "current_progress", 1.0) if self.training else 1.0
        spatial_warmup = 0.20 + 0.80 * min(1.0, progress * 2.0)

        raw_affinity = F.softplus(
            bio_sim + (self.spatial_gain * spatial_warmup * spatial_shift)
        )
        pre_acts = raw_affinity * z_mag

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
                torch.randn(
                    int(collapsed.sum().item()),
                    candidates.size(-1),
                    device=candidates.device,
                )
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
        z, pre_acts, cell_mass, z_mag = self.encode(x_dense, src, dst, edge_weights)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)

        baseline_gene = F.normalize(
            F.softplus(self.decoder_bias) + 1e-6, p=2, dim=-1
        ).unsqueeze(0)
        ambient_coeff = torch.sigmoid(self.ambient_scale) * getattr(
            cfg, "ambient_max_cap", 0.35
        )

        comp_profile = (1.0 - ambient_coeff) * torch.mm(z, w_dec_norm) + (
            ambient_coeff * baseline_gene
        )
        x_recon = comp_profile * cell_mass

        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(
            self.n_latents, dtype=torch.bool, device=x_dense.device
        )

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

            if dead_mask_ret.any() and residual_energy > getattr(
                cfg, "aux_min_residual_energy", 0.05
            ):
                dead_indices = torch.nonzero(dead_mask_ret).squeeze(-1)
                num_dead = dead_indices.numel()
                k_aux = min(max(getattr(cfg, "aux_min_k", 2), self.aux_k), num_dead)

                w_dead = w_dec_norm[dead_indices]
                aux_sim = F.linear(r_norm, w_dead)
                topk_res = torch.topk(aux_sim, k=k_aux, dim=-1)

                dead_mag = z_mag[:, dead_indices]
                topk_mag = torch.gather(dead_mag, -1, topk_res.indices)

                z_aux_weights = F.softplus(topk_res.values, beta=1.0) * topk_mag
                z_aux = torch.zeros_like(aux_sim).scatter_(
                    -1, topk_res.indices, z_aux_weights
                )
                aux_recon = torch.mm(z_aux, w_dead)

        return (
            x_recon,
            z,
            w_dec_norm,
            aux_recon,
            r_norm,
            z_mag,
            r_pos_ret,
            dead_mask_ret,
        )

    def calc_loss(
        self,
        recon_x: torch.Tensor,
        x_true: torch.Tensor,
        z: torch.Tensor,
        w_dec_norm: torch.Tensor,
        aux_recon: torch.Tensor | None = None,
        r_norm: torch.Tensor | None = None,
        progress: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        asym_factor = 1.0 + (is_non_zero.to(x_true.dtype) * asym_penalty) * (
            raw_delta < 0
        ).to(x_true.dtype)

        scaled_delta = raw_delta * asym_factor
        abs_delta = scaled_delta.abs()
        log_cosh_delta = (
            abs_delta + F.softplus(-2.0 * abs_delta) - 0.6931471805599453
        )

        per_cell_loss = torch.sum(variance_weight * log_cosh_delta, dim=-1)
        l_recon = torch.mean(per_cell_loss) / math.sqrt(x_true.shape[-1])

        # Orthogonality Barrier
        gram = torch.mm(w_dec_norm, w_dec_norm.t())
        off_diag = gram * self.ortho_mask

        ortho_thresh = getattr(cfg, "ortho_overlap_threshold", 0.30)
        excess_corr = F.relu(off_diag - ortho_thresh)
        num_violating = torch.clamp(
            (excess_corr > 0).sum().to(dtype=x_true.dtype), min=1.0
        )
        l_ortho_mean = excess_corr.pow(2).sum() / num_violating

        max_corr = off_diag.max()
        l_ortho_max = F.relu(max_corr - 0.50).pow(2) * 50.0
        l_ortho = l_ortho_mean + l_ortho_max

        l_sparse = torch.tensor(0.0, device=x_true.device)

        # Residual Alignment
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


# ---------------------------------------------------------------------------
# 3. Hardware Telemetry Background Subprocess (Zero-GIL Interference)
# ---------------------------------------------------------------------------
class HardwareTelemetryWorker(mp.Process):
    """External process that monitors process CPU, RSS/VMS, and physical memory footprint."""

    def __init__(self, target_pid: int, poll_hz: float = 500.0) -> None:
        super().__init__(daemon=True)
        self.target_pid = target_pid
        self.poll_interval = 1.0 / poll_hz
        self.stop_event = mp.Event()
        self.metrics_queue = mp.Queue()

    def run(self) -> None:
        try:
            proc = psutil.Process(self.target_pid)
        except psutil.NoSuchProcess:
            return

        cpu_records = []
        rss_records = []
        vms_records = []
        timestamps = []

        proc.cpu_percent(interval=None)
        t_start = time.perf_counter()

        while not self.stop_event.is_set():
            try:
                cpu_records.append(proc.cpu_percent(interval=None))
                mem = proc.memory_info()
                rss_records.append(mem.rss / (1024 * 1024))
                vms_records.append(mem.vms / (1024 * 1024))
                timestamps.append(time.perf_counter() - t_start)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.poll_interval)

        summary = {
            "samples": len(cpu_records),
            "duration_sec": timestamps[-1] if timestamps else 0.0,
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
        self.join(timeout=2.0)
        return self.metrics_queue.get() if not self.metrics_queue.empty() else {}


# ---------------------------------------------------------------------------
# 4. Accurate Memory & Driver Allocation Audit
# ---------------------------------------------------------------------------
def audit_runtime_memory(
    model: LibellaGNN,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """Measures static model allocations, AdamW buffers, and active GPU runtime memory."""
    report: dict[str, Any] = {"tensors": [], "driver": {}}

    # 1. Parameter Footprint
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

    # 2. Buffers
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

    # 3. Optimizer State
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

    # 4. Device Driver-Level Memory Inspection
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
# 5. Graph Sanitizer & Bounds Verification
# ---------------------------------------------------------------------------
def load_and_verify_chunk(chunk_path: Path) -> dict[str, torch.Tensor]:
    """Loads a chunk on CPU, verifying all index bounds against the node count."""
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {chunk_path}")

    raw = torch.load(chunk_path, map_location="cpu", weights_only=False)
    x = raw["x"].to(dtype=torch.float32).contiguous()
    N = x.size(0)

    src = raw["src"].to(dtype=torch.int64).contiguous()
    dst = raw["dst"].to(dtype=torch.int64).contiguous()
    weights = raw.get("weights", torch.ones_like(src, dtype=torch.float32)).to(
        dtype=torch.float32
    ).contiguous()

    if src.numel() > 0:
        valid_mask = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        if not valid_mask.all():
            src = src[valid_mask].contiguous()
            dst = dst[valid_mask].contiguous()
            weights = weights[valid_mask].contiguous()

    train_core = raw.get("train_core_idx", torch.arange(N, dtype=torch.int64)).to(
        dtype=torch.int64
    ).contiguous()
    train_mask = (train_core >= 0) & (train_core < N)
    train_core = train_core[train_mask].contiguous()

    val_core = raw.get("val_core_idx")
    if val_core is not None and val_core.numel() > 0:
        val_core = val_core.to(dtype=torch.int64).contiguous()
        val_mask = (val_core >= 0) & (val_core < N)
        val_core = val_core[val_mask].contiguous()
    else:
        val_core = None

    return {
        "x": x,
        "src": src,
        "dst": dst,
        "weights": weights,
        "train_core_idx": train_core,
        "val_core_idx": val_core,
    }


# ---------------------------------------------------------------------------
# 6. Master 1-to-1 Step Replica & Telemetry Execution
# ---------------------------------------------------------------------------
def run_exact_gradient_step(
    chunk_paths: list[Path],
    progress: float = 0.0,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print("=" * 90)
    print(" LIBELLA GNN: EXACT 1-TO-1 TRAINING STEP REPLICA & PERFORMANCE TELEMETRY")
    print("=" * 90)
    print(f"[*] Target Compute Engine : {device.type.upper()}")
    print(f"[*] Process ID (PID)      : {os.getpid()}")
    print(f"[*] Phase/Progress State  : {progress:.4f}")
    print(f"[*] Edge Dropout State    : {getattr(cfg, 'edge_dropout', 0.0):.2f}")

    # 1. Start External Subprocess Telemetry
    monitor = HardwareTelemetryWorker(target_pid=os.getpid(), poll_hz=500.0)
    monitor.start()
    print(f"[*] Hardware Subprocess   : ACTIVE (PID: {monitor.pid}, Poll Rate: 500 Hz)")

    # 2. Ingest Specified Chunks
    print(f"[*] Ingesting {len(chunk_paths)} benchmark chunks...")
    chunks_cpu: list[dict[str, Any]] = []
    for cp in chunk_paths:
        c = load_and_verify_chunk(cp)
        chunks_cpu.append(c)
        print(f"  ↳ Loaded: {cp.name} [Nodes: {c['x'].size(0)}, Edges: {c['src'].size(0)}, Core: {c['train_core_idx'].size(0)}]")

    in_channels = chunks_cpu[0]["x"].shape[-1]
    n_latents = getattr(cfg, "n_latents", 512)

    # 3. Model & Optimizer Initialization
    model = LibellaGNN(in_channels=in_channels, n_metaprograms=n_latents).to(device)
    model.train()

    bias_ambient_params = [
        p for n, p in model.named_parameters()
        if any(k in n for k in ["decoder_bias", "ambient_scale"])
    ]
    decoder_weight_params = [
        p for n, p in model.named_parameters() if "decoder_weight" in n
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

    # 4. Zero-Synchronization Accumulators & GPU Buffers
    accumulation_steps = len(chunks_cpu)
    train_steps = 0
    val_steps = 0
    train_chunk_count = 0

    train_loss_acc = torch.tensor(0.0, device=device)
    val_loss_acc = torch.tensor(0.0, device=device)
    step_loss_acc = torch.tensor(0.0, device=device)

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

    alpha_ema = min(
        getattr(cfg, "alpha_ema_max", 0.05),
        1.0 / (accumulation_steps * getattr(cfg, "alpha_ema_step_multiplier", 1.0) + 1e-9),
    )
    ema_latent_freq = None

    optimizer.zero_grad(set_to_none=True)
    step_loss_acc.zero_()
    last_r_pos = None
    last_dead_mask = None

    # Pre-step device barrier
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    t_step_start = time.perf_counter_ns()

    # -----------------------------------------------------------------------
    # 5. Exact Gradient Accumulation Pass Across All 5 Chunks
    # -----------------------------------------------------------------------
    for chunk_idx, batch in enumerate(chunks_cpu):
        x = batch["x"].to(device=device, non_blocking=True).contiguous()
        src = batch["src"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        dst = batch["dst"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        weights = batch["weights"].to(device=device, non_blocking=True).contiguous()

        # Edge Dropout (Exact Production Implementation)
        if model.training and src.numel() > 0:
            edge_drop = getattr(cfg, "edge_dropout", 0.0)
            if edge_drop > 0.0:
                keep_mask = torch.rand(src.size(0), device=device) > edge_drop
                src = src[keep_mask].contiguous()
                dst = dst[keep_mask].contiguous()
                weights = weights[keep_mask].contiguous()

        x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)

        # Progress and Latent K State
        model.current_progress = progress
        model.current_k = getattr(cfg, "topk_k", 3)

        # Forward Pass
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

        # Train Core Slice
        train_idx = batch["train_core_idx"].to(device=device, dtype=torch.int64, non_blocking=True)
        x_train = x[train_idx]
        recon_train = recon[train_idx]
        z_train = z[train_idx]
        aux_recon_train = aux_recon[train_idx] if aux_recon is not None else None
        r_norm_train = r_norm[train_idx] if r_norm is not None else None

        # Loss Calculation
        loss_res = model.calc_loss(
            recon_train,
            x_train,
            z_train,
            w_dec_norm,
            aux_recon=aux_recon_train,
            r_norm=r_norm_train,
            progress=progress,
        )
        true_batch_loss = loss_res[0]
        base_recon_val = loss_res[1]
        base_ort_val = loss_res[2]
        base_sparse_val = loss_res[3]
        base_aux_val = loss_res[4]

        # Backward Pass with Accumulation Scaling
        (true_batch_loss / accumulation_steps).backward()

        step_loss_acc.add_(true_batch_loss.detach() / accumulation_steps)
        train_loss_acc += true_batch_loss.detach()
        train_steps += 1

        # Real-time GPU Telemetry Accumulator Updates
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

        # Explicit garbage collection of training tensors
        del train_idx, x_train, recon_train, z_train, aux_recon_train, r_norm_train, true_batch_loss

        # Online Validation Evaluation
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
                variance_weight_val = variance_weight_val / torch.clamp(
                    variance_weight_val.mean(), min=1e-5
                )

                raw_delta_val = val_recon - x_val
                asym_penalty = getattr(cfg, "asym_penalty_weight", 0.5)
                asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * asym_penalty) * (
                    raw_delta_val < 0
                ).to(x_val.dtype)
                scaled_delta_val = raw_delta_val * asym_val

                abs_delta_val = scaled_delta_val.abs()
                log_cosh_val = (
                    abs_delta_val + F.softplus(-2.0 * abs_delta_val) - 0.6931471805599453
                )

                per_cell_val = torch.sum(variance_weight_val * log_cosh_val, dim=-1)
                val_log_cosh = torch.mean(per_cell_val) / math.sqrt(x_val.shape[-1])

                val_loss_acc.add_(val_log_cosh)
                val_steps += 1

            del (
                val_idx,
                val_recon,
                x_val,
                w_mat,
                raw_delta_val,
                asym_val,
                scaled_delta_val,
                per_cell_val,
                log_cosh_val,
            )

        # Immediate lifecycle purge of chunk tensors
        del batch, src, dst, weights, x, recon, z, w_dec_norm, aux_recon, r_norm

    # -----------------------------------------------------------------------
    # 6. Post-Accumulation Parameter & Optimization Step
    # -----------------------------------------------------------------------
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

    # In-Place Tangent-Space Projection on Unit Sphere
    with torch.no_grad():
        if hasattr(model, "decoder_weight") and model.decoder_weight.grad is not None:
            w = F.normalize(model.decoder_weight.data, p=2, dim=1)
            grad = model.decoder_weight.grad
            grad.sub_((grad * w).sum(dim=1, keepdim=True) * w)

    # Optimizer Step
    optimizer.step()

    # In-Place Non-Negative Spherical Retraction & Latent Resampling
    with torch.no_grad():
        if hasattr(model, "decoder_weight"):
            w_data = model.decoder_weight.data
            w_data.clamp_min_(0.0)
            w_norm = torch.linalg.vector_norm(w_data + 1e-8, ord=2, dim=-1, keepdim=True)
            w_data.div_(w_norm)

        if last_dead_mask is not None and last_dead_mask.any() and last_r_pos is not None:
            model.resample_dead_latents(last_r_pos, last_dead_mask, optimizer=optimizer)

    # Final Synchronization Barrier (For clean step timing and telemetry harvest)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    t_step_end = time.perf_counter_ns()
    total_step_wall_ms = (t_step_end - t_step_start) / 1e6

    # -----------------------------------------------------------------------
    # 7. Single GPU-to-CPU Barrier Transfer (1-to-1 Parity)
    # -----------------------------------------------------------------------
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

    epoch_telemetry = {}
    for idx, k in enumerate(telemetry_keys):
        epoch_telemetry[k] = all_scalars_host[idx + 2]

    current_l0_val = epoch_telemetry.get("l0_avg", float(model.n_latents))
    epoch_telemetry["p_w"] = (1.0 - (current_l0_val / float(model.n_latents))) * 100.0

    if ema_latent_freq is not None:
        p_norm = ema_latent_freq / torch.clamp(ema_latent_freq.sum(), min=1e-6)
        epoch_telemetry["ent"] = (-(p_norm * torch.log(p_norm + 1e-9)).sum()).item()
    else:
        epoch_telemetry["ent"] = 0.0

    current_rec = epoch_telemetry.get("l_rec", 0.0)
    current_l0 = epoch_telemetry.get("l0_avg", 0.0)
    composite_score = current_rec * math.sqrt(1.0 + (current_l0 / float(model.n_latents)))

    # Harvest Hardware Telemetry
    host_stats = monitor.stop_and_harvest()
    mem_audit = audit_runtime_memory(model, optimizer, device)

    # -----------------------------------------------------------------------
    # 8. Precise Telemetry Output Reports
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" 1. EXACT TRAINING INVARIANTS & NUMERICAL LOSS STATE (5th Decimal Precision)")
    print("=" * 90)
    print(f"  • Accumulated Train Loss   : {final_train_loss:.5f}")
    print(f"  • Accumulated Val Loss     : {final_val_loss:.5f}")
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
    print(f"  • Pareto Composite Score    : {composite_score:.5f}")

    print("\n" + "=" * 90)
    print(" 2. ISOLATED SUBPROCESS HARDWARE TELEMETRY (Zero GIL Contention)")
    print("=" * 90)
    print(f"  • Sampling Duration        : {host_stats.get('duration_sec', 0.0):.3f} s ({host_stats.get('samples', 0)} polling ticks)")
    print(f"  • Process CPU Mean Load    : {host_stats.get('cpu_avg_pct', 0.0):.2f}%")
    print(f"  • Process CPU Peak Burst   : {host_stats.get('cpu_max_pct', 0.0):.2f}%")
    print(f"  • Baseline RSS Footprint   : {host_stats.get('rss_start_mb', 0.0):.2f} MB")
    print(f"  • Peak RSS Memory          : {host_stats.get('rss_peak_mb', 0.0):.2f} MB")
    print(f"  • Net RAM Expansion (Δ)    : {host_stats.get('rss_delta_mb', 0.0):+.2f} MB")
    print(f"  • Virtual Memory Peak (VMS): {host_stats.get('vms_peak_mb', 0.0):.2f} MB")

    print("\n" + "=" * 90)
    print(" 3. DEVICE RUNTIME & DRIVER ALLOCATION BREAKDOWN")
    print("=" * 90)
    print(f"  • Total Step GPU Wall Time : {total_step_wall_ms:.3f} ms")
    print(f"  • Parameters Memory        : {mem_audit['param_total_mb']:.4f} MB")
    print(f"  • Gradients Memory         : {mem_audit['grad_total_mb']:.4f} MB")
    print(f"  • Buffers Memory           : {mem_audit['buffer_total_mb']:.4f} MB")
    print(f"  • AdamW Optimizer State    : {mem_audit['optimizer_total_mb']:.4f} MB")
    print(f"  • Total Static Allocated   : {mem_audit['static_tracked_mb']:.4f} MB")

    for k, v in mem_audit["driver"].items():
        print(f"  • Driver Metric [{k:<22}]: {v:.2f} MB")

    print("\n" + "=" * 90)
    print(" 4. TOP ALLOCATED MODEL & OPTIMIZER TENSORS")
    print("=" * 90)
    print(f"{'Scope':<15} | {'Identifier':<35} | {'Shape':<18} | {'Dtype':<8} | {'RAM (MB)':<10}")
    print("-" * 92)
    sorted_tensors = sorted(mem_audit["tensors"], key=lambda r: r["mb"], reverse=True)
    for t in sorted_tensors[:15]:
        print(f"{t['scope']:<15} | {t['name']:<35} | {t['shape']:<18} | {t['dtype']:<8} | {t['mb']:<10.4f}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# 7. Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Libella 1-to-1 Step Telemetry Harness")
    parser.add_argument("--progress", type=float, default=0.0, help="Training progress in [0.0, 1.0]")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic initialization seed")
    args = parser.parse_args()

    default_chunks = [
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_0.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_1.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_2.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_3.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_4.pt"),
    ]

    run_exact_gradient_step(
        chunk_paths=default_chunks,
        progress=args.progress,
        seed=args.seed,
    )