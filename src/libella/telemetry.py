#!/usr/bin/env python3
"""Libella Spatial GNN - 1-1 Training Step Replica & Zero-Interference Telemetry.

Runs 1 exact gradient accumulation step across the 5 benchmark chunks while monitoring
host CPU, RSS/VMS memory, per-tensor allocations, and micro-stage compute latency.
"""

from __future__ import annotations

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
# 1. Libella Source Resolution & Dynamic Imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
for candidate in [
    PROJECT_ROOT / "libella" / "src",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "libella",
    PROJECT_ROOT,
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from libella.config import cfg
except ImportError:
    class FallbackCfg:
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

    cfg = FallbackCfg()


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_mps_shapes(
    x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensures 16-byte memory alignment and contiguous strides for Metal kernels."""
    return x.contiguous(), src.contiguous(), dst.contiguous(), weights.contiguous()


# ---------------------------------------------------------------------------
# 2. Libella Architecture
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
        profiler: StageTimer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x_dense.size(0)
        has_edges = src.numel() > 0

        t0 = time.perf_counter_ns()
        if has_edges:
            src = src.contiguous()
            dst = dst.contiguous()
            edge_weights = edge_weights.contiguous()

        cell_mass = torch.clamp(
            torch.linalg.vector_norm(x_dense, ord=2, dim=-1, keepdim=True), min=1e-5
        )
        x_norm = x_dense / cell_mass
        h_self = self.self_enc(x_norm)
        if profiler:
            profiler.record("encode/self_enc", time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
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
        if profiler:
            profiler.record("encode/spatial_msg_pass", time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
        h_fused = F.layer_norm(h_self + h_sp, [self.hidden_dim])
        z_mag = self.mag_head(h_fused)

        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        bio_sim = F.linear(x_norm, w_dec_norm)
        spatial_shift = self.spatial_gate_head(h_sp)

        progress = getattr(self, "current_progress", 1.0) if self.training else 1.0
        spatial_warmup = 0.20 + 0.80 * min(1.0, progress * 2.0)

        raw_affinity = F.softplus(bio_sim + (self.spatial_gain * spatial_warmup * spatial_shift))
        pre_acts = raw_affinity * z_mag

        target_k = getattr(self, "current_k", self.k)
        topk_vals, topk_indices = torch.topk(pre_acts, k=target_k, dim=-1)
        z_sparse = torch.zeros_like(pre_acts).scatter_(-1, topk_indices, topk_vals)
        if profiler:
            profiler.record("encode/topk_activation", time.perf_counter_ns() - t0)

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
        profiler: StageTimer | None = None,
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
        z, pre_acts, cell_mass, z_mag = self.encode(x_dense, src, dst, edge_weights, profiler=profiler)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)

        t0 = time.perf_counter_ns()
        baseline_gene = F.normalize(F.softplus(self.decoder_bias) + 1e-6, p=2, dim=-1).unsqueeze(0)
        ambient_coeff = torch.sigmoid(self.ambient_scale) * getattr(cfg, "ambient_max_cap", 0.35)

        comp_profile = (1.0 - ambient_coeff) * torch.mm(z, w_dec_norm) + (ambient_coeff * baseline_gene)
        x_recon = comp_profile * cell_mass
        if profiler:
            profiler.record("forward/decode_recon", time.perf_counter_ns() - t0)

        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(self.n_latents, dtype=torch.bool, device=x_dense.device)

        if self.training:
            t0 = time.perf_counter_ns()
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
            if profiler:
                profiler.record("forward/aux_dead_revival", time.perf_counter_ns() - t0)

        return x_recon, z, w_dec_norm, aux_recon, r_norm, z_mag, r_pos_ret, dead_mask_ret

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
        asym_factor = 1.0 + (is_non_zero.to(x_true.dtype) * asym_penalty) * (raw_delta < 0).to(x_true.dtype)

        scaled_delta = raw_delta * asym_factor
        abs_delta = scaled_delta.abs()
        log_cosh_delta = abs_delta + F.softplus(-2.0 * abs_delta) - 0.6931471805599453

        per_cell_loss = torch.sum(variance_weight * log_cosh_delta, dim=-1)
        l_recon = torch.mean(per_cell_loss) / math.sqrt(x_true.shape[-1])

        # Orthogonality Barrier
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
# 3. Telemetry Subprocess & Zero-Overhead Profiler
# ---------------------------------------------------------------------------
@dataclass
class StageTimer:
    stages_ns: dict[str, int] = field(default_factory=dict)

    def record(self, name: str, delta_ns: int) -> None:
        self.stages_ns[name] = self.stages_ns.get(name, 0) + delta_ns


class ExternalProcessMonitor(mp.Process):
    """Monitors host CPU, RSS memory, and VMS memory at 500Hz without acquiring Python GIL."""

    def __init__(self, target_pid: int, interval_sec: float = 0.002) -> None:
        super().__init__(daemon=True)
        self.target_pid = target_pid
        self.interval = interval_sec
        self.stop_event = mp.Event()
        self.queue = mp.Queue()

    def run(self) -> None:
        try:
            proc = psutil.Process(self.target_pid)
        except psutil.NoSuchProcess:
            return

        cpu_samples = []
        rss_samples = []
        vms_samples = []
        proc.cpu_percent(interval=None)

        while not self.stop_event.is_set():
            try:
                cpu_samples.append(proc.cpu_percent(interval=None))
                mem = proc.memory_info()
                rss_samples.append(mem.rss / (1024 * 1024))
                vms_samples.append(mem.vms / (1024 * 1024))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

        summary = {
            "samples": len(cpu_samples),
            "cpu_avg": float(np.mean(cpu_samples)) if cpu_samples else 0.0,
            "cpu_max": float(np.max(cpu_samples)) if cpu_samples else 0.0,
            "rss_start_mb": float(rss_samples[0]) if rss_samples else 0.0,
            "rss_peak_mb": float(np.max(rss_samples)) if rss_samples else 0.0,
            "rss_delta_mb": float(rss_samples[-1] - rss_samples[0]) if rss_samples else 0.0,
            "vms_peak_mb": float(np.max(vms_samples)) if vms_samples else 0.0,
        }
        self.queue.put(summary)

    def harvest(self) -> dict[str, Any]:
        self.stop_event.set()
        self.join(timeout=2.0)
        return self.queue.get() if not self.queue.empty() else {}


def audit_memory_footprint(
    model: LibellaGNN,
    loaded_batches: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    records = []

    for name, p in model.named_parameters():
        mb = (p.nelement() * p.element_size()) / (1024 * 1024)
        records.append({
            "scope": "Parameter",
            "identifier": name,
            "shape": str(list(p.shape)),
            "dtype": str(p.dtype).replace("torch.", ""),
            "mb": round(mb, 4),
        })
        if p.grad is not None:
            g_mb = (p.grad.nelement() * p.grad.element_size()) / (1024 * 1024)
            records.append({
                "scope": "Gradient",
                "identifier": f"grad/{name}",
                "shape": str(list(p.grad.shape)),
                "dtype": str(p.grad.dtype).replace("torch.", ""),
                "mb": round(g_mb, 4),
            })

    for name, buf in model.named_buffers():
        mb = (buf.nelement() * buf.element_size()) / (1024 * 1024)
        records.append({
            "scope": "Buffer",
            "identifier": name,
            "shape": str(list(buf.shape)),
            "dtype": str(buf.dtype).replace("torch.", ""),
            "mb": round(mb, 4),
        })

    for idx, b in enumerate(loaded_batches):
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                mb = (v.nelement() * v.element_size()) / (1024 * 1024)
                records.append({
                    "scope": f"Batch[{idx}]",
                    "identifier": k,
                    "shape": str(list(v.shape)),
                    "dtype": str(v.dtype).replace("torch.", ""),
                    "mb": round(mb, 4),
                })

    for g_idx, group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(group["params"]):
            state = optimizer.state.get(p, {})
            for sk, sv in state.items():
                if isinstance(sv, torch.Tensor):
                    mb = (sv.nelement() * sv.element_size()) / (1024 * 1024)
                    records.append({
                        "scope": f"AdamW_G{g_idx}",
                        "identifier": f"p{p_idx}/{sk}",
                        "shape": str(list(sv.shape)),
                        "dtype": str(sv.dtype).replace("torch.", ""),
                        "mb": round(mb, 4),
                    })

    return records


# ---------------------------------------------------------------------------
# 4. Strict Chunk Validation & Execution Engine
# ---------------------------------------------------------------------------
def sanitize_chunk(chunk: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Sanitizes raw chunk tensors to eliminate MPS index out-of-bounds errors."""
    x = chunk["x"].to(dtype=torch.float32).contiguous()
    N = x.size(0)

    src = chunk["src"].to(dtype=torch.int64).contiguous()
    dst = chunk["dst"].to(dtype=torch.int64).contiguous()
    weights = chunk.get("weights", torch.ones_like(src, dtype=torch.float32)).to(dtype=torch.float32).contiguous()

    # Mask invalid edges referencing nodes outside local subgraph
    if src.numel() > 0:
        valid_mask = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        src = src[valid_mask].contiguous()
        dst = dst[valid_mask].contiguous()
        weights = weights[valid_mask].contiguous()

    train_core = chunk.get("train_core_idx", torch.arange(N, dtype=torch.int64)).to(dtype=torch.int64).contiguous()
    train_mask = (train_core >= 0) & (train_core < N)
    train_core = train_core[train_mask].contiguous()

    val_core = chunk.get("val_core_idx")
    if val_core is not None and val_core.numel() > 0:
        val_core = val_core.to(dtype=torch.int64).contiguous()
        val_mask = (val_core >= 0) & (val_core < N)
        val_core = val_core[val_mask].contiguous()

    return {
        "x": x,
        "src": src,
        "dst": dst,
        "weights": weights,
        "train_core_idx": train_core,
        "val_core_idx": val_core,
    }


def main() -> None:
    chunk_paths = [
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_0.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_1.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_2.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_3.pt"),
        Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_4.pt"),
    ]

    print("=" * 85)
    print(" LIBELLA GNN: EXACT TRAINING STEP & DEEP HARDWARE TELEMETRY")
    print("=" * 85)

    device = get_device()
    print(f"[*] Target Compute Device : {device.type.upper()}")
    print(f"[*] Main Process PID      : {os.getpid()}")

    # Launch non-blocking external observer subprocess
    monitor = ExternalProcessMonitor(target_pid=os.getpid(), interval_sec=0.002)
    monitor.start()
    print(f"[*] Subprocess Monitor    : ACTIVE (PID: {monitor.pid})")

    # Load and sanitize chunks
    sanitized_chunks: list[dict[str, torch.Tensor]] = []
    for cp in chunk_paths:
        if not cp.exists():
            print(f"  [!] Chunk path not found: {cp}. Generating fallback...")
            N_nodes, E_edges, genes = 6194, 24000, 500
            raw = {
                "x": torch.randn(N_nodes, genes).abs(),
                "src": torch.randint(0, N_nodes, (E_edges,), dtype=torch.int64),
                "dst": torch.randint(0, N_nodes, (E_edges,), dtype=torch.int64),
                "weights": torch.rand(E_edges, dtype=torch.float32),
                "train_core_idx": torch.arange(0, int(N_nodes * 0.8), dtype=torch.int64),
                "val_core_idx": torch.arange(int(N_nodes * 0.8), N_nodes, dtype=torch.int64),
            }
        else:
            raw = torch.load(cp, map_location="cpu", weights_only=False)
        sanitized_chunks.append(sanitize_chunk(raw))

    in_channels = sanitized_chunks[0]["x"].shape[-1]
    n_latents = getattr(cfg, "n_latents", 512)
    print(f"[*] Chunks Verified       : {len(sanitized_chunks)} | In-Channels: {in_channels} | Latents: {n_latents}")

    # Initialize Model & Exact Optimizer Groupings
    model = LibellaGNN(in_channels=in_channels, n_metaprograms=n_latents).to(device)

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

    timer = StageTimer()
    meta_batch_size = len(sanitized_chunks)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_loss_acc = torch.tensor(0.0, device=device)
    last_r_pos = None
    last_dead_mask = None
    device_chunks = []

    t_wall_start = time.perf_counter_ns()

    # -----------------------------------------------------------------------
    # Gradient Accumulation Over 5 Batches
    # -----------------------------------------------------------------------
    for b_idx, batch in enumerate(sanitized_chunks):
        t0 = time.perf_counter_ns()
        x = batch["x"].to(device=device, non_blocking=True).contiguous()
        src = batch["src"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        dst = batch["dst"].to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
        weights = batch["weights"].to(device=device, dtype=torch.float32, non_blocking=True).contiguous()

        x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
        device_chunks.append({"x": x, "src": src, "dst": dst, "weights": weights})
        timer.record("host_to_device_transfer", time.perf_counter_ns() - t0)

        model.current_progress = 0.0
        model.current_k = getattr(cfg, "topk_k", 3)

        (
            recon,
            z,
            w_dec_norm,
            aux_recon,
            r_norm,
            z_mag,
            r_pos,
            dead_mask,
        ) = model(x, src, dst, weights, profiler=timer)

        last_r_pos = r_pos
        last_dead_mask = dead_mask

        t0 = time.perf_counter_ns()
        train_idx = batch["train_core_idx"].to(device=device, dtype=torch.int64, non_blocking=True)
        x_train = x[train_idx]
        recon_train = recon[train_idx]
        z_train = z[train_idx]
        aux_recon_train = aux_recon[train_idx] if aux_recon is not None else None
        r_norm_train = r_norm[train_idx] if r_norm is not None else None

        loss_res = model.calc_loss(
            recon_train,
            x_train,
            z_train,
            w_dec_norm,
            aux_recon=aux_recon_train,
            r_norm=r_norm_train,
            progress=0.0,
        )
        batch_loss = loss_res[0]
        timer.record("loss_computation", time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
        (batch_loss / meta_batch_size).backward()
        timer.record("backward_autograd", time.perf_counter_ns() - t0)

        step_loss_acc.add_(batch_loss.detach() / meta_batch_size)

    # -----------------------------------------------------------------------
    # Step Updates (Clipping, Tangent Projection, Step, Retraction)
    # -----------------------------------------------------------------------
    t0 = time.perf_counter_ns()
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
        torch.nn.utils.clip_grad_norm_(recon_params, max_norm=getattr(cfg, "grad_clip_recon", 5.0))
    if spatial_params:
        torch.nn.utils.clip_grad_norm_(spatial_params, max_norm=getattr(cfg, "grad_clip_spatial", 15.0))
    timer.record("gradient_clipping", time.perf_counter_ns() - t0)

    # Tangent Projection
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        if hasattr(model, "decoder_weight") and model.decoder_weight.grad is not None:
            w = F.normalize(model.decoder_weight.data, p=2, dim=1)
            grad = model.decoder_weight.grad
            grad.sub_((grad * w).sum(dim=1, keepdim=True) * w)
    timer.record("tangent_space_projection", time.perf_counter_ns() - t0)

    # Optimizer Step
    t0 = time.perf_counter_ns()
    optimizer.step()
    timer.record("optimizer_adamw_step", time.perf_counter_ns() - t0)

    # Spherical Retraction & Latent Resampling
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        if hasattr(model, "decoder_weight"):
            w_data = model.decoder_weight.data
            w_data.clamp_min_(0.0)
            w_norm = torch.linalg.vector_norm(w_data + 1e-8, ord=2, dim=-1, keepdim=True)
            w_data.div_(w_norm)

        if last_dead_mask is not None and last_dead_mask.any() and last_r_pos is not None:
            model.resample_dead_latents(last_r_pos, last_dead_mask, optimizer=optimizer)
    timer.record("spherical_retraction", time.perf_counter_ns() - t0)

    # Benchmark Barrier
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    t_wall_end = time.perf_counter_ns()
    total_wall_ms = (t_wall_end - t_wall_start) / 1e6

    # Collect Telemetry
    host_stats = monitor.harvest()
    memory_audit = audit_memory_footprint(model, device_chunks, optimizer)

    # -----------------------------------------------------------------------
    # Diagnostic Reports
    # -----------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 1. EXTERNAL SUBPROCESS HOST METRICS (Zero GIL / Zero Lock Contention)")
    print("=" * 85)
    print(f"  • Telemetry Poll Samples : {host_stats.get('samples', 0)}")
    print(f"  • CPU Utilization (Avg)  : {host_stats.get('cpu_avg', 0.0):.2f}%")
    print(f"  • CPU Utilization (Peak) : {host_stats.get('cpu_max', 0.0):.2f}%")
    print(f"  • Process RSS Baseline   : {host_stats.get('rss_start_mb', 0.0):.2f} MB")
    print(f"  • Process RSS Peak       : {host_stats.get('rss_peak_mb', 0.0):.2f} MB")
    print(f"  • Net RAM Expansion (Δ)  : {host_stats.get('rss_delta_mb', 0.0):+.2f} MB")
    print(f"  • Virtual Memory Peak    : {host_stats.get('vms_peak_mb', 0.0):.2f} MB")

    print("\n" + "=" * 85)
    print(f" 2. PIPELINE STAGE LATENCIES (Total Compute Step: {total_wall_ms:.2f} ms)")
    print("=" * 85)
    print(f"{'Pipeline Execution Stage':<35} | {'Latency (ms)':<15} | {'% of Total':<10}")
    print("-" * 68)
    for stage, ns in timer.stages_ns.items():
        ms = ns / 1e6
        pct = (ms / total_wall_ms) * 100.0 if total_wall_ms > 0 else 0.0
        print(f"{stage:<35} | {ms:<15.3f} | {pct:<9.2f}%")

    print("\n" + "=" * 85)
    print(" 3. TOP TENSOR RESIDENCY MAP (Parameters, Gradients, AdamW States)")
    print("=" * 85)
    print(f"{'Scope':<15} | {'Identifier':<35} | {'Shape':<20} | {'Dtype':<8} | {'RAM (MB)':<10}")
    print("-" * 95)
    sorted_mem = sorted(memory_audit, key=lambda r: r["mb"], reverse=True)
    total_mb = sum(r["mb"] for r in memory_audit)
    for r in sorted_mem[:20]:
        print(f"{r['scope']:<15} | {r['identifier']:<35} | {r['shape']:<20} | {r['dtype']:<8} | {r['mb']:<10.4f}")
    print("-" * 95)
    print(f"  Total Explicitly Tracked Tensor Memory: {total_mb:.2f} MB")

    print("\n" + "=" * 85)
    print(" 4. NUMERICAL STATE & INVARIANT VERIFICATION")
    print("=" * 85)
    print(f"  • Step Accumulated Loss  : {step_loss_acc.item():.6f}")
    print(f"  • Dynamic Zero Weight EMA : {model.dynamic_w_ema.item():.4f}")
    print(f"  • Ambient Decouple Coeff  : {torch.sigmoid(model.ambient_scale).item() * 0.35:.4f}")
    print(f"  • Spherical Atom Unit Norm: {torch.allclose(torch.linalg.vector_norm(model.decoder_weight.data, ord=2, dim=-1), torch.ones(n_latents, device=device), atol=1e-3)}")
    print(f"  • Active Latent Ratio     : {(model.steps_since_active < model.dead_step_threshold).sum().item()}/{n_latents}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()