#!/usr/bin/env python3
"""Libella Full-Epoch Master Evolutionary Pilot Engine.

Runs Baseline and 5 evolved architectures across full epochs with:
1. Complete OLS PhaseTracker elastic governor (Phase 1 discovery + Phase 2 squeeze/breathe).
2. Full synchronized 4-way sharpening schedules (Alpha, Scale, Temp, Dynamic Prob Blend).
3. Decoupled CosineAnnealingLR scheduling and Pareto composite score checkpointing.
4. Comprehensive trajectory telemetry export (CSV) and final comparative health scorecard.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. OPTIMIZED NUMERICAL OPERATORS & UTILITIES
# =============================================================================

def entmax_bisect_fn(X: torch.Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 25) -> torch.Tensor:
    """Exact native PyTorch implementation of Entmax-alpha with no-grad bisection."""
    if abs(alpha - 1.0) < 1e-4:
        return F.softmax(X, dim=dim)
    if abs(alpha - 2.0) < 1e-4:
        sorted_logits, _ = torch.sort(X, descending=True, dim=dim)
        z = torch.cumsum(sorted_logits, dim=dim)
        k = torch.arange(1, X.size(dim) + 1, device=X.device, dtype=X.dtype)
        bound = 1 + k * sorted_logits > z
        rho = torch.sum(bound.to(X.dtype), dim=dim, keepdim=True)
        tau = (torch.gather(z, dim, (rho - 1).long()) - 1) / rho
        return torch.clamp(X - tau, min=0.0)

    p = alpha
    with torch.no_grad():
        max_X = X.max(dim=dim, keepdim=True).values
        tau_lo = max_X - (1.0 / (p - 1.0))
        tau_hi = max_X

        for _ in range(n_iter):
            tau = (tau_lo + tau_hi) / 2.0
            v = torch.clamp((p - 1.0) * (X - tau), min=0.0)
            sum_v = (v ** (1.0 / (p - 1.0))).sum(dim=dim, keepdim=True)
            mask = sum_v < 1.0
            tau_hi = torch.where(mask, tau, tau_hi)
            tau_lo = torch.where(mask, tau_lo, tau)

        tau_star = (tau_lo + tau_hi) / 2.0

    v = torch.clamp((p - 1.0) * (X - tau_star), min=0.0)
    res = v ** (1.0 / (p - 1.0))
    return res / (res.sum(dim=dim, keepdim=True) + 1e-9)


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Scatter softmax over graph edge destinations."""
    src_safe = torch.clamp(src, min=-60.0, max=60.0)
    exp_val = torch.exp(src_safe)
    sum_val = torch.zeros(num_nodes, dtype=src.dtype, device=src.device).scatter_add(0, index, exp_val)
    return exp_val / (sum_val[index] + 1e-9)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_mps_shapes(
    x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor, batch_size: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads node and edge shapes to fixed buckets for Apple Silicon MPS optimization."""
    n_max = batch_size * 4
    e_max = n_max * 10
    node_bucket = max(1024, int(round((n_max * 0.20) / 1024) * 1024))
    edge_bucket = max(4096, int(round((e_max * 0.20) / 4096) * 4096))

    N = x.size(0)
    E = src.size(0)
    N_pad = ((N + node_bucket - 1) // node_bucket) * node_bucket
    E_pad = ((E + edge_bucket - 1) // edge_bucket) * edge_bucket

    if E_pad > E and N_pad == N:
        N_pad += node_bucket

    if N_pad > N:
        x_dummy = torch.zeros(N_pad - N, x.size(1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, x_dummy], dim=0)

    if E_pad > E:
        dummy_idx = torch.full((E_pad - E,), N, dtype=src.dtype, device=src.device)
        dummy_w = torch.zeros(E_pad - E, dtype=weights.dtype, device=weights.device)
        src = torch.cat([src, dummy_idx], dim=0)
        dst = torch.cat([dst, dummy_idx], dim=0)
        weights = torch.cat([weights, dummy_w], dim=0)

    return x, src, dst, weights


# =============================================================================
# 2. HIGH-RESOLUTION DICTIONARY HEALTH TELEMETRY
# =============================================================================

def compute_svd_effective_rank(mat: torch.Tensor) -> float:
    """Continuous Roy-Vetterli effective rank via singular value entropy."""
    if mat.ndim != 2 or mat.size(0) == 0 or mat.size(1) == 0:
        return 0.0
    try:
        _, s, _ = torch.linalg.svd(mat.float(), full_matrices=False)
        s_sum = s.sum()
        if s_sum <= 1e-9:
            return 0.0
        p = s / s_sum
        p = p[p > 1e-9]
        entropy = -(p * torch.log(p)).sum().item()
        return float(math.exp(entropy))
    except Exception:
        return 0.0


def compute_gram_overlap(anchors: torch.Tensor) -> Tuple[float, float]:
    """Calculates max and mean off-diagonal cosine overlap between dictionary topics."""
    K = anchors.size(0)
    if K <= 1:
        return 0.0, 0.0
    anc_norm = F.normalize(anchors.float(), p=2, dim=-1)
    gram = torch.mm(anc_norm, anc_norm.t())
    eye_mask = 1.0 - torch.eye(K, device=anchors.device)
    off_diag = gram * eye_mask
    max_overlap = off_diag.max().item()
    mean_overlap = (off_diag.sum() / (K * (K - 1))).item()
    return float(max_overlap), float(mean_overlap)


def compute_gene_entropy(anchors: torch.Tensor) -> float:
    """Mean Shannon entropy of gene distributions within dictionary programs."""
    p = anchors.float() / (anchors.float().sum(dim=-1, keepdim=True) + 1e-9)
    ent = -(p * torch.log(p + 1e-12)).sum(dim=-1)
    return float(ent.mean().item())


# =============================================================================
# 3. COMPLETE OLS PHASE TRACKER (DYNAMIC GOVERNOR)
# =============================================================================

class PhaseTracker:
    """Adaptive Elastic Phase Controller with Dual-Horizon OLS Oscillation Filtering."""
    def __init__(
        self,
        cycle_window: int = 6,
        target_pw: float = 70.0,
        rel_tolerance: float = 0.06,
        max_p1_epochs: int = 20,
    ) -> None:
        self.phase: int = 1
        self.cycle_window: int = cycle_window
        self.target_pw: float = target_pw
        self.rel_tolerance: float = rel_tolerance
        self.max_p1_epochs: int = max_p1_epochs

        self.rec_history: List[float] = []
        self.pw_history: List[float] = []

        self.best_rec_loss: float = float("inf")
        self.p1_baseline_rec: Optional[float] = None

        self.pressure: float = 0.0
        self.squeeze_momentum: float = 0.05
        self.breathing_cooldown: int = 0

        self.saturation_streak: int = 0
        self.required_saturation_streak: int = 4

    @staticmethod
    def _fit_ols(series: List[float]) -> Tuple[float, float, float]:
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
        return 0.5 * (1.0 - math.cos(math.pi * self.pressure))

    def step(self, rec_loss: float, pw: float, epoch: int) -> bool:
        """Evaluates epoch telemetry, adjusting squeeze pressure elastically."""
        self.rec_history.append(rec_loss)
        self.pw_history.append(pw)

        if rec_loss < self.best_rec_loss:
            self.best_rec_loss = rec_loss

        if len(self.rec_history) < self.cycle_window:
            return False

        window_rec = self.rec_history[-self.cycle_window:]
        window_pw = self.pw_history[-self.cycle_window:]

        rec_slope, rec_mu, rec_sigma = self._fit_ols(window_rec)
        pw_slope, pw_mu, _ = self._fit_ols(window_pw)

        # -------------------------------------------------------------
        # PHASE 1: Manifold Discovery & Plateau Detection
        # -------------------------------------------------------------
        if self.phase == 1:
            if epoch >= self.max_p1_epochs:
                self.force_phase2(epoch, rec_mu)
                return False

            relative_drop_rate = (-rec_slope * self.cycle_window) / max(1e-5, rec_mu)
            if relative_drop_rate < 0.008:
                self.force_phase2(epoch, rec_mu)
            return False

        # -------------------------------------------------------------
        # PHASE 2: Elastic Squeeze & Breathe Dynamic Governor
        # -------------------------------------------------------------
        if self.phase == 2:
            dynamic_budget = max(self.best_rec_loss * self.rel_tolerance, 2.5 * rec_sigma)
            loss_ceiling = self.best_rec_loss + dynamic_budget
            overshoot = rec_loss - loss_ceiling

            if overshoot > 0.0:
                severity = min(2.0, overshoot / max(1e-5, dynamic_budget))
                release_amount = 0.04 * severity
                self.pressure = max(0.10, self.pressure - release_amount)
                self.squeeze_momentum = 0.008
                self.breathing_cooldown = 2
                self.saturation_streak = 0
            else:
                if self.breathing_cooldown > 0:
                    self.breathing_cooldown -= 1
                else:
                    self.squeeze_momentum = min(0.035, self.squeeze_momentum + 0.002)
                    self.pressure = min(1.0, self.pressure + self.squeeze_momentum)

            if self.pressure >= 0.95 and pw >= (self.target_pw - 3.0):
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
        """Transitions tracker directly to Phase 2."""
        if self.phase == 1:
            self.phase = 2
            self.p1_baseline_rec = current_baseline
            if self.best_rec_loss == float("inf") or current_baseline < self.best_rec_loss:
                self.best_rec_loss = current_baseline


# =============================================================================
# 4. UNSTRIPPED GROUND TRUTH LIBELLA CORE ARCHITECTURE
# =============================================================================

class FullLibellaCore(nn.Module):
    """Ground Truth Complete Libella Neural Architecture."""
    def __init__(
        self,
        in_channels: int,
        n_metaprograms: int,
        init_components: Optional[np.ndarray] = None,
        hidden_dim: int = 128,
        k_hops: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_metaprograms = n_metaprograms
        self.hidden_dim = hidden_dim
        self.k_hops = k_hops

        # 1. Identity & Context Encoders
        self.id_enc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(hidden_dim)
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.lin_appnp = nn.Linear(hidden_dim, hidden_dim)

        # 2. GATv2 & Bilateral Graph Propagation
        self.gat_w_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_w_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_w_edge = nn.Linear(1, hidden_dim, bias=True)
        self.gat_a = nn.Linear(hidden_dim, 1, bias=False)
        self.att_temp = nn.Parameter(torch.tensor(-2.25))
        self.mp_update = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_proj = nn.Linear(hidden_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # 3. Graph Cross-Attention & Gating
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.sp_norm = nn.LayerNorm(hidden_dim)
        self.topic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_metaprograms)
        )

        # 4. Tissue Plasticity Engine: Spatial Bridge
        self.spatial_bridge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_metaprograms * in_channels)
        )

        # 5. Parameterized Dictionary & Prior Buffers
        self.dict_temp = nn.Parameter(torch.tensor(0.20))
        self.register_buffer("ortho_mask", 1.0 - torch.eye(n_metaprograms, dtype=torch.float32))
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))

        if init_components is not None:
            active_mask = (init_components > 0)
            base_logits = np.where(active_mask, 2.0, -2.0)
            init_logits = base_logits + np.random.randn(*base_logits.shape) * 0.1
            self.topic_gene_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
            self.register_buffer("anchor_logits", torch.tensor(init_logits, dtype=torch.float32).clone())
        else:
            self.topic_gene_logits = nn.Parameter(torch.randn(n_metaprograms, in_channels))
            self.register_buffer("anchor_logits", torch.ones(n_metaprograms, in_channels))

        # Dynamic runtime variables
        self.current_scale: float = 8.0
        self.current_alpha: float = 1.2
        self.current_temp: float = 1.5
        self.current_progress: float = 0.0

    def compute_anchors(self, macro_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 2.0
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)
        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)
        anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()
        return anchors_raw, dynamic_logits

    def encode_graph(
        self, x_dense: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = x_dense.size(0)
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        h_id = self.id_enc(x_dense)
        h_0 = self.lin_appnp(self.ctx_enc(x_dense))
        macro_ctx = h_0.mean(dim=0)

        anchors_raw, dynamic_logits = self.compute_anchors(macro_ctx)

        # Bilateral Graph Edge Physics
        if len(src) > 0:
            with torch.no_grad():
                bio_h = torch.mm(x_dense, anchors_raw.detach().t())
                diff = bio_h[src] - bio_h[dst]
                dist = (diff * diff).sum(dim=1)
            decay = torch.exp(-F.softplus(self.gamma) * dist)
            W_bil = edge_weights * decay
        else:
            W_bil = edge_weights

        # APPNP + GATv2 Multi-Hop Aggregation
        alpha = torch.sigmoid(self.alpha_proj(h_0)) * 0.85 + 0.10
        inv_alpha = 1.0 - alpha
        h_0_scaled = h_0 * alpha

        h_ctx = h_0
        for _ in range(self.k_hops):
            out = torch.zeros_like(h_ctx)
            if len(src) > 0:
                h_edge = self.gat_w_src(h_ctx)[src] + self.gat_w_dst(h_ctx)[dst] + self.gat_w_edge(W_bil.unsqueeze(1))
                e_raw = self.gat_a(F.leaky_relu(h_edge)).squeeze(-1)
                tau = torch.clamp(F.softplus(self.att_temp), min=0.05)
                alpha_att = scatter_softmax(e_raw / tau, dst, N)
                msg = h_ctx[src] * alpha_att.unsqueeze(1)
                out.index_add_(0, dst, msg)
            agg = F.silu(self.mp_update(out))
            h_ctx = agg * inv_alpha + h_0_scaled

        # Transformer Graph Cross-Attention
        Q = self.q_proj(h_id)
        K = self.k_proj(h_ctx)
        V = self.v_proj(h_ctx)

        idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x_dense.device.type == "mps" else torch.int64)
        self_loops = torch.arange(N, dtype=idx_dtype, device=x_dense.device)
        src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
        dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

        cross_scores = (Q[dst_with_self] * K[src_with_self]).sum(dim=-1) / (self.hidden_dim ** 0.5)
        cross_att = scatter_softmax(cross_scores, dst_with_self, N)

        pulled_msg = (V[src_with_self] * cross_att.unsqueeze(1)).contiguous()
        ctx_pulled = torch.zeros_like(Q)
        ctx_pulled.index_add_(0, dst_with_self, pulled_msg)

        h_final = h_id + self.context_gate(ctx_pulled)
        h_norm = F.normalize(self.sp_norm(h_final), p=2, dim=-1)

        # Hybrid Transcriptomic Sim + GNN Shift
        t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
        x_norm = F.normalize(x_dense, p=2, dim=-1)
        bio_sim = torch.mm(x_norm, t_proj_weights.t())

        gnn_shift_raw = self.topic_proj(h_norm)
        gnn_shift_norm = F.normalize(gnn_shift_raw, p=2, dim=-1)
        base_logits = bio_sim + (0.5 * gnn_shift_norm)

        if self.training:
            base_logits = base_logits + torch.randn_like(base_logits) * 0.05
        logits = base_logits * self.current_scale

        return logits, anchors_raw, dynamic_logits

    def forward(
        self, x_dense: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, anchors_raw, dynamic_logits = self.encode_graph(x_dense, src, dst, edge_weights)
        sparse_prob = entmax_bisect_fn(logits, alpha=self.current_alpha, dim=1)
        if self.training:
            smooth_prob = F.softmax(logits / self.current_temp, dim=1)
            progress = getattr(self, "current_progress", 0.0)
            smooth_weight = 0.50 - (0.45 * progress)
            sparse_weight = 1.0 - smooth_weight
            prob = (sparse_weight * sparse_prob) + (smooth_weight * smooth_prob)
        else:
            prob = sparse_prob
        frac = prob * x_dense.sum(dim=1, keepdim=True)
        return frac, anchors_raw, logits, dynamic_logits

    def calc_loss(
        self,
        recon_c: torch.Tensor,
        x_c: torch.Tensor,
        anchors: torch.Tensor,
        epoch: int,
        total_epochs: int,
        f_train: Optional[torch.Tensor] = None,
        target_f_dist: Optional[torch.Tensor] = None,
        kl_weight: float = 5.0,
        dynamic_logits: Optional[torch.Tensor] = None,
        train_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_pos = torch.clamp((x_c > 0).float().sum(), min=1.0)
        num_zeros = (x_c == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(current_dynamic_w, weight=0.1)

        is_non_zero = (x_c > 0)
        if self.training:
            zero_mask = torch.rand_like(x_c) < 0.05
            active_mask = (is_non_zero | zero_mask).to(x_c.dtype)
            masked_w_mat = torch.where(is_non_zero, current_dynamic_w, 1.0) * active_mask
        else:
            masked_w_mat = torch.where(is_non_zero, current_dynamic_w, 1.0)

        raw_delta = recon_c - x_c
        asymmetry_factor = 1.0 + (is_non_zero.to(x_c.dtype) * 2.0) * (raw_delta < 0).float()
        scaled_delta = torch.clamp(raw_delta * asymmetry_factor, min=-30.0, max=30.0)

        l_recon = torch.sum(masked_w_mat * torch.log(torch.cosh(scaled_delta + 1e-6))) / max(1, x_c.numel())

        # Prior Anchor Regularization Loss (Matched Temperature Scaling)
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        anc_norm = F.normalize(anchors, p=2, dim=1)
        ref_norm = F.normalize(F.softmax(self.anchor_logits / safe_temp, dim=-1), p=2, dim=1)
        l_anc = 1.0 - (anc_norm * ref_norm).sum(dim=1).mean()

        # Orthogonality and Gene Entropy
        latent_ortho = torch.mm(anc_norm, anc_norm.t()) * self.ortho_mask
        l_ortho = (F.relu(latent_ortho.max(dim=1)[0] - 0.25) ** 2).mean()
        collapse_penalty = (F.relu(anchors - 0.80) ** 2).sum(dim=1).mean()
        gene_entropy = -(anchors * torch.log(anchors + 1e-9)).sum(dim=1).mean()

        # Information Maximization (Tsallis + KL)
        tsallis_h = torch.tensor(0.0, device=x_c.device)
        kl_marginal = torch.tensor(0.0, device=x_c.device)

        if f_train is not None:
            f_norm = f_train / torch.clamp(f_train.sum(dim=1, keepdim=True), min=1e-6)
            f_safe = torch.clamp(f_norm, min=1e-5, max=1.0)
            alpha_ent = 1.5
            tsallis_h = (1.0 - (f_safe ** alpha_ent).sum(dim=1).mean()) / (alpha_ent - 1.0)

            p_mean = torch.clamp(f_norm.mean(dim=0), min=1e-5, max=1.0)
            if target_f_dist is not None:
                kl_target = torch.clamp(target_f_dist, min=1e-5)
                kl_marginal = (p_mean * (torch.log(p_mean + 1e-9) - torch.log(kl_target + 1e-9))).sum()
            else:
                uniform_prior = torch.ones(self.n_metaprograms, device=x_c.device) / self.n_metaprograms
                kl_marginal = (p_mean * (torch.log(p_mean + 1e-9) - torch.log(uniform_prior))).sum()

        progress = epoch / max(1, total_epochs - 1)
        recon_mag = l_recon.item()
        anc_scale = recon_mag * 0.1 * max(0.05, 1.0 - progress)
        tsallis_scale = recon_mag * 0.05 * max(0.0, (progress - 0.5) * 2.0)
        kl_scale = recon_mag * 0.05

        total_loss = (
            l_recon
            + (l_anc * anc_scale)
            + ((l_ortho + collapse_penalty) * (recon_mag * 0.05))
            + (gene_entropy * (recon_mag * 0.01))
            + (tsallis_h * tsallis_scale)
            + (kl_weight * kl_marginal * kl_scale)
        )

        loss_breakdown = {
            "loss_total": total_loss.item(),
            "loss_recon": l_recon.item(),
            "loss_anc": l_anc.item(),
            "loss_ortho": l_ortho.item(),
            "loss_im": (tsallis_h * tsallis_scale + kl_weight * kl_marginal * kl_scale).item(),
            "dyn_w": current_dynamic_w.item(),
        }
        return total_loss, loss_breakdown


# =============================================================================
# 5. FIVE EVOLVED ARCHITECTURAL ALTERNATIVES
# =============================================================================

class Alt1_JacobianMatchedSTE(FullLibellaCore):
    """Alt 1: Jacobian-Matched Sharp STE + Logit Margin Prior Loss (Problem 1 Fix)."""
    def compute_anchors(self, macro_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 2.0
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)
        anchors_raw = sharp_anchors
        return anchors_raw, dynamic_logits

    def calc_loss(self, recon_c, x_c, anchors, epoch, total_epochs, f_train=None, target_f_dist=None, kl_weight=5.0, dynamic_logits=None, train_idx=None):
        total_loss, loss_dict = super().calc_loss(recon_c, x_c, anchors, epoch, total_epochs, f_train, target_f_dist, kl_weight, dynamic_logits, train_idx)
        if dynamic_logits is not None:
            is_pos_marker = (self.anchor_logits > 0)
            pos_loss = F.relu(2.0 - dynamic_logits[is_pos_marker]) ** 2
            neg_loss = F.relu(dynamic_logits[~is_pos_marker] - (-2.0)) ** 2
            l_anc_margin = (pos_loss.mean() + neg_loss.mean()) * 0.05

            progress = epoch / max(1, total_epochs - 1)
            anc_scale = loss_dict["loss_recon"] * 0.1 * max(0.05, 1.0 - progress)
            total_loss = total_loss - (loss_dict["loss_anc"] * anc_scale) + (l_anc_margin * anc_scale)
            loss_dict["loss_anc"] = l_anc_margin.item()
            loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict


class Alt2_LowRankFactoredBridge(FullLibellaCore):
    """Alt 2: Low-Rank Factored Bridge + Bounded Shift Dynamics (Problem 2 Fix)."""
    def __init__(self, in_channels: int, n_metaprograms: int, init_components=None, hidden_dim: int = 128, rank: int = 4):
        super().__init__(in_channels, n_metaprograms, init_components, hidden_dim)
        self.rank = rank
        self.spatial_bridge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_metaprograms * rank)
        )
        self.global_basis = nn.Parameter(torch.randn(rank, in_channels) * 0.05)

    def compute_anchors(self, macro_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_coeffs = torch.tanh(self.spatial_bridge(macro_ctx)).view(self.n_metaprograms, self.rank)
        dict_shift = torch.mm(u_coeffs, self.global_basis) * 0.35
        dynamic_logits = self.topic_gene_logits + dict_shift
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)
        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()
        return anchors_raw, dynamic_logits


class Alt3_SparseEntmaxPlasticity(FullLibellaCore):
    """Alt 3: Sparse Simplex Entmax Anchors + Learnable Topic Plasticity Gates (Problem 1 & 2 Fix)."""
    def __init__(self, in_channels: int, n_metaprograms: int, init_components=None, hidden_dim: int = 128):
        super().__init__(in_channels, n_metaprograms, init_components, hidden_dim)
        self.plasticity_gates = nn.Parameter(torch.zeros(n_metaprograms, 1))

    def compute_anchors(self, macro_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 1.0
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)

        anchors_learned = entmax_bisect_fn(dynamic_logits / safe_temp, alpha=1.5, dim=-1)
        anchors_prior = entmax_bisect_fn(self.anchor_logits / safe_temp, alpha=1.5, dim=-1)

        gate = torch.sigmoid(self.plasticity_gates)
        anchors_raw = gate * anchors_prior + (1.0 - gate) * anchors_learned
        return anchors_raw, dynamic_logits


class Alt4_DualStreamGradHighway(FullLibellaCore):
    """Alt 4: Dual-Stream Flow: Sparse Attribution + Dense Gradient Highway (Problem 1 & 2 Fix)."""
    def forward(self, x_dense: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weights: torch.Tensor):
        logits, anchors_raw, dynamic_logits = self.encode_graph(x_dense, src, dst, edge_weights)
        sparse_prob = entmax_bisect_fn(logits, alpha=self.current_alpha, dim=1)
        smooth_prob = F.softmax(logits / 1.5, dim=1)

        mag = x_dense.sum(dim=1, keepdim=True)
        frac = sparse_prob * mag
        if self.training:
            self._smooth_frac = smooth_prob * mag
        else:
            self._smooth_frac = frac
        return frac, anchors_raw, logits, dynamic_logits

    def calc_loss(self, recon_c, x_c, anchors, epoch, total_epochs, f_train=None, target_f_dist=None, kl_weight=5.0, dynamic_logits=None, train_idx=None):
        total_loss, loss_dict = super().calc_loss(recon_c, x_c, anchors, epoch, total_epochs, f_train, target_f_dist, kl_weight, dynamic_logits, train_idx)
        if self.training and hasattr(self, "_smooth_frac") and self._smooth_frac is not None and train_idx is not None:
            smooth_recon = self._smooth_frac[train_idx] @ anchors
            l_aux = F.mse_loss(smooth_recon, x_c) * 0.15
            total_loss = total_loss + l_aux
            loss_dict["loss_aux_highway"] = l_aux.item()
            loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict


class Alt5_ContinuousQDeformed(FullLibellaCore):
    """Alt 5: Continuous q-Deformed Simplex + Latent-Space Microenvironment Conditioning (Problem 1 & 2 Fix)."""
    def __init__(self, in_channels: int, n_metaprograms: int, init_components=None, hidden_dim: int = 128):
        super().__init__(in_channels, n_metaprograms, init_components, hidden_dim)
        self.latent_macro_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True)
        )

    def compute_anchors(self, macro_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        anchors_raw = entmax_bisect_fn(self.topic_gene_logits / safe_temp, alpha=1.25, dim=-1)
        return anchors_raw, self.topic_gene_logits

    def encode_graph(self, x_dense: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weights: torch.Tensor):
        N = x_dense.size(0)
        h_id = self.id_enc(x_dense)
        h_0 = self.lin_appnp(self.ctx_enc(x_dense))
        macro_ctx = h_0.mean(dim=0)

        anchors_raw, dynamic_logits = self.compute_anchors(macro_ctx)

        if len(src) > 0:
            with torch.no_grad():
                bio_h = torch.mm(x_dense, anchors_raw.detach().t())
                dist = ((bio_h[src] - bio_h[dst]) ** 2).sum(dim=1)
            decay = torch.exp(-F.softplus(self.gamma) * dist)
            W_bil = edge_weights * decay
        else:
            W_bil = edge_weights

        alpha = torch.sigmoid(self.alpha_proj(h_0)) * 0.85 + 0.10
        h_ctx = h_0
        for _ in range(self.k_hops):
            out = torch.zeros_like(h_ctx)
            if len(src) > 0:
                h_edge = self.gat_w_src(h_ctx)[src] + self.gat_w_dst(h_ctx)[dst] + self.gat_w_edge(W_bil.unsqueeze(1))
                e_raw = self.gat_a(F.leaky_relu(h_edge)).squeeze(-1)
                alpha_att = scatter_softmax(e_raw / torch.clamp(F.softplus(self.att_temp), min=0.05), dst, N)
                out.index_add_(0, dst, h_ctx[src] * alpha_att.unsqueeze(1))
            h_ctx = F.silu(self.mp_update(out)) * (1.0 - alpha) + (h_0 * alpha)

        Q, K, V = self.q_proj(h_id), self.k_proj(h_ctx), self.v_proj(h_ctx)
        idx_dtype = src.dtype if len(src) > 0 else torch.int64
        self_loops = torch.arange(N, dtype=idx_dtype, device=x_dense.device)
        src_all = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
        dst_all = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

        cross_scores = (Q[dst_all] * K[src_all]).sum(dim=-1) / (self.hidden_dim ** 0.5)
        cross_att = scatter_softmax(cross_scores, dst_all, N)
        ctx_pulled = torch.zeros_like(Q)
        ctx_pulled.index_add_(0, dst_all, (V[src_all] * cross_att.unsqueeze(1)).contiguous())

        macro_mod = self.latent_macro_gate(macro_ctx).unsqueeze(0)
        h_final = h_id + self.context_gate(ctx_pulled) + macro_mod
        h_norm = F.normalize(self.sp_norm(h_final), p=2, dim=-1)

        t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
        bio_sim = torch.mm(F.normalize(x_dense, p=2, dim=-1), t_proj_weights.t())
        gnn_shift_norm = F.normalize(self.topic_proj(h_norm), p=2, dim=-1)
        base_logits = bio_sim + (0.5 * gnn_shift_norm)
        logits = base_logits * self.current_scale
        return logits, anchors_raw, dynamic_logits


# =============================================================================
# 6. DATA LOADERS & ARTIFACT PARSERS
# =============================================================================

def load_pilot_artifacts(
    genes_path: str | Path, priors_path: str | Path, chunks_dir: str | Path, n_chunks: int = 10
) -> Tuple[List[str], np.ndarray, int, List[Path]]:
    g_p = Path(genes_path)
    p_p = Path(priors_path)
    c_p = Path(chunks_dir)

    if not g_p.exists():
        raise FileNotFoundError(f"Gene vocabulary not found at: {g_p}")
    with open(g_p, "r", encoding="utf-8") as f:
        common_genes = json.load(f)

    if not p_p.exists():
        raise FileNotFoundError(f"Prior dictionary not found at: {p_p}")
    try:
        priors_data = joblib.load(p_p)
    except Exception:
        with open(p_p, "rb") as f:
            priors_data = pickle.load(f)

    if isinstance(priors_data, dict):
        init_components = priors_data.get("components", priors_data.get("init_components", priors_data.get("priors")))
    elif isinstance(priors_data, np.ndarray):
        init_components = priors_data
    elif isinstance(priors_data, torch.Tensor):
        init_components = priors_data.cpu().numpy()
    else:
        raise ValueError(f"Unknown prior container type: {type(priors_data)}")

    optimal_k = init_components.shape[0]
    chunk_files = sorted(list(c_p.glob("*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No .pt chunks found in {c_p}")

    selected_chunks = chunk_files[:n_chunks]
    return common_genes, init_components.astype(np.float32), optimal_k, selected_chunks


# =============================================================================
# 7. FULL-EPOCH MASTER COMPARATIVE BENCHMARK RUNNER
# =============================================================================

def run_pilot_suite(
    common_genes_path: str,
    priors_path: str,
    chunks_dir: str,
    out_dir: str,
    n_epochs: int = 300,
):
    device = get_device()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "model_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 122)
    print("🔬 LIBELLA FULL-EPOCH EVOLUTIONARY ENGINE (PHASETRACKER & SHARPENING ACTIVE)")
    print(f"   Compute Device: {device} | Total Epoch Budget: {n_epochs} | Output Dir: {out_path.resolve()}")
    print("=" * 122)

    common_genes, init_components, optimal_k, chunk_files = load_pilot_artifacts(
        common_genes_path, priors_path, chunks_dir, n_chunks=10
    )
    in_channels = len(common_genes)
    print(f"[✓] Successfully loaded {len(chunk_files)} spatial chunks ({in_channels} genes, Base K={optimal_k}).")

    # Cache pre-tensorized graph chunks in RAM
    cached_graph_chunks = []
    for cf in chunk_files:
        data = torch.load(cf, map_location="cpu", weights_only=False)
        cached_graph_chunks.append({
            "x": data["x"].float(),
            "src": data["src"].int(),
            "dst": data["dst"].int(),
            "weights": data["weights"].float(),
            "train_idx": data["train_core_idx"].long(),
            "val_idx": data["val_core_idx"].long() if "val_core_idx" in data else torch.tensor([], dtype=torch.long),
            "patient_name": data.get("patient_name", cf.stem)
        })

    # Instantiate all 6 model variants
    models: Dict[str, FullLibellaCore] = {
        "0. Baseline Libella (Ground Truth)": FullLibellaCore(in_channels, optimal_k, init_components).to(device),
        "1. Alt 1 (Jacobian-Matched STE)": Alt1_JacobianMatchedSTE(in_channels, optimal_k, init_components).to(device),
        "2. Alt 2 (Low-Rank Factored Bridge)": Alt2_LowRankFactoredBridge(in_channels, optimal_k, init_components).to(device),
        "3. Alt 3 (Sparse Entmax + Gating)": Alt3_SparseEntmaxPlasticity(in_channels, optimal_k, init_components).to(device),
        "4. Alt 4 (Dual-Stream Grad Highway)": Alt4_DualStreamGradHighway(in_channels, optimal_k, init_components).to(device),
        "5. Alt 5 (Continuous q-Deformed Simplex)": Alt5_ContinuousQDeformed(in_channels, optimal_k, init_components).to(device),
    }

    # Setup decoupled optimizers & Cosine Annealing Schedulers
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
    best_composite_scores: Dict[str, float] = {name: float("inf") for name in models}
    early_stopped_models: set = set()

    for name, m in models.items():
        base_params = [p for n, p in m.named_parameters() if "topic_gene_logits" not in n and "global_basis" not in n]
        dict_params = [p for n, p in m.named_parameters() if "topic_gene_logits" in n or "global_basis" in n]
        opt = torch.optim.AdamW([
            {"params": base_params, "lr": 1e-3, "weight_decay": 1e-4},
            {"params": dict_params, "lr": 1e-3, "weight_decay": 1e-4}
        ])
        optimizers[name] = opt
        schedulers[name] = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    trackers: Dict[str, PhaseTracker] = {name: PhaseTracker(max_p1_epochs=20) for name in models}
    step_records: List[Dict[str, Any]] = []

    print("\n[➤] Starting Full-Epoch Training Loops...\n")

    for epoch in range(1, n_epochs + 1):
        active_models = [m for m in models.keys() if m not in early_stopped_models]
        if not active_models:
            print(f"\n[✓] All architectures saturated and completed training by Epoch {epoch-1}.")
            break

        print(f"\n▶ Epoch {epoch:03d}/{n_epochs:03d} (Active Models: {len(active_models)}/6)")

        for model_name in active_models:
            model = models[model_name]
            optimizer = optimizers[model_name]
            scheduler = schedulers[model_name]
            tracker = trackers[model_name]

            # 1. Failsafe Transition Check: Force Phase 2 if nearing epoch budget
            epochs_remaining = n_epochs - epoch
            if tracker.phase == 1 and epochs_remaining <= 20:
                tracker.force_phase2(epoch, tracker.best_rec_loss)

            # 2. Update Model Physics Schedules from PhaseTracker
            prog = tracker.get_progress()
            model.current_progress = prog
            model.current_scale = 8.0 + ((16.0 - 8.0) * (prog ** 0.8))
            model.current_temp = 0.3 + ((1.5 - 0.3) * ((1.0 - prog) ** 1.5))
            model.current_alpha = 1.2 + ((1.7 - 1.2) * prog)

            model.train()
            ema_mean = None
            alpha_ema = 0.01

            epoch_rec_loss_sum = 0.0
            epoch_pw_sum = 0.0
            epoch_gw_sum = 0.0
            epoch_dict_grad_sum = 0.0
            n_chunks = len(cached_graph_chunks)

            for chunk_idx, chunk in enumerate(cached_graph_chunks):
                x = chunk["x"].to(device, non_blocking=True)
                src = chunk["src"].to(device, non_blocking=True)
                dst = chunk["dst"].to(device, non_blocking=True)
                weights = chunk["weights"].to(device, non_blocking=True)
                train_idx = chunk["train_idx"].to(device, non_blocking=True)

                if len(src) > 0:
                    keep_mask = torch.rand(src.size(0), device=device) > 0.40
                    src, dst, weights = src[keep_mask], dst[keep_mask], weights[keep_mask]

                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != "mps":
                    src, dst = src.to(torch.int64), dst.to(torch.int64)

                optimizer.zero_grad(set_to_none=True)

                fracs, pure_anchors, cell_logits, dynamic_logits = model(x, src, dst, weights)
                f_train = fracs[train_idx]
                x_train = x[train_idx]

                # Online EMA marginal target
                p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                current_p_mean = p_train.mean(dim=0)
                if ema_mean is None:
                    ema_mean = current_p_mean.detach()
                else:
                    ema_mean = alpha_ema * current_p_mean.detach() + (1 - alpha_ema) * ema_mean

                uniform_prior = torch.ones_like(current_p_mean) / optimal_k
                ideal_c = torch.clamp(uniform_prior * 2.0 - ema_mean, min=1e-5)
                target_f_dist = ideal_c / ideal_c.sum()

                ema_entropy = -torch.sum(ema_mean * torch.log(ema_mean + 1e-9))
                collapse_ratio = torch.clamp(1.0 - (ema_entropy / math.log(optimal_k)), min=0.0, max=1.0)
                dynamic_kl_w = 0.10 + (collapse_ratio * 3.0)

                recon = f_train @ pure_anchors
                loss, loss_dict = model.calc_loss(
                    recon, x_train, pure_anchors, epoch, n_epochs,
                    f_train=f_train, target_f_dist=target_f_dist,
                    kl_weight=dynamic_kl_w, dynamic_logits=dynamic_logits,
                    train_idx=train_idx
                )
                loss.backward()

                # Decoupled Gradient Clipping & Force Vector Decomposition
                base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n and "global_basis" not in n and p.grad is not None]
                dict_params = [p for n, p in model.named_parameters() if ("topic_gene_logits" in n or "global_basis" in n) and p.grad is not None]

                dict_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in dict_params]), 2).item() if dict_params else 0.0

                if base_params:
                    torch.nn.utils.clip_grad_norm_([p for p in base_params], max_norm=5.0)
                if dict_params:
                    torch.nn.utils.clip_grad_norm_([p for p in dict_params], max_norm=5.0)

                optimizer.step()

                # Dictionary Health Diagnostics
                with torch.no_grad():
                    pw = p_train.max(dim=1).values.mean().item() * 100.0
                    gw = pure_anchors.max(dim=1).values.mean().item() * 100.0
                    dead_topics_count = (f_train.sum(dim=0) < 1e-4).float().sum().item()
                    dead_pct = (dead_topics_count / optimal_k) * 100.0

                    eff_rank = compute_svd_effective_rank(pure_anchors)
                    max_ovlp, mean_ovlp = compute_gram_overlap(pure_anchors)
                    gene_ent = compute_gene_entropy(pure_anchors)

                    safe_temp = torch.clamp(model.dict_temp, min=0.25, max=1.0)
                    anc_norm = F.normalize(pure_anchors, p=2, dim=1)
                    init_norm = F.normalize(F.softmax(model.anchor_logits / safe_temp, dim=-1), p=2, dim=1)
                    t0_drift = 1.0 - (anc_norm * init_norm).sum(dim=1).mean().item()

                epoch_rec_loss_sum += loss_dict["loss_recon"]
                epoch_pw_sum += pw
                epoch_gw_sum += gw
                epoch_dict_grad_sum += dict_grad_norm

                step_records.append({
                    "epoch": epoch,
                    "chunk": chunk_idx + 1,
                    "model": model_name,
                    "phase": tracker.phase,
                    "pressure": tracker.pressure,
                    "progress": prog,
                    "scale": model.current_scale,
                    "temp": model.current_temp,
                    "alpha": model.current_alpha,
                    "loss_total": loss_dict["loss_total"],
                    "loss_recon": loss_dict["loss_recon"],
                    "loss_anc": loss_dict["loss_anc"],
                    "loss_ortho": loss_dict["loss_ortho"],
                    "loss_im": loss_dict.get("loss_im", 0.0),
                    "p_w": pw,
                    "g_w": gw,
                    "dict_grad_norm": dict_grad_norm,
                    "dead_topics_pct": dead_pct,
                    "dead_topics_count": int(dead_topics_count),
                    "svd_effective_rank": eff_rank,
                    "effective_rank_ratio": eff_rank / optimal_k,
                    "gram_max_overlap": max_ovlp,
                    "gram_mean_overlap": mean_ovlp,
                    "anchor_drift_t0": t0_drift,
                    "gene_entropy_mean": gene_ent,
                })

            scheduler.step()

            # 3. Epoch Boundary Tracker Step & Pareto Evaluation
            avg_rec = epoch_rec_loss_sum / n_chunks
            avg_pw = epoch_pw_sum / n_chunks
            avg_gw = epoch_gw_sum / n_chunks
            avg_grad = epoch_dict_grad_sum / n_chunks

            is_saturated = tracker.step(avg_rec, avg_pw, epoch)

            # Pareto Composite Score Checkpointing
            composite_score = avg_rec / max(1.0, math.sqrt(avg_pw / 100.0))
            if composite_score < best_composite_scores[model_name]:
                best_composite_scores[model_name] = composite_score
                clean_name = model_name.split()[1].lower()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "composite_score": composite_score,
                    "rec_loss": avg_rec,
                    "pw": avg_pw,
                    "gw": avg_gw,
                }, ckpt_dir / f"{clean_name}_best_pareto.pt")

            status_tag = f"P{tracker.phase} [Prog: {prog*100:4.1f}% | Press: {tracker.pressure:.2f}]"
            print(
                f"  {model_name:<38} | {status_tag} | "
                f"Rec: {avg_rec:<6.4f} | ||g||: {avg_grad:<8.2e} | "
                f"P_W: {avg_pw:<4.1f}% | G_W: {avg_gw:<4.1f}% | SVD: {eff_rank:<4.1f}/{optimal_k}"
            )

            if is_saturated:
                print(f"  ↳ [✓] Model '{model_name}' saturated at P_W={avg_pw:.1f}% (Phase 2 complete). Terminating gracefully.")
                early_stopped_models.add(model_name)

        # Periodic interim flush of trajectory CSV every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs:
            df_interim = pd.DataFrame(step_records)
            df_interim.to_csv(out_path / "trajectory_comparison.csv", index=False)

    # 4. Final Comprehensive Output Export
    df_final_steps = pd.DataFrame(step_records)
    final_csv = out_path / "trajectory_comparison.csv"
    df_final_steps.to_csv(final_csv, index=False)
    print(f"\n[✓] Complete full-epoch trajectory saved to: {final_csv}")

    # Build Final Scorecard from Best Pareto Epochs
    summary_rows = []
    for model_name in models.keys():
        m_df = df_final_steps[df_final_steps["model"] == model_name]
        last_epoch_num = m_df["epoch"].max()
        last_epoch_df = m_df[m_df["epoch"] == last_epoch_num]

        summary_rows.append({
            "Model Architecture": model_name,
            "Total Epochs Run": int(last_epoch_num),
            "Best Composite Score": best_composite_scores[model_name],
            "Final Recon Loss": last_epoch_df["loss_recon"].mean(),
            "Dict Grad Norm (Plasticity)": last_epoch_df["dict_grad_norm"].mean(),
            "Cell Sharpness (P_W %)": last_epoch_df["p_w"].mean(),
            "Gene Sharpness (G_W %)": last_epoch_df["g_w"].mean(),
            "Dead Topic % (Atrophy)": last_epoch_df["dead_topics_pct"].mean(),
            "SVD Effective Rank": last_epoch_df["svd_effective_rank"].mean(),
            "Max Topic Overlap": last_epoch_df["gram_max_overlap"].mean(),
            "Anchor Drift (T0 Adaptation)": last_epoch_df["anchor_drift_t0"].mean(),
            "Gene Entropy (Nats)": last_epoch_df["gene_entropy_mean"].mean(),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("Best Composite Score")
    summary_csv = out_path / "final_benchmark_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[✓] Final comparative evaluation scorecard saved to: {summary_csv}")

    # Display Terminal Summary Table
    print("\n" + "=" * 130)
    print("🏆 FINAL ARCHITECTURAL BENCHMARK SCORECARD (FULL-EPOCH OPTIMIZATION)")
    print("=" * 130)
    print(
        f"{'Model Architecture':<38} | {'Epochs':<6} | {'Pareto':<7} | {'Recon':<7} | "
        f"{'||g_dict||':<10} | {'P_W %':<6} | {'G_W %':<6} | {'Dead %':<6} | {'SVD Rank':<8} | {'T0 Drift'}"
    )
    print("-" * 130)
    for _, row in df_summary.iterrows():
        print(
            f"{row['Model Architecture']:<38} | "
            f"{row['Total Epochs Run']:<6d} | "
            f"{row['Best Composite Score']:<7.3f} | "
            f"{row['Final Recon Loss']:<7.4f} | "
            f"{row['Dict Grad Norm (Plasticity)']:<10.2e} | "
            f"{row['Cell Sharpness (P_W %)']:<6.1f} | "
            f"{row['Gene Sharpness (G_W %)']:<6.1f} | "
            f"{row['Dead Topic % (Atrophy)']:<6.1f} | "
            f"{row['SVD Effective Rank']:<8.2f} | "
            f"{row['Anchor Drift (T0 Adaptation)']:.4f}"
        )
    print("=" * 130 + "\n")


# =============================================================================
# 8. MAIN CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Libella Full-Epoch Evolutionary Pilot Engine")
    parser.add_argument(
        "--genes",
        type=str,
        default="/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/common_genes.json",
        help="Path to common_genes.json",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default="/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/global_cnmf_priors.pkl",
        help="Path to global_cnmf_priors.pkl",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/temp_training_chunks",
        help="Path to temp_training_chunks directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/full_pilot_results",
        help="Output directory for generated logs, checkpoints, and CSVs",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Total epochs for training")
    args = parser.parse_args()

    run_pilot_suite(
        common_genes_path=args.genes,
        priors_path=args.priors,
        chunks_dir=args.chunks,
        out_dir=args.out_dir,
        n_epochs=args.epochs,
    )