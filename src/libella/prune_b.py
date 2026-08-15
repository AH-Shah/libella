#!/usr/bin/env python3
"""
Libella Spatial GNN: 3-Step Pruning Architecture Pilot
Integrates:
  1. Dynamic Slot Masking (-1e9 masking on deactivated slots)
  2. Biological Signal-to-Noise Gating (Raw Fold >= 1.25x & Cell Pct >= 0.5%)
  3. Background Noise Sink Channel (K+1 Null Channel)
"""

from __future__ import annotations

import gc
import json
import math
import joblib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Resolve Libella Root & Direct Module Imports
# -----------------------------------------------------------------------------
for parent in [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
    if (parent / "libella").is_dir() or (parent / "config.py").is_file():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

try:
    from libella.config import RunConfig, cfg, init_env, paths
    from libella.data import make_meta_batches, pad_mps_shapes
    from libella.utils import PhaseTracker, get_device, scatter_softmax
except ImportError:
    @dataclass
    class RunConfig:
        hidden_dim: int = 128
        k_hops: int = 2
        dict_temp: float = 0.30
        att_temp: float = 0.50
        gnn_shift_weight: float = 0.20
        train_noise: float = 0.05
        inference_scale: float = 12.0
        inference_alpha: float = 1.33
        inference_temp: float = 0.25
        scale_start: float = 6.0
        scale_end: float = 18.0
        temp_start: float = 0.60
        temp_end: float = 0.15
        alpha_start: float = 1.05
        alpha_end: float = 1.45
        lr_base: float = 1e-3
        lr_anchor: float = 5e-3
        wd_base: float = 1e-4
        wd_anchor: float = 0.0
        epochs: int = 50
        batch_size: int = 4096
        meta_batch_size: int = 4
        edge_dropout: float = 0.10
        grad_clip: float = 1.0
        kl_base: float = 0.50
        kl_collapse_weight: float = 3.0
        hub_threshold: float = 0.30
        kl_weight: float = 1.0
        ortho_overlap_threshold: float = 0.60
        anchor_peak_threshold: float = 0.40
        ortho_weight: float = 0.50
        tsallis_alpha: float = 1.5
        zero_mask_rate: float = 0.05
        delta_clamp: float = 30.0
        suffix: str = "pruning_pilot"

    cfg = RunConfig()

    def get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        max_val = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        max_val.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
        src_exp = torch.exp(src - max_val[index])
        sum_val = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        sum_val.index_add_(0, index, src_exp)
        return src_exp / (sum_val[index] + 1e-9)

    def pad_mps_shapes(
        x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return x, src, dst, weights

    def make_meta_batches(cache: List[Dict[str, Any]], meta_batch_size: int = 4) -> List[List[Dict[str, Any]]]:
        indices = np.random.permutation(len(cache))
        return [[cache[i] for i in indices[j : j + meta_batch_size]] for j in range(0, len(cache), meta_batch_size)]

    class PhaseTracker:
        def __init__(
            self,
            cycle_window: int = 6,
            target_pw: float = 70.0,
            rel_tolerance: float = 0.06,
            max_p1_epochs: int = 15,
        ) -> None:
            self.phase: int = 1
            self.cycle_window: int = cycle_window
            self.target_pw: float = target_pw
            self.rel_tolerance: float = rel_tolerance
            self.max_p1_epochs: int = max_p1_epochs
            self.rec_history: List[float] = []
            self.pw_history: List[float] = []
            self.best_rec_loss: float = float("inf")
            self.p1_baseline_rec: float | None = None
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
            if self.phase == 1:
                return 0.0
            return 0.5 * (1.0 - math.cos(math.pi * self.pressure))

        def step(self, epoch_telemetry: Dict[str, float], epoch: int) -> bool:
            current_rec = float(epoch_telemetry.get("l_rec", 0.0))
            current_pw = float(epoch_telemetry.get("p_w", 0.0))
            self.rec_history.append(current_rec)
            self.pw_history.append(current_pw)

            if current_rec < self.best_rec_loss:
                self.best_rec_loss = current_rec

            if len(self.rec_history) < self.cycle_window:
                return False

            window_rec = self.rec_history[-self.cycle_window :]
            window_pw = self.pw_history[-self.cycle_window :]
            rec_slope, rec_mu, rec_sigma = self._fit_ols(window_rec)
            pw_slope, _, _ = self._fit_ols(window_pw)

            if self.phase == 1:
                if epoch >= self.max_p1_epochs:
                    self.force_phase2(epoch, rec_mu)
                    return False
                relative_drop_rate = (-rec_slope * self.cycle_window) / max(1e-5, rec_mu)
                if relative_drop_rate < 0.008:
                    self.force_phase2(epoch, rec_mu)
                return False

            if self.phase == 2:
                dynamic_budget = max(self.best_rec_loss * self.rel_tolerance, 2.5 * rec_sigma)
                loss_ceiling = self.best_rec_loss + dynamic_budget
                overshoot = current_rec - loss_ceiling

                if overshoot > 0.0:
                    severity = min(2.0, overshoot / max(1e-5, dynamic_budget))
                    self.pressure = max(0.10, self.pressure - (0.04 * severity))
                    self.squeeze_momentum = 0.008
                    self.breathing_cooldown = 2
                    self.saturation_streak = 0
                else:
                    if self.breathing_cooldown > 0:
                        self.breathing_cooldown -= 1
                    else:
                        self.squeeze_momentum = min(0.035, self.squeeze_momentum + 0.002)
                        self.pressure = min(1.0, self.pressure + self.squeeze_momentum)

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
            if self.phase == 1:
                self.phase = 2
                self.p1_baseline_rec = current_baseline
                if self.best_rec_loss == float("inf") or current_baseline < self.best_rec_loss:
                    self.best_rec_loss = current_baseline

try:
    init_env()
except Exception:
    pass


# =============================================================================
# 2. Libella GNN Model with Masking and Background Sink Channel
# =============================================================================

class LibellaGNN(nn.Module):
    """Core Libella Spatial GNN with Dynamic Masking and Background Noise Sink."""

    def __init__(
        self,
        in_channels: int,
        n_metaprograms: int,
        init_components: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.k_hops = cfg.k_hops
        self.n_metaprograms = n_metaprograms
        self.in_channels = in_channels

        # STEP 1: Dynamic Slot Mask Buffer
        self.register_buffer("active_topic_mask", torch.ones(n_metaprograms, dtype=torch.bool))

        self.ctx_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.lin_appnp = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.id_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(self.hidden_dim),
        )

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.sp_norm = nn.LayerNorm(self.hidden_dim)

        self.topic_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, n_metaprograms),
        )

        self.spatial_bridge = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, n_metaprograms * in_channels),
        )

        self.dict_temp = nn.Parameter(torch.tensor(cfg.dict_temp))
        self.register_buffer("ortho_mask", 1.0 - torch.eye(n_metaprograms, dtype=torch.float32))

        if init_components is not None:
            active_mask = init_components > 0
            base_logits = np.where(active_mask, 2.0, -2.0)
            noise = np.random.randn(*base_logits.shape) * 0.1
            init_logits = base_logits + noise
            self.topic_gene_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
            self.register_buffer("anchor_logits", torch.tensor(init_logits, dtype=torch.float32).clone())
        else:
            self.topic_gene_logits = nn.Parameter(torch.randn(n_metaprograms, in_channels))
            self.register_buffer("anchor_logits", torch.ones(n_metaprograms, in_channels))

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.alpha_proj = nn.Linear(self.hidden_dim, 1)
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))

        self.gat_w_src = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gat_w_dst = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gat_w_edge = nn.Linear(1, self.hidden_dim, bias=True)
        self.gat_a = nn.Linear(self.hidden_dim, 1, bias=False)
        self.att_temp = nn.Parameter(torch.tensor(cfg.att_temp))
        self.mp_update = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        h_id = self.id_enc(x_dense)
        h_0 = self.lin_appnp(self.ctx_enc(x_dense))

        macro_ctx = h_0.mean(dim=0)
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 2.0
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)

        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        safe_temp = torch.clamp(getattr(self, "dict_temp", torch.tensor(0.30)), min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)
        anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()

        N = h_0.size(0)

        if len(src) > 0:
            with torch.no_grad():
                bio_h = torch.mm(x_dense, anchors_raw.detach().t())
                diff = bio_h[src] - bio_h[dst]
                dist = (diff * diff).sum(dim=1)
            decay = torch.exp(-F.softplus(self.gamma) * dist)
        else:
            decay = torch.ones_like(edge_weights)

        W_bil = edge_weights * decay
        alpha = torch.sigmoid(self.alpha_proj(h_0)) * 0.85 + 0.10
        inv_alpha = 1.0 - alpha
        h_0_scaled = h_0 * alpha

        h_ctx = h_0
        for _ in range(self.k_hops):
            out = torch.zeros_like(h_ctx)
            if len(src) > 0:
                h_src_proj = self.gat_w_src(h_ctx)
                h_dst_proj = self.gat_w_dst(h_ctx)
                edge_proj = self.gat_w_edge(W_bil.unsqueeze(1))
                h_edge = h_src_proj[src] + h_dst_proj[dst] + edge_proj

                e_raw = self.gat_a(F.leaky_relu(h_edge)).squeeze(-1)
                tau = torch.clamp(F.softplus(self.att_temp), min=0.05)
                e_scaled = e_raw / tau

                alpha_att = scatter_softmax(e_scaled, dst, N)
                msg = h_ctx[src] * alpha_att.unsqueeze(1)
                out.index_add_(0, dst, msg)

            agg = F.silu(self.mp_update(out))
            h_ctx = agg * inv_alpha + h_0_scaled

        Q = self.q_proj(h_id)
        K = self.k_proj(h_ctx)
        V = self.v_proj(h_ctx)

        idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x_dense.device.type == "mps" else torch.int64)
        self_loops = torch.arange(N, dtype=idx_dtype, device=x_dense.device)
        src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
        dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

        q_dst = Q[dst_with_self]
        k_src = K[src_with_self]
        v_src = V[src_with_self]

        cross_scores = (q_dst * k_src).sum(dim=-1) / (self.hidden_dim**0.5)
        cross_att = scatter_softmax(cross_scores, dst_with_self, N)

        pulled_msg = (v_src * cross_att.unsqueeze(1)).contiguous()
        ctx_pulled = torch.zeros_like(Q)
        ctx_pulled.index_add_(0, dst_with_self, pulled_msg)

        h_final = h_id + self.context_gate(ctx_pulled)
        h_norm = F.normalize(self.sp_norm(h_final), p=2, dim=-1)

        t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
        x_norm = F.normalize(x_dense, p=2, dim=-1)
        bio_sim = torch.mm(x_norm, t_proj_weights.t())

        gnn_shift_raw = self.topic_proj(h_norm)
        gnn_shift_norm = F.normalize(gnn_shift_raw, p=2, dim=-1)

        base_logits = bio_sim + (cfg.gnn_shift_weight * gnn_shift_norm)
        noise = torch.randn_like(base_logits) * cfg.train_noise if self.training else 0.0
        base_logits = base_logits + noise

        # STEP 1: Apply Dynamic Topic Masking (-1e9 to deactivated slots)
        mask_val = torch.tensor(-1e9, device=base_logits.device, dtype=base_logits.dtype)
        masked_base_logits = torch.where(self.active_topic_mask.unsqueeze(0), base_logits, mask_val)

        current_scale = getattr(self, "current_scale", cfg.inference_scale)
        logits = masked_base_logits * current_scale

        return logits, anchors_raw

    def forward(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, anchors_raw = self.encode(x_dense, src, dst, edge_weights)
        current_alpha = getattr(self, "current_alpha", cfg.inference_alpha)
        current_temp = getattr(self, "current_temp", cfg.inference_temp)
        progress = getattr(self, "current_progress", 1.0)

        # STEP 3: Append Background Noise Sink Channel (K + 1)
        null_logit = torch.zeros(logits.size(0), 1, device=logits.device, dtype=logits.dtype)
        augmented_logits = torch.cat([logits, null_logit], dim=1)

        sparse_prob_aug = entmax_bisect(augmented_logits, alpha=current_alpha, dim=1)

        if self.training:
            smooth_prob_aug = F.softmax(augmented_logits / current_temp, dim=1)
            smooth_weight = 0.50 - (0.45 * progress)
            sparse_weight = 1.0 - smooth_weight
            prob_aug = (sparse_weight * sparse_prob_aug) + (smooth_weight * smooth_prob_aug)
        else:
            prob_aug = sparse_prob_aug

        # Slice to real K programs for tissue reconstruction
        prob = prob_aug[:, : self.n_metaprograms]

        mag = x_dense.sum(dim=1, keepdim=True)
        frac = prob * mag
        return frac, anchors_raw

    def calc_loss(
        self,
        recon_c: torch.Tensor,
        x_c: torch.Tensor,
        anchors: torch.Tensor,
        ortho_mat: torch.Tensor | None,
        ep: int,
        total_epochs: int,
        f_train: torch.Tensor | None = None,
        target_f_dist: torch.Tensor | None = None,
        kl_weight: float = cfg.kl_weight,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates regularized reconstruction loss over active topics."""
        num_pos = torch.clamp((x_c > 0).float().sum(), min=1.0)
        num_zeros = (x_c == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(current_dynamic_w, weight=0.1)

        is_non_zero = x_c > 0
        if self.training:
            zero_mask = torch.rand_like(x_c) < 0.05
            active_mask_x = (is_non_zero | zero_mask).to(x_c.dtype)
            masked_w_mat = torch.where(is_non_zero, current_dynamic_w, 1.0) * active_mask_x
        else:
            masked_w_mat = torch.where(is_non_zero, current_dynamic_w, 1.0)

        raw_delta = recon_c - x_c
        asymmetry_factor = 1.0 + (is_non_zero.to(x_c.dtype) * 2.0) * (raw_delta < 0).float()
        scaled_delta = torch.clamp(raw_delta * asymmetry_factor, min=-30.0, max=30.0)

        l_recon_sum = torch.sum(masked_w_mat * torch.log(torch.cosh(scaled_delta + 1e-6)))
        total_elements = max(1, x_c.shape[0] * x_c.shape[1])
        l_recon = l_recon_sum / total_elements

        # Dictionary losses computed strictly on active slots
        active_mask = self.active_topic_mask
        k_active = int(active_mask.sum().item())

        if k_active > 0:
            active_anchors = anchors[active_mask]
            anc_norm = F.normalize(active_anchors, p=2, dim=1)

            with torch.no_grad():
                ref_probs = F.softmax(self.anchor_logits[active_mask], dim=-1)
                ref_norm = F.normalize(ref_probs, p=2, dim=1)

            l_anc = 1.0 - (anc_norm * ref_norm).sum(dim=1).mean()

            if k_active > 1:
                latent_ortho = torch.mm(anc_norm, anc_norm.t()) * (1.0 - torch.eye(k_active, device=x_c.device))
                max_overlap = latent_ortho.max(dim=1)[0]
                l_ortho = (F.relu(max_overlap - cfg.ortho_overlap_threshold) ** 2).mean()
            else:
                l_ortho = torch.tensor(0.0, device=x_c.device)

            peak_excess = F.relu(active_anchors - cfg.anchor_peak_threshold)
            collapse_penalty = (peak_excess**2).sum(dim=1).mean()
            gene_entropy = -(active_anchors * torch.log(active_anchors + 1e-9)).sum(dim=1).mean()
        else:
            l_anc = torch.tensor(0.0, device=x_c.device)
            l_ortho = torch.tensor(0.0, device=x_c.device)
            collapse_penalty = torch.tensor(0.0, device=x_c.device)
            gene_entropy = torch.tensor(0.0, device=x_c.device)

        im_loss = torch.tensor(0.0, device=x_c.device)
        tsallis_val = 0.0

        if f_train is not None and k_active > 0:
            f_active = f_train[:, active_mask]
            f_sum = f_active.sum(dim=1, keepdim=True)
            f_norm = f_active / torch.clamp(f_sum, min=1e-6)

            alpha_ent = getattr(cfg, "tsallis_alpha", 1.5)
            f_safe = torch.clamp(f_norm, min=1e-5, max=1.0)

            if abs(alpha_ent - 1.0) > 1e-4:
                tsallis_h = (1.0 - (f_safe**alpha_ent).sum(dim=1).mean()) / (alpha_ent - 1.0)
            else:
                tsallis_h = -(f_safe * torch.log(f_safe)).sum(dim=1).mean()

            tsallis_val = tsallis_h.item()
            p_mean = torch.clamp(f_norm.mean(dim=0), min=1e-5, max=1.0)

            if target_f_dist is not None:
                kl_target = torch.clamp(target_f_dist[active_mask], min=1e-5)
                kl_target = kl_target / kl_target.sum()
                kl_marginal = (p_mean * (torch.log(p_mean + 1e-9) - torch.log(kl_target + 1e-9))).sum()
            else:
                uniform_prior = torch.ones(k_active, device=x_c.device) / k_active
                kl_marginal = (p_mean * (torch.log(p_mean + 1e-9) - torch.log(uniform_prior))).sum()
        else:
            tsallis_h = torch.tensor(0.0, device=x_c.device)
            kl_marginal = torch.tensor(0.0, device=x_c.device)

        progress = ep / max(1, total_epochs - 1)
        with torch.no_grad():
            recon_mag = l_recon.item()
            lock_weight = max(0.05, 1.0 - progress)
            anc_scale = recon_mag * 0.1 * lock_weight
            kl_scale = recon_mag * 0.05
            tsallis_weight = max(0.0, (progress - 0.5) * 2.0)
            tsallis_scale = recon_mag * 0.05 * tsallis_weight

        im_loss = (tsallis_h * tsallis_scale) + (kl_weight * kl_marginal * kl_scale)
        scaled_anc = l_anc * anc_scale
        scaled_ortho = (l_ortho + collapse_penalty) * (recon_mag * 0.05)
        scaled_gene_ent = gene_entropy * (recon_mag * 0.01)

        base_loss = l_recon + scaled_anc + scaled_ortho + scaled_gene_ent

        self._last_losses = {
            "rec": l_recon.item(),
            "anc": l_anc.item(),
            "ort": l_ortho.item(),
            "im": im_loss.item(),
            "base": base_loss.item(),
            "dyn_w": current_dynamic_w.item(),
            "kl_w": kl_weight,
            "tsallis_val": tsallis_val,
        }

        return (base_loss + im_loss, l_recon.detach(), l_anc.detach(), l_ortho.detach())


# =============================================================================
# 3. STEP 2: Biological Signal-to-Noise Gating Evaluator
# =============================================================================

@torch.no_grad()
def prune_inactive_topics(
    model: LibellaGNN,
    training_cache: List[Dict[str, Any]],
    device: torch.device,
    min_fold: float = 1.25,
    min_cell_pct: float = 0.005,
) -> int:
    """Evaluates biological enrichment and cell volume at the end of Phase 1."""
    model.eval()
    all_assigned_topics: List[torch.Tensor] = []
    all_raw_counts: List[torch.Tensor] = []

    print("\n" + "=" * 80)
    print("[➤] EXECUTING PHASE 1 BIOLOGICAL SIGNAL-TO-NOISE PRUNING CHECK")
    print(f"    Criteria: Marker Raw Fold >= {min_fold:.2f}x | Cell Volume >= {min_cell_pct*100:.1f}%")
    print("=" * 80)

    for b in training_cache:
        chunk = torch.load(b["chunk_file"], map_location="cpu", weights_only=False)
        x = chunk["x"].to(device=device, non_blocking=True)
        src = chunk["src"].to(device=device, non_blocking=True)
        dst = chunk["dst"].to(device=device, non_blocking=True)
        weights = chunk["weights"].to(device=device, non_blocking=True)

        x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
        if device.type != "mps":
            src = src.to(torch.int64)
            dst = dst.to(torch.int64)

        fracs, _ = model(x, src, dst, weights)
        train_idx = chunk["train_core_idx"].to(device=device, non_blocking=True)

        f_train = fracs[train_idx]
        x_train = x[train_idx]


        max_val, assigned = f_train.max(dim=1)
        assigned = torch.where(max_val > 1e-4, assigned, torch.tensor(-1, device=assigned.device))
        all_assigned_topics.append(assigned.cpu())
        all_raw_counts.append(x_train.cpu())

        del chunk, x, src, dst, weights, fracs, train_idx, f_train, x_train
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    assigned_all = torch.cat(all_assigned_topics)
    x_all = torch.cat(all_raw_counts)
    total_cells = len(assigned_all)

    safe_temp = torch.clamp(getattr(model, "dict_temp", torch.tensor(0.30)), min=0.25, max=1.0).item()
    anchors = F.softmax(model.topic_gene_logits / safe_temp, dim=-1).cpu()

    surviving_count = 0
    pruned_reasons: List[str] = []

    for k in range(model.n_metaprograms):
        if not model.active_topic_mask[k]:
            continue

        cell_mask = (assigned_all == k)
        cell_count = int(cell_mask.sum().item())
        cell_pct = cell_count / max(1, total_cells)

        # Criterion 1: Minimum cell volume (>= 0.5%)
        if cell_count < (total_cells * min_cell_pct):
            model.active_topic_mask[k] = False
            pruned_reasons.append(
                f"Topic {k:02d}: Pruned by Cell Volume ({cell_pct*100:.2f}% < {min_cell_pct*100:.1f}%, n={cell_count})"
            )
            continue

        # Criterion 2: Marker raw fold enrichment over background (>= 1.25x)
        top_markers = torch.topk(anchors[k], k=10).indices
        expr_in = x_all[cell_mask][:, top_markers].mean()
        expr_out = x_all[~cell_mask][:, top_markers].mean() if (~cell_mask).sum() > 0 else torch.tensor(1.0)
        fold_enrichment = (expr_in / (expr_out + 1e-6)).item()

        if fold_enrichment < min_fold:
            model.active_topic_mask[k] = False
            pruned_reasons.append(
                f"Topic {k:02d}: Pruned by Marker Fold ({fold_enrichment:.2f}x < {min_fold:.2f}x, n={cell_count})"
            )
        else:
            surviving_count += 1
            print(f"  [✓ RETAINED] Topic {k:02d}: Cells = {cell_count:,} ({cell_pct*100:.2f}%) | Raw Fold = {fold_enrichment:.2f}x")

    for r in pruned_reasons:
        print(f"  [✗ PRUNED]   {r}")

    print("-" * 80)
    print(f"[✓] Biological Pruning Complete: Retained {surviving_count}/{model.n_metaprograms} Genuine Programs.")
    print("=" * 80 + "\n")
    return surviving_count


# =============================================================================
# 4. Training Engine & Prefetcher
# =============================================================================

def prefetch_batches(
    meta_batches: List[List[Dict[str, Any]]]
) -> Iterator[Tuple[List[Dict[str, Any]], List[Any]]]:
    """Synchronous batch prefetching without thread lock overhead."""
    for meta_meta in meta_batches:
        chunks = [torch.load(b["chunk_file"], map_location="cpu", weights_only=False) for b in meta_meta]
        yield meta_meta, chunks


def train_pruned_gnn(
    training_cache: List[Dict[str, Any]],
    common_genes: List[str],
    optimal_k: int,
    init_components: np.ndarray | None,
    out_dir: Path,
) -> Tuple[LibellaGNN, Dict[str, Any]]:
    """Master orchestrator for training the 3-step pruning Libella model."""
    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "libella_model_pruned_optimal.pt"

    print("\n" + "=" * 80)
    print("      INITIALIZING LIBELLA 3-STEP PRUNING ARCHITECTURE TRAINING")
    print(f"      Device: {device} | Genes: {len(common_genes)} | Starting Slots: {optimal_k}")
    print("=" * 80)

    model = LibellaGNN(
        in_channels=len(common_genes),
        n_metaprograms=optimal_k,
        init_components=init_components,
    ).to(device)

    base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
    anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
            {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    tracker = PhaseTracker(max_p1_epochs=15)
    ema_mean = None
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    best_composite_score = float("inf")
    history: Dict[str, List[Any]] = {"train_loss": [], "val_loss": [], "autopsy_metrics": []}
    max_entropy_scalar = float(np.log(optimal_k))
    pruning_executed = False

    for epoch in tqdm(range(cfg.epochs), desc="Training Libella [Pruned]", leave=True):
        model.train()
        train_steps, val_steps, train_chunk_count = 0, 0, 0
        train_loss_acc = torch.tensor(0.0, device=device)
        val_loss_acc = torch.tensor(0.0, device=device)
        epoch_p_mean_sum = torch.zeros(optimal_k, device=device)

        gpu_telemetry = {
            "ent": torch.tensor(0.0, device=device),
            "col_r": torch.tensor(0.0, device=device),
            "kl_w": torch.tensor(0.0, device=device),
            "g_w": torch.tensor(0.0, device=device),
            "p_w": torch.tensor(0.0, device=device),
            "l_rec": torch.tensor(0.0, device=device),
            "l_anc": torch.tensor(0.0, device=device),
            "l_ort": torch.tensor(0.0, device=device),
        }

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps = len(meta_batches)
        alpha_ema = min(0.001, 1.0 / (total_steps * 5.0 + 1e-9))
        nan_detected = False

        for step, (meta_meta, chunk_iter) in enumerate(prefetch_batches(meta_batches)):
            optimizer.zero_grad(set_to_none=True)

            for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):
                x = batch["x"].to(device=device, non_blocking=True)
                src = batch["src"].to(device=device, non_blocking=True)
                dst = batch["dst"].to(device=device, non_blocking=True)
                weights = batch["weights"].to(device=device, non_blocking=True)

                if model.training and len(src) > 0:
                    keep_mask = torch.rand(src.size(0), device=device) > cfg.edge_dropout
                    src = src[keep_mask]
                    dst = dst[keep_mask]
                    weights = weights[keep_mask]

                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != "mps":
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

                prog = tracker.get_progress()
                model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * (prog**0.8))
                model.current_temp = cfg.temp_end + ((cfg.temp_start - cfg.temp_end) * ((1.0 - prog) ** 1.5))
                model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * prog)
                model.current_progress = prog

                fracs, pure_anchors = model(x, src, dst, weights)

                train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)
                f_train = fracs[train_idx]
                x_train = x[train_idx]

                p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                current_p_mean = p_train.mean(dim=0)

                # Prior distribution target calculated over active topics
                uniform_prior = torch.zeros_like(current_p_mean)
                active_mask = model.active_topic_mask
                k_act = active_mask.sum().clamp(min=1)
                uniform_prior[active_mask] = 1.0 / k_act.float()

                if ema_mean is None:
                    ema_mean = current_p_mean.detach()
                else:
                    ema_mean = alpha_ema * current_p_mean.detach() + (1.0 - alpha_ema) * ema_mean

                ideal_c = torch.clamp(uniform_prior * 2.0 - ema_mean, min=1e-5)
                target_f_dist = ideal_c / ideal_c.sum()

                active_ema = ema_mean[active_mask]
                active_ema_norm = active_ema / active_ema.sum().clamp(min=1e-6)
                ema_entropy = -torch.sum(active_ema_norm * torch.log(active_ema_norm + 1e-9))
                collapse_ratio = torch.clamp(1.0 - (ema_entropy / math.log(max(2, int(k_act.item())))), min=0.0, max=1.0)

                peak_p = ema_mean.max()
                hub_multiplier = F.relu((peak_p / cfg.hub_threshold) - 1.0) * 10.0
                dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + hub_multiplier

                recon = f_train @ pure_anchors

                true_batch_loss, base_recon_val, base_anc_val, base_ort_val = model.calc_loss(
                    recon,
                    x_train,
                    pure_anchors,
                    None,
                    epoch,
                    cfg.epochs,
                    f_train=f_train,
                    target_f_dist=target_f_dist,
                    kl_weight=dynamic_kl_w,
                )

                if torch.isnan(true_batch_loss) or torch.isinf(true_batch_loss):
                    nan_detected = True
                    break

                (true_batch_loss / len(meta_meta)).backward()

                train_loss_acc += true_batch_loss.detach()
                train_steps += 1

                gpu_telemetry["g_w"] += pure_anchors.max(dim=1).values.mean().detach() * 100.0
                gpu_telemetry["p_w"] += p_train.max(dim=1).values.mean().detach() * 100.0
                gpu_telemetry["ent"] += ema_entropy.detach()
                gpu_telemetry["col_r"] += collapse_ratio.detach()
                gpu_telemetry["kl_w"] += dynamic_kl_w.detach()
                gpu_telemetry["l_rec"] += base_recon_val.detach()
                gpu_telemetry["l_anc"] += base_anc_val.detach()
                gpu_telemetry["l_ort"] += base_ort_val.detach()
                epoch_p_mean_sum += current_p_mean.detach()
                train_chunk_count += 1

                val_core_idx = batch["val_core_idx"]
                if val_core_idx.numel() > 0:
                    val_idx = val_core_idx.to(device=device, non_blocking=True)
                    with torch.no_grad():
                        f_val = fracs[val_idx]
                        x_val = x[val_idx]
                        val_recon = f_val @ pure_anchors
                        is_non_zero_val = x_val > 0
                        w_mat = torch.where(is_non_zero_val, model.dynamic_w_ema, 1.0)
                        zero_mask_val = torch.where(is_non_zero_val, 1.0, cfg.zero_mask_rate).to(x_val.dtype)
                        raw_delta_val = val_recon - x_val
                        asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * 2.0) * (raw_delta_val < 0).to(x_val.dtype)
                        scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-cfg.delta_clamp, max=cfg.delta_clamp)
                        val_loss_sum = torch.sum(w_mat * zero_mask_val * torch.log(torch.cosh(scaled_delta_val + 1e-6)))
                        val_loss_acc += (val_loss_sum / max(1, x_val.numel())).detach()
                        val_steps += 1

            if nan_detected:
                optimizer.zero_grad(set_to_none=True)
                break

            b_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n and p.grad is not None]
            if b_params:
                torch.nn.utils.clip_grad_norm_(b_params, max_norm=cfg.grad_clip)

            a_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n and p.grad is not None]
            if a_params:
                torch.nn.utils.clip_grad_norm_(a_params, max_norm=cfg.grad_clip)

            optimizer.step()

        if nan_detected:
            print(f"\n[!] NaN gradient detected at Epoch {epoch}. Halting.")
            break

        history["train_loss"].append((train_loss_acc / max(1, train_steps)).item())
        history["val_loss"].append((val_loss_acc / max(1, val_steps)).item())
        scheduler.step()

        epoch_telemetry = {k: (v / max(1, train_chunk_count)).item() for k, v in gpu_telemetry.items()}
        epoch_p_mean = (epoch_p_mean_sum / max(1, train_chunk_count)).cpu()
        top_topic_val, top_topic_idx = epoch_p_mean.max(dim=0)
        epoch_telemetry["top_t_pct"] = top_topic_val.item() * 100.0
        epoch_telemetry["top_t_id"] = top_topic_idx.item()

        n_active_slots = int(model.active_topic_mask.sum().item())

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(history["train_loss"][-1], 4),
            "val_loss": round(history["val_loss"][-1], 4),
            "pure_rec": round(epoch_telemetry["l_rec"], 4),
            "p_w": round(epoch_telemetry["p_w"], 2),
            "g_w": round(epoch_telemetry["g_w"], 2),
            "n_active_slots": n_active_slots,
            "top_topic_pct": round(epoch_telemetry["top_t_pct"], 2),
        }
        history["autopsy_metrics"].append(epoch_metrics)

        current_rec = epoch_telemetry["l_rec"]
        current_pw = epoch_telemetry["p_w"]
        composite_score = current_rec / max(1.0, math.sqrt(current_pw / 100.0))

        if composite_score < best_composite_score and not nan_detected:
            best_composite_score = composite_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_composite_score": best_composite_score,
                    "metrics": epoch_metrics,
                    "history": history,
                },
                ckpt_path,
            )

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            tqdm.write(
                f"[Ep {epoch+1:02d}] Rec: {epoch_telemetry['l_rec']:.3f} | "
                f"Val: {history['val_loss'][-1]:.3f} | P_W: {epoch_telemetry['p_w']:.1f}% | "
                f"Active Slots: {n_active_slots}/{optimal_k} | TopT: {epoch_telemetry['top_t_pct']:.1f}%"
            )


        epochs_remaining = cfg.epochs - epoch - 1
        if tracker.phase == 1 and epochs_remaining <= 15:
            tracker.force_phase2(epoch, epoch_telemetry["l_rec"])

        was_phase_1 = tracker.phase == 1
        is_done = tracker.step(epoch_telemetry, epoch)

        # STEP 2: Trigger biological pruning on Phase 2 entry or Epoch 15 cutoff
        if (was_phase_1 and tracker.phase == 2 and not pruning_executed) or (epoch == 15 and not pruning_executed):
            prune_inactive_topics(model, training_cache, device, min_fold=1.25, min_cell_pct=0.005)
            pruning_executed = True
            model.train()

        if is_done:
            tqdm.write(f"[✓] Convergence achieved at Epoch {epoch+1}.")
            break

    if ckpt_path.exists():
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    return model, history


# =============================================================================
# 5. Main Execution Entrypoint
# =============================================================================

def run_pilot():
    chunk_paths = [
        Path(f"/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/temp_training_chunks/benchmark_data_chunk_{i}.pt")
        for i in range(10)
    ]
    genes_path = Path("/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/common_genes.json")
    priors_path = Path("/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/global_cnmf_priors.pkl")
    out_dir = Path("/Users/Hemato/project_3/libella/src/libella/libella_loss_pilot_output/pruned_optimal").resolve()

    print("\n--- 1. Loading Training Artefacts & Initial Priors ---")
    with open(genes_path, "r") as f:
        common_genes = json.load(f)

    with open(priors_path, "rb") as f:
        prior_data = joblib.load(f)

    if isinstance(prior_data, dict):
        init_components = prior_data.get("components", prior_data.get("init_components", None))
        optimal_k = prior_data.get("optimal_k", init_components.shape[0] if init_components is not None else 38)
    elif isinstance(prior_data, np.ndarray):
        init_components = prior_data
        optimal_k = init_components.shape[0]
    elif isinstance(prior_data, tuple):
        init_components = prior_data[0]
        optimal_k = prior_data[1]
    else:
        init_components = None
        optimal_k = 38

    training_cache = [{"patient_name": "benchmark_cohort", "chunk_file": cp} for cp in chunk_paths if cp.exists()]

    if len(training_cache) == 0:
        raise FileNotFoundError("Could not find training chunk files. Verify chunk paths.")

    model, hist = train_pruned_gnn(
        training_cache=training_cache,
        common_genes=common_genes,
        optimal_k=optimal_k,
        init_components=init_components,
        out_dir=out_dir,
    )

    print(f"\n[✓] 3-Step Pruning Pilot Complete! Final Pareto model saved to:\n    {out_dir / 'libella_model_pruned_optimal.pt'}\n")


if __name__ == "__main__":
    run_pilot()