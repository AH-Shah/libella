"""Spatial Graph Neural Network architecture for Libella (Top-K Hard Sparsity SAE)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import cfg
from .utils import PhaseTracker, scatter_softmax


class LibellaGNN(nn.Module):
    """Core Libella Spatial GNN architecture with Top-K Hard Sparsity and Residual AuxK Revival."""

    def __init__(
        self,
        in_channels: int,
        n_metaprograms: int,
        init_components: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.k_hops = cfg.k_hops
        self.n_latents = n_metaprograms
        self.in_channels = in_channels
        self.k = getattr(cfg, "topk_k", 3)  # Hard Top-K target (L0 = k)

        # --- 1. Context & Identity Encoders ---
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

        # --- 2. Cross-Attention Projections with Linear-Gated Highway ---
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.cross_temp = nn.Parameter(torch.tensor(getattr(cfg, "cross_temp_init", -2.0)))

        self.context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.sp_norm = nn.LayerNorm(self.hidden_dim)

        # --- 3. Dual-Stream Projections ---
        # Magnitude Stream: Unconstrained positive scaling per latent
        self.mag_enc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_latents),
            nn.Softplus(beta=1.0),
        )

        # Spatial Context Gating Stream with Learnable Gain
        self.gate_spatial_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_latents),
        )
        self.spatial_gain = nn.Parameter(
            torch.tensor(float(getattr(cfg, "spatial_gain_init", 1.0)))
        )

        # --- 4. Oblique Unit-Norm Decoder Dictionary ---
        if init_components is not None:
            dec_init = torch.tensor(init_components, dtype=torch.float32)
            if dec_init.shape != (self.n_latents, in_channels):
                dec_init = torch.randn(self.n_latents, in_channels)
            dec_init = F.normalize(dec_init, p=2, dim=-1)
        else:
            dec_init = F.normalize(torch.randn(self.n_latents, in_channels), p=2, dim=-1)

        self.decoder_weight = nn.Parameter(dec_init)
        self.decoder_bias = nn.Parameter(torch.zeros(in_channels))
        self.ambient_scale = nn.Parameter(torch.tensor(getattr(cfg, "ambient_scale_init", 0.50)))

        # Buffers and aux state tracking
        self.register_buffer("ortho_mask", 1.0 - torch.eye(self.n_latents, dtype=torch.float32))
        self.register_buffer("steps_since_active", torch.zeros(self.n_latents, dtype=torch.int64))
        self.dead_step_threshold = getattr(cfg, "dead_step_threshold", 12)
        self.aux_k = getattr(cfg, "aux_k", 4)
        self.ortho_sample_size = getattr(cfg, "ortho_sample_size", min(256, self.n_latents))

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.alpha_proj = nn.Linear(self.hidden_dim, 1)
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))

        # --- 5. Spatial GAT Components ---
        self.gat_w_src = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gat_w_dst = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gat_w_edge = nn.Linear(1, self.hidden_dim, bias=True)
        self.gat_a = nn.Linear(self.hidden_dim, 1, bias=False)
        self.att_temp = nn.Parameter(torch.tensor(getattr(cfg, "att_temp_init", -2.0)))
        self.mp_update = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        N = x_dense.size(0)

        # 1. Depth Disentanglement
        cell_mass = torch.clamp(x_dense.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        x_norm = x_dense / cell_mass

        h_id = self.id_enc(x_norm)
        h_0 = self.lin_appnp(self.ctx_enc(x_norm))

        # 2. Bilateral Edge Decay
        if len(src) > 0:
            with torch.no_grad():
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1)
            decay = torch.sigmoid(
                (cos_sim - getattr(cfg, "edge_sim_threshold", 0.60))
                * getattr(cfg, "edge_decay_slope", 20.0)
            )
            W_bil = edge_weights * decay
        else:
            W_bil = edge_weights

        # 3. K-Hop Spatial Message Passing Loop (FIXED GAT ATTENTION)
        alpha = (
            torch.sigmoid(self.alpha_proj(h_0)) * getattr(cfg, "appnp_alpha_scale", 0.85)
            + getattr(cfg, "appnp_alpha_offset", 0.10)
        )
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

                # FIX 1: Use direct LeakyReLU output with learnable gain (DO NOT divide by sqrt(d))
                e_raw = self.gat_a(F.leaky_relu(h_edge, negative_slope=0.2)).squeeze(-1)
                # Gain multiplier (e.g. 2.0) keeps softmax selective and derivatives healthy
                alpha_att = scatter_softmax(e_raw * 2.0, dst, N)

                msg = h_ctx[src] * alpha_att.unsqueeze(1)
                out.index_add_(0, dst, msg)

            agg = F.silu(self.mp_update(out))
            # Residual GAT connection preserves feature energy across hops
            h_ctx = agg * inv_alpha + h_0_scaled

        # 4. Cross-Attention Highway (FIXED COSINE SCALING & RESIDUAL)
        Q = self.q_proj(h_id)
        K = self.k_proj(h_ctx)
        V = self.v_proj(h_ctx)

        if len(src) > 0:
            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)
            
            # FIX 2: Multiply cosine similarity by sharp temperature scale (e.g. 4.0 to 10.0)
            # Instead of dividing by sqrt(d), scale cosine range [-1, 1] -> [-4.0, +4.0]
            cross_scores = (Q_norm[dst] * K_norm[src]).sum(dim=-1) * 4.0
            cross_att = scatter_softmax(cross_scores, dst, N)
            
            pulled_msg = (V[src] * cross_att.unsqueeze(1)).contiguous()
            ctx_pulled = torch.zeros_like(Q)
            ctx_pulled.index_add_(0, dst, pulled_msg)
        else:
            ctx_pulled = V

        # 5. Dual-Stream Fusion with Residual Context Highway (FIX 3)
        h_id_norm = self.sp_norm(h_id)
        # Add h_ctx residual directly so spatial features bypass cross-attention bottlenecks
        ctx_combined = ctx_pulled + h_ctx
        ctx_norm = F.layer_norm(ctx_combined, [self.hidden_dim])
        
        gate_coeff = torch.sigmoid(self.context_gate(ctx_norm))
        h_final = (1.0 - 0.5 * gate_coeff) * h_id_norm + (1.0 + 0.5 * gate_coeff) * ctx_norm

        # 6. Latent Pre-Activations
        z_mag = self.mag_enc(h_final)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        bio_sim = torch.mm(x_norm, w_dec_norm.t())
        spatial_shift = self.gate_spatial_proj(h_final)

        # Baseline spatial warmup schedule
        progress = getattr(self, "current_progress", 1.0) if self.training else 1.0
        spatial_warmup = 0.20 + 0.80 * min(1.0, progress * 2.0)

        raw_affinity = F.softplus(bio_sim + (self.spatial_gain * spatial_warmup * spatial_shift))
        pre_acts = raw_affinity * z_mag

        # 7. Top-K Hard Sparsity Operator
        target_k = getattr(self, "current_k", self.k)
        topk_vals, topk_indices = torch.topk(pre_acts, k=target_k, dim=-1)
        z_sparse = torch.zeros_like(pre_acts).scatter(-1, topk_indices, topk_vals)

        return z_sparse, pre_acts, cell_mass, z_mag
        
    @torch.no_grad()
    def resample_dead_latents(
        self, 
        r_pos: torch.Tensor, 
        dead_mask: torch.Tensor, 
        optimizer: torch.optim.Optimizer | None = None
    ) -> int:
        if not dead_mask.any():
            return 0

        dead_indices = torch.nonzero(dead_mask).squeeze(-1)
        num_dead = dead_indices.numel()
        cell_res_energy = r_pos.norm(p=2, dim=-1)

        # Select candidate cells with highest residual energy
        k_resample = min(num_dead, (cell_res_energy > 0.05).sum().item())
        if k_resample == 0:
            return 0

        worst_cells = torch.topk(cell_res_energy, k=k_resample, dim=0).indices
        target_dead_ids = dead_indices[:k_resample]

        # 1. Base candidate vectors from single-cell residuals
        candidates = r_pos[worst_cells].clone()

        # 2. Add small random noise to break symmetry / 1-hot collinearity
        noise = torch.randn_like(candidates) * 0.02
        candidates = F.relu(candidates + noise)

        # 3. Project out components parallel to existing healthy dictionary atoms
        healthy_mask = ~dead_mask
        if healthy_mask.any():
            w_healthy = F.normalize(self.decoder_weight.data[healthy_mask], p=2, dim=-1)
            # Gram-Schmidt projection: v' = v - (v · u) u
            proj = torch.mm(candidates, w_healthy.t()) # (k_resample, n_healthy)
            candidates = candidates - torch.mm(proj, w_healthy)
            candidates = F.relu(candidates) # Retain non-negativity

        # 4. Final safety normalization (fallback to random non-negative if fully collapsed)
        norms = candidates.norm(p=2, dim=-1, keepdim=True)
        collapsed = (norms < 1e-4).squeeze(-1)
        if collapsed.any():
            candidates[collapsed] = F.relu(torch.randn(collapsed.sum(), candidates.size(-1), device=candidates.device))
        
        new_atoms = F.normalize(candidates, p=2, dim=-1)
        self.decoder_weight.data[target_dead_ids] = new_atoms

        # Reset tracking state
        self.steps_since_active[target_dead_ids] = 0

        # Flush Adam momentum buffers for reset slices
        if optimizer is not None:
            state = optimizer.state.get(self.decoder_weight, None)
            if state is not None:
                if 'exp_avg' in state:
                    state['exp_avg'][target_dead_ids] = 0.0
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'][target_dead_ids] = 0.0

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

        # Ambient Baseline Decoupling
        baseline_gene = F.normalize(F.softplus(self.decoder_bias) + 1e-6, p=2, dim=-1).unsqueeze(0)
        ambient_coeff = torch.sigmoid(self.ambient_scale) * getattr(cfg, "ambient_max_cap", 0.40)

        comp_profile = (1.0 - ambient_coeff) * torch.mm(z, w_dec_norm) + (ambient_coeff * baseline_gene)
        x_recon = comp_profile * cell_mass

        # AuxK Dead Latent Tracking & Residual Preparation
        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(self.n_latents, dtype=torch.bool, device=x_dense.device)

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
            residual_energy = r_pos.norm(p=2, dim=-1).mean()

            # Continuous auxiliary alignment for dormant atoms
            if dead_mask_ret.any() and residual_energy > getattr(cfg, "aux_min_residual_energy", 1e-3):
                dead_indices = torch.nonzero(dead_mask_ret).squeeze(-1)
                num_dead = dead_indices.numel()
                k_aux = min(max(getattr(cfg, "aux_min_k", 1), self.aux_k), num_dead)

                w_dead = w_dec_norm[dead_indices]
                aux_sim = torch.mm(r_norm, w_dead.t())
                topk_res = torch.topk(aux_sim, k=k_aux, dim=-1)

                dead_mag = z_mag[:, dead_indices]
                topk_mag = torch.gather(dead_mag, -1, topk_res.indices)

                z_aux_weights = F.softplus(topk_res.values, beta=1.0) * topk_mag
                z_aux = torch.zeros_like(aux_sim).scatter(-1, topk_res.indices, z_aux_weights)
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
        # 1. Variance-Weighted Asymmetric Log-Cosh Loss
        is_non_zero = x_true > 0
        num_pos = torch.clamp(is_non_zero.float().sum(), min=1.0)
        num_zeros = (x_true == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(
                current_dynamic_w, weight=getattr(cfg, "dynamic_w_ema_weight", 0.05)
            )

        w_mat = torch.where(is_non_zero, self.dynamic_w_ema, 1.0)
        variance_weight = w_mat * (1.0 + torch.log1p(x_true))
        variance_weight = variance_weight / torch.clamp(variance_weight.mean(), min=1e-5)

        raw_delta = recon_x - x_true
        asym_penalty = getattr(cfg, "asym_penalty_weight", 0.5)
        asym_factor = 1.0 + (is_non_zero.float() * asym_penalty) * (raw_delta < 0).float()

        delta_clamp = getattr(cfg, "delta_clamp", 30.0)
        scaled_delta = torch.clamp(raw_delta * asym_factor, min=-delta_clamp, max=delta_clamp)

        l_recon = torch.sum(
            variance_weight * torch.log(torch.cosh(scaled_delta + 1e-6))
        ) / max(1, x_true.numel())

        # In calc_loss():
        gram = torch.mm(w_dec_norm, w_dec_norm.t())
        off_diag = gram * self.ortho_mask

        ortho_thresh = getattr(cfg, 'ortho_overlap_threshold', 0.30)
        excess_corr = F.relu(off_diag - ortho_thresh)

        # Mean violation loss
        num_violating = torch.clamp((excess_corr > 0).float().sum(), min=1.0)
        l_ortho_mean = excess_corr.pow(2).sum() / num_violating

        # Soft log-barrier instead of hard (max - 0.5)^2 * 200
        # -log(1 - x) smoothly approaches infinity as correlation approaches 0.95
        max_corr = torch.clamp(off_diag.max(), max=0.8)
        l_ortho_barrier = -torch.log(1.0 - max_corr + 1e-5)

        l_ortho = l_ortho_mean + 0.5 * l_ortho_barrier

        # 3. Top-K Sparsity (Exact L0 enforced; auxiliary regularization remains 0.0)
        l_sparse = torch.tensor(0.0, device=x_true.device)

        # 4. Normalized MSE Residual Alignment Loss (AuxK)
        if aux_recon is not None and r_norm is not None:
            res_energy = torch.clamp(r_norm.pow(2).sum(dim=-1).mean(), min=1e-4)
            aux_error = (aux_recon - r_norm).pow(2).sum(dim=-1).mean()
            l_aux = aux_error / res_energy
        else:
            l_aux = torch.tensor(0.0, device=x_true.device)

        # 5. Combined Dynamic Schedules
        base_ortho = getattr(cfg, "ortho_weight", 1.0)
        ortho_min = getattr(cfg, "ortho_min_scale", 0.2)
        current_ortho = base_ortho * (ortho_min + (1.0 - ortho_min) * progress)
        aux_weight = getattr(cfg, "aux_weight", 1.0)

        total_loss = l_recon + (current_ortho * l_ortho) + (aux_weight * l_aux)

        return total_loss, l_recon.detach(), l_ortho.detach(), l_sparse.detach(), l_aux.detach()

    @torch.no_grad()
    def get_deep_telemetry(self) -> dict[str, float]:
        """Harvests telemetry: parameter/gradient norms, SVD spectrum, effective rank, and correlation."""
        stats: dict[str, float] = {}
        total_g_norm_sq = 0.0

        for name, param in self.named_parameters():
            p_clean = name.replace(".", "/")
            stats[f"param_norm/{p_clean}"] = param.detach().norm(2).item()
            if param.grad is not None:
                g_norm = param.grad.detach().norm(2).item()
                total_g_norm_sq += g_norm**2
                stats[f"grad_norm/{p_clean}"] = g_norm
                stats[f"grad_zeros/{p_clean}_pct"] = (
                    (param.grad == 0).float().mean().item() * 100.0
                )

        stats["grad_norm/global_l2"] = total_g_norm_sq**0.5

        if hasattr(self, "decoder_weight"):
            w = F.normalize(self.decoder_weight, p=2, dim=1)

            # 1. Off-diagonal Gram correlations
            sim = torch.mm(w, w.t())
            off_diag_mask = ~torch.eye(w.size(0), dtype=torch.bool, device=w.device)
            off_diag_vals = sim.masked_select(off_diag_mask)
            if off_diag_vals.numel() > 0:
                stats["dict/max_cross_corr"] = off_diag_vals.max().item()
                stats["dict/mean_cross_corr"] = off_diag_vals.abs().mean().item()

            # 2. SVD Spectrum and Effective Rank (CPU offloaded for MPS/CUDA stability)
            w_cpu = w.detach().cpu()
            s = torch.linalg.svdvals(w_cpu)
            eff_rank = (s.sum() ** 2) / torch.clamp((s**2).sum(), min=1e-9)
            stats["dict/effective_rank"] = eff_rank.item()
            stats["dict/svd_sigma_1"] = s[0].item()
            stats["dict/svd_sigma_2"] = s[1].item() if s.numel() > 1 else 0.0
            stats["dict/svd_sigma_3"] = s[2].item() if s.numel() > 2 else 0.0

        # 3. Ambient Baseline Absorption Percentage
        if hasattr(self, "ambient_scale"):
            lr_mult = getattr(cfg, "ambient_lr_mult", 1.0)
            max_cap = getattr(cfg, "ambient_max_cap", 0.40)
            amb_pct = torch.sigmoid(self.ambient_scale * lr_mult).item() * max_cap * 100.0
            stats["model/ambient_absorption_pct"] = amb_pct

        # 4. Latent Activity Health
        dead_count = (self.steps_since_active >= self.dead_step_threshold).sum().item()
        stats["latents/dead_count"] = float(dead_count)
        stats["latents/active_pct"] = (1.0 - (dead_count / self.n_latents)) * 100.0

        return stats