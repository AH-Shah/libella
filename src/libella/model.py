"""Spatial Graph Neural Network architecture for Libella (Top-K Hard Sparsity SAE)."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg


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
        if src.numel() > 0 and not src.is_contiguous():
            src = src.contiguous()
            dst = dst.contiguous()

        N = x_dense.size(0)

        # 1. Depth Disentanglement
        cell_mass = torch.clamp(x_dense.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        x_norm = x_dense / cell_mass

        # 2. Self Feature Extraction
        h_self = self.self_enc(x_norm)

        # 3. Bilateral Edge Filtering & Symmetric Normalization
        if src.numel() > 0:
            with torch.no_grad():
                # Direct dot product without materializing full broadcasted intermediates
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)
                decay = torch.sigmoid(
                    (cos_sim - getattr(cfg, "edge_sim_threshold", 0.50))
                    * getattr(cfg, "edge_decay_slope", 15.0)
                )

                # Symmetric Laplacian normalization
                deg = torch.zeros((N, 1), device=x_dense.device, dtype=x_dense.dtype)
                deg.index_add_(0, dst, torch.ones((dst.size(0), 1), device=x_dense.device, dtype=x_dense.dtype))
                norm_inv_sqrt = 1.0 / torch.clamp(deg.sqrt(), min=1.0)
                edge_norm = norm_inv_sqrt[src] * norm_inv_sqrt[dst]

            W_bil = edge_weights.unsqueeze(1) * decay

            # Feature-aware directional gate
            gate_in = torch.cat([h_self[src] - h_self[dst], W_bil], dim=-1)
            gate = self.edge_gate(gate_in)

            # Pre-fuse static edge modulator across hops to save kernel launches
            static_edge_mod = gate * edge_norm

            # 4. Selective Graph Message Passing with buffer reuse
            h_sp = h_self
            agg = torch.empty((N, self.hidden_dim), device=x_dense.device, dtype=x_dense.dtype)
            for _ in range(self.k_hops):
                h_proj = self.spatial_lin(h_sp)
                msg = h_proj[src] * static_edge_mod
                agg.zero_()
                agg.index_add_(0, dst, msg)
                h_sp = h_sp + F.silu(agg)
        else:
            h_sp = h_self

        # 5. Fusion of Self + Spatial Context
        h_fused = F.layer_norm(h_self + h_sp, [self.hidden_dim])

        # 6. Unconstrained Magnitude & Spatial Gating Shifts
        z_mag = self.mag_head(h_fused)

        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        bio_sim = torch.mm(x_norm, w_dec_norm.t())
        spatial_shift = self.spatial_gate_head(h_sp)

        progress = getattr(self, "current_progress", 1.0) if self.training else 1.0
        spatial_warmup = 0.20 + 0.80 * min(1.0, progress * 2.0)

        raw_affinity = F.softplus(bio_sim + (self.spatial_gain * spatial_warmup * spatial_shift))
        pre_acts = raw_affinity * z_mag

        # 7. Top-K Hard Sparsity
        target_k = getattr(self, "current_k", self.k)
        topk_vals, topk_indices = torch.topk(pre_acts, k=target_k, dim=-1)
        z_sparse = torch.zeros(pre_acts.shape, dtype=pre_acts.dtype, device=pre_acts.device).scatter(
            -1, topk_indices, topk_vals
        )

        return z_sparse, pre_acts, cell_mass, z_mag

    @torch.no_grad()
    def resample_dead_latents(
        self,
        r_pos: torch.Tensor,
        dead_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Resamples dead atoms with Gram-Schmidt projection without CPU sync stalls."""
        dead_indices = torch.nonzero(dead_mask).squeeze(-1)
        num_dead = dead_indices.numel()
        if num_dead == 0:
            return 0

        cell_res_energy = r_pos.norm(p=2, dim=-1)
        valid_res_mask = cell_res_energy > 0.05
        num_valid_res = int(valid_res_mask.sum())
        k_resample = min(num_dead, num_valid_res)
        if k_resample == 0:
            return 0

        worst_cells = torch.topk(cell_res_energy, k=k_resample, dim=0).indices
        target_dead_ids = dead_indices[:k_resample]

        candidates = r_pos[worst_cells].clone()
        noise = torch.randn_like(candidates) * 0.02
        candidates = F.relu(candidates + noise)

        healthy_mask = ~dead_mask
        if healthy_mask.sum() > 0:
            w_healthy = F.normalize(self.decoder_weight.data[healthy_mask], p=2, dim=-1)
            proj = torch.mm(candidates, w_healthy.t())
            candidates = candidates - torch.mm(proj, w_healthy)
            candidates = F.relu(candidates)

        norms = candidates.norm(p=2, dim=-1, keepdim=True)
        collapsed = (norms < 1e-4).squeeze(-1)
        if collapsed.sum() > 0:
            candidates[collapsed] = F.relu(
                torch.randn(int(collapsed.sum()), candidates.size(-1), device=candidates.device)
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

        # Baseline Decoupling
        baseline_gene = F.normalize(F.softplus(self.decoder_bias) + 1e-6, p=2, dim=-1).unsqueeze(0)
        ambient_coeff = torch.sigmoid(self.ambient_scale) * getattr(cfg, "ambient_max_cap", 0.35)

        comp_profile = (1.0 - ambient_coeff) * torch.mm(z, w_dec_norm) + (ambient_coeff * baseline_gene)
        x_recon = comp_profile * cell_mass

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

            if dead_mask_ret.any() and residual_energy > getattr(cfg, "aux_min_residual_energy", 0.05):
                dead_indices = torch.nonzero(dead_mask_ret).squeeze(-1)
                num_dead = dead_indices.numel()
                k_aux = min(max(getattr(cfg, "aux_min_k", 2), self.aux_k), num_dead)

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
        # 1. Variance-Weighted Cell-Averaged Asymmetric Log-Cosh Loss
        is_non_zero = x_true > 0
        num_pos = torch.clamp(is_non_zero.float().sum(), min=1.0)
        num_zeros = (x_true == 0).float().sum()
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
        asym_factor = 1.0 + (is_non_zero.float() * asym_penalty) * (raw_delta < 0).float()

        delta_clamp = getattr(cfg, "delta_clamp", 30.0)
        scaled_delta = torch.clamp(raw_delta * asym_factor, min=-delta_clamp, max=delta_clamp)

        # Numerically stable, zero-allocation log(cosh(u)) = |u| + softplus(-2|u|) - ln(2)
        abs_delta = scaled_delta.abs()
        log_cosh_val = abs_delta + F.softplus(-2.0 * abs_delta) - 0.6931471805599453

        per_cell_loss = torch.sum(variance_weight * log_cosh_val, dim=-1)
        l_recon = torch.mean(per_cell_loss) / math.sqrt(x_true.shape[-1])

        # 2. Strict Orthogonality Barrier (In-place diagonal zeroing)
        gram = torch.mm(w_dec_norm, w_dec_norm.t())
        gram.fill_diagonal_(0.0)

        ortho_thresh = getattr(cfg, "ortho_overlap_threshold", 0.30)
        excess_corr = F.relu(gram - ortho_thresh)
        num_violating = torch.clamp((excess_corr > 0).float().sum(), min=1.0)
        l_ortho_mean = excess_corr.pow(2).sum() / num_violating

        max_corr = gram.max()
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
        """Harvests parameter/gradient norms, SVD spectrum, effective rank, and correlation."""
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

            sim = torch.mm(w, w.t())
            off_diag_mask = ~torch.eye(w.size(0), dtype=torch.bool, device=w.device)
            off_diag_vals = sim.masked_select(off_diag_mask)
            if off_diag_vals.numel() > 0:
                stats["dict/max_cross_corr"] = off_diag_vals.max().item()
                stats["dict/mean_cross_corr"] = off_diag_vals.abs().mean().item()

            w_cpu = w.detach().cpu()
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