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
        self.k = getattr(cfg, "topk_k", 6)

        # 1. Identity Encoder (Self Signal)
        self.self_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 2. Directional Feature-Conditioned Spatial Filter
        self.spatial_lin = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + 1),
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
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

        # 1. Bounds Guard & Alignment for MPS
        if has_edges:
            src = src.to(dtype=torch.int64).contiguous()
            dst = dst.to(dtype=torch.int64).contiguous()
            edge_weights = edge_weights.to(dtype=torch.float32).contiguous()
            valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
            if not valid.all():
                src = src[valid]
                dst = dst[valid]
                edge_weights = edge_weights[valid]
                has_edges = src.numel() > 0

        # 2. Depth Normalization (Conserves Total Cell Mass)
        cell_mass = torch.clamp(
            torch.linalg.vector_norm(x_dense, ord=2, dim=-1, keepdim=True), min=1e-5
        )
        x_norm = x_dense / cell_mass

        # 3. Self Feature Extraction
        h_self = self.self_enc(x_norm)

        # 4. Bilateral Edge Filtering (Hard Cutoff for V1 Sharpness)
        if has_edges:
            with torch.no_grad():
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)
                # Hard Bilateral Cutoff: Zero out edges across sharp transcriptomic boundaries (< 0.45)
                edge_sim_thresh = float(getattr(cfg, "edge_sim_threshold", 0.45))
                edge_mask = (cos_sim > edge_sim_thresh).to(dtype=x_dense.dtype)
                decay = torch.sigmoid(
                    (cos_sim - edge_sim_thresh) * getattr(cfg, "edge_decay_slope", 15.0)
                ) * edge_mask

                deg = torch.zeros((N, 1), dtype=x_dense.dtype, device=x_dense.device)
                deg.index_add_(0, dst, torch.ones((dst.size(0), 1), dtype=x_dense.dtype, device=x_dense.device))
                edge_norm = torch.rsqrt(torch.clamp(deg[src], min=1.0)) * torch.rsqrt(torch.clamp(deg[dst], min=1.0))

            W_bil = edge_weights.unsqueeze(1) * decay
            gate = torch.sigmoid(self.edge_gate(torch.cat([h_self[src] - h_self[dst], W_bil], dim=-1)))

            h_sp = h_self
            for _ in range(self.k_hops):
                msg = self.spatial_lin(h_sp)[src] * gate * edge_norm
                h_sp = h_sp + F.silu(torch.zeros_like(h_sp).index_add_(0, dst, msg))
        else:
            h_sp = h_self

        # 5. APPNP Teleport Identity Anchor (85% Self + 15% Spatial Context)
        alpha_teleport = float(getattr(cfg, "appnp_alpha", 0.85))
        h_fused = F.layer_norm(
            alpha_teleport * h_self + (1.0 - alpha_teleport) * h_sp, [self.hidden_dim]
        )

        # 6. Direct Latent Prediction (Unbounded for Magnitude Growth)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        bio_sim = F.linear(x_norm, w_dec_norm)
        spatial_logits = self.spatial_gate_head(h_fused)

        # Clean ReLU floor. Network backprop will scale spatial_logits naturally to reconstruct x_norm.
        raw_acts = F.relu(bio_sim + spatial_logits)

        # 7. Top-K Hard Sparsity
        target_k = getattr(self, "current_k", max(6, self.k))
        topk_vals, topk_indices = torch.topk(raw_acts, k=target_k, dim=-1)
        
        z_sparse = torch.zeros_like(raw_acts).scatter_(-1, topk_indices, topk_vals)

        # Return raw_acts as the last tuple item so telemetry tracking z_mag doesn't crash
        return z_sparse, raw_acts, cell_mass, raw_acts

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

        # In-place noise injection to prevent allocation thrashing
        candidates = r_pos[worst_cells].clone()
        noise = torch.randn_like(candidates) * 0.02
        candidates = F.relu(candidates.add_(noise))

        healthy_mask = ~dead_mask
        if healthy_mask.any():
            w_healthy = F.normalize(self.decoder_weight.data[healthy_mask], p=2, dim=-1)
            # Direct GEMM without explicit transpose view allocations
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

        # 2. Baseline Decoupling (Strict 15% Cap so Sparse Latents DO the reconstruction work)
        baseline_gene = F.normalize(F.softplus(self.decoder_bias) + 1e-6, p=2, dim=-1).unsqueeze(0)
        ambient_coeff = torch.sigmoid(self.ambient_scale) * getattr(cfg, "ambient_max_cap", 0.15)

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
                dead_mask_ret = self.steps_since_active >= getattr(self, "dead_step_threshold", 20)

            x_norm = F.normalize(x_dense, p=2, dim=-1)
            x_recon_norm = F.normalize(F.relu(x_recon), p=2, dim=-1)
            r_pos = F.relu(x_norm - x_recon_norm)
            r_pos_ret = r_pos.detach()
            r_norm = F.normalize(r_pos + 1e-6, p=2, dim=-1).detach()
            residual_energy = torch.linalg.vector_norm(r_pos, ord=2, dim=-1).mean()

            # Exact dual-gate AuxK condition
            if dead_mask_ret.any() and residual_energy > getattr(cfg, "aux_min_residual_energy", 0.05):
                dead_indices = torch.nonzero(dead_mask_ret).squeeze(-1)
                num_dead = dead_indices.numel()
                k_aux = min(max(getattr(cfg, "aux_min_k", 2), self.aux_k), num_dead)

                w_dead = w_dec_norm[dead_indices]
                aux_sim = F.relu(F.linear(r_norm, w_dead))
                topk_res = torch.topk(aux_sim, k=k_aux, dim=-1)

                z_aux = torch.zeros_like(aux_sim).scatter_(-1, topk_res.indices, topk_res.values)
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

        # Numerically stable, overflow-proof Log-Cosh implementation:
        # log(cosh(u)) = |u| + softplus(-2|u|) - log(2)
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
        
        # 1. Vectorized Parameter & Gradient Norm Extraction
        param_items = list(self.named_parameters())
        p_names = [name.replace(".", "/") for name, _ in param_items]
        
        # Calculate L2 norms on GPU
        p_norms = torch.stack([torch.linalg.vector_norm(p.detach(), ord=2) for _, p in param_items])
        
        g_tensors = [p.grad.detach() for _, p in param_items if p.grad is not None]
        g_indices = [i for i, (_, p) in enumerate(param_items) if p.grad is not None]
        
        if g_tensors:
            g_norms = torch.stack([torch.linalg.vector_norm(g, ord=2) for g in g_tensors])
            g_zero_pcts = torch.stack([(g == 0).float().mean() * 100.0 for g in g_tensors])
            total_g_norm = torch.linalg.vector_norm(g_norms, ord=2)
            
            # Single host transfer for all gradient statistics
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

        # 2. Dictionary Cross-Correlation & Spectral Analysis
        if hasattr(self, "decoder_weight"):
            w = F.normalize(self.decoder_weight, p=2, dim=1)
            sim = torch.mm(w, w.t())
            off_diag_mask = ~torch.eye(w.size(0), dtype=torch.bool, device=w.device)
            off_diag_vals = sim.masked_select(off_diag_mask)
            
            if off_diag_vals.numel() > 0:
                stats["dict/max_cross_corr"] = off_diag_vals.max().item()
                stats["dict/mean_cross_corr"] = off_diag_vals.abs().mean().item()

            # Offload SVD computation to CPU float32 without blocking the GPU stream
            w_cpu = w.detach().to(device="cpu", dtype=torch.float32)
            s = torch.linalg.svdvals(w_cpu)
            eff_rank = (s.sum() ** 2) / torch.clamp((s**2).sum(), min=1e-9)
            stats["dict/effective_rank"] = eff_rank.item()
            stats["dict/svd_sigma_1"] = s[0].item()
            stats["dict/svd_sigma_2"] = s[1].item() if s.numel() > 1 else 0.0
            stats["dict/svd_sigma_3"] = s[2].item() if s.numel() > 2 else 0.0

        # 3. Ambient Scale & Dead Latent Counters
        if hasattr(self, "ambient_scale"):
            lr_mult = getattr(cfg, "ambient_lr_mult", 1.0)
            max_cap = getattr(cfg, "ambient_max_cap", 0.35)
            amb_pct = torch.sigmoid(self.ambient_scale * lr_mult).item() * max_cap * 100.0
            stats["model/ambient_absorption_pct"] = amb_pct

        dead_count = (self.steps_since_active >= self.dead_step_threshold).sum().item()
        stats["latents/dead_count"] = float(dead_count)
        stats["latents/active_pct"] = (1.0 - (dead_count / self.n_latents)) * 100.0

        return stats