"""Spatial Graph Neural Network architecture for Libella (Top-K Hard Sparsity SAE)."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg


class SafePadeActivation(nn.Module):
    """
    Direct Safe-Padé rational activation unit (Yin & Yu, 2026).
    Smoothly approximates non-linear thresholding without hard gradient cliffs.
    """
    def __init__(self, p_deg: int = 3, q_deg: int = 2) -> None:
        super().__init__()
        self.p_coeffs = nn.Parameter(torch.randn(p_deg + 1) * 0.02)
        self.q_coeffs = nn.Parameter(torch.randn(q_deg) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_val = self.p_coeffs[0]
        for i in range(1, len(self.p_coeffs)):
            p_val = p_val + self.p_coeffs[i] * (x ** i)
        q_val = 1.0
        abs_x = torch.abs(x)
        for i in range(len(self.q_coeffs)):
            q_val = q_val + torch.abs(self.q_coeffs[i]) * (abs_x ** (i + 1))
        return p_val / q_val


class ExactLaPruneFunction(torch.autograd.Function):
    """
    Exact-budget differentiable Top-K layer with normalized second-moment hardness control (Antczak et al., 2026).
    Executes a 2D Newton-Raphson root solve in forward and an exact 2x2 IFT VJP in backward.
    """
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        k_target: torch.Tensor,
        gamma: torch.Tensor | float,
        max_iters: int = 25,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        B, N = scores.shape
        device = scores.device
        dtype = scores.dtype

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, device=device, dtype=dtype)
        if gamma.dim() == 0:
            gamma = gamma.view(1, 1).expand(B, 1)

        a = k_target / float(N)
        beta = a + (1.0 - a) * gamma
        target_m1 = k_target
        target_m2 = beta * k_target

        mean_s = scores.mean(dim=-1, keepdim=True)
        std_s = scores.std(dim=-1, keepdim=True).clamp(min=1e-4)

        b = mean_s + std_s * torch.clamp(1.0 - 2.0 * a, min=-2.5, max=2.5)
        tau = torch.zeros((B, 1), device=device, dtype=dtype)

        for _ in range(max_iters):
            t = torch.exp(tau).clamp(min=1e-5, max=100.0)
            u = (scores - b) / t
            p = torch.where(u <= 0.0, 0.5 * torch.exp(u), 1.0 - 0.5 * torch.exp(-u))
            q = 0.5 * torch.exp(-torch.abs(u))

            F1 = p.sum(dim=-1, keepdim=True) - target_m1
            F2 = (p ** 2).sum(dim=-1, keepdim=True) - target_m2

            if torch.max(F1.abs()) < tol and torch.max(F2.abs()) < tol:
                break

            inv_t = 1.0 / t
            J11 = -inv_t * q.sum(dim=-1, keepdim=True)
            J12 = -(u * q).sum(dim=-1, keepdim=True)
            J21 = -2.0 * inv_t * (p * q).sum(dim=-1, keepdim=True)
            J22 = -2.0 * (u * p * q).sum(dim=-1, keepdim=True)

            det = J11 * J22 - J12 * J21
            det_stable = torch.where(det.abs() < 1e-7, torch.sign(det + 1e-7) * 1e-7, det)

            delta_b = -(J22 * F1 - J12 * F2) / det_stable
            delta_tau = -(-J21 * F1 + J11 * F2) / det_stable

            b = b + delta_b.clamp(min=-5.0, max=5.0)
            tau = (tau + delta_tau.clamp(min=-2.0, max=2.0)).clamp(min=-10.0, max=5.0)

        t_final = torch.exp(tau).clamp(min=1e-5, max=100.0)
        u_final = (scores - b) / t_final
        p_final = torch.where(u_final <= 0.0, 0.5 * torch.exp(u_final), 1.0 - 0.5 * torch.exp(-u_final))

        ctx.save_for_backward(p_final, t_final, scores, b, k_target, gamma)
        return p_final

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        p, t, scores, b, k_target, gamma = ctx.saved_tensors
        B, N = scores.shape

        u = (scores - b) / t
        q = 0.5 * torch.exp(-torch.abs(u))
        inv_t = 1.0 / t

        v_b = -inv_t * (grad_output * q).sum(dim=-1, keepdim=True)
        v_tau = -(grad_output * u * q).sum(dim=-1, keepdim=True)

        J11 = -inv_t * q.sum(dim=-1, keepdim=True)
        J12 = -(u * q).sum(dim=-1, keepdim=True)
        J21 = -2.0 * inv_t * (p * q).sum(dim=-1, keepdim=True)
        J22 = -2.0 * (u * p * q).sum(dim=-1, keepdim=True)

        det = J11 * J22 - J12 * J21
        det_stable = torch.where(det.abs() < 1e-7, torch.sign(det + 1e-7) * 1e-7, det)

        lambda_1 = (v_b * J22 - v_tau * J21) / det_stable
        lambda_2 = (-v_b * J12 + v_tau * J11) / det_stable

        grad_scores = inv_t * q * (grad_output - lambda_1 - 2.0 * lambda_2 * p)

        a = k_target / float(N)
        c_a = 2.0 * a + (1.0 - 2.0 * a) * gamma
        grad_k = lambda_1 + c_a * lambda_2

        grad_gamma = lambda_2 * (1.0 - a) * k_target

        if ctx.needs_input_grad[2]:
            return grad_scores, grad_k, grad_gamma, None, None
        return grad_scores, grad_k, None, None, None


class LibellaGNN(nn.Module):
    """Core Libella Spatial GNN architecture with Top-K Hard Sparsity and Residual AuxK Revival."""

    def __init__(
        self,
        in_channels: int,
        n_metaprograms: int,
        init_components: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = getattr(cfg, "hidden_dim", 128)
        self.k_hops = getattr(cfg, "k_hops", 2)
        self.n_latents = n_metaprograms
        self.in_channels = in_channels
        self.target_k = float(getattr(cfg, "topk_k", 38))
        self.k = self.target_k
        self.max_k = float(self.target_k * 2.0)
        self.laprune_gamma = float(getattr(cfg, "laprune_gamma", 0.90))

        # SoftSAE Dynamic Sparsity & Cosine Scoring Parameterization
        self.k_predictor = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        init_scale = math.log(math.sqrt(in_channels))
        self.b_scale = nn.Parameter(torch.full((n_metaprograms,), init_scale))
        self.b_enc = nn.Parameter(torch.zeros(n_metaprograms))
        self.pade_gate = SafePadeActivation(p_deg=3, q_deg=2)

        # 1. Identity Encoder (Self Signal)
        self.self_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 2. Asymmetric Key-Query Spatial Attention
        self.spatial_lin = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.head_dim = max(16, self.hidden_dim // 2)
        self.q_proj = nn.Linear(self.hidden_dim, self.head_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.head_dim)

        self.edge_gate = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Tanh(),
        )
        
        # Phase 3: Bipolar GLU Feature-Wise Gating
        self.gate_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.val_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 3. Spatial Gate Head (Direct Rational Modulation)
        self.spatial_gate_head = nn.Linear(self.hidden_dim, self.n_latents)
        self.spatial_gain = nn.Parameter(
            torch.tensor(float(getattr(cfg, "spatial_gain_init", 1.0)))
        )

        # 4. Untied Encoder & Decoder Dictionaries
        if init_components is not None:
            dec_init = torch.tensor(init_components, dtype=torch.float32)
            if dec_init.shape != (self.n_latents, in_channels):
                dec_init = torch.randn(self.n_latents, in_channels)
        else:
            dec_init = torch.randn(self.n_latents, in_channels)

        dec_init = F.normalize(dec_init, p=2, dim=-1)

        self.decoder_weight = nn.Parameter(dec_init.clone())
        self.encoder_weight = nn.Parameter(dec_init.clone())
        self.decoder_bias = nn.Parameter(torch.zeros(in_channels))

        # Buffers & Aux State Tracking
        self.register_buffer("ortho_mask", 1.0 - torch.eye(self.n_latents, dtype=torch.float32))
        self.register_buffer("steps_since_active", torch.zeros(self.n_latents, dtype=torch.int64))
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("routing_mean", torch.zeros(self.n_latents, dtype=torch.float32))
        self.register_buffer("routing_std", torch.ones(self.n_latents, dtype=torch.float32))

        self.dead_step_threshold = getattr(cfg, "dead_step_threshold", 20)

        self.aux_k = getattr(cfg, "aux_k", 4)
        self.ortho_sample_size = getattr(cfg, "ortho_sample_size", min(256, self.n_latents))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Enforces unit-norm constraints on dictionaries after optimizer.step()."""
        self.decoder_weight.data = F.normalize(self.decoder_weight.data, p=2, dim=-1)
        self.encoder_weight.data = F.normalize(self.encoder_weight.data, p=2, dim=-1)

    def encode(
        self,
        x_dense: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        N = x_dense.size(0)

        # 1. Depth Disentanglement (True Biological Mass)
        cell_mass = torch.clamp(x_dense.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        x_norm = x_dense / cell_mass
        x_centered = x_norm - self.decoder_bias

        # 2. Pure Biological Cosine Scoring
        w_enc_dir = F.normalize(self.encoder_weight, p=2, dim=1)
        cosine_sim = torch.mm(x_centered, w_enc_dir.t())
        bio_scores = torch.exp(self.b_scale) * cosine_sim + self.b_enc

        # 3. Spatial GNN Message Passing (APPNP with Leaky Bipolar Gating)
        h_self = self.self_enc(x_centered)

        # 3. Bilateral Edge Filtering & Symmetric Normalization
        if len(src) > 0:
            with torch.no_grad():
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)
                decay = torch.sigmoid(
                    (cos_sim - getattr(cfg, "edge_sim_threshold", 0.50))
                    * getattr(cfg, "edge_decay_slope", 15.0)
                )

                # Symmetric Laplacian normalization
                deg = torch.zeros((N, 1), device=x_dense.device)
                deg.index_add_(0, dst, torch.ones((len(dst), 1), device=x_dense.device))
                norm_inv_sqrt = 1.0 / torch.clamp(deg.sqrt(), min=1.0)
                edge_norm = norm_inv_sqrt[src] * norm_inv_sqrt[dst]

            W_bil = edge_weights.unsqueeze(1) * decay

            # 1. Asymmetric Key-Query Gating
            q = self.q_proj(h_self)
            k = self.k_proj(h_self)
            kq_sim = (q[dst] * k[src]).sum(dim=-1, keepdim=True) / math.sqrt(self.head_dim)

            gate_in = torch.cat([kq_sim, W_bil], dim=-1)
            raw_gate = self.edge_gate(gate_in)
            
            # Phase 1: Sparse Bipolar Attention with Differentiable Softshrink
            dead_zone = float(getattr(cfg, "bipolar_deadzone", 0.20))
            gate = F.softshrink(raw_gate, lambd=dead_zone)

            # 4. Selective Bipolar Message Passing & APPNP
            H_0 = h_self
            H_k = H_0
            
            for _ in range(self.k_hops):
                h_proj = self.spatial_lin(H_k)
                msg = h_proj[src] * gate * edge_norm
                agg = torch.zeros_like(H_k)
                agg.index_add_(0, dst, msg)

                # Phase 3: Bipolar GLU Feature-Wise Gating
                gate_val = torch.sigmoid(self.gate_proj(agg)) 
                val = torch.tanh(self.val_proj(agg))
                H_mixed = gate_val * val
                
                spatial_prog = getattr(self, "current_spatial_progress", 1.0) if self.training else 1.0
                alpha_id = 0.50 + (0.30 * spatial_prog)
                alpha_sp = 1.0 - alpha_id
                
                H_k = (alpha_id * H_0) + (alpha_sp * H_mixed)
                
            h_sp = H_k
        else:
            h_sp = h_self
            decay = None
            raw_gate = None

        # 4. Spatial Modulation Generation
        spatial_prog = getattr(self, "current_spatial_progress", 1.0) if self.training else 1.0
        alpha_id = 0.50 + (0.30 * spatial_prog)
        alpha_sp = 1.0 - alpha_id
        
        h_pure_spatial = (h_sp - (alpha_id * h_self)) / max(alpha_sp, 1e-3)
        spatial_shift = torch.tanh(self.spatial_gate_head(h_pure_spatial))

        spatial_warmup = 0.10 + 0.90 * spatial_prog
        unleashed_gain = F.softplus(self.spatial_gain)
        spatial_gain_knob = torch.clamp(
            1.0 + (unleashed_gain * spatial_warmup * spatial_shift), 
            min=0.05, 
            max=2.5
        )

        # 5. Modulate Scores BEFORE Routing (Only modulate positive biological evidence)
        routed_scores = torch.where(bio_scores > 0.0, bio_scores * spatial_gain_knob, bio_scores)

        # 6. Exact-K Differentiable Routing & Activation
        k_ratio = self.k_predictor(x_centered)
        k_i_float = 4.0 + k_ratio * (self.max_k - 4.0)

        if self.training:
            gamma_scale = getattr(self, "current_gamma_progress", 1.0)
            gamma_effective = max(0.05, self.laprune_gamma * gamma_scale)
            soft_mask = ExactLaPruneFunction.apply(routed_scores, k_i_float, gamma_effective)
            z_sparse = F.relu(self.pade_gate(routed_scores)) * soft_mask
        else:
            k_discrete = torch.clamp(k_i_float.round().long(), min=1, max=self.n_latents)
            sorted_scores, _ = torch.sort(routed_scores, dim=-1, descending=True)
            hard_thresh = torch.gather(sorted_scores, dim=1, index=k_discrete - 1)
            hard_mask = (routed_scores >= hard_thresh).float()
            z_sparse = F.relu(self.pade_gate(routed_scores)) * hard_mask

        return z_sparse, routed_scores, cell_mass, None, decay, src, dst, raw_gate, k_i_float

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
        cell_res_energy = r_pos.norm(p=2, dim=-1)

        k_resample = min(num_dead, (cell_res_energy > 0.05).sum().item())
        if k_resample == 0:
            return 0

        worst_cells = torch.topk(cell_res_energy, k=k_resample, dim=0).indices
        target_dead_ids = dead_indices[:k_resample]

        candidates = r_pos[worst_cells].clone()
        noise = torch.randn_like(candidates) * 0.02
        candidates = F.relu(candidates + noise)

        healthy_mask = ~dead_mask
        if healthy_mask.any():
            w_healthy = F.normalize(self.decoder_weight.data[healthy_mask], p=2, dim=-1)
            proj = torch.mm(candidates, w_healthy.t())
            candidates = candidates - torch.mm(proj, w_healthy)
            candidates = F.relu(candidates)

        norms = candidates.norm(p=2, dim=-1, keepdim=True)
        collapsed = (norms < 1e-4).squeeze(-1)
        if collapsed.any():
            candidates[collapsed] = F.relu(
                torch.randn(collapsed.sum(), candidates.size(-1), device=candidates.device)
            )

        new_atoms = F.normalize(candidates, p=2, dim=-1)
        self.decoder_weight.data[target_dead_ids] = new_atoms
        self.steps_since_active[target_dead_ids] = 0
        self.routing_mean[target_dead_ids] = 0.0
        self.routing_std[target_dead_ids] = 1.0

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
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        z, pre_acts, cell_mass, _, decay, src, dst, raw_gate, k_i_float = self.encode(
            x_dense, src, dst, edge_weights
        )
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=-1)

        # Direct Magnitude-Bypass Decoder Reconstruction
        x_recon_centered = torch.mm(z, w_dec_norm)
        x_recon = (x_recon_centered + self.decoder_bias) * cell_mass

        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(self.n_latents, dtype=torch.bool, device=x_dense.device)

        if self.training:
            with torch.no_grad():
                active_in_batch = (z > 0.01).any(dim=0)
                self.steps_since_active.add_(1)
                self.steps_since_active.masked_fill_(active_in_batch, 0)
                dead_mask_ret = self.steps_since_active >= self.dead_step_threshold

            x_norm = F.normalize(x_dense, p=2, dim=-1)
            x_recon_norm = F.normalize(F.relu(x_recon) + 1e-6, p=2, dim=-1)
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
                aux_scores = torch.exp(self.b_scale[dead_indices]) * aux_sim + self.b_enc[dead_indices]
                topk_res = torch.topk(aux_scores, k=k_aux, dim=-1)

                z_aux_weights = F.relu(topk_res.values)
                z_aux = torch.zeros_like(aux_scores).scatter(-1, topk_res.indices, z_aux_weights)
                aux_recon = torch.mm(z_aux, w_dead)

        return (
            x_recon,
            z,
            w_dec_norm,
            aux_recon,
            r_norm,
            cell_mass,
            r_pos_ret,
            dead_mask_ret,
            decay,
            raw_gate,
            k_i_float,
        )

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
        edge_decay: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
        dst: torch.Tensor | None = None,
        z_full: torch.Tensor | None = None,
        raw_gate: torch.Tensor | None = None,
        k_i_float: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # 1. Numerically Stable Log-Cosh (Handles the baseline distribution perfectly)
        abs_delta = torch.abs(scaled_delta)
        stable_log_cosh = abs_delta + torch.log1p(torch.exp(-2.0 * abs_delta)) - math.log(2.0)

        # 2. The 1.5 Power Law (Forces the network to respect peaks for R2 safely)
        peak_penalty = (abs_delta + 1e-6).pow(1.5) * 0.05

        per_cell_loss = torch.sum(variance_weight * (stable_log_cosh + peak_penalty), dim=-1)
        l_recon = torch.mean(per_cell_loss) / math.sqrt(x_true.shape[-1])

        # 2. Strict Orthogonality Barrier
        gram = torch.mm(w_dec_norm, w_dec_norm.t())
        off_diag = gram * self.ortho_mask

        ortho_thresh = getattr(cfg, "ortho_overlap_threshold", 0.30)
        excess_corr = F.relu(off_diag.abs() - ortho_thresh)
        l_ortho_mean = excess_corr.pow(2).mean()

        max_corr = off_diag.max()
        l_ortho_max = F.relu(max_corr - 0.50).pow(2) * 50.0
        l_ortho = l_ortho_mean + l_ortho_max

        # SoftSAE Continuous Differentiable Sparsity Budget (Eq. 11)
        if k_i_float is not None:
            mean_k = k_i_float.mean()
            beta_soft = 5.0
            l_budget = (1.0 / beta_soft) * F.softplus(beta_soft * (mean_k - self.target_k))
        else:
            l_budget = torch.tensor(0.0, device=x_true.device)

        l_sparse = l_budget

        # 3. Residual Alignment
        if aux_recon is not None and r_norm is not None:
            res_energy = torch.clamp(r_norm.pow(2).sum(dim=-1).mean(), min=1e-4)
            aux_error = (aux_recon - r_norm).pow(2).sum(dim=-1).mean()
            l_aux = aux_error / res_energy
        else:
            l_aux = torch.tensor(0.0, device=x_true.device)

        # 5. Strict, Continuous Boundary Targets (One-Sided Hinge Loss)
        l_sharp = torch.tensor(0.0, device=x_true.device)
        if src is not None and len(src) > 0 and edge_decay is not None and raw_gate is not None:
            mean_gate = raw_gate.mean(dim=-1, keepdim=True)

            boundary_mask = (edge_decay < 0.40).float()
            internal_mask = (edge_decay > 0.60).float()

            n_boundary = torch.clamp(boundary_mask.sum(), min=1.0)
            n_internal = torch.clamp(internal_mask.sum(), min=1.0)

            # TARGETS: Boundary <= -0.35 (Repel), Internal >= +0.25 (Attract)
            # F.relu ensures loss is exactly 0 if the gate successfully crosses the target threshold.
            bnd_violation = F.relu(mean_gate + 0.35)  # Penalizes if mean_gate > -0.35
            int_violation = F.relu(0.25 - mean_gate)  # Penalizes if mean_gate < +0.25

            l_boundary = (boundary_mask * bnd_violation.pow(2)).sum() / n_boundary
            l_internal = (internal_mask * int_violation.pow(2)).sum() / n_internal

            l_sharp = l_boundary + l_internal

        base_ortho = getattr(cfg, "ortho_weight", 8.0)
        ortho_min = getattr(cfg, "ortho_min_scale", 0.50)
        global_prog = getattr(self, "current_global_progress", progress)
        current_ortho = base_ortho * (ortho_min + (1.0 - ortho_min) * global_prog)
        aux_weight = getattr(cfg, "aux_weight", 0.50)
        sharp_weight = getattr(cfg, "sharp_weight", 50)
        budget_weight = getattr(cfg, "softsae_budget_weight", 1.0)

        total_loss = (
            l_recon
            + (current_ortho * l_ortho)
            + (budget_weight * l_budget)
            + (aux_weight * l_aux)
            + (sharp_weight * l_sharp)
        )

        return (
            total_loss,
            l_recon.detach(),
            l_ortho.detach(),
            l_sparse.detach(),
            l_aux.detach(),
            l_sharp.detach(),
            l_budget.detach(),
        )

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

        if hasattr(self, "encoder_weight"):
            w_enc = F.normalize(self.encoder_weight, p=2, dim=1)
            stats["dict/encoder_norm"] = self.encoder_weight.detach().norm(2).item()
            w_enc_cpu = w_enc.detach().cpu()
            s_enc = torch.linalg.svdvals(w_enc_cpu)
            stats["dict/encoder_effective_rank"] = (
                ((s_enc.sum() ** 2) / torch.clamp((s_enc**2).sum(), min=1e-9)).item()
            )

        if hasattr(self, "decoder_weight"):
            w_dec = F.normalize(self.decoder_weight, p=2, dim=1)
            stats["dict/decoder_norm"] = self.decoder_weight.detach().norm(2).item()

            sim = torch.mm(w_dec, w_dec.t())
            off_diag_mask = ~torch.eye(w_dec.size(0), dtype=torch.bool, device=w_dec.device)
            off_diag_vals = sim.masked_select(off_diag_mask)
            if off_diag_vals.numel() > 0:
                stats["dict/max_cross_corr"] = off_diag_vals.max().item()
                stats["dict/mean_cross_corr"] = off_diag_vals.abs().mean().item()

            w_dec_cpu = w_dec.detach().cpu()
            s_dec = torch.linalg.svdvals(w_dec_cpu)
            eff_rank = (s_dec.sum() ** 2) / torch.clamp((s_dec**2).sum(), min=1e-9)
            stats["dict/effective_rank"] = eff_rank.item()
            stats["dict/svd_sigma_1"] = s_dec[0].item()
            stats["dict/svd_sigma_2"] = s_dec[1].item() if s_dec.numel() > 1 else 0.0
            stats["dict/svd_sigma_3"] = s_dec[2].item() if s_dec.numel() > 2 else 0.0

        # Untied Encoder-Decoder Feature Alignment Telemetry
        if hasattr(self, "encoder_weight") and hasattr(self, "decoder_weight"):
            w_enc = F.normalize(self.encoder_weight, p=2, dim=1)
            w_dec = F.normalize(self.decoder_weight, p=2, dim=1)
            enc_dec_diag = (w_enc * w_dec).sum(dim=-1)
            stats["dict/enc_dec_alignment_mean"] = enc_dec_diag.mean().item()
            stats["dict/enc_dec_alignment_min"] = enc_dec_diag.min().item()
            stats["dict/enc_dec_alignment_max"] = enc_dec_diag.max().item()

        dead_count = (self.steps_since_active >= self.dead_step_threshold).sum().item()
        stats["latents/dead_count"] = float(dead_count)
        stats["latents/active_pct"] = (1.0 - (dead_count / self.n_latents)) * 100.0

        if hasattr(self, "routing_mean") and hasattr(self, "routing_std"):
            stats["routing/mean_l2"] = self.routing_mean.norm(2).item()
            stats["routing/std_mean"] = self.routing_std.mean().item()
            stats["routing/std_min"] = self.routing_std.min().item()
            stats["routing/std_max"] = self.routing_std.max().item()

        # Cosine Scoring, Safe-Padé & LaPrune Exact Telemetry
        if hasattr(self, "b_scale"):
            stats["softsae/b_scale_mean"] = self.b_scale.mean().item()
            stats["softsae/b_enc_mean"] = self.b_enc.mean().item()
        if hasattr(self, "pade_gate"):
            stats["rsae/pade_p_norm"] = self.pade_gate.p_coeffs.norm(2).item()
            stats["rsae/pade_q_norm"] = self.pade_gate.q_coeffs.norm(2).item()
            stats["rsae/pade_p0"] = self.pade_gate.p_coeffs[0].item()
            stats["rsae/pade_q0"] = self.pade_gate.q_coeffs[0].item()
        if hasattr(self, "laprune_gamma"):
            gamma_scale = getattr(self, "current_gamma_progress", 1.0)
            stats["laprune/gamma_effective"] = float(self.laprune_gamma) * gamma_scale
            stats["schedules/global_progress"] = float(getattr(self, "current_global_progress", 1.0))
            stats["schedules/spatial_progress"] = float(getattr(self, "current_spatial_progress", 1.0))
            stats["schedules/squeeze_progress"] = float(getattr(self, "current_progress", 0.0))

        return stats