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
    Inputs are softly bounded to [-1, 1] to prevent runaway polynomial oscillation (Runge's Phenomenon).
    """
    def __init__(self, p_deg: int = 3, q_deg: int = 2) -> None:
        super().__init__()
        # Initialize near Identity function (f(x) = x)
        p_init = torch.zeros(p_deg + 1)
        p_init[1] = 1.0  
        
        self.p_coeffs = nn.Parameter(p_init + torch.randn(p_deg + 1) * 0.01)
        self.q_coeffs = nn.Parameter(torch.abs(torch.randn(q_deg) * 0.01))
        
        # RSAE C_in scale to map raw scores into the safe [-1, 1] polynomial design space
        self.c_in = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Softly bind inputs into [-1, 1] to prevent x^3 from exploding/inverting
        # This forces Padé to behave purely as a smooth shape inside its stable domain
        x_safe = torch.tanh(x / F.softplus(self.c_in))
        
        p_val = self.p_coeffs[0]
        for i in range(1, len(self.p_coeffs)):
            p_val = p_val + self.p_coeffs[i] * (x_safe ** i)
            
        q_val = 1.0
        abs_x = torch.abs(x_safe)
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
        self.target_k = float(getattr(cfg, "target_k", 38.0))
        # Limit dynamic budget bounds to [0.5x, 1.5x] of K goal
        self.min_k = 0.5 * self.target_k
        self.max_k = 1.5 * self.target_k
        self.laprune_gamma = float(getattr(cfg, "laprune_gamma", 0.90))

        # SoftSAE Dynamic Sparsity & Cosine Scoring Parameterization
        self.k_predictor = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Start scale at 1.0 (e^0 = 1.0) so Padé operates in its optimal [-1, 1] domain
        self.b_scale = nn.Parameter(torch.zeros(n_metaprograms))
        self.b_enc = nn.Parameter(torch.zeros(n_metaprograms))
        self.pade_gate = SafePadeActivation(p_deg=3, q_deg=2)

        # --- NATIVE ARCHITECTURE (Unified on n_latents) ---
        self.head_dim = max(16, self.hidden_dim // 2)
        self.q_proj = nn.Linear(self.n_latents, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_latents, self.head_dim, bias=False)
        
        self.sign_tau = nn.Parameter(torch.tensor(-2.0))
        self.acmp_beta = nn.Parameter(torch.tensor(0.0))
        self.ac_delta = nn.Parameter(torch.tensor(0.0))
        
        self.listen_gate = nn.Linear(self.n_latents, 1)
        self.broadcast_gate = nn.Linear(self.n_latents, 1)
        
        with torch.no_grad():
            # Start at 50/50 probability (0.0 raw) to maximize initial gradient flow
            nn.init.constant_(self.acmp_beta, 0.0)
            nn.init.constant_(self.listen_gate.bias, 0.0)
            nn.init.constant_(self.broadcast_gate.bias, 0.0)

        self.last_listen_prob = None
        self.last_broadcast_prob = None

        # Spatial Gate Head operates on latent delta residual
        self.spatial_gate_head = nn.Linear(self.n_latents, self.n_latents)
        with torch.no_grad():
            nn.init.normal_(self.spatial_gate_head.weight, std=0.02)
            nn.init.zeros_(self.spatial_gate_head.bias)

        # Learnable spatial gain knob (sigmoid(0.0) * 0.10 = 0.05 baseline modulation)
        self.spatial_gain = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("last_a_ij_density", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_a_ij_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_spatial_delta_ratio", torch.tensor(0.0), persistent=False)

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

        # EMA Buffers for Jaccard Redundancy Tracking
        self.register_buffer("coact_ema", torch.zeros((self.n_latents, self.n_latents), dtype=torch.float32))
        self.register_buffer("marginal_ema", torch.zeros(self.n_latents, dtype=torch.float32))
        self.coact_ema_initialized = False
        self.ema_momentum = 0.95  # Remembers roughly the last 20 batches

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
        torch.Tensor | None,
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

        # 1. BIOLOGY & RAW LOGIT SCORING (Scale σ ≈ sqrt(in_channels))
        cell_mass = torch.clamp(x_dense.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        x_norm = x_dense / cell_mass
        x_centered = x_norm - self.decoder_bias

        w_enc_dir = F.normalize(self.encoder_weight, p=2, dim=1)
        cosine_sim = torch.mm(x_centered, w_enc_dir.t())
        bio_scores = torch.exp(self.b_scale) * cosine_sim + self.b_enc

        # 2. DYNAMIC K-BUDGET ESTIMATION [0.5x -> 1.5x of target_k]
        k_ratio = self.k_predictor(x_centered)
        k_i_float = self.min_k + k_ratio * (self.max_k - self.min_k)

        # 3. EXACT-K ROUTING: Standardize Inputs for Solver Stability & Safe-Padé Amplitude
        bio_mean = bio_scores.mean(dim=-1, keepdim=True)
        bio_std = bio_scores.std(dim=-1, keepdim=True).clamp(min=1e-4)
        laprune_inputs = (bio_scores - bio_mean) / bio_std

        if self.training:
            gamma_scale = getattr(self, "current_gamma_progress", 1.0)
            gamma_effective = max(0.05, self.laprune_gamma * gamma_scale)
            soft_mask = ExactLaPruneFunction.apply(laprune_inputs, k_i_float, gamma_effective)
            # Safe-Padé rational pre-conditioner with ReLU positivity guard
            z_canonical = F.relu(self.pade_gate(bio_scores)) * soft_mask
        else:
            k_discrete = torch.clamp(k_i_float.round().long(), min=1, max=self.n_latents)
            # Double-argsort guarantees EXACTLY K non-zeros by breaking ties deterministically
            rank = torch.argsort(torch.argsort(bio_scores, dim=-1, descending=True), dim=-1)
            hard_mask = (rank < k_discrete).float()
            z_canonical = F.relu(self.pade_gate(bio_scores)) * hard_mask

        # 4. TOPOLOGY & ACMP WITH REINFORCED 5% GRADIENT FIREWALL
        H_0 = torch.tanh(bio_scores)
        
        # Spatial Gradient Firewall: Passes forward activations, throttles backward gradients into SAE to 5%
        rho_spatial = getattr(cfg, "spatial_gradient_scale", 0.2)
        H_0_spatial = H_0.detach() + rho_spatial * (H_0 - H_0.detach())

        if len(src) > 0:
            W_bil = edge_weights.unsqueeze(1)

            # Cosine Attention directly on firewall-shielded latents
            q = F.normalize(self.q_proj(H_0_spatial), p=2, dim=-1)
            k = F.normalize(self.k_proj(H_0_spatial), p=2, dim=-1)
            sim = (q[dst] * k[src]).sum(dim=-1, keepdim=True)

            # Reconnect tau for learned temperature sharpness
            tau = F.softplus(self.sign_tau) + 1e-3
            a_ij_raw = torch.sigmoid(sim / tau)
            beta = torch.sigmoid(self.acmp_beta)
            edge_sign = torch.tanh((a_ij_raw - beta) * 4.0)
            edge_mag = torch.exp(torch.clamp(torch.abs(sim / tau), max=20.0))

            mag_sum = torch.zeros((N, 1), device=x_dense.device)
            mag_sum.index_add_(0, dst, edge_mag)
            normalized_mag = edge_mag / torch.clamp(mag_sum[dst], min=1e-5)
            A_ij = edge_sign * normalized_mag * W_bil

            if self.training:
                with torch.no_grad():
                    self.last_a_ij_density.copy_((torch.abs(A_ij) > 0.01).float().mean())
                    self.last_a_ij_mean.copy_(A_ij.mean())

            H_k = H_0_spatial
            delta = torch.sigmoid(self.ac_delta) * 0.45
            eta = 0.5  # Explicit Euler step size to prevent overshooting

            for _ in range(self.k_hops):
                # Strict hard cap at 0.40 to prevent semantic oversmoothing and dictionary collapse
                g_listen = 0.40 * torch.sigmoid(self.listen_gate(H_k))
                g_broadcast = 0.40 * torch.sigmoid(self.broadcast_gate(H_k))
                W_edge = A_ij * g_broadcast[src] * g_listen[dst]

                msg = W_edge * (H_k[src] - H_k[dst])
                laplacian_agg = torch.zeros_like(H_k)
                laplacian_agg.index_add_(0, dst, msg)

                ac_force = H_k * (1.0 - H_k**2)
                H_k = torch.tanh(H_k + eta * laplacian_agg + delta * ac_force)

            # Track gate probabilities for L1 sparsity regularizer
            self.last_listen_prob = g_listen
            self.last_broadcast_prob = g_broadcast

            # THE GRAPH RESIDUAL: Pure neighborhood deviation from firewall state
            delta_h = H_k - H_0_spatial
        else:
            self.last_listen_prob = None
            self.last_broadcast_prob = None
            delta_h = torch.zeros_like(H_0_spatial)
            A_ij = None

        # Track GNN activity telemetry
        if self.training:
            with torch.no_grad():
                d_ratio = delta_h.norm(p=2, dim=-1).mean() / (H_0_spatial.norm(p=2, dim=-1).mean() + 1e-5)
                self.last_spatial_delta_ratio.copy_(d_ratio)

        # 5. ZERO-CENTERED CONTEXTUAL MODULATION (Center BEFORE Tanh)
        raw_context = self.spatial_gate_head(delta_h)
        centered_context = raw_context - raw_context.mean(dim=-1, keepdim=True)
        spatial_context = torch.tanh(centered_context)

        spatial_prog = getattr(self, "current_spatial_progress", 1.0) if self.training else 1.0
        # Cap modulation at 15% maximum
        alpha = torch.sigmoid(self.spatial_gain) * 0.15 * spatial_prog

        z_contextual = z_canonical * (1.0 + alpha * spatial_context)

        return z_contextual, z_canonical, bio_scores, cell_mass, spatial_context, delta_h, src, dst, A_ij, k_i_float

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

        # 1. Update BOTH dictionaries simultaneously
        self.decoder_weight.data[target_dead_ids] = new_atoms
        self.encoder_weight.data[target_dead_ids] = new_atoms.clone()

        # 2. Reset score scale and bias for revived latents (Scale = 0.0 -> e^0 = 1.0)
        self.b_scale.data[target_dead_ids] = 0.0
        self.b_enc.data[target_dead_ids] = 0.0

        self.steps_since_active[target_dead_ids] = 0

        # 3. Reset optimizer moments
        if optimizer is not None:
            params_to_reset = [self.decoder_weight, self.encoder_weight, self.b_scale, self.b_enc]
            for param in params_to_reset:
                state = optimizer.state.get(param, None)
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
        (
            z_contextual,
            z_canonical,
            bio_scores,
            cell_mass,
            spatial_context,
            delta_h,
            src,
            dst,
            A_ij,
            k_i_float,
        ) = self.encode(x_dense, src, dst, edge_weights)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=-1)

        # Direct Magnitude-Bypass Decoder Reconstruction
        x_recon_centered = torch.mm(z_contextual, w_dec_norm)
        x_recon = (x_recon_centered + self.decoder_bias) * cell_mass

        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(self.n_latents, dtype=torch.bool, device=x_dense.device)

        if self.training:
            with torch.no_grad():
                # Use top-K indices from raw scores to determine genuine usage
                k_track = k_i_float.round().clamp(1, self.n_latents).long()
                sorted_scores, _ = torch.sort(bio_scores, dim=-1, descending=True)
                hard_thresh = torch.gather(sorted_scores, dim=1, index=k_track - 1)
                active_in_batch = (bio_scores >= hard_thresh).any(dim=0)

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
            z_contextual,
            w_dec_norm,
            aux_recon,
            r_norm,
            cell_mass,
            r_pos_ret,
            dead_mask_ret,
            spatial_context,
            A_ij,
            k_i_float,
            delta_h,
            z_canonical,
            bio_scores,
        )

    def calc_loss(
        self,
        recon_x: torch.Tensor,
        x_true: torch.Tensor,
        z: torch.Tensor,
        w_dec_norm: torch.Tensor,
        routed_scores: torch.Tensor | None = None,
        k_i_float: torch.Tensor | None = None,
        aux_recon: torch.Tensor | None = None,
        r_norm: torch.Tensor | None = None,
        ghost_logits: torch.Tensor | None = None,
        ghost_weights: torch.Tensor | None = None,
        progress: float = 1.0,
        spatial_shift: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
        dst: torch.Tensor | None = None,
        z_full: torch.Tensor | None = None,
        A_ij: torch.Tensor | None = None,
        x_full: torch.Tensor | None = None,
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

        abs_delta = torch.abs(scaled_delta)
        stable_log_cosh = abs_delta + torch.log1p(torch.exp(-2.0 * abs_delta)) - math.log(2.0)
        peak_penalty = (abs_delta + 1e-6).pow(1.5) * 0.05

        per_cell_loss = torch.sum(variance_weight * (stable_log_cosh + peak_penalty), dim=-1)
        l_recon = torch.mean(per_cell_loss) / math.sqrt(x_true.shape[-1])

        # 2. DUAL-HINGE ALIGNMENT LOSS (500x hard boundary wall)
        w_enc_norm = F.normalize(self.encoder_weight, p=2, dim=-1)
        align_cos = (w_enc_norm * w_dec_norm).sum(dim=-1)
        # Linear hinge below 0.85 (gentle tether) + Impenetrable cliff below 0.75
        l_align_linear = F.relu(0.85 - align_cos)
        l_align_hard = F.relu(0.75 - align_cos).pow(2) * 500.0
        l_align = torch.mean(l_align_linear + l_align_hard)

        # 3. L1 GATE SPARSITY PENALTY (Forces selective neighborhood listening)
        if self.last_listen_prob is not None and self.last_broadcast_prob is not None:
            l_gate_sparse = self.last_listen_prob.mean() + self.last_broadcast_prob.mean()
        else:
            l_gate_sparse = torch.tensor(0.0, device=x_true.device)

        # ---------------------------------------------------------
        # JACCARD-SCALED REDUNDANCY PENALTY
        # ---------------------------------------------------------
        with torch.no_grad():
            # 1. Compute Exact Hard Support (Ignore soft-routed noise)
            if routed_scores is not None and k_i_float is not None:
                k_discrete = torch.clamp(k_i_float.round().long(), min=1, max=self.n_latents)
                sorted_scores, _ = torch.sort(routed_scores, dim=-1, descending=True)
                hard_thresh = torch.gather(sorted_scores, dim=1, index=k_discrete - 1)
                hard_mask = (routed_scores >= hard_thresh).float()
            else:
                hard_mask = (z > 1e-4).float()

            # 2. Batch Co-activation & Marginals
            batch_size = max(1.0, float(hard_mask.size(0)))
            batch_coact = torch.mm(hard_mask.t(), hard_mask) / batch_size
            batch_marginals = hard_mask.sum(dim=0) / batch_size

            # 3. EMA Updates with Warm Start
            if self.training:
                if not self.coact_ema_initialized:
                    self.coact_ema.copy_(batch_coact)
                    self.marginal_ema.copy_(batch_marginals)
                    self.coact_ema_initialized = True
                else:
                    self.coact_ema.lerp_(batch_coact, 1.0 - self.ema_momentum)
                    self.marginal_ema.lerp_(batch_marginals, 1.0 - self.ema_momentum)

            # 4. Compute Jaccard Overlap: P(A & B) / (P(A) + P(B) - P(A & B))
            m_i = self.marginal_ema.unsqueeze(1)  # (N, 1)
            m_j = self.marginal_ema.unsqueeze(0)  # (1, N)
            # Add eps to prevent division by zero
            union_prob = torch.clamp(m_i + m_j - self.coact_ema, min=1e-5)
            jaccard_ema = self.coact_ema / union_prob

        # 5. Decoder Weight Similarity (POSITIVE ONLY)
        gram_dec = torch.mm(w_dec_norm, w_dec_norm.t())
        positive_sim = F.relu(gram_dec * self.ortho_mask)
        
        ortho_thresh = getattr(cfg, "ortho_overlap_threshold", 0.30)
        excess_sim = F.relu(positive_sim - ortho_thresh)
        
        # 6. Apply Jaccard Scaling
        redundancy_matrix = jaccard_ema * excess_sim.pow(2)
        l_ortho_mean = redundancy_matrix.sum() / max(1.0, self.ortho_mask.sum())
        
        # 7. Clone Guard (Emergency brake for literal twins)
        max_corr = positive_sim.max()
        l_ortho_max = F.relu(max_corr - 0.60).pow(2) * 20.0

        l_ortho = l_ortho_mean + l_ortho_max

        # 4. Batch-Mean Quadratic K-Budget Loss (Penalizes batch-level deviation, allows per-cell variance)
        if k_i_float is not None:
            mean_k = k_i_float.mean()
            l_budget = (mean_k - self.target_k).pow(2) / self.target_k
        else:
            l_budget = torch.tensor(0.0, device=x_true.device)

        l_sparse = l_budget

        # 5. Residual Alignment
        if aux_recon is not None and r_norm is not None:
            res_energy = torch.clamp(r_norm.pow(2).sum(dim=-1).mean(), min=1e-4)
            aux_error = (aux_recon - r_norm).pow(2).sum(dim=-1).mean()
            l_aux = aux_error / res_energy
        else:
            l_aux = torch.tensor(0.0, device=x_true.device)

        # Loss Compilation
        base_ortho = getattr(cfg, "ortho_weight", 15.0)
        ortho_min = getattr(cfg, "ortho_min_scale", 0.50)
        global_prog = getattr(self, "current_global_progress", progress)
        current_ortho = base_ortho * (ortho_min + (1.0 - ortho_min) * global_prog)

        aux_weight = getattr(cfg, "aux_weight", 0.50)
        budget_weight = getattr(cfg, "softsae_budget_weight", 1.0)
        align_weight = getattr(cfg, "enc_dec_align_weight", 15.0)
        gate_weight = getattr(cfg, "gate_sparsity_weight", 0.05)

        total_loss = (
            l_recon
            + (current_ortho * l_ortho)
            + (budget_weight * l_budget)
            + (aux_weight * l_aux)
            + (align_weight * l_align)
            + (gate_weight * l_gate_sparse)
        )

        return (
            total_loss,
            l_recon.detach(),
            l_ortho.detach(),
            l_budget.detach(),
            l_aux.detach(),
            l_align.detach(),
            l_gate_sparse.detach(),
        )

    def calc_spatial_loss(
        self,
        delta_h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        raw_gate: torch.Tensor | None,
    ) -> torch.Tensor:
        """Supervises the neighborhood-induced delta embeddings using gate sign routing."""
        d_norm = F.normalize(delta_h, p=2, dim=-1, eps=1e-6)

        if len(src) > 0 and raw_gate is not None:
            sim = (d_norm[src] * d_norm[dst]).sum(dim=-1, keepdim=True)
            gate_sign = torch.sign(raw_gate.detach())

            mask_homo = (gate_sign > 0.0).float()
            mask_hetero = (gate_sign < 0.0).float()

            pull_loss = mask_homo * F.relu(0.60 - sim)
            push_loss = mask_hetero * F.relu(sim - 0.10)

            l_spatial = (pull_loss.sum() + push_loss.sum()) / max(1.0, float(len(src)))
        else:
            l_spatial = torch.tensor(0.0, device=delta_h.device)

        return l_spatial

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

        if hasattr(self, "coact_ema") and hasattr(self, "marginal_ema"):
            off_diag_coact = self.coact_ema * self.ortho_mask
            stats["dict/coact_ema_mean"] = off_diag_coact.mean().item()
            stats["dict/coact_ema_max"] = off_diag_coact.max().item()

            m_i = self.marginal_ema.unsqueeze(1)
            m_j = self.marginal_ema.unsqueeze(0)
            union_p = torch.clamp(m_i + m_j - self.coact_ema, min=1e-5)
            jaccard_mat = (self.coact_ema / union_p) * self.ortho_mask
            stats["dict/jaccard_ema_mean"] = jaccard_mat.mean().item()
            stats["dict/jaccard_ema_max"] = jaccard_mat.max().item()

        # Untied Encoder-Decoder Feature Alignment Telemetry
        if hasattr(self, "encoder_weight") and hasattr(self, "decoder_weight"):
            w_enc = F.normalize(self.encoder_weight, p=2, dim=1)
            w_dec = F.normalize(self.decoder_weight, p=2, dim=1)
            align_cos = (w_enc * w_dec).sum(dim=-1)
            
            stats["dict/alignment_mean"] = align_cos.mean().item()
            stats["dict/alignment_min"] = align_cos.min().item()
            
            quantiles = torch.quantile(align_cos, torch.tensor([0.10, 0.90], device=align_cos.device))
            stats["dict/alignment_p10"] = quantiles[0].item()
            stats["dict/alignment_p90"] = quantiles[1].item()

        dead_count = (self.steps_since_active >= self.dead_step_threshold).sum().item()
        stats["latents/dead_count"] = float(dead_count)
        stats["latents/active_pct"] = (1.0 - (dead_count / self.n_latents)) * 100.0

        # SignGT, ACMP, Spatial Gain, Delta Ratio, and CSNN Telemetry
        if hasattr(self, "last_spatial_delta_ratio"):
            stats["spatial/delta_ratio"] = self.last_spatial_delta_ratio.item()
        if hasattr(self, "spatial_gain"):
            stats["spatial/gain_raw"] = self.spatial_gain.item()
            stats["spatial/effective_gain"] = (torch.sigmoid(self.spatial_gain) * 0.10).item()
        if hasattr(self, "sign_tau"):
            stats["sign_gt/tau_effective"] = (F.softplus(self.sign_tau) + 1e-3).item()
            stats["sign_gt/tau_raw"] = self.sign_tau.item()
        if hasattr(self, "acmp_beta"):
            stats["acmp/beta_effective"] = torch.sigmoid(self.acmp_beta).item()
            stats["acmp/beta_raw"] = self.acmp_beta.item()
        if hasattr(self, "ac_delta"):
            stats["acmp/delta_effective"] = (torch.sigmoid(self.ac_delta) * 0.45).item()
            stats["acmp/delta_raw"] = self.ac_delta.item()
        if hasattr(self, "listen_gate"):
            gate_w_listen = self.listen_gate.weight.detach()
            gate_w_broad = self.broadcast_gate.weight.detach()
            stats["csnn/listen_gate_norm"] = gate_w_listen.norm(2).item()
            stats["csnn/broadcast_gate_norm"] = gate_w_broad.norm(2).item()
        if hasattr(self, "last_a_ij_mean"):
            stats["graph/a_ij_mean"] = self.last_a_ij_mean.item()
            stats["graph/a_ij_active_density"] = self.last_a_ij_density.item()

        # Cosine Scoring, Safe-Padé & LaPrune Exact Telemetry
        if hasattr(self, "b_scale"):
            stats["softsae/b_scale_mean"] = self.b_scale.mean().item()
            stats["softsae/b_enc_mean"] = self.b_enc.mean().item()
        if hasattr(self, "pade_gate"):
            stats["rsae/pade_p_norm"] = self.pade_gate.p_coeffs.norm(2).item()
            stats["rsae/pade_q_norm"] = self.pade_gate.q_coeffs.norm(2).item()
            stats["rsae/pade_p0"] = self.pade_gate.p_coeffs[0].item()
            stats["rsae/pade_q0"] = self.pade_gate.q_coeffs[0].item()

            # Dynamic Padé Monotonicity and Rank Inversion Audit (1D Grid over [0, 10])
            with torch.enable_grad():
                s_grid = torch.linspace(
                    0.0, 10.0, 500, device=self.pade_gate.p_coeffs.device, requires_grad=True
                )
                y_grid = self.pade_gate(s_grid)
                dy_ds = torch.autograd.grad(y_grid.sum(), s_grid)[0]

                neg_mask = dy_ds < 0.0
                stats["pade/negative_derivative_pct"] = neg_mask.float().mean().item() * 100.0
                stats["pade/rank_inversion_pct"] = neg_mask.float().mean().item() * 100.0
                stats["pade/derivative_min"] = dy_ds.min().item()
                stats["pade/output_min"] = y_grid.min().item()
                stats["pade/output_max"] = y_grid.max().item()

        if hasattr(self, "laprune_gamma"):
            gamma_scale = getattr(self, "current_gamma_progress", 1.0)
            stats["laprune/gamma_effective"] = float(self.laprune_gamma) * gamma_scale
            stats["schedules/global_progress"] = float(getattr(self, "current_global_progress", 1.0))
            stats["schedules/spatial_progress"] = float(getattr(self, "current_spatial_progress", 1.0))
            stats["schedules/squeeze_progress"] = float(getattr(self, "current_progress", 0.0))
            stats["schedules/gamma_progress"] = float(gamma_scale)

        return stats