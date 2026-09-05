"""Spatial Graph Neural Network architecture for Libella (Top-K Hard Sparsity SAE)."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg
from .utils import ExactLaPruneFunction, SafePadeActivation

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
        self.laprune_gamma = float(getattr(cfg, "laprune_gamma", 0.95))

        # SoftSAE Dynamic Sparsity & Cosine Scoring Parameterization
        self.k_predictor = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim // 2),
            nn.RMSNorm(self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Start scale at 1.0 (e^0 = 1.0) so Padé operates in its optimal [-1, 1] domain
        self.b_scale = nn.Parameter(torch.zeros(n_metaprograms))
        self.b_enc = nn.Parameter(torch.zeros(n_metaprograms))
        self.pade_gate = SafePadeActivation(p_deg=3, q_deg=2)

        # 1. Microsoft Differential Transformer: 2x Head Dimension
        self.head_dim = max(16, self.hidden_dim // 2)
        self.q_proj = nn.Linear(self.n_latents, 2 * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_latents, 2 * self.head_dim, bias=False)

        # Node-projected 1D lambda logit for edge-adaptive noise subtraction
        self.lambda_node_proj = nn.Linear(self.n_latents, 1)
        with torch.no_grad():
            nn.init.normal_(self.lambda_node_proj.weight, std=0.1)
            nn.init.constant_(self.lambda_node_proj.bias, 0.5)

        # --- REVISED: Geometric RBF Physical Distance Scales ---
        # Use relative log-offsets centered at 0.0 to prevent 1/tau^2 gradient vanishing
        self.register_buffer("tau_1_base", torch.tensor(1.0))
        self.register_buffer("tau_2_base", torch.tensor(0.1))
        self.delta_tau_1 = nn.Parameter(torch.tensor(0.0))
        self.delta_tau_2 = nn.Parameter(torch.tensor(0.0))

        # 2. Alibaba Qwen: Element-wise, Query-Dependent Gate
        self.qwen_norm = nn.RMSNorm(self.n_latents)
        self.qwen_gate = nn.Linear(self.n_latents, self.n_latents)

        self.sign_tau = nn.Parameter(torch.tensor(-2.0))
        self.ac_delta = nn.Parameter(torch.tensor(0.0))

        self.listen_gate = nn.Linear(self.n_latents, 1)
        self.broadcast_gate = nn.Linear(self.n_latents, 1)
        
        with torch.no_grad():
            # Start at ~82% probability (+1.5 raw) to prevent early spatial freezing
            nn.init.constant_(self.listen_gate.bias, 1.5)
            nn.init.constant_(self.broadcast_gate.bias, 1.5)

        self.last_listen_prob = None
        self.last_broadcast_prob = None

        # Initialize spatial gate head cleanly
        self.spatial_gate_head = nn.Linear(self.n_latents, self.n_latents)
        with torch.no_grad():
            nn.init.normal_(self.spatial_gate_head.weight, std=0.01)
            nn.init.zeros_(self.spatial_gate_head.bias)

        self.register_buffer("last_a_ij_density", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_a_ij_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_lambda_ij_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_lambda_ij_std", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_lambda_ij_min", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_lambda_ij_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_spatial_delta_ratio", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_k_float_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_k_float_std", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_bio_scores_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_pade_out_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_edge_mag_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_delta_h_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_qwen_gate_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_spatial_context_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_r_pos_energy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_aux_recon_energy", torch.tensor(0.0), persistent=False)

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
        # Bias for the [0, 1] normalized space
        self.encoder_bias = nn.Parameter(torch.zeros(in_channels))
        # Bias for the [0, 5000] raw counts space
        self.decoder_bias = nn.Parameter(torch.zeros(in_channels))

        # Buffers & Aux State Tracking
        self.register_buffer("ortho_mask", 1.0 - torch.eye(self.n_latents, dtype=torch.float32))
        self.register_buffer("steps_since_active", torch.zeros(self.n_latents, dtype=torch.int64))
        self.register_buffer("dynamic_w_ema", torch.tensor(1.0, dtype=torch.float32))

        # EMA Buffers for Jaccard Redundancy Tracking
        self.register_buffer("coact_ema", torch.zeros((self.n_latents, self.n_latents), dtype=torch.float32))
        self.register_buffer("marginal_ema", torch.zeros(self.n_latents, dtype=torch.float32))
        self.coact_ema_initialized = False
        self.ema_momentum = 0.95 

        self.dead_step_threshold = getattr(cfg, "dead_step_threshold", 200)

        self.aux_k = getattr(cfg, "aux_k", 4)
        self.ortho_sample_size = getattr(cfg, "ortho_sample_size", min(256, self.n_latents))

    @torch.no_grad()
    def set_empirical_rbf_scales(self, tau_1: float, tau_2: float) -> None:
        """Sets empirical physical distance bases; learnable deltas remain centered at 0.0."""
        self.tau_1_base.copy_(torch.tensor(float(tau_1)))
        self.tau_2_base.copy_(torch.tensor(float(tau_2)))
        self.delta_tau_1.zero_()
        self.delta_tau_2.zero_()

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
        spatial: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,              
        torch.Tensor,              
        torch.Tensor,              
        torch.Tensor,              
        torch.Tensor,              
        torch.Tensor,             
        torch.Tensor,       
        torch.Tensor,          
        torch.Tensor | None,    
        torch.Tensor,         
        torch.Tensor | None,    
    ]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        N = x_dense.size(0)

        # 1. BIOLOGY & RAW LOGIT SCORING (Scale σ ≈ sqrt(in_channels))
        cell_mass = torch.clamp(x_dense.norm(p=2, dim=-1, keepdim=True), min=1e-5)
        x_norm = x_dense / cell_mass
        x_centered = x_norm - self.encoder_bias

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

        k_discrete = torch.clamp(k_i_float.round().long(), min=1, max=self.n_latents)
        rank = torch.argsort(torch.argsort(bio_scores, dim=-1, descending=True), dim=-1)
        hard_mask = (rank < k_discrete).float()

        if self.training:
            gamma_scale = getattr(self, "current_gamma_progress", 1.0)
            gamma_effective = max(0.05, self.laprune_gamma * gamma_scale)
            soft_mask = ExactLaPruneFunction.apply(laprune_inputs, k_i_float, gamma_effective)
            # STE: forward uses hard_mask, backward uses soft_mask
            z_canonical = F.relu(self.pade_gate(bio_scores)) * (hard_mask.detach() - soft_mask.detach() + soft_mask)
        else:
            z_canonical = F.relu(self.pade_gate(bio_scores)) * hard_mask

        if self.training:
            with torch.no_grad():
                self.last_k_float_mean.copy_(k_i_float.mean())
                self.last_k_float_std.copy_(k_i_float.std())
                self.last_bio_scores_max.copy_(bio_scores.max())
                self.last_pade_out_max.copy_(z_canonical.max())

        # 4. TOPOLOGY & ACMP WITH REINFORCED 5% GRADIENT FIREWALL
        H_0 = torch.tanh(bio_scores)
        
        # Spatial Gradient Firewall: Passes forward activations, throttles backward gradients into SAE to 5%
        rho_spatial = getattr(cfg, "spatial_gradient_scale", 0.5)
        H_0_spatial = H_0

        if len(src) > 0:
            W_bil = edge_weights.unsqueeze(1)

            # Microsoft DiffAttn: Split Projections & Scale
            Q = F.normalize(self.q_proj(H_0_spatial), p=2, dim=-1)
            K = F.normalize(self.k_proj(H_0_spatial), p=2, dim=-1)
            Q1, Q2 = Q.chunk(2, dim=-1)
            K1, K2 = K.chunk(2, dim=-1)

            scale = 1.0 / math.sqrt(self.head_dim)
            # --- REVISED: Adaptive RBF scales with log-offset parameterization ---
            tau_1 = self.tau_1_base * torch.exp(torch.clamp(self.delta_tau_1, min=-4.0, max=4.0))
            tau_2 = self.tau_2_base * torch.exp(torch.clamp(self.delta_tau_2, min=-4.0, max=4.0))

            if spatial is not None:
                dist_sq = (spatial[src] - spatial[dst]).pow(2).sum(dim=-1, keepdim=True)
                rbf_1 = torch.exp(torch.clamp(-dist_sq / tau_1, min=-30.0))
                rbf_2 = torch.exp(torch.clamp(-dist_sq / tau_2, min=-30.0))
            else:
                rbf_1 = 5.0
                rbf_2 = 30.0

            # 1. Geometric RBF-Scaled Dual Stream Similarities
            sim_1 = (Q1[dst] * K1[src]).sum(dim=-1, keepdim=True) * scale * rbf_1
            sim_2 = (Q2[dst] * K2[src]).sum(dim=-1, keepdim=True) * scale * rbf_2

            # 2. Node-Factorized Edge-Adaptive Lambda (N, 1) -> (E, 1)
            lambda_node_raw = self.lambda_node_proj(H_0_spatial)
            lambda_edge_raw = (lambda_node_raw[src] + lambda_node_raw[dst]) / 2.0
            lambda_ij = 0.20 + 0.65 * torch.sigmoid(lambda_edge_raw)

            if self.training:
                with torch.no_grad():
                    self.last_lambda_ij_mean.copy_(lambda_ij.mean())
                    self.last_lambda_ij_std.copy_(lambda_ij.std())
                    self.last_lambda_ij_min.copy_(lambda_ij.min())
                    self.last_lambda_ij_max.copy_(lambda_ij.max())

            # 3. Differential Similarity & Phase Shift
            diff_sim = sim_1 - lambda_ij * sim_2
            tau = F.softplus(self.sign_tau) + 1e-3
            edge_sign = torch.tanh(diff_sim / tau)

            # 4. Magnitude Clamping (Max ~54.6)
            edge_mag = torch.exp(torch.clamp(torch.abs(diff_sim / tau), max=4.0))

            # 5. Signed Un-normalized Adjacency & Degree Normalization Across Neighbors
            A_unnorm = edge_sign * edge_mag
            deg_abs = torch.zeros((N, 1), device=x_dense.device)
            deg_abs.index_add_(0, dst, torch.abs(A_unnorm))
            A_diff = A_unnorm / torch.clamp(deg_abs[dst], min=1e-5)
            A_ij = A_diff * W_bil

            if getattr(self, "current_epoch", 0) < 10:
                A_ij = torch.clamp(A_ij, min=-10.0, max=10.0)

            if self.training:
                with torch.no_grad():
                    self.last_a_ij_density.copy_((torch.abs(A_ij) > 0.01).float().mean())
                    self.last_a_ij_mean.copy_(A_ij.mean())
                    self.last_edge_mag_max.copy_(edge_mag.max())

            H_k = H_0_spatial
            delta = torch.sigmoid(self.ac_delta) * 0.8
            eta = 0.5  # Explicit Euler step size to prevent overshooting

            for _ in range(self.k_hops):
                raw_listen = torch.sigmoid(self.listen_gate(H_k))
                raw_broadcast = torch.sigmoid(self.broadcast_gate(H_k))
                g_listen = raw_listen * 0.95 + 0.05
                g_broadcast = raw_broadcast * 0.95 + 0.05
                W_edge = A_ij * g_broadcast[src] * g_listen[dst]

                msg = W_edge * (H_k[src] - H_k[dst])

                # ALLOCATE FRESH INSIDE THE LOOP
                laplacian_agg = torch.zeros_like(H_0_spatial)
                laplacian_agg.index_add_(0, dst, msg)

                ac_force = H_k * (1.0 - H_k.square())
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
            edge_sign = None

        # Track GNN activity telemetry
        if self.training:
            with torch.no_grad():
                d_ratio = delta_h.norm(p=2, dim=-1).mean() / (H_0_spatial.norm(p=2, dim=-1).mean() + 1e-5)
                self.last_spatial_delta_ratio.copy_(d_ratio)

        # 5. VECTORIZED CONTEXTUAL MODULATION (DeepSeek Soft Severance + Qwen Gating + Vector Alpha)
        rho_recon_to_gnn = getattr(cfg, "recon_to_gnn_scale", 0.50)
        delta_h_for_sae = delta_h
        g_qwen = torch.sigmoid(self.qwen_gate(self.qwen_norm(H_0_spatial)))
        delta_h_gated = delta_h_for_sae * g_qwen

        # Raw context from un-normalized residual
        raw_context = self.spatial_gate_head(delta_h_gated)
        spatial_prog = getattr(self, "current_spatial_progress", 1.0) if self.training else 1.0

        # Full unconstrained dynamic range: GNN can fully purge (-1.0) or reinforce (+1.0)
        alpha_vec = spatial_prog * torch.tanh(raw_context)
        z_contextual = z_canonical * (1.0 + alpha_vec)

        if self.training:
            with torch.no_grad():
                self.last_delta_h_max.copy_(delta_h.abs().max())
                self.last_qwen_gate_mean.copy_(g_qwen.mean())
                self.last_spatial_context_max.copy_(alpha_vec.abs().max())

        return z_contextual, z_canonical, bio_scores, cell_mass, alpha_vec, delta_h, src, dst, A_ij, k_i_float, edge_sign, hard_mask

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
        spatial: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,       
        torch.Tensor,       
        torch.Tensor,       
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,       
        torch.Tensor | None,
        torch.Tensor,       
        torch.Tensor,       
        torch.Tensor | None,
        torch.Tensor,       
        torch.Tensor,       
        torch.Tensor,       
        torch.Tensor,       
        torch.Tensor | None,
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
            edge_sign,
            hard_mask,
        ) = self.encode(x_dense, src, dst, edge_weights, spatial=spatial)
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=-1)

        # Direct Magnitude-Bypass Decoder Reconstruction with Depth Re-injection
        x_recon = torch.mm(z_contextual, w_dec_norm) * cell_mass + self.decoder_bias

        aux_recon = None
        r_norm = None
        r_pos_ret = None
        dead_mask_ret = torch.zeros(self.n_latents, dtype=torch.bool, device=x_dense.device)

        if self.training:
            with torch.no_grad():
                # Re-use precomputed hard support directly to avoid redundant argsort passes
                active_in_batch = (hard_mask > 0).any(dim=0)

                self.steps_since_active.add_(1)
                self.steps_since_active.masked_fill_(active_in_batch, 0)
                dead_mask_ret = (self.steps_since_active >= self.dead_step_threshold).detach()

                x_norm = F.normalize(x_dense.detach(), p=2, dim=-1)
                x_recon_norm = F.normalize(F.relu(x_recon.detach()) + 1e-6, p=2, dim=-1)
                r_pos = F.relu(x_norm - x_recon_norm)
                r_pos_ret = r_pos
                r_norm = F.normalize(r_pos + 1e-6, p=2, dim=-1)
                residual_energy = r_pos.norm(p=2, dim=-1).mean()
                self.last_r_pos_energy.copy_(residual_energy)
                self.last_aux_recon_energy.zero_()

            if dead_mask_ret.any() and residual_energy > getattr(cfg, "aux_min_residual_energy", 0.05):
                with torch.no_grad():
                    dead_indices = torch.nonzero(dead_mask_ret).squeeze(-1)
                    num_dead = dead_indices.numel()
                    k_aux = min(max(getattr(cfg, "aux_min_k", 2), self.aux_k), num_dead)

                    w_dead = w_dec_norm[dead_indices].detach()
                    aux_sim = torch.mm(r_norm, w_dead.t())
                    aux_scores = torch.exp(self.b_scale[dead_indices]) * aux_sim + self.b_enc[dead_indices]
                    topk_res = torch.topk(aux_scores, k=k_aux, dim=-1)

                    z_aux_weights = F.relu(topk_res.values)
                    z_aux = torch.zeros_like(aux_scores).scatter_(-1, topk_res.indices, z_aux_weights)
                    aux_recon = torch.mm(z_aux, w_dead)
                    self.last_aux_recon_energy.copy_(aux_recon.norm(p=2, dim=-1).mean())

                    del dead_indices, w_dead, aux_sim, aux_scores, topk_res, z_aux_weights, z_aux

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
            edge_sign,
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
        train_mask: torch.Tensor | None = None,
        ghost_logits: torch.Tensor | None = None,
        ghost_weights: torch.Tensor | None = None,
        progress: float = 1.0,
        src: torch.Tensor | None = None,
        dst: torch.Tensor | None = None,
        z_full: torch.Tensor | None = None,
        A_ij: torch.Tensor | None = None,
        x_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Variance-Weighted Cell-Averaged Asymmetric Log-Cosh Loss
        is_non_zero = (x_true > 0).detach()
        mask = train_mask if train_mask is not None else torch.ones((x_true.size(0), 1), device=x_true.device)
        valid_nodes = torch.clamp(mask.sum(), min=1.0)
        num_pos = torch.clamp((is_non_zero.float() * mask).sum(), min=1.0)
        num_zeros = ((x_true == 0).float() * mask).sum().detach()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            with torch.no_grad():
                self.dynamic_w_ema.lerp_(
                    current_dynamic_w, weight=getattr(cfg, "dynamic_w_ema_weight", 0.10)
                )

        w_mat = torch.where(is_non_zero, self.dynamic_w_ema.detach(), 1.0)
        variance_weight = (w_mat * (1.0 + torch.log1p(x_true))).detach()
        variance_weight = (variance_weight / torch.clamp(variance_weight.mean(), min=1e-5)).detach()

        raw_delta = recon_x - x_true
        asym_penalty = getattr(cfg, "asym_penalty_weight", 0.50)
        asym_factor = (1.0 + (is_non_zero.float() * asym_penalty) * (raw_delta.detach() < 0).float()).detach()

        delta_clamp = getattr(cfg, "delta_clamp", 30.0)
        scaled_delta = torch.clamp(raw_delta * asym_factor, min=-delta_clamp, max=delta_clamp)

        abs_delta = torch.abs(scaled_delta)
        stable_log_cosh = abs_delta + torch.log1p(torch.exp(-2.0 * abs_delta)) - math.log(2.0)
        peak_penalty = (abs_delta + 1e-6).pow(1.5) * 0.05

        per_cell_loss = torch.sum(variance_weight * (stable_log_cosh + peak_penalty), dim=-1, keepdim=True)
        l_recon = torch.sum(per_cell_loss * mask) / (valid_nodes * math.sqrt(x_true.shape[-1]))

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

        # JACCARD-SCALED REDUNDANCY PENALTY
        with torch.no_grad():
            # 1. Compute Exact Hard Support (Ignore soft-routed noise)
            if routed_scores is not None and k_i_float is not None:
                k_discrete = torch.clamp(k_i_float.round().long(), min=1, max=self.n_latents)
                rank = torch.argsort(torch.argsort(routed_scores, dim=-1, descending=True), dim=-1)
                hard_mask = (rank < k_discrete).float()
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

        # 4. Phase-Gated Asymmetric Charbonnier K-Budget Loss
        if k_i_float is not None:
            mean_k = (k_i_float * mask).sum() / valid_nodes
            k_err = mean_k - self.target_k
            
            # 1. Asymmetric Multiplier
            asym_factor = torch.where(k_err > 0, 2.0, 1.0)
            
            # 2. Charbonnier / Smoothed L1 Penalty
            eps = 0.1
            smooth_l1 = torch.sqrt(k_err.pow(2) + eps**2) - eps
            
            # 3. Dynamic Phase Pressure
            squeeze = getattr(self, "current_progress", 1.0)
            pressure = 1.0 + 9.0 * squeeze
            
            l_budget = (asym_factor * smooth_l1 * pressure) / self.target_k
        else:
            l_budget = torch.tensor(0.0, device=x_true.device)

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
        gate_prog = getattr(self, "current_gate_progress", 1.0) if self.training else 1.0
        gate_weight = getattr(cfg, "gate_sparsity_weight", 0.05) * gate_prog

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
        edge_sign: torch.Tensor | None,
        src: torch.Tensor,
        dst: torch.Tensor,
        x_norm: torch.Tensor,
        edge_mask_float: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if src.numel() == 0 or edge_sign is None or edge_sign.numel() == 0:
            return delta_h.new_zeros(())

        N = x_norm.size(0)

        with torch.no_grad():
            bio_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1, keepdim=True)

            ones_sim = torch.ones_like(bio_sim)
            degree = torch.zeros((N, 1), device=x_norm.device).index_add_(0, dst, ones_sim)
            degree_clamped = degree.clamp(min=1)

            node_bio_sum = torch.zeros((N, 1), device=x_norm.device).index_add_(0, dst, bio_sim)
            local_mean = (node_bio_sum / degree_clamped)[dst]

            var_components = (bio_sim - local_mean).square()
            node_var_sum = torch.zeros((N, 1), device=x_norm.device).index_add_(0, dst, var_components)

            local_std = torch.clamp(torch.sqrt(node_var_sum / degree_clamped)[dst], min=0.05)

            z_score = (bio_sim - local_mean) / local_std
            y_target = torch.tanh(z_score)

            variance_mask = (node_var_sum / degree_clamped)[dst] > 1e-4

        # 1. Edge Anchor (Directly trains Q/K attention matrices)
        loss_edges = F.mse_loss(edge_sign, y_target, reduction="none")

        # 2. PDE Smoothing Supervision
        d_norm = F.normalize(delta_h, p=2, dim=-1, eps=1e-6)
        spatial_sim = (d_norm[src] * d_norm[dst]).sum(dim=-1, keepdim=True)
        loss_pde = F.huber_loss(spatial_sim, y_target, delta=0.5, reduction="none")

        total_raw_loss = loss_edges + loss_pde
        mask = variance_mask.float() if edge_mask_float is None else variance_mask.float() * edge_mask_float
        return (total_raw_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

    
