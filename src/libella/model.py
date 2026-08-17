"""Spatial Graph Neural Network architecture for Libella."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect

from .config import cfg
from .utils import scatter_softmax, PhaseTracker


class LibellaGNN(nn.Module):
    """Core Libella Spatial GNN architecture."""
    def __init__(
        self, 
        in_channels: int, 
        n_metaprograms: int, 
        init_components: np.ndarray | None = None
    ) -> None: 
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.k_hops = cfg.k_hops
        self.n_latents = n_metaprograms
        self.in_channels = in_channels

        # --- Context & Identity Encoders ---
        self.ctx_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.lin_appnp = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.id_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(self.hidden_dim)
        )

        # --- Cross-Attention Projections ---
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        self.context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.sp_norm = nn.LayerNorm(self.hidden_dim)

        # 1. Spatially Enriched Intensity Stream (Imputes dropouts via graph context)
        self.mag_enc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_latents),
            nn.Softplus(beta=1.0)
        )

        # 2. Spatial Context Gating Stream with Learnable Gradient Gain
        self.gate_spatial_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_latents)
        )
        self.spatial_gain = nn.Parameter(torch.tensor(cfg.spatial_gain_init))

        

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

        # --- 5. Buffers & Hyperparameters ---
        self.register_buffer('ortho_mask', 1.0 - torch.eye(self.n_latents, dtype=torch.float32))
        self.register_buffer('steps_since_active', torch.zeros(self.n_latents, dtype=torch.int64))
        self.dead_step_threshold = getattr(cfg, 'dead_step_threshold', 100)
        self.aux_k = getattr(cfg, 'aux_k', min(32, max(4, self.n_latents // 16)))
        self.ortho_sample_size = getattr(cfg, 'ortho_sample_size', min(256, self.n_latents))

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.alpha_proj = nn.Linear(self.hidden_dim, 1)
        self.register_buffer('dynamic_w_ema', torch.tensor(1.0, dtype=torch.float32))

        # --- Spatial GAT Components ---
        self.gat_w_src = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gat_w_dst = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gat_w_edge = nn.Linear(1, self.hidden_dim, bias=True)
        self.gat_a = nn.Linear(self.hidden_dim, 1, bias=False)
        self.att_temp = nn.Parameter(torch.tensor(float(getattr(cfg, 'att_temp', 1.0))))
        self.mp_update = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(
        self, 
        x_dense: torch.Tensor, 
        src: torch.Tensor, 
        dst: torch.Tensor, 
        edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        # 1. Depth Disentanglement
        cell_mass = torch.clamp(x_dense.sum(dim=-1, keepdim=True), min=1e-5)
        x_norm = F.normalize(x_dense, p=2, dim=-1)

        h_id = self.id_enc(x_norm)
        h_0 = self.lin_appnp(self.ctx_enc(x_norm))
        N = h_0.size(0)

        # 2. Smooth Bilateral Edge Decay (Preserves sparse cell graph connectivity)
        if len(src) > 0:
            with torch.no_grad():
                cos_sim = (x_norm[src] * x_norm[dst]).sum(dim=-1)

            decay = torch.sigmoid((cos_sim - cfg.edge_sim_threshold) * cfg.edge_decay_slope)
        else:
            decay = torch.ones_like(edge_weights)
            
        W_bil = edge_weights * decay

        # 3. K-Hop Spatial Message Passing Loop
        alpha = torch.sigmoid(self.alpha_proj(h_0)) * cfg.appnp_alpha_scale + cfg.appnp_alpha_offset
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
                tau = torch.clamp(F.softplus(self.att_temp) + cfg.att_temp_min, min=cfg.att_temp_min, max=cfg.att_temp_max)
                alpha_att = scatter_softmax(e_raw / tau, dst, N)
                
                msg = h_ctx[src] * alpha_att.unsqueeze(1)
                out.index_add_(0, dst, msg)
                
            agg = F.silu(self.mp_update(out))
            h_ctx = agg * inv_alpha + h_0_scaled

        # 4. Identity-Context Cross-Attention Bottleneck
        Q = self.q_proj(h_id)
        K = self.k_proj(h_ctx)
        V = self.v_proj(h_ctx)

        idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x_dense.device.type == 'mps' else torch.int64)
        self_loops = torch.arange(N, dtype=idx_dtype, device=x_dense.device)
        
        src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
        dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops
            
        cross_scores = (Q[dst_with_self] * K[src_with_self]).sum(dim=-1) / (self.hidden_dim ** 0.5)
        cross_att = scatter_softmax(cross_scores, dst_with_self, N)
        
        pulled_msg = (V[src_with_self] * cross_att.unsqueeze(1)).contiguous()
        ctx_pulled = torch.zeros_like(Q)
        ctx_pulled.index_add_(0, dst_with_self, pulled_msg)

        # Direct spatial highway connection to strengthen GNN backpropagation
        h_final = h_id + self.context_gate(ctx_pulled) + ctx_pulled
        h_norm = F.normalize(self.sp_norm(h_final), p=2, dim=-1)

        # 5. Decoupled Dual-Stream Projections with Dictionary-Coupled Gating
        z_mag = self.mag_enc(h_norm)
        
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        bio_sim = torch.mm(x_norm, w_dec_norm.t())
        
        spatial_shift = self.gate_spatial_proj(h_norm)
        base_logits = bio_sim + (self.spatial_gain * spatial_shift)
        
        # Dynamic scale: starts at 8.0 for exploration, ramps to 14.0 for sharp cell boundaries
        current_scale = getattr(self, 'current_scale', cfg.scale_end)
        gate_logits = base_logits * current_scale

        return z_mag, gate_logits, cell_mass

    def forward(
        self, 
        x_dense: torch.Tensor, 
        src: torch.Tensor, 
        dst: torch.Tensor, 
        edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        # Unpack the 3 tensors returned by encode
        z_mag, gate_logits, cell_mass = self.encode(x_dense, src, dst, edge_weights)


        # 1. Retrieve dynamic alpha scheduled by the training loop
        current_alpha = float(getattr(self, 'current_alpha', cfg.alpha_start))
        current_temp = getattr(self, 'current_temp', cfg.temp_end)
        progress = getattr(self, 'current_progress', 1.0)
        
        # 2. Simplex gating providing non-vanishing gradients across co-active latents
        gate_probs = entmax_bisect(gate_logits, alpha=current_alpha, dim=-1)


        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)

        # 3. Bio-SAE Activation: Independent magnitude per gene program
        z = gate_probs * z_mag

        # 4. Depth-scaled reconstruction: latents model cell-type programs; bias models ambient baseline
        baseline_gene_rate = F.softmax(self.decoder_bias, dim=-1).unsqueeze(0)
        x_recon = (torch.mm(z, w_dec_norm) + baseline_gene_rate) * cell_mass

        # --- OPTIMIZATION 2: AuxK Dead Latent Routing ---
        aux_recon = None
        r_norm = None

        if self.training:
            with torch.no_grad():
                active_in_batch = (gate_probs > cfg.active_latent_threshold).any(dim=0)
                self.steps_since_active.add_(1)
                self.steps_since_active.masked_fill_(active_in_batch, 0)
                dead_mask = self.steps_since_active >= self.dead_step_threshold

            x_norm = F.normalize(x_dense, p=2, dim=-1)
            x_recon_norm = F.normalize(F.relu(x_recon), p=2, dim=-1)
            r_norm = (x_norm - x_recon_norm).detach()
            residual_energy = r_norm.norm(p=2, dim=-1).mean()

            if dead_mask.any() and residual_energy > cfg.aux_min_residual_energy:
                dead_indices = torch.nonzero(dead_mask).squeeze(-1)
                num_dead = dead_indices.numel()
                k_aux = min(max(cfg.aux_min_k, cfg.aux_k), num_dead)

                w_dead = w_dec_norm[dead_indices]
                aux_logits = torch.mm(r_norm, w_dead.t())
                # Top-K on raw alignment ensures dead atoms always receive gradient updates
                topk_aux = torch.topk(aux_logits, k=k_aux, dim=-1)
                topk_weights = F.softplus(topk_aux.values)
                z_aux = torch.zeros_like(aux_logits).scatter(-1, topk_aux.indices, topk_weights)
                aux_recon = torch.mm(z_aux, w_dead)

        return x_recon, z, w_dec_norm, aux_recon, r_norm, z_mag

    def calc_loss(
        self, 
        recon_x: torch.Tensor, 
        x_true: torch.Tensor, 
        z: torch.Tensor,
        w_dec_norm: torch.Tensor,
        aux_recon: torch.Tensor | None = None,
        r_norm: torch.Tensor | None = None,
        progress: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Asymmetric Log-Cosh Loss with Dynamic Zero-Weighting
        is_non_zero = (x_true > 0)
        num_pos = torch.clamp(is_non_zero.float().sum(), min=1.0)
        num_zeros = (x_true == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(current_dynamic_w, weight=cfg.dynamic_w_ema_weight)

        w_mat = torch.where(is_non_zero, self.dynamic_w_ema, 1.0)
        w_mat = w_mat / torch.clamp(w_mat.mean(), min=1e-5)

        raw_delta = recon_x - x_true
        asym_factor = 1.0 + (is_non_zero.float() * cfg.asym_penalty_weight) * (raw_delta < 0).float()
        
        delta_clamp = getattr(cfg, 'delta_clamp', 30.0)
        scaled_delta = torch.clamp(raw_delta * asym_factor, min=-delta_clamp, max=delta_clamp)

        l_recon = torch.sum(w_mat * torch.log(torch.cosh(scaled_delta + 1e-6))) / max(1, x_true.numel())

        # 2. Sharp Log-Barrier Gram Decorrelation for Non-Negative Dictionary Atoms
        gram = torch.mm(w_dec_norm, w_dec_norm.t())
        off_diag = gram * self.ortho_mask
        
        excess_corr = F.relu(off_diag - cfg.ortho_overlap_threshold)
        l_ortho = torch.exp(cfg.ortho_barrier_scale * excess_corr).sub(1.0).sum() / (self.n_latents * (self.n_latents - 1))

        # 3. Sparsity Loss
        l_sparse = z.mean()

        # 4. AuxK Residual Loss: Normalized cosine alignment with stable gradient scaling
        if aux_recon is not None and r_norm is not None:
            aux_recon_norm = F.normalize(aux_recon, p=2, dim=-1)
            cos_align = (aux_recon_norm * r_norm).sum(dim=-1).clamp(min=-1.0, max=1.0)
            l_aux = (1.0 - cos_align).mean()
        else:
            l_aux = torch.tensor(0.0, device=x_true.device)

        # 5. Combined Dynamic Schedules
        base_l1 = cfg.l1_coeff
        base_ortho = cfg.ortho_weight
        aux_weight = cfg.aux_weight

        sparsity_multiplier = cfg.sparsity_min_scale + (1.0 - cfg.sparsity_min_scale) * (progress ** cfg.sparsity_prog_pow)
        current_l1 = base_l1 * sparsity_multiplier
        current_ortho = base_ortho * (cfg.ortho_min_scale + (1.0 - cfg.ortho_min_scale) * progress)

        total_loss = l_recon + (current_ortho * l_ortho) + (current_l1 * l_sparse) + (aux_weight * l_aux)

        return total_loss, l_recon.detach(), l_ortho.detach(), l_sparse.detach(), l_aux.detach()

    @torch.no_grad()
    def get_deep_telemetry(self) -> dict[str, float]:
        """Self-inspecting telemetry harvest with exact off-diagonal correlation stats."""
        stats = {}
        total_g_norm_sq = 0.0

        for name, param in self.named_parameters():
            p_clean = name.replace('.', '/')
            stats[f"param_norm/{p_clean}"] = param.detach().norm(2).item()
            if param.grad is not None:
                g_norm = param.grad.detach().norm(2).item()
                total_g_norm_sq += (g_norm ** 2)
                stats[f"grad_norm/{p_clean}"] = g_norm
                stats[f"grad_zeros/{p_clean}_pct"] = (param.grad == 0).float().mean().item() * 100.0

        stats["grad_norm/global_l2"] = total_g_norm_sq ** 0.5

        if hasattr(self, 'decoder_weight'):
            w = F.normalize(self.decoder_weight, p=2, dim=1)
            sim = torch.mm(w, w.t())
            off_diag_mask = ~torch.eye(w.size(0), dtype=torch.bool, device=w.device)
            off_diag_vals = sim.masked_select(off_diag_mask)
            if off_diag_vals.numel() > 0:
                stats["dict/max_cross_corr"] = off_diag_vals.max().item()
                stats["dict/mean_cross_corr"] = off_diag_vals.abs().mean().item()

        return stats