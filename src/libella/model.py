"""Spatial Graph Neural Network architecture for Libella."""

import numpy as np
import torch
import torch.nn.functional as F
from entmax import entmax_bisect
from torch import nn

from .config import cfg
from .utils import scatter_softmax


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

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        

        self.context_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.sp_norm = nn.LayerNorm(self.hidden_dim)
        

        self.n_latents = n_metaprograms  # Overcomplete latent dimension (M)
        self.in_channels = in_channels    # Number of input genes (D)

        # 1. Stabilized Local Magnitude Stream
        self.mag_enc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_latents),
            nn.LayerNorm(self.n_latents),
            nn.Softplus(beta=1.0)
        )

        # 2. Context Gating Stream (takes h_norm from untouched GNN)
        self.gate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_latents)
        )

        # 3. Learnable Jump Thresholds for strict L0 boundary control
        self.jump_threshold = nn.Parameter(torch.full((self.n_latents,), 0.1, dtype=torch.float32))

        # 4. Oblique Unit-Norm Decoder Dictionary (M x D)
        dec_weight = torch.randn(self.n_latents, in_channels)
        dec_weight = F.normalize(dec_weight, p=2, dim=1)
        self.decoder_weight = nn.Parameter(dec_weight)
        self.decoder_bias = nn.Parameter(torch.zeros(in_channels))

        # 5. Mask for Oblique Orthogonality
        self.register_buffer('ortho_mask', 1.0 - torch.eye(self.n_latents, dtype=torch.float32))

        
        if init_components is not None:
            active_mask = (init_components > 0)


            base_logits = np.where(active_mask, 2.0, -2.0)
            
            noise = np.random.randn(*base_logits.shape) * 0.1
            init_logits = base_logits + noise
            
            self.topic_gene_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
            self.register_buffer('anchor_logits', torch.tensor(init_logits, dtype=torch.float32).clone())

        else:
            self.topic_gene_logits = nn.Parameter(torch.randn(n_metaprograms, in_channels))
            self.register_buffer('anchor_logits', torch.ones(n_metaprograms, in_channels))

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.alpha_proj = nn.Linear(self.hidden_dim, 1)
        self.register_buffer('dynamic_w_ema', torch.tensor(1.0, dtype=torch.float32))
        


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
        edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        # 1. Depth Disentanglement: Extract mass and compute normalized input
        cell_mass = torch.clamp(x_dense.sum(dim=-1, keepdim=True), min=1e-5)
        x_norm = F.normalize(x_dense, p=2, dim=-1)

        # 2. Local Identity and Initial Context representations
        h_id = self.id_enc(x_norm)
        h_0 = self.lin_appnp(self.ctx_enc(x_norm))
        

        macro_ctx = h_0.mean(dim=0)
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 2.0
        
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)
        
        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        

        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)
        
        anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()

        
        N = h_0.size(0)
        
        # 2. GNN Edge Decay 
        if len(src) > 0:
            with torch.no_grad():
                bio_h = torch.mm(x_dense, anchors_raw.detach().t())
                # 🚨 MPS FIX: Use direct multiplication instead of .pow(2)
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

        idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x_dense.device.type == 'mps' else torch.int64)
        self_loops = torch.arange(N, dtype=idx_dtype, device=x_dense.device)
        
        src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
        dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

            
        q_dst = Q[dst_with_self]
        k_src = K[src_with_self]
        v_src = V[src_with_self]
        
        cross_scores = (q_dst * k_src).sum(dim=-1) / (self.hidden_dim ** 0.5)
        cross_att = scatter_softmax(cross_scores, dst_with_self, N)
        

        pulled_msg = (v_src * cross_att.unsqueeze(1)).contiguous()
        ctx_pulled = torch.zeros_like(Q)
        ctx_pulled.index_add_(0, dst_with_self, pulled_msg)

        h_final = h_id + self.context_gate(ctx_pulled)
        h_norm = F.normalize(self.sp_norm(h_final), p=2, dim=-1)

        # 1. Compute Decoupled Streams
        # A. Local Magnitude Stream (unaffected by neighbor graph smoothing)
        z_mag = self.mag_enc(x_norm)

        # B. Spatial Context Gating Stream (conditioned by spatial graph)
        gate_logits = self.gate_proj(h_norm)

        return z_mag, gate_logits, cell_mass

    def forward(
        self, 
        x_dense: torch.Tensor, 
        src: torch.Tensor, 
        dst: torch.Tensor, 
        edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mag, gate_logits, cell_mass = self.encode(x_dense, src, dst, edge_weights)

        # 1. Compute Gate Probabilities (Sigmoid or Entmax)
        gate_probs = torch.sigmoid(gate_logits)

        # 2. Straight-Through JumpReLU Operator
        theta = torch.clamp(self.jump_threshold, min=0.01, max=0.99)
        hard_mask = (gate_probs > theta).float()
        
        # STE: Forward uses exact hard step; Backward passes gradient through sigmoid
        jump_gate = hard_mask.detach() + gate_probs - gate_probs.detach()

        # 3. Final Sparse Latent Code (magnitude isolated from gate)
        z = z_mag * jump_gate

        # 4. Decode with Unit-Norm Oblique Projection
        w_dec_norm = F.normalize(self.decoder_weight, p=2, dim=1)
        x_recon_norm = torch.mm(z, w_dec_norm) + self.decoder_bias

        # 5. Restore Cell-Specific Mass
        x_recon = x_recon_norm * cell_mass

        return x_recon, z, w_dec_norm

    def calc_loss(
        self, 
        recon_x: torch.Tensor, 
        x_true: torch.Tensor, 
        z: torch.Tensor,
        w_dec_norm: torch.Tensor,
        progress: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Asymmetric Log-Cosh Loss with Dynamic Zero-Weighting
        is_non_zero = (x_true > 0)
        num_pos = torch.clamp(is_non_zero.float().sum(), min=1.0)
        num_zeros = (x_true == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(current_dynamic_w, weight=0.1)

        w_mat = torch.where(is_non_zero, self.dynamic_w_ema, 1.0)
        raw_delta = recon_x - x_true
        asym_factor = 1.0 + (is_non_zero.float() * 2.0) * (raw_delta < 0).float()
        scaled_delta = torch.clamp(raw_delta * asym_factor, min=-30.0, max=30.0)

        l_recon = torch.sum(w_mat * torch.log(torch.cosh(scaled_delta + 1e-6))) / max(1, x_true.numel())

        # 2. Oblique Orthogonality Loss on Unit-Norm Dictionary (Strided/Sampled)
        cosine_sim = torch.mm(w_dec_norm, w_dec_norm.t()) * self.ortho_mask
        l_ortho = (F.relu(cosine_sim - 0.10) ** 2).sum() / (self.n_latents * (self.n_latents - 1))

        # 3. Sparsity Penalty (encourages gate thresholds to settle sharply)
        l_sparse = z.mean()

        # Dynamic Loss Balancing
        total_loss = l_recon + (10.0 * l_ortho) + (cfg.l1_coeff * l_sparse)

        return total_loss, l_recon.detach(), l_ortho.detach(), l_sparse.detach()
        