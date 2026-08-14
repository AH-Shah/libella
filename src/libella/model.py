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
        

        self.topic_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, n_metaprograms)
        )

        self.spatial_bridge = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, n_metaprograms * in_channels)
        )
        self.dict_temp = nn.Parameter(torch.tensor(cfg.dict_temp))
        self.n_metaprograms = n_metaprograms
        self.in_channels = in_channels

        
        self.register_buffer('ortho_mask', 1.0 - torch.eye(n_metaprograms, dtype=torch.float32))

        
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()
            
        h_id = self.id_enc(x_dense)
        h_0 = self.lin_appnp(self.ctx_enc(x_dense))
        

        macro_ctx = h_0.mean(dim=0)
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 2.0
        
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)
        
        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        

        safe_temp = torch.clamp(getattr(self, 'dict_temp', torch.tensor(0.30)), min=0.25, max=1.0)
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

        
        t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
        x_norm = F.normalize(x_dense, p=2, dim=-1)
        
        bio_sim = torch.mm(x_norm, t_proj_weights.t())
        

        gnn_shift_raw = self.topic_proj(h_norm)
        gnn_shift_norm = F.normalize(gnn_shift_raw, p=2, dim=-1)

        base_logits = bio_sim + (cfg.gnn_shift_weight * gnn_shift_norm)
        
        noise = torch.randn_like(base_logits) * cfg.train_noise if self.training else 0.0
        base_logits = base_logits + noise
        
        current_scale = getattr(self, 'current_scale', cfg.inference_scale)
        logits = base_logits * current_scale
        
        return logits, anchors_raw

    def forward(
        self, 
        x_dense: torch.Tensor, 
        src: torch.Tensor, 
        dst: torch.Tensor, 
        edge_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, anchors_raw = self.encode(x_dense, src, dst, edge_weights)

        current_alpha = getattr(self, 'current_alpha', cfg.inference_alpha)
        current_temp = getattr(self, 'current_temp', cfg.inference_temp)
        progress = getattr(self, 'current_progress', 1.0)
        
        sparse_prob = entmax_bisect(logits, alpha=current_alpha, dim=1)
        
        if self.training:
            smooth_prob = F.softmax(logits / current_temp, dim=1)
            
            smooth_weight = 0.50 - (0.45 * progress)
            sparse_weight = 1.0 - smooth_weight
            
            prob = (sparse_weight * sparse_prob) + (smooth_weight * smooth_prob)
        else:
            prob = sparse_prob 
        
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
        kl_weight: float = cfg.kl_weight
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate regularized reconstruction loss."""
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

        l_recon_sum = torch.sum(masked_w_mat * torch.log(torch.cosh(scaled_delta + 1e-6)))
        
        # Direct scalar division without creating GPU tensor objects
        l_recon = l_recon_sum / max(1, x_c.shape[0])

        anc_norm = F.normalize(anchors, p=2, dim=1)
        ref_probs = F.softmax(self.anchor_logits, dim=-1)
        ref_norm = F.normalize(ref_probs, p=2, dim=1)
        l_anc = 1.0 - (anc_norm * ref_norm).sum(dim=1).mean()

        peak_excess = F.relu(anchors - cfg.anchor_peak_threshold)
        collapse_penalty = (peak_excess ** 2).sum(dim=1).mean()
        gene_entropy = -(anchors * torch.log(anchors + 1e-9)).sum(dim=1).mean()

        raw_t_norm = F.normalize(anchors, p=2, dim=-1)
        latent_ortho = torch.mm(raw_t_norm, raw_t_norm.t())
        
        
        latent_ortho = latent_ortho * self.ortho_mask

        max_overlap = latent_ortho.max(dim=1)[0]
        

        l_ortho = (F.relu(max_overlap - cfg.ortho_overlap_threshold) ** 2).mean()
        scaled_ortho = (l_ortho + collapse_penalty) * cfg.ortho_weight
        scaled_gene_ent = gene_entropy * 0.1

        im_loss = torch.tensor(0.0, device=x_c.device)
        tsallis_val = 0.0
        
        if f_train is not None:
            f_norm = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
            
            alpha_ent = cfg.tsallis_alpha
            tsallis_h = (1.0 - (f_norm ** alpha_ent).sum(dim=1).mean()) / (alpha_ent - 1.0)
            tsallis_val = tsallis_h.item()
            
            p_mean = torch.clamp(f_norm.mean(dim=0), min=1e-7)
            
            if target_f_dist is not None:
                kl_marginal = (p_mean * (torch.log(p_mean) - torch.log(target_f_dist + 1e-9))).sum()
            else:
                K_topics = anchors.shape[0]
                uniform_prior = torch.ones(K_topics, device=x_c.device) / K_topics
                kl_marginal = (p_mean * (torch.log(p_mean) - torch.log(uniform_prior))).sum()
                
        progress = ep / max(1, total_epochs - 1)
        
        with torch.no_grad():
            recon_mag = l_recon.item()
            

            lock_weight = max(0.05, 1.0 - (progress))
        
            anc_scale = recon_mag * 0.1 * lock_weight 
            
            kl_scale = recon_mag * 0.05
            
            tsallis_weight = max(0.0, (progress - 0.5) * 2.0)
            tsallis_scale = recon_mag * 0.05 * tsallis_weight

        im_loss = (tsallis_h * tsallis_scale) + (kl_weight * kl_marginal * kl_scale)

        scaled_anc = l_anc * anc_scale        
        scaled_ortho = l_ortho * (recon_mag * 0.05)
        scaled_gene_ent = gene_entropy * (recon_mag * 0.01)
        
        base_loss = l_recon + scaled_anc + scaled_ortho + scaled_gene_ent


        self._last_losses = {
            'rec': l_recon.item(), 'anc': l_anc.item(), 
            'ort': l_ortho.item(), 'im': im_loss.item(), 'base': base_loss.item(),
            'dyn_w': current_dynamic_w.item(), 'kl_w': kl_weight, 
            'tsallis_val': tsallis_val
        }
        
        return base_loss + im_loss, l_recon.detach()