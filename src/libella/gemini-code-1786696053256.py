#!/usr/bin/env python3
"""Libella High-Resolution Forensic Pilot (Cache-Immune & Gene-Normalized).

Guarantees exact gene-normalized loss computation by dynamically overriding
stale cached modules:
    total_elements = max(1, x_c.shape[0] * x_c.shape[1])
    l_recon = l_recon_sum / total_elements
"""

import argparse
import gc
import importlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from entmax import entmax_bisect

# --- 1. FORCE PURGE & RELOAD STALE PYTHON MODULES FROM MEMORY ---
for mod in list(sys.modules.keys()):
    if mod.startswith("libella"):
        del sys.modules[mod]

import libella.config
import libella.data
import libella.model
import libella.utils

importlib.reload(libella.config)
importlib.reload(libella.data)
importlib.reload(libella.model)
importlib.reload(libella.utils)

from libella.config import cfg
from libella.data import make_meta_batches, pad_mps_shapes
from libella.model import LibellaGNN
from libella.utils import get_device, PhaseTracker, scatter_softmax


# =============================================================================
# 2. RUNTIME MONKEY-PATCH: GUARANTEE GENE-NORMALIZED LOSS
# =============================================================================

def patch_normalized_calc_loss(
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
    """Guaranteed gene-normalized loss with subgradient-stabilized Tsallis entropy."""
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
    
    # 🚨 STRICT TOTAL ELEMENTS (CELLS * GENES) NORMALIZATION
    total_elements = max(1, x_c.shape[0] * x_c.shape[1])
    l_recon = l_recon_sum / total_elements

    anc_norm = F.normalize(anchors, p=2, dim=1)

    with torch.no_grad():
        raw_temp = getattr(self, 'dict_temp', torch.tensor(cfg.dict_temp, device=anchors.device))
        safe_temp = torch.clamp(raw_temp, min=0.25, max=1.0)
        ref_probs = F.softmax(self.anchor_logits / safe_temp, dim=-1)
        ref_norm = F.normalize(ref_probs, p=2, dim=1)

    l_anc = 1.0 - (anc_norm * ref_norm).sum(dim=1).mean()

    peak_excess = F.relu(anchors - cfg.anchor_peak_threshold)
    collapse_penalty = (peak_excess ** 2).sum(dim=1).mean()
    gene_entropy = -(anchors * torch.log(anchors + 1e-9)).sum(dim=1).mean()

    raw_t_norm = F.normalize(anchors, p=2, dim=-1)
    latent_ortho = torch.mm(raw_t_norm, raw_t_norm.t()) * self.ortho_mask
    max_overlap = latent_ortho.max(dim=1)[0]
    
    l_ortho = (F.relu(max_overlap - cfg.ortho_overlap_threshold) ** 2).mean()

    im_loss = torch.tensor(0.0, device=x_c.device)
    tsallis_val = 0.0
    
    if f_train is not None:
        # Stabilized Tsallis Subgradient Clamping (Fix 1)
        f_norm = f_train / torch.clamp(f_train.sum(dim=1, keepdim=True), min=1e-6)
        alpha_ent = getattr(cfg, "tsallis_alpha", 1.5)
        f_safe = torch.clamp(f_norm, min=1e-5, max=1.0)
        
        if abs(alpha_ent - 1.0) > 1e-4:
            tsallis_h = (1.0 - (f_safe ** alpha_ent).sum(dim=1).mean()) / (alpha_ent - 1.0)
        else:
            tsallis_h = -(f_safe * torch.log(f_safe)).sum(dim=1).mean()
            
        tsallis_val = tsallis_h.item()
        p_mean = torch.clamp(f_norm.mean(dim=0), min=1e-5, max=1.0)
        
        if target_f_dist is not None:
            kl_marginal = (p_mean * (torch.log(p_mean) - torch.log(target_f_dist + 1e-9))).sum()
        else:
            K_topics = anchors.shape[0]
            uniform_prior = torch.ones(K_topics, device=x_c.device) / K_topics
            kl_marginal = (p_mean * (torch.log(p_mean) - torch.log(uniform_prior))).sum()
            
    progress = ep / max(1, total_epochs - 1)
    
    with torch.no_grad():
        recon_mag = l_recon.item()
        lock_weight = max(0.05, 1.0 - progress)
        anc_scale = recon_mag * 0.1 * lock_weight 
        kl_scale = recon_mag * 0.05
        tsallis_weight = max(0.0, (progress - 0.5) * 2.0)
        tsallis_scale = recon_mag * 0.05 * tsallis_weight

    if f_train is not None:
        im_loss = (tsallis_h * tsallis_scale) + (kl_weight * kl_marginal * kl_scale)

    scaled_anc = l_anc * anc_scale        
    scaled_ortho = (l_ortho + collapse_penalty) * cfg.ortho_weight * (recon_mag * 0.05)
    scaled_gene_ent = gene_entropy * (recon_mag * 0.01)
    
    base_loss = l_recon + scaled_anc + scaled_ortho + scaled_gene_ent

    self._last_losses = {
        'rec': l_recon.item(), 'anc': l_anc.item(), 
        'ort': l_ortho.item(), 'im': im_loss.item(), 'base': base_loss.item(),
        'dyn_w': current_dynamic_w.item(), 'kl_w': kl_weight, 
        'tsallis_val': tsallis_val
    }
    
    return base_loss + im_loss, l_recon.detach()

# Bind directly onto class
LibellaGNN.calc_loss = patch_normalized_calc_loss


# =============================================================================
# 3. UNIVERSAL ARTIFACT LOADER
# =============================================================================

def robust_load_artifact(file_path: Path | str) -> Any:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found at: {p}")

    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        pass
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        pass
    try:
        return np.load(p, allow_pickle=True)
    except Exception:
        pass

    raise RuntimeError(f"Could not deserialize artifact at: {p}")


def load_priors_and_genes(
    common_genes_path: str | Path, priors_path: str | Path
) -> tuple[list[str], np.ndarray | None, int]:
    genes_p = Path(common_genes_path)
    if not genes_p.exists():
        raise FileNotFoundError(f"Common genes file missing at: {genes_p}")

    with open(genes_p, "r", encoding="utf-8") as f:
        common_genes = json.load(f)

    priors_data = robust_load_artifact(priors_path)
    init_components = None
    optimal_k = getattr(cfg, "k_components", 38)

    if isinstance(priors_data, dict):
        init_components = priors_data.get(
            "components",
            priors_data.get("init_components", priors_data.get("priors", None)),
        )
        optimal_k = priors_data.get(
            "optimal_k",
            init_components.shape[0] if init_components is not None else optimal_k,
        )
    elif isinstance(priors_data, np.ndarray):
        init_components = priors_data
        optimal_k = init_components.shape[0]
    elif isinstance(priors_data, torch.Tensor):
        init_components = priors_data.detach().cpu().numpy()
        optimal_k = init_components.shape[0]

    n_extra_slots = getattr(cfg, "extra_topics", 0)
    if n_extra_slots > 0 and init_components is not None:
        extra_slots = np.zeros((n_extra_slots, init_components.shape[1]), dtype=np.float32)
        init_components = np.vstack([init_components, extra_slots])
        optimal_k += n_extra_slots

    return common_genes, init_components, optimal_k


# =============================================================================
# 4. HIGH-RESOLUTION STATISTICAL PROFILER
# =============================================================================

class HighResTelemetryRecorder:
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def record(self, key: str, value: float | torch.Tensor | np.ndarray | None):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            val = value.detach().item() if value.numel() == 1 else value.detach().float().norm().item()
        elif isinstance(value, (np.ndarray, np.number)):
            val = float(value.item() if value.size == 1 else np.linalg.norm(value))
        else:
            val = float(value)

        if not math.isnan(val) and not math.isinf(val):
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(val)

    def record_tensor(self, name: str, t: torch.Tensor | None, grad: torch.Tensor | None = None):
        if t is None or not isinstance(t, torch.Tensor):
            return
        with torch.no_grad():
            t_det = t.detach().float()
            numel = max(1, t_det.numel())
            l2 = t_det.norm(2).item()
            rms = math.sqrt((t_det ** 2).sum().item() / numel)
            mean = t_det.mean().item()
            std = t_det.std().item() if numel > 1 else 0.0
            zero_frac = ((t_det == 0).sum().item() / numel) * 100.0

            self.record(f"fwd__{name}__l2", l2)
            self.record(f"fwd__{name}__rms", rms)
            self.record(f"fwd__{name}__mean", mean)
            self.record(f"fwd__{name}__std", std)
            self.record(f"fwd__{name}__zero_pct", zero_frac)

            if grad is not None and isinstance(grad, torch.Tensor):
                g_det = grad.detach().float()
                g_numel = max(1, g_det.numel())
                g_l2 = g_det.norm(2).item()
                g_rms = math.sqrt((g_det ** 2).sum().item() / g_numel)
                ratio = g_l2 / (l2 + 1e-9)

                self.record(f"bwd__{name}__l2", g_l2)
                self.record(f"bwd__{name}__rms", g_rms)
                self.record(f"bwd__{name}__grad_to_act", ratio)

    def mean(self, key: str, default: float = 0.0) -> float:
        vals = self.metrics.get(key, [])
        return float(np.mean(vals)) if vals else default


# =============================================================================
# 5. HIGH-RESOLUTION PILOT AUDIT ENGINE
# =============================================================================

def run_high_res_pilot(
    chunk_dir_path: str, common_genes_path: str, priors_path: str
):
    device = get_device()
    chunk_dir = Path(chunk_dir_path)
    chunk_files = sorted(list(chunk_dir.glob("*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found at: {chunk_dir}")

    common_genes, init_components, optimal_k = load_priors_and_genes(
        common_genes_path, priors_path
    )
    in_channels = len(common_genes)

    print("=" * 118)
    print("🔬 LIBELLA FORENSIC PILOT (EXPLICIT GENE-NORMALIZED & CACHE-IMMUNE)")
    print(f"   Target Device: {device} | Cache Chunks: {len(chunk_files)} | Genes: {in_channels}")
    print(f"   Latent Topics (K): {optimal_k} | Priors: {'Initialized' if init_components is not None else 'Random'}")
    print("=" * 118)

    model = LibellaGNN(
        in_channels=in_channels,
        n_metaprograms=optimal_k,
        init_components=init_components,
    ).to(device)
    model.train()

    base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
    anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
        {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor},
    ])

    initial_logits = model.topic_gene_logits.detach().clone()
    tracker = PhaseTracker()
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    max_entropy_scalar = float(np.log(optimal_k))

    training_cache = []
    for f in chunk_files:
        patient_name = f.stem.split("_chunk_")[0] if "_chunk_" in f.stem else f.stem
        training_cache.append({"patient_name": patient_name, "chunk_file": f})

    meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
    total_steps_per_epoch = len(meta_batches)
    alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9))
    ema_mean = None

    rec = HighResTelemetryRecorder()

    print("\n[➤] Profiling Full 1-Epoch Sub-Graph Dynamics & Autograd Flow...\n")

    for step_idx, meta_meta in enumerate(meta_batches):
        optimizer.zero_grad(set_to_none=True)
        batch_chunks = [torch.load(b["chunk_file"], map_location="cpu", weights_only=False) for b in meta_meta]
        n_chunks_in_batch = len(batch_chunks)

        for chunk_idx, batch in enumerate(batch_chunks):
            x = batch["x"].to(device=device, non_blocking=True)
            src = batch["src"].to(device=device, non_blocking=True)
            dst = batch["dst"].to(device=device, non_blocking=True)
            weights = batch["weights"].to(device=device, non_blocking=True)
            train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)

            if len(src) > 0:
                keep_mask = torch.rand(src.size(0), device=device) > cfg.edge_dropout
                src = src[keep_mask]
                dst = dst[keep_mask]
                weights = weights[keep_mask]

            x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
            if device.type != "mps":
                src = src.to(torch.int64)
                dst = dst.to(torch.int64)

            squeeze_progress = tracker.get_progress()
            model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * squeeze_progress)
            model.current_temp = cfg.temp_start - ((cfg.temp_start - cfg.temp_end) * squeeze_progress)
            model.current_alpha = cfg.alpha_start

            # Forward pass
            h_id_lin = model.id_enc[0](x)
            h_id_glu = F.glu(h_id_lin, dim=-1)
            h_id = model.id_enc[2](h_id_glu)
            h_id.retain_grad()

            h_ctx_lin = model.ctx_enc[0](x)
            h_ctx_ln = model.ctx_enc[1](h_ctx_lin)
            h_ctx_act = model.ctx_enc[2](h_ctx_ln)
            h_0 = model.lin_appnp(h_ctx_act)
            h_0.retain_grad()

            macro_ctx = h_0.mean(dim=0)
            macro_ctx.retain_grad()

            bridge_raw = model.spatial_bridge(macro_ctx)
            dict_shift = torch.tanh(bridge_raw) * 2.0
            dict_shift.retain_grad()

            dynamic_logits = model.topic_gene_logits + dict_shift.view(model.n_metaprograms, -1)
            dynamic_logits.retain_grad()

            raw_temp = getattr(model, "dict_temp", torch.tensor(cfg.dict_temp, device=device))
            safe_temp = torch.clamp(raw_temp, min=0.25, max=1.0)
            soft_anchors = F.softmax(dynamic_logits, dim=-1)
            sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)

            anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()
            anchors_raw.retain_grad()

            N = h_0.size(0)
            if len(src) > 0:
                with torch.no_grad():
                    bio_h = torch.mm(x, anchors_raw.detach().t())
                    diff = bio_h[src] - bio_h[dst]
                    dist = (diff * diff).sum(dim=1)
                gamma_decay = F.softplus(model.gamma)
                decay = torch.exp(-gamma_decay * dist)
                rec.record("atom__gamma_val", gamma_decay.item())
                rec.record("atom__bio_dist_mean", dist.mean().item())
                rec.record("atom__decay_scale", decay.mean().item())
            else:
                decay = torch.ones_like(weights)

            W_bil = weights * decay
            W_bil.retain_grad()

            alpha = torch.sigmoid(model.alpha_proj(h_0)) * 0.85 + 0.10
            inv_alpha = 1.0 - alpha
            h_0_scaled = h_0 * alpha

            h_ctx = h_0
            for hop in range(model.k_hops):
                out = torch.zeros_like(h_ctx)
                if len(src) > 0:
                    h_src_proj = model.gat_w_src(h_ctx)
                    h_dst_proj = model.gat_w_dst(h_ctx)
                    edge_proj = model.gat_w_edge(W_bil.unsqueeze(1))
                    h_edge = h_src_proj[src] + h_dst_proj[dst] + edge_proj

                    e_raw = model.gat_a(F.leaky_relu(h_edge)).squeeze(-1)
                    tau = torch.clamp(F.softplus(model.att_temp), min=0.05)
                    alpha_att = scatter_softmax(e_raw / tau, dst, N)

                    msg = h_ctx[src] * alpha_att.unsqueeze(1)
                    out.index_add_(0, dst, msg)

                    att_entropy = -(alpha_att * torch.log(alpha_att + 1e-12)).sum() / max(1, N)
                    rec.record(f"atom__hop_{hop}_att_entropy", att_entropy.item())
                    rec.record(f"atom__hop_{hop}_att_max", alpha_att.max().item())

                agg = F.silu(model.mp_update(out))
                h_ctx = agg * inv_alpha + h_0_scaled

            h_ctx.retain_grad()

            Q = model.q_proj(h_id)
            K = model.k_proj(h_ctx)
            V = model.v_proj(h_ctx)
            Q.retain_grad(); K.retain_grad(); V.retain_grad()

            idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x.device.type == "mps" else torch.int64)
            self_loops = torch.arange(N, dtype=idx_dtype, device=x.device)
            src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
            dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

            cross_scores = (Q[dst_with_self] * K[src_with_self]).sum(dim=-1) / (model.hidden_dim ** 0.5)
            cross_att = scatter_softmax(cross_scores, dst_with_self, N)
            ctx_pulled = torch.zeros_like(Q)
            ctx_pulled.index_add_(0, dst_with_self, (V[src_with_self] * cross_att.unsqueeze(1)).contiguous())
            ctx_pulled.retain_grad()

            gate_out = model.context_gate(ctx_pulled)
            gate_out.retain_grad()

            h_final = h_id + gate_out
            h_final.retain_grad()

            with torch.no_grad():
                h_id_norms = h_id.norm(p=2, dim=-1)
                gate_norms = gate_out.norm(p=2, dim=-1)
                ctx_id_ratio = gate_norms / (h_id_norms + 1e-9)
                rec.record("atom__ctx_to_id_ratio_mean", ctx_id_ratio.mean().item())

            h_sp = model.sp_norm(h_final)
            h_norm = F.normalize(h_sp, p=2, dim=-1)
            h_norm.retain_grad()

            t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
            x_norm = F.normalize(x, p=2, dim=-1)
            bio_sim = torch.mm(x_norm, t_proj_weights.t())
            bio_sim.retain_grad()

            gnn_shift_raw = model.topic_proj(h_norm)
            gnn_shift_norm = F.normalize(gnn_shift_raw, p=2, dim=-1)
            gnn_shift_norm.retain_grad()

            base_logits = bio_sim + (cfg.gnn_shift_weight * gnn_shift_norm)
            noise = torch.randn_like(base_logits) * cfg.train_noise
            base_logits_noisy = base_logits + noise
            logits = base_logits_noisy * model.current_scale
            logits.retain_grad()

            sparse_prob = entmax_bisect(logits, alpha=model.current_alpha, dim=1)
            smooth_prob = F.softmax(logits / model.current_temp, dim=1)
            smooth_weight = 0.50 - (0.45 * squeeze_progress)
            sparse_weight = 1.0 - smooth_weight
            prob = (sparse_weight * sparse_prob) + (smooth_weight * smooth_prob)
            prob.retain_grad()

            fracs = prob * x.sum(dim=1, keepdim=True)
            f_train = fracs[train_idx]
            x_train = x[train_idx]
            f_train.retain_grad()

            with torch.no_grad():
                active_topics_per_cell = (f_train > 1e-4).float().sum(dim=1).mean().item()
                exact_zero_pct = (sparse_prob == 0.0).float().mean().item() * 100.0
                rec.record("atom__active_topics_per_cell", active_topics_per_cell)
                rec.record("atom__entmax_zero_pct", exact_zero_pct)

            p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
            current_p_mean = p_train.mean(dim=0)
            uniform_prior = torch.ones_like(current_p_mean) / optimal_k

            if ema_mean is None:
                ema_mean = current_p_mean.detach()
            else:
                ema_mean = alpha_ema * current_p_mean.detach() + (1 - alpha_ema) * ema_mean

            ideal_c = torch.clamp(uniform_prior * 2.0 - ema_mean, min=1e-5)
            target_f_dist = ideal_c / ideal_c.sum()

            ema_entropy = -torch.sum(ema_mean * torch.log(ema_mean + 1e-9))
            collapse_ratio = torch.clamp(1.0 - (ema_entropy / max_entropy_scalar), min=0.0, max=1.0)
            peak_p = ema_mean.max()
            hub_multiplier = F.relu((peak_p / cfg.hub_threshold) - 1.0) * 10.0
            dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + hub_multiplier

            recon = f_train @ anchors_raw
            recon.retain_grad()

            # Execute gene-normalized loss
            loss, base_recon_val = model.calc_loss(
                recon, x_train, anchors_raw, None,
                ep=0, total_epochs=cfg.epochs,
                f_train=f_train, target_f_dist=target_f_dist,
                kl_weight=dynamic_kl_w
            )

            scaled_loss = loss / n_chunks_in_batch
            scaled_loss.backward()

            rec.record_tensor("x_dense", x)
            rec.record_tensor("h_id", h_id, h_id.grad)
            rec.record_tensor("h_0", h_0, h_0.grad)
            rec.record_tensor("macro_ctx", macro_ctx, macro_ctx.grad)
            rec.record_tensor("dict_shift", dict_shift, dict_shift.grad)
            rec.record_tensor("dynamic_logits", dynamic_logits, dynamic_logits.grad)
            rec.record_tensor("anchors_raw", anchors_raw, anchors_raw.grad)
            rec.record_tensor("W_bil", W_bil, W_bil.grad)
            rec.record_tensor("h_ctx_final", h_ctx, h_ctx.grad)
            rec.record_tensor("Q", Q, Q.grad)
            rec.record_tensor("K", K, K.grad)
            rec.record_tensor("V", V, V.grad)
            rec.record_tensor("ctx_pulled", ctx_pulled, ctx_pulled.grad)
            rec.record_tensor("gate_out", gate_out, gate_out.grad)
            rec.record_tensor("h_final", h_final, h_final.grad)
            rec.record_tensor("h_norm", h_norm, h_norm.grad)
            rec.record_tensor("bio_sim", bio_sim, bio_sim.grad)
            rec.record_tensor("gnn_shift_norm", gnn_shift_norm, gnn_shift_norm.grad)
            rec.record_tensor("logits", logits, logits.grad)
            rec.record_tensor("prob", prob, prob.grad)
            rec.record_tensor("f_train", f_train, f_train.grad)
            rec.record_tensor("recon", recon, recon.grad)

            rec.record("loss__total", loss.item())
            rec.record("loss__recon", model._last_losses.get("rec", 0.0))
            rec.record("loss__anc", model._last_losses.get("anc", 0.0))
            rec.record("loss__ortho", model._last_losses.get("ort", 0.0))
            rec.record("loss__im", model._last_losses.get("im", 0.0))
            rec.record("loss__tsallis", model._last_losses.get("tsallis_val", 0.0))
            rec.record("loss__dyn_w", model._last_losses.get("dyn_w", 1.0))

        # Decoupled parameter clipping
        base_named_params = [(n, p) for n, p in model.named_parameters() if "topic_gene_logits" not in n and p.grad is not None]
        anchor_named_params = [(n, p) for n, p in model.named_parameters() if "topic_gene_logits" in n and p.grad is not None]

        base_grad_list = [p.grad.detach() for _, p in base_named_params]
        anchor_grad_list = [p.grad.detach() for _, p in anchor_named_params]

        base_grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in base_grad_list]), 2).item() if base_grad_list else 0.0
        anchor_grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in anchor_grad_list]), 2).item() if anchor_grad_list else 0.0

        clip_scale_base = min(1.0, cfg.grad_clip / (base_grad_norm + 1e-6))
        clip_scale_anchor = min(1.0, cfg.grad_clip / (anchor_grad_norm + 1e-6))

        rec.record("clip__scale_base", clip_scale_base)
        rec.record("clip__scale_anchor", clip_scale_anchor)
        rec.record("clip__base_grad_norm", base_grad_norm)
        rec.record("clip__anchor_grad_norm", anchor_grad_norm)

        if base_named_params:
            torch.nn.utils.clip_grad_norm_([p for _, p in base_named_params], max_norm=cfg.grad_clip)
        if anchor_named_params:
            torch.nn.utils.clip_grad_norm_([p for _, p in anchor_named_params], max_norm=cfg.grad_clip)

        # Exact AdamW Vector Dynamics
        anchor_group = optimizer.param_groups[1]
        lr = anchor_group["lr"]
        wd = anchor_group["weight_decay"]
        beta1, beta2 = anchor_group.get("betas", (0.9, 0.999))
        eps = anchor_group.get("eps", 1e-8)

        p_tensor = model.topic_gene_logits
        g_clipped = p_tensor.grad.detach()
        param_state = optimizer.state[p_tensor]

        raw_step = param_state.get("step", 0)
        current_step = (int(raw_step.item()) if isinstance(raw_step, torch.Tensor) else int(raw_step)) + 1

        if "exp_avg" in param_state:
            exp_avg = param_state["exp_avg"].clone()
            exp_avg_sq = param_state["exp_avg_sq"].clone()
            exp_avg.mul_(beta1).add_(g_clipped, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_clipped, g_clipped, value=1.0 - beta2)
        else:
            exp_avg = g_clipped * (1.0 - beta1)
            exp_avg_sq = (g_clipped * g_clipped) * (1.0 - beta2)

        bias_correction1 = 1.0 - (beta1 ** current_step)
        bias_correction2 = 1.0 - (beta2 ** current_step)

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        adam_grad_step = -lr * (exp_avg / bias_correction1) / denom
        adam_wd_step = -lr * wd * p_tensor.detach()

        norm_grad_force = adam_grad_step.norm().item()
        norm_wd_force = adam_wd_step.norm().item()
        wd_to_grad_ratio = norm_wd_force / (norm_grad_force + 1e-12)
        cos_sim_grad_wd = F.cosine_similarity(adam_grad_step.view(1, -1), adam_wd_step.view(1, -1)).item()

        rec.record("adamw__grad_force", norm_grad_force)
        rec.record("adamw__wd_force", norm_wd_force)
        rec.record("adamw__wd_grad_ratio", wd_to_grad_ratio)
        rec.record("adamw__cos_sim", cos_sim_grad_wd)

        optimizer.step()

        if (step_idx + 1) % max(1, len(meta_batches) // 5) == 0 or step_idx == len(meta_batches) - 1:
            print(
                f"  [Meta-Batch {step_idx+1:02d}/{len(meta_batches):02d}] "
                f"Loss: {rec.mean('loss__total'):.4f} (Rec: {rec.mean('loss__recon'):.4f}) | "
                f"Clip Scales: (Base={clip_scale_base:.3f}, Anchor={clip_scale_anchor:.3f}) | "
                f"AdamW WD/Grad: {wd_to_grad_ratio:.3f}x"
            )

    # Statistical synthesis
    delta_logits = (model.topic_gene_logits.detach() - initial_logits).norm().item()
    max_logit_delta = (model.topic_gene_logits.detach() - initial_logits).abs().max().item()

    print("\n" + "=" * 118)
    print("📊 HIGH-RESOLUTION PILOT AUDIT RESULTS (GENE-NORMALIZED & VERIFIED)")
    print("=" * 118)

    def print_hdr(title: str):
        print(f"\n{'─' * 118}\n▶ {title}\n{'─' * 118}")

    def eval_flag(val: float, bounds: tuple[float, float], lower_is_better: bool = False) -> str:
        low, high = bounds
        if low <= val <= high:
            return "🟢 HEALTHY"
        elif (val < low and not lower_is_better) or (val > high and lower_is_better):
            return "🔴 CHOKEPOINT"
        return "🟡 CAUTION"

    print_hdr("A. FORWARD TENSOR ACTIVATION & STRUCTURAL TOPOLOGY")
    print(f"{'Sub-Layer Atom':<20} | {'Role & Mechanism':<32} | {'RMS Norm':<10} | {'Zero %':<8} | {'Status':<14}")
    print("─" * 118)

    fwd_spec = [
        ("x_dense", "Raw Count Matrix X", (0.01, 10.0), False),
        ("h_id", "id_enc (Linear + GLU + LN)", (0.1, 5.0), False),
        ("h_0", "lin_appnp(ctx_enc(X))", (0.1, 5.0), False),
        ("macro_ctx", "Graph Context Pooled Mean", (0.01, 2.0), False),
        ("dict_shift", "Spatial Bridge Shift tanh*2.0", (0.05, 2.0), False),
        ("dynamic_logits", "topic_logits + dict_shift", (0.5, 10.0), False),
        ("anchors_raw", "STE Sharp/Softmax Anchor Matrix", (0.001, 0.5), False),
        ("W_bil", "Bilateral Edge Decay Weights", (0.01, 1.0), False),
        ("h_ctx_final", "APPNP + GATv2 k-hop Aggregation", (0.1, 5.0), False),
        ("ctx_pulled", "Cross-Attention Pulled Context", (0.01, 5.0), False),
        ("gate_out", "context_gate(ctx_pulled)", (0.01, 5.0), False),
        ("h_final", "h_id + gate_out (Pure Residual)", (0.1, 5.0), False),
        ("h_norm", "sp_norm Unit-Sphere (L2)", (0.05, 1.5), False),
        ("bio_sim", "Cosine Sim(X_norm, T_norm)", (0.01, 1.0), False),
        ("gnn_shift_norm", "Topic Projection Vector (L2)", (0.05, 1.5), False),
        ("logits", "Hybrid Scaled Shifted Logits", (0.5, 25.0), False),
        ("prob", "Entmax1.5 / Softmax Blend", (0.01, 0.5), False),
        ("recon", "Reconstructed Expression (F @ A)", (0.01, 10.0), False),
    ]

    for name, desc, bounds, lib in fwd_spec:
        rms = rec.mean(f"fwd__{name}__rms")
        zp = rec.mean(f"fwd__{name}__zero_pct")
        print(f"{name:<20} | {desc:<32} | {rms:<10.4e} | {zp:<7.1f}% | {eval_flag(rms, bounds, lib):<14}")

    print_hdr("B. LOSS COMPONENTS (GENE-NORMALIZED SCALE)")
    print(f" • Total Loss (base + im):      {rec.mean('loss__total'):.4f}")
    print(f" • Pure Reconstruction Loss:    {rec.mean('loss__recon'):.4f} (Target: ~5.0 - 15.0)")
    print(f" • Anchor Regularization Loss:  {rec.mean('loss__anc'):.4f}")
    print(f" • Orthogonality Loss:          {rec.mean('loss__ortho'):.4f}")
    print(f" • Information Maximization:    {rec.mean('loss__im'):.4f}")

    print_hdr("C. BACKPROPAGATION GRADIENT CHAIN")
    print(f"{'Tensor Gradient':<20} | {'Autograd Pathway':<32} | {'||∇L||_2':<11} | {'Grad/Act Ratio':<14} | {'Status':<14}")
    print("─" * 118)

    bwd_spec = [
        ("recon", "∂L / ∂(Reconstruction)", (1e-7, 1.0)),
        ("f_train", "∂L / ∂(Topic Proportions F)", (1e-4, 500.0)),
        ("prob", "∂L / ∂(Cell Topic Prob P)", (1e-5, 50.0)),
        ("logits", "∂L / ∂(Logits via Entmax/SM)", (1e-6, 10.0)),
        ("bio_sim", "∂L / ∂(Bio Cosine Similarity)", (1e-6, 10.0)),
        ("gnn_shift_norm", "∂L / ∂(GNN Topic Shift)", (1e-6, 10.0)),
        ("h_norm", "∂L / ∂(LayerNorm Feature H)", (1e-6, 10.0)),
        ("gate_out", "∂L / ∂(context_gate output)", (1e-6, 10.0)),
        ("ctx_pulled", "∂L / ∂(Cross-Attention Pulled)", (1e-6, 10.0)),
        ("h_ctx_final", "∂L / ∂(GNN Aggregation)", (1e-6, 10.0)),
        ("anchors_raw", "∂L / ∂(Raw Anchors Matrix)", (1e-5, 100.0)),
        ("dynamic_logits", "∂L / ∂(Dynamic Logits)", (1e-6, 10.0)),
        ("dict_shift", "∂L / ∂(Spatial Bridge Shift)", (1e-6, 10.0)),
        ("macro_ctx", "∂L / ∂(Macro Graph Context)", (1e-6, 10.0)),
        ("h_0", "∂L / ∂(Context Encoder Root)", (1e-6, 10.0)),
        ("h_id", "∂L / ∂(Identity Encoder Root)", (1e-6, 10.0)),
    ]

    for name, desc, bounds in bwd_spec:
        g_l2 = rec.mean(f"bwd__{name}__l2")
        g_ratio = rec.mean(f"bwd__{name}__grad_to_act")
        print(f"{name:<20} | {desc:<32} | {g_l2:<11.4e} | {g_ratio:<14.4e} | {eval_flag(g_l2, bounds):<14}")

    print_hdr("D. DECOUPLED CLIPPING ISOLATION & TRUE ADAMW FORCE")
    clip_b = rec.mean("clip__scale_base")
    clip_a = rec.mean("clip__scale_anchor")
    norm_b = rec.mean("clip__base_grad_norm")
    norm_a = rec.mean("clip__anchor_grad_norm")

    print(f" • Base Backbone Grad Norm:     ||g_base||   = {norm_b:.4e} (Clip Scale: {clip_b:.4f}) ➔ {eval_flag(clip_b, (0.5, 1.0))}")
    print(f" • Anchor Dictionary Grad Norm: ||g_anchor|| = {norm_a:.4e} (Clip Scale: {clip_a:.4f}) ➔ {eval_flag(clip_a, (0.5, 1.0))}")

    f_grad = rec.mean("adamw__grad_force")
    f_wd = rec.mean("adamw__wd_force")
    ratio_wd = rec.mean("adamw__wd_grad_ratio")
    cos_sim = rec.mean("adamw__cos_sim")

    print(f"\n • Exact AdamW Step Decomposition on 'topic_gene_logits':")
    print(f"   ↳ Normalized Data Gradient Force:  ||-η · m̂/(√v̂ + ε)|| = {f_grad:.6e}")
    print(f"   ↳ Weight Decay Vector Force:        ||-η · λ · θ||     = {f_wd:.6e}")
    print(f"   ↳ True Force Ratio (WD / Data Grad): {ratio_wd:.3f}x (Cosine Alignment = {cos_sim:+.3f})")

    print_hdr("E. TOTAL PARAMETER DISPLACEMENT OVER 1 FULL EPOCH")
    print(f" • Cumulative Logit Matrix L2 Drift: ||ΔW||_2 = {delta_logits:.6e}")
    print(f" • Maximum Single Coordinate Shift:  max|Δw|  = {max_logit_delta:.6e}")
    print("=" * 118 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Libella High-Resolution Forensic Pilot (Gene-Normalized)")
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks",
        help="Path to directory containing .pt training chunks",
    )
    parser.add_argument(
        "--common-genes",
        type=str,
        default="/Users/Hemato/project_3/benchmark/libella_output/run/common_genes.json",
        help="Path to common_genes.json",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default="/Users/Hemato/project_3/benchmark/libella_output/run/global_cnmf_priors.pkl",
        help="Path to global_cnmf_priors.pkl",
    )

    args = parser.parse_args()
    run_high_res_pilot(args.chunk_dir, args.common_genes, args.priors)