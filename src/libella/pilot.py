#!/usr/bin/env python3
"""Libella Master Forensic & Architectural Autopsy Pilot Engine.

Unified multi-domain diagnostic and telemetry suite:
1. High-Resolution Live Forensic Profiler (Forward tensors, backward autograd flow,
   decoupled gradient clipping, AdamW force vector decomposition, gene-normalized loss).
2. Trajectory Health & Checkpoint Autopsy (SVD effective rank, condition numbers,
   Gram matrix collinearity, attention dynamics, 12-panel dashboard visualization).
"""

import argparse
import gc
import json
import math
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from entmax import entmax_bisect

# Direct imports from Libella architecture
from libella.config import cfg, paths
from libella.data import make_meta_batches, pad_mps_shapes
from libella.model import LibellaGNN
from libella.utils import get_device, PhaseTracker, scatter_softmax

# Plot aesthetic configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# =============================================================================
# 1. UNIVERSAL ARTIFACT & PRIOR LOADERS
# =============================================================================

def robust_load_artifact(file_path: Path | str) -> Any:
    """Deserializes checkpoints, priors, and dictionary artifacts across formats."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found at: {p}")

    # PyTorch loader
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        pass

    # Pickle loader
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # Joblib loader
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        pass

    # NumPy binary loader
    try:
        return np.load(p, allow_pickle=True)
    except Exception:
        pass

    raise RuntimeError(f"Universal deserializer could not open artifact at: {p}")


def load_priors_and_genes(
    common_genes_path: str | Path, priors_path: str | Path
) -> Tuple[List[str], Optional[np.ndarray], int]:
    """Loads consensus gene space and initialized topic dictionary components."""
    genes_p = Path(common_genes_path)
    if not genes_p.exists():
        raise FileNotFoundError(f"Consensus gene vocabulary missing at: {genes_p}")

    with open(genes_p, "r", encoding="utf-8") as f:
        common_genes = json.load(f)

    priors_data = robust_load_artifact(priors_path)
    init_components = None
    optimal_k = getattr(cfg, "k_components", getattr(cfg, "n_dict_components", 38))

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
# 2. MATHEMATICAL DIAGNOSTIC HELPERS
# =============================================================================

def matrix_effective_rank(mat: torch.Tensor) -> float:
    """Computes continuous effective rank via singular value entropy (Roy & Vetterli)."""
    if mat.ndim != 2:
        return 0.0
    try:
        _, s, _ = torch.linalg.svd(mat.float(), full_matrices=False)
        s_sum = s.sum()
        if s_sum <= 1e-9:
            return 0.0
        p = s / s_sum
        p = p[p > 1e-9]
        entropy = -(p * torch.log(p)).sum().item()
        return math.exp(entropy)
    except Exception:
        return 0.0


def matrix_condition_number(mat: torch.Tensor) -> float:
    """Calculates condition number (s_max / s_min) of a linear transformation."""
    try:
        s = torch.linalg.svdvals(mat.float())
        s_max = s[0].item()
        s_min = s[-1].item()
        return (s_max / (s_min + 1e-9)) if s_min > 1e-9 else 1e6
    except Exception:
        return 1.0


# =============================================================================
# 3. HIGH-RESOLUTION TELEMETRY & TENSOR PROFILER
# =============================================================================

class HighResTelemetryRecorder:
    """Profiles live forward activations, backward autograd flow, and optimizer force vectors."""
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record(self, key: str, value: Any):
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

    def record_tensor(self, name: str, t: Optional[torch.Tensor], grad: Optional[torch.Tensor] = None):
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
# 4. HIGH-RESOLUTION FORENSIC PILOT AUDIT
# =============================================================================

def run_forensic_audit(
    chunk_dir_path: str | Path,
    common_genes_path: str | Path,
    priors_path: str | Path
) -> HighResTelemetryRecorder:
    """Executes full live 1-epoch autograd and tensor dynamics profiling."""
    device = get_device()
    chunk_dir = Path(chunk_dir_path)
    chunk_files = sorted(list(chunk_dir.glob("*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No .pt training chunks found at: {chunk_dir}")

    common_genes, init_components, optimal_k = load_priors_and_genes(
        common_genes_path, priors_path
    )
    in_channels = len(common_genes)

    print("=" * 118)
    print("🔬 LIBELLA MASTER FORENSIC PILOT (LIVE AUTOGRAD & TENSOR PROFILING)")
    print(f"   Compute Device: {device} | SSD Chunks: {len(chunk_files)} | Gene Space: {in_channels}")
    print(f"   Metaprograms (K): {optimal_k} | Initialization: {'CNMF Priors' if init_components is not None else 'Random Normal'}")
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

    print("\n[➤] Tracing Sub-Graph Micro-Layers, Retention Gradients, and Optimizer Physics...\n")

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

            # -------------------------------------------------------------
            # EXPLICIT FORWARD PASS WITH RETENTION HOOKS
            # -------------------------------------------------------------
            # Identity encoder
            h_id_lin = model.id_enc[0](x)
            h_id_glu = F.glu(h_id_lin, dim=-1)
            h_id = model.id_enc[2](h_id_glu)
            h_id.retain_grad()

            # Context encoder
            h_ctx_lin = model.ctx_enc[0](x)
            h_ctx_ln = model.ctx_enc[1](h_ctx_lin)
            h_ctx_act = model.ctx_enc[2](h_ctx_ln)
            h_0 = model.lin_appnp(h_ctx_act)
            h_0.retain_grad()

            # Spatial bridge
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

            # Straight-through estimator
            anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()
            anchors_raw.retain_grad()

            # Bilateral edge physics
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

            # APPNP Teleport + GATv2 Propagation
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

            # Transformer Cross-Attention
            Q = model.q_proj(h_id)
            K = model.k_proj(h_ctx)
            V = model.v_proj(h_ctx)
            Q.retain_grad()
            K.retain_grad()
            V.retain_grad()

            idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x.device.type == "mps" else torch.int64)
            self_loops = torch.arange(N, dtype=idx_dtype, device=x.device)
            src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
            dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

            cross_scores = (Q[dst_with_self] * K[src_with_self]).sum(dim=-1) / (model.hidden_dim ** 0.5)
            cross_att = scatter_softmax(cross_scores, dst_with_self, N)
            ctx_pulled = torch.zeros_like(Q)
            ctx_pulled.index_add_(0, dst_with_self, (V[src_with_self] * cross_att.unsqueeze(1)).contiguous())
            ctx_pulled.retain_grad()

            # Context gate & LayerNorm
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

            # Hybrid cosine biological projection
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

            # Online EMA marginal distribution
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

            # Loss computation
            loss_out = model.calc_loss(
                recon, x_train, anchors_raw, None,
                ep=0, total_epochs=cfg.epochs,
                f_train=f_train, target_f_dist=target_f_dist,
                kl_weight=dynamic_kl_w
            )
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out

            scaled_loss = loss / n_chunks_in_batch
            scaled_loss.backward()

            # -------------------------------------------------------------
            # TELEMETRY RECORDING: FORWARD ACTIVATIONS & BACKWARD GRADIENTS
            # -------------------------------------------------------------
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
            if hasattr(model, "_last_losses"):
                rec.record("loss__recon", model._last_losses.get("rec", 0.0))
                rec.record("loss__anc", model._last_losses.get("anc", 0.0))
                rec.record("loss__ortho", model._last_losses.get("ort", 0.0))
                rec.record("loss__im", model._last_losses.get("im", 0.0))
                rec.record("loss__tsallis", model._last_losses.get("tsallis_val", 0.0))
                rec.record("loss__dyn_w", model._last_losses.get("dyn_w", 1.0))

        # -------------------------------------------------------------
        # DECOUPLED GRADIENT CLIPPING AUDIT
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # EXACT ADAMW VECTOR DYNAMICS AUDIT
        # -------------------------------------------------------------
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

        if (step_idx + 1) % max(1, len(meta_batches) // 4) == 0 or step_idx == len(meta_batches) - 1:
            print(
                f"  [Meta-Batch {step_idx+1:02d}/{len(meta_batches):02d}] "
                f"Loss: {rec.mean('loss__total'):.4f} (Rec: {rec.mean('loss__recon'):.4f}) | "
                f"Clip: (Base={clip_scale_base:.3f}, Anch={clip_scale_anchor:.3f}) | "
                f"WD/Grad: {wd_to_grad_ratio:.3f}x"
            )

    # -----------------------------------------------------------------
    # FORENSIC REPORT GENERATION
    # -----------------------------------------------------------------
    delta_logits = (model.topic_gene_logits.detach() - initial_logits).norm().item()
    max_logit_delta = (model.topic_gene_logits.detach() - initial_logits).abs().max().item()

    print("\n" + "=" * 118)
    print("📊 HIGH-RESOLUTION FORENSIC AUDIT SCORECARD (GENE-NORMALIZED & VERIFIED)")
    print("=" * 118)

    def print_hdr(title: str):
        print(f"\n{'─' * 118}\n▶ {title}\n{'─' * 118}")

    def eval_flag(val: float, bounds: Tuple[float, float], lower_is_better: bool = False) -> str:
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
        ("h_final", "h_id + gate_out (Residual)", (0.1, 5.0), False),
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

    print_hdr("B. LOSS DECOMPOSITION (GENE-NORMALIZED SCALE)")
    print(f" • Total Loss (base + im):      {rec.mean('loss__total'):.4f}")
    print(f" • Pure Reconstruction Loss:    {rec.mean('loss__recon'):.4f} (Target: ~5.0 - 15.0)")
    print(f" • Anchor Regularization Loss:  {rec.mean('loss__anc'):.4f}")
    print(f" • Orthogonality Loss:          {rec.mean('loss__ortho'):.4f}")
    print(f" • Information Maximization:    {rec.mean('loss__im'):.4f}")
    print(f" • Tsallis Entropy Metric:      {rec.mean('loss__tsallis'):.4f}")

    print_hdr("C. BACKPROPAGATION AUTOGRAD CHAIN")
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

    print_hdr("D. DECOUPLED CLIPPING & ADAMW DYNAMICS")
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

    print_hdr("E. TOTAL PARAMETER DISPLACEMENT OVER FULL AUDIT")
    print(f" • Cumulative Logit Matrix L2 Drift: ||ΔW||_2 = {delta_logits:.6e}")
    print(f" • Maximum Single Coordinate Shift:  max|Δw|  = {max_logit_delta:.6e}")
    print("=" * 118 + "\n")

    return rec


# =============================================================================
# 5. TRAJECTORY AUTOPSY & HISTORICAL CHECKPOINT ENGINE
# =============================================================================

class TrajectoryAutopsy:
    """Evaluates checkpoint trajectories, mathematical conditioning, and failure modes."""
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        self.ckpt_files = self._discover_checkpoints()
        if not self.ckpt_files:
            raise ValueError(f"No valid .pt checkpoints found in {self.checkpoint_dir}")

        print(f"[i] Found {len(self.ckpt_files)} checkpoints in: {self.checkpoint_dir}")
        self.history: List[Dict[str, Any]] = []

    def _discover_checkpoints(self) -> List[Path]:
        files = list(self.checkpoint_dir.glob("epoch_*.pt")) + list(self.checkpoint_dir.glob("*.pt"))
        files = list(set(files))

        def extract_epoch(p: Path) -> int:
            match = re.search(r'epoch_?(\d+)', p.name)
            if match:
                return int(match.group(1))
            match_num = re.search(r'(\d+)', p.name)
            return int(match_num.group(1)) if match_num else 0

        valid_files = [f for f in files if re.search(r'\d+', f.stem)]
        return sorted(valid_files, key=extract_epoch)

    def run_autopsy(self) -> pd.DataFrame:
        """Parses weights, losses, SVD ranks, and attention parameters across all epochs."""
        print("[➤] Extracting trajectory metrics and tensor dynamics...")
        first_anchor_logits: Optional[torch.Tensor] = None

        for ckpt_path in self.ckpt_files:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            metrics = ckpt.get("metrics", {})
            epoch = ckpt.get("epoch", int(re.search(r'\d+', ckpt_path.stem).group(0)))

            entry: Dict[str, Any] = {"epoch": epoch, "file": ckpt_path.name}

            # Losses and training telemetry
            entry["train_loss"] = metrics.get("train_loss", np.nan)
            entry["val_loss"] = metrics.get("val_loss", np.nan)
            loss_components = metrics.get("loss_components", {})
            entry["l_rec"] = loss_components.get("rec", np.nan)
            entry["l_anc"] = loss_components.get("anc", np.nan)
            entry["l_ort"] = loss_components.get("ort", np.nan)

            entry["entropy"] = metrics.get("entropy", np.nan)
            entry["collapse_ratio"] = metrics.get("collapse_ratio", np.nan)
            entry["kl_weight"] = metrics.get("kl_weight", np.nan)
            entry["g_w"] = metrics.get("g_w", np.nan)
            entry["p_w"] = metrics.get("p_w", np.nan)
            entry["top_topic_pct"] = metrics.get("top_topic_pct", np.nan)
            entry["top_topic_id"] = metrics.get("top_topic_id", np.nan)

            # Dictionary geometry & Gram matrix
            if "topic_gene_logits" in state_dict:
                t_logits = state_dict["topic_gene_logits"].float()
                K, G = t_logits.shape
                entry["K_topics"] = K
                entry["G_genes"] = G

                d_temp = state_dict.get("dict_temp", torch.tensor(0.30)).float().clamp(0.25, 1.0).item()
                entry["dict_temp"] = d_temp

                anchors = F.softmax(t_logits / d_temp, dim=-1)
                anchor_peaks = anchors.max(dim=1).values
                entry["anchor_peak_mean"] = anchor_peaks.mean().item() * 100.0
                entry["anchor_peak_min"] = anchor_peaks.min().item() * 100.0

                gene_ent = -(anchors * torch.log(anchors + 1e-9)).sum(dim=1)
                entry["gene_entropy_mean"] = gene_ent.mean().item()

                anc_norm = F.normalize(anchors, p=2, dim=1)
                gram = torch.mm(anc_norm, anc_norm.t())
                eye_mask = 1.0 - torch.eye(K, dtype=torch.float32)
                off_diag_gram = gram * eye_mask

                entry["topic_max_overlap"] = off_diag_gram.max().item()
                entry["topic_mean_overlap"] = (off_diag_gram.sum() / max(1, K * (K - 1))).item()
                entry["topic_effective_rank"] = matrix_effective_rank(anc_norm)
                entry["topic_effective_rank_ratio"] = entry["topic_effective_rank"] / K

                if first_anchor_logits is None:
                    first_anchor_logits = t_logits.clone()
                    entry["anchor_drift"] = 0.0
                else:
                    first_anc_norm = F.normalize(F.softmax(first_anchor_logits / d_temp, dim=-1), p=2, dim=1)
                    cos_drift = 1.0 - (anc_norm * first_anc_norm).sum(dim=1).mean().item()
                    entry["anchor_drift"] = cos_drift

            # Spatial bridge
            if "spatial_bridge.3.weight" in state_dict:
                entry["spatial_bridge_norm"] = torch.norm(state_dict["spatial_bridge.3.weight"].float()).item()
            elif "spatial_bridge.0.weight" in state_dict:
                entry["spatial_bridge_norm"] = torch.norm(state_dict["spatial_bridge.0.weight"].float()).item()
            else:
                entry["spatial_bridge_norm"] = 0.0

            # GATv2 & Bilateral physics
            if "att_temp" in state_dict:
                tau = F.softplus(state_dict["att_temp"].float()).clamp(min=0.05).item()
                entry["gat_att_temp"] = tau

            if "gamma" in state_dict:
                gamma_decay = F.softplus(state_dict["gamma"].float()).item()
                entry["gamma_decay"] = gamma_decay

            if "alpha_proj.weight" in state_dict:
                entry["alpha_proj_norm"] = torch.norm(state_dict["alpha_proj.weight"].float()).item()
                if "alpha_proj.bias" in state_dict:
                    base_alpha = torch.sigmoid(state_dict["alpha_proj.bias"].float()).item() * 0.85 + 0.10
                    entry["appnp_base_alpha"] = base_alpha

            for layer_k in ["gat_w_src.weight", "gat_w_dst.weight", "gat_w_edge.weight", "gat_a.weight", "mp_update.weight"]:
                if layer_k in state_dict:
                    entry[layer_k.replace(".weight", "_norm")] = torch.norm(state_dict[layer_k].float()).item()

            # Cross-Attention Conditioning & Gating
            for proj_k in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]:
                if proj_k in state_dict:
                    mat = state_dict[proj_k].float()
                    entry[proj_k.replace(".weight", "_norm")] = torch.norm(mat).item()
                    entry[proj_k.replace(".weight", "_cond")] = matrix_condition_number(mat)
                    entry[proj_k.replace(".weight", "_eff_rank")] = matrix_effective_rank(mat)

            if "context_gate.0.weight" in state_dict and "id_enc.0.weight" in state_dict:
                gate_norm = torch.norm(state_dict["context_gate.0.weight"].float()).item()
                id_norm = torch.norm(state_dict["id_enc.0.weight"].float()).item()
                entry["gate_to_id_ratio"] = gate_norm / max(1e-9, id_norm)

            if "dynamic_w_ema" in state_dict:
                entry["dynamic_w_ema"] = state_dict["dynamic_w_ema"].float().item()

            self.history.append(entry)

        df = pd.DataFrame(self.history).sort_values("epoch").reset_index(drop=True)
        return df

    def print_health_report(self, df: pd.DataFrame) -> None:
        """Evaluates architectural anomalies and prints diagnostic scorecard."""
        latest = df.iloc[-1]
        earliest = df.iloc[0]

        print("\n" + "=" * 90)
        print("               LIBELLA TRAINING TRAJECTORY AUTOPSY REPORT")
        print(f"      Evaluated {len(df)} Checkpoints from Epoch {int(earliest['epoch'])} to {int(latest['epoch'])}")
        print("=" * 90)

        print("\n[1] METRIC & PARAMETER TRAJECTORY SUMMARY:")
        summary_cols = [
            ("Reconstruction Loss", "l_rec", ".4f"),
            ("Val Loss", "val_loss", ".4f"),
            ("Cell Sharpness (P_W %)", "p_w", ".1f"),
            ("Gene Peak (G_W %)", "g_w", ".1f"),
            ("Topic Max Overlap", "topic_max_overlap", ".3f"),
            ("Topic Effective Rank", "topic_effective_rank", ".1f"),
            ("Anchor Drift from T0", "anchor_drift", ".4f"),
            ("GAT Att Temp (Tau)", "gat_att_temp", ".3f"),
            ("Edge Decay (Gamma)", "gamma_decay", ".3f"),
            ("Gate-to-ID Ratio", "gate_to_id_ratio", ".3f"),
            ("Top Topic Utilization", "top_topic_pct", ".1f"),
        ]

        print(f"{'Metric':<28} | {'Epoch ' + str(int(earliest['epoch'])):<12} | {'Epoch ' + str(int(latest['epoch'])):<12} | {'Delta':<12} | {'Status'}")
        print("-" * 90)

        for name, col, fmt in summary_cols:
            if col in df.columns and not df[col].isna().all():
                v0 = earliest[col]
                v1 = latest[col]
                delta = v1 - v0

                status = "✓ OK"
                if col == "topic_max_overlap" and v1 > 0.40:
                    status = "⚠️ REDUNDANT"
                elif col == "topic_effective_rank" and "K_topics" in latest and (v1 / latest["K_topics"]) < 0.40:
                    status = "🚨 COLLAPSED"
                elif col == "gat_att_temp" and (v1 < 0.06 or v1 > 5.0):
                    status = "⚠️ EXTREME TAU"
                elif col == "gamma_decay" and v1 < 0.05:
                    status = "⚠️ UNIFORM GRAPH"
                elif col == "gate_to_id_ratio" and (v1 < 0.05 or v1 > 10.0):
                    status = "⚠️ UNBALANCED GATE"
                elif col == "top_topic_pct" and v1 > 35.0:
                    status = "🚨 DOMINATED"
                elif col == "p_w" and v1 < 45.0:
                    status = "⚠️ BLURRY CELLS"

                print(f"{name:<28} | {format(v0, fmt):<12} | {format(v1, fmt):<12} | {format(delta, fmt):<12} | {status}")

        print("-" * 90)

        print("\n[2] COMPONENT HEALTH & ANOMALY AUDIT:")
        issues = []
        strengths = []

        if "val_loss" in df.columns and not df["val_loss"].isna().all():
            val_delta = latest["val_loss"] - df["val_loss"].min()
            if val_delta > 0.05:
                issues.append(f"[OVERFITTING] Val loss increased by +{val_delta:.4f} from best checkpoint.")
            else:
                strengths.append("[GENERALIZATION] Validation loss tracked stably with training loss.")

        if "topic_max_overlap" in latest and latest["topic_max_overlap"] > 0.35:
            issues.append(f"[TOPIC COLLAPSE] High maximum topic collinearity detected ({latest['topic_max_overlap']:.3f} > 0.35). Some metaprograms are redundant.")
        else:
            strengths.append(f"[ORTHOGONALITY] Topic dictionary is well-separated (Max overlap: {latest.get('topic_max_overlap', 0):.3f}).")

        if "topic_effective_rank_ratio" in latest:
            ratio = latest["topic_effective_rank_ratio"]
            if ratio < 0.50:
                issues.append(f"[DIMENSIONALITY] Dictionary effective rank ratio is {ratio:.1%} (< 50%). Model is under-utilizing K topic capacity.")
            else:
                strengths.append(f"[RANK] Effective topic utilization is healthy ({ratio:.1%} of full K capacity).")

        if "gat_att_temp" in latest:
            tau = latest["gat_att_temp"]
            if tau < 0.08:
                issues.append(f"[GAT ATTENTION FREEZE] Attention temperature tau={tau:.3f} is near minimum clamp.")
            elif tau > 3.0:
                issues.append(f"[GAT ATTENTION DILUTION] Attention temperature tau={tau:.3f} is diffuse. Graph edges approach uniform average.")
            else:
                strengths.append(f"[GAT DYNAMICS] Attention temperature is in optimal regime (tau={tau:.3f}).")

        if "gamma_decay" in latest:
            gamma = latest["gamma_decay"]
            if gamma < 0.10:
                issues.append(f"[SPATIAL BLUR] Softplus(gamma)={gamma:.3f} is low; bilateral decay is barely pruning transcriptomic edges.")
            elif gamma > 10.0:
                issues.append(f"[GRAPH SEVERING] Softplus(gamma)={gamma:.3f} is aggressive; spatial neighbors are excessively downweighted.")
            else:
                strengths.append(f"[BILATERAL GRAPH] Spatial decay parameter gamma={gamma:.3f} maintains structured graph propagation.")

        if "gate_to_id_ratio" in latest:
            ratio = latest["gate_to_id_ratio"]
            if ratio < 0.10:
                issues.append(f"[CONTEXT GATE SUPPRESSED] Gate-to-ID ratio is {ratio:.3f}. Spatial context is being largely ignored.")
            elif ratio > 5.0:
                issues.append(f"[IDENTITY DROWNED] Gate-to-ID ratio is {ratio:.3f}. Spatial neighborhood dominates intrinsic state.")
            else:
                strengths.append(f"[CROSS-ATTENTION] Balanced Context Gate vs Intrinsic Identity (Ratio: {ratio:.3f}).")

        if "q_proj_cond" in latest and latest["q_proj_cond"] > 100.0:
            issues.append(f"[ILL-CONDITIONED Q] Query projection condition number is high ({latest['q_proj_cond']:.1f}).")

        for s in strengths:
            print(f"  [✓ PASS] {s}")
        if not issues:
            print("  [✓ EXCELLENT] No critical training dysfunctions or collapses detected.")
        else:
            for iss in issues:
                print(f"  [!] {iss}")

        print("=" * 90 + "\n")

    def generate_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """Generates a 12-panel publication-quality diagnostic figure."""
        print(f"[➤] Rendering 12-panel visual dashboard: {output_path.name}...")
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.28)
        epochs = df["epoch"].to_numpy()

        # 1. Loss Dynamics
        ax1 = fig.add_subplot(gs[0, 0])
        if "train_loss" in df and not df["train_loss"].isna().all():
            ax1.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        if "val_loss" in df and not df["val_loss"].isna().all():
            ax1.plot(epochs, df["val_loss"], label="Val Loss", color="#d62728", lw=2, ls="--")
        if "l_rec" in df and not df["l_rec"].isna().all():
            ax1.plot(epochs, df["l_rec"], label="Pure Recon (L_rec)", color="#2ca02c", lw=1.5, ls=":")
        ax1.set_title("1. Loss & Reconstruction Trajectory", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")

        # 2. Regularization Balances
        ax2 = fig.add_subplot(gs[0, 1])
        if "l_anc" in df and not df["l_anc"].isna().all():
            ax2.plot(epochs, df["l_anc"], label="Anchor Loss (L_anc)", color="#9467bd", lw=1.8)
        if "l_ort" in df and not df["l_ort"].isna().all():
            ax2.plot(epochs, df["l_ort"], label="Ortho Penalty (L_ort)", color="#8c564b", lw=1.8)
        if "kl_weight" in df and not df["kl_weight"].isna().all():
            ax2_twin = ax2.twinx()
            ax2_twin.plot(epochs, df["kl_weight"], label="Dynamic KL Weight", color="#e377c2", lw=1.5, ls="--")
            ax2_twin.set_ylabel("KL Weight", color="#e377c2")
            ax2_twin.grid(False)
        ax2.set_title("2. Regularization & Loss Anchoring", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Penalty Scale")
        ax2.legend(loc="upper left")

        # 3. Cell & Gene Sharpness Trajectories
        ax3 = fig.add_subplot(gs[0, 2])
        if "p_w" in df and not df["p_w"].isna().all():
            ax3.plot(epochs, df["p_w"], label="Cell Sharpness (P_W)", color="#ff7f0e", lw=2.2)
        if "g_w" in df and not df["g_w"].isna().all():
            ax3.plot(epochs, df["g_w"], label="Gene Peak (G_W)", color="#17becf", lw=2.2)
        ax3.axhline(70.0, color="#ff7f0e", ls=":", alpha=0.6, label="P_W Target (70%)")
        ax3.set_title("3. Cell & Gene Sharpness Squeeze", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Max Peak Probability (%)")
        ax3.legend(loc="lower right")

        # 4. Dictionary Orthogonality
        ax4 = fig.add_subplot(gs[1, 0])
        if "topic_max_overlap" in df:
            ax4.plot(epochs, df["topic_max_overlap"], label="Max Pairwise Overlap", color="#d62728", lw=2)
        if "topic_mean_overlap" in df:
            ax4.plot(epochs, df["topic_mean_overlap"], label="Mean Overlap", color="#7f7f7f", lw=1.5)
        ax4.axhline(0.25, color="red", ls="--", alpha=0.5, label="Target Ortho Bound (0.25)")
        ax4.set_title("4. Topic Separation & Redundancy", fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Cosine Overlap")
        ax4.legend(loc="upper left")

        # 5. Effective Rank of Metaprograms
        ax5 = fig.add_subplot(gs[1, 1])
        if "topic_effective_rank" in df:
            ax5.plot(epochs, df["topic_effective_rank"], label="Effective Topic Rank", color="#2ca02c", lw=2.2)
            if "K_topics" in df:
                k_val = df["K_topics"].iloc[0]
                ax5.axhline(k_val, color="black", ls=":", alpha=0.5, label=f"Max K Capacity ({k_val})")
        ax5.set_title("5. Dictionary Dimensionality Health", fontweight="bold")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Continuous Rank")
        ax5.legend(loc="lower right")

        # 6. Prior Drift & Spatial Bridge
        ax6 = fig.add_subplot(gs[1, 2])
        if "anchor_drift" in df:
            ax6.plot(epochs, df["anchor_drift"], label="Anchor Drift from T0", color="#9467bd", lw=2)
        if "spatial_bridge_norm" in df:
            ax6_twin = ax6.twinx()
            ax6_twin.plot(epochs, df["spatial_bridge_norm"], label="Spatial Bridge ||W||", color="#bcbd22", lw=1.5, ls="--")
            ax6_twin.set_ylabel("Bridge L2 Norm", color="#bcbd22")
            ax6_twin.grid(False)
        ax6.set_title("6. Dictionary Plasticity & Spatial Shift", fontweight="bold")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("1 - Cosine(T, T0)")
        ax6.legend(loc="upper left")

        # 7. GATv2 Attention Tau & Edge Decay Gamma
        ax7 = fig.add_subplot(gs[2, 0])
        if "gat_att_temp" in df:
            ax7.plot(epochs, df["gat_att_temp"], label="GAT Temp (Tau)", color="#e377c2", lw=2)
        if "gamma_decay" in df:
            ax7.plot(epochs, df["gamma_decay"], label="Edge Decay (Gamma)", color="#8c564b", lw=2, ls="-.")
        ax7.set_title("7. GATv2 Attention & Edge Physics", fontweight="bold")
        ax7.set_xlabel("Epoch")
        ax7.set_ylabel("Value")
        ax7.legend(loc="upper right")

        # 8. Transformer Attention Projections
        ax8 = fig.add_subplot(gs[2, 1])
        if "q_proj_norm" in df:
            ax8.plot(epochs, df["q_proj_norm"], label="||Q_proj||", lw=1.8)
        if "k_proj_norm" in df:
            ax8.plot(epochs, df["k_proj_norm"], label="||K_proj||", lw=1.8)
        if "v_proj_norm" in df:
            ax8.plot(epochs, df["v_proj_norm"], label="||V_proj||", lw=1.8)
        ax8.set_title("8. Cross-Attention Projection Norms", fontweight="bold")
        ax8.set_xlabel("Epoch")
        ax8.set_ylabel("Frobenius Norm")
        ax8.legend(loc="upper left")

        # 9. Context Gate vs Single-Cell Identity
        ax9 = fig.add_subplot(gs[2, 2])
        if "gate_to_id_ratio" in df:
            ax9.plot(epochs, df["gate_to_id_ratio"], label="Gate-to-ID Ratio", color="#17becf", lw=2)
            ax9.axhline(1.0, color="gray", ls="--", alpha=0.6, label="Parity (1.0)")
            ax9.set_yscale("log")
        ax9.set_title("9. Spatial Context vs Cell Identity Gate", fontweight="bold")
        ax9.set_xlabel("Epoch")
        ax9.set_ylabel("Ratio (Log Scale)")
        ax9.legend(loc="upper right")

        # 10. Topic Dominance / Monopolization
        ax10 = fig.add_subplot(gs[3, 0])
        if "top_topic_pct" in df:
            ax10.plot(epochs, df["top_topic_pct"], label="Dominant Topic %", color="#d62728", lw=2)
            ax10.axhline(25.0, color="orange", ls=":", label="Warning (25%)")
            ax10.axhline(40.0, color="red", ls="--", label="Collapse (40%)")
        ax10.set_title("10. Topic Monopolization Risk", fontweight="bold")
        ax10.set_xlabel("Epoch")
        ax10.set_ylabel("Max Cell Share (%)")
        ax10.legend(loc="upper left")

        # 11. EMA Entropy & Collapse Ratio
        ax11 = fig.add_subplot(gs[3, 1])
        if "entropy" in df and not df["entropy"].isna().all():
            ax11.plot(epochs, df["entropy"], label="Global Topic Entropy", color="#3399e6", lw=2)
        if "collapse_ratio" in df and not df["collapse_ratio"].isna().all():
            ax11_twin = ax11.twinx()
            ax11_twin.plot(epochs, df["collapse_ratio"], label="Collapse Ratio", color="#e63333", lw=1.5, ls="--")
            ax11_twin.set_ylabel("Collapse Ratio", color="#e63333")
            ax11_twin.grid(False)
        ax11.set_title("11. Information Entropy Dynamics", fontweight="bold")
        ax11.set_xlabel("Epoch")
        ax11.set_ylabel("Entropy (Nats)")
        ax11.legend(loc="lower left")

        # 12. Projection Head Conditioning
        ax12 = fig.add_subplot(gs[3, 2])
        if "q_proj_cond" in df:
            ax12.plot(epochs, df["q_proj_cond"], label="Cond(Q)", color="#1f77b4", lw=1.5)
        if "k_proj_cond" in df:
            ax12.plot(epochs, df["k_proj_cond"], label="Cond(K)", color="#ff7f0e", lw=1.5)
        if "v_proj_cond" in df:
            ax12.plot(epochs, df["v_proj_cond"], label="Cond(V)", color="#2ca02c", lw=1.5)
        ax12.set_yscale("log")
        ax12.set_title("12. Linear Head Conditioning", fontweight="bold")
        ax12.set_xlabel("Epoch")
        ax12.set_ylabel("Condition Number (Log)")
        ax12.legend(loc="upper left")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] Multi-panel visual report saved to: {output_path}")


# =============================================================================
# 6. MASTER EXECUTION CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Libella Master Forensic & Architectural Autopsy Pilot Engine")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forensic", "autopsy", "both"],
        default="both",
        help="Diagnostic mode: 'forensic' (live step autograd), 'autopsy' (checkpoint trajectory), or 'both'",
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks",
        help="Directory of .pt training subgraphs for forensic live audit",
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
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="/Users/Hemato/project_3/benchmark/libella_output/run/autopsy_checkpoints",
        help="Path to directory containing epoch_*.pt checkpoints",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for generated logs and figures (defaults to checkpoint parent)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif Path(args.ckpt_dir).exists():
        out_dir = Path(args.ckpt_dir).resolve().parent
    else:
        out_dir = Path("./libella_master_audit").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forensic Live Step Audit
    if args.mode in ["forensic", "both"]:
        if Path(args.chunk_dir).exists() and Path(args.common_genes).exists() and Path(args.priors).exists():
            run_forensic_audit(args.chunk_dir, args.common_genes, args.priors)
        else:
            print(f"[!] Forensic audit bypassed: One or more paths do not exist (Chunk dir, genes, or priors).")

    # 2. Checkpoint Trajectory Autopsy
    if args.mode in ["autopsy", "both"]:
        ckpt_path = Path(args.ckpt_dir)
        if ckpt_path.exists():
            report_csv = out_dir / "trajectory_autopsy_metrics.csv"
            plot_png = out_dir / "trajectory_autopsy_dashboard.png"

            autopsy = TrajectoryAutopsy(ckpt_path)
            df = autopsy.run_autopsy()

            df.to_csv(report_csv, index=False)
            print(f"[✓] Complete trajectory metrics saved to: {report_csv}")

            autopsy.print_health_report(df)
            autopsy.generate_dashboard(df, plot_png)
        else:
            print(f"[!] Autopsy audit bypassed: Checkpoint directory not found at {ckpt_path}.")


if __name__ == "__main__":
    main()