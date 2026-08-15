#!/usr/bin/env python3
"""Libella Forensic Autograd Diagnostic:
Exact mathematical audit of gradient flow, Softmax Jacobian attenuation,
decoupled gradient clipping isolation, and true AdamW second-moment weight decay force.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from entmax import entmax_bisect

# --- Libella Pipeline Imports ---
from libella.config import cfg
from libella.data import make_meta_batches, pad_mps_shapes
from libella.model import LibellaGNN
from libella.utils import get_device, PhaseTracker, scatter_softmax


def robust_load_artifact(file_path: Path | str) -> Any:
    """Universal loader supporting torch.save, pickle, joblib, and numpy formats."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # 1. Try PyTorch loader (handles torch.save .pkl / .pt / .pth)
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        pass

    # 2. Try standard pickle
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3. Try joblib
    try:
        import joblib
        return joblib.load(p)
    except Exception:
        pass

    # 4. Try numpy
    try:
        return np.load(p, allow_pickle=True)
    except Exception:
        pass

    raise RuntimeError(f"Failed to deserialize prior artifact at: {p}")


def load_priors_and_genes(
    common_genes_path: str | Path, priors_path: str | Path
) -> tuple[list[str], np.ndarray | None, int]:
    """Load gene dictionary and cNMF spatial prior matrices."""
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


def run_chokepoint_diagnostics(
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

    print("=" * 95)
    print("🔬 LIBELLA FORENSIC AUTOGRAD & DECOUPLED CLIPPING AUDIT")
    print(f"   Target Device: {device} | Found {len(chunk_files)} Chunks")
    print(f"   Genes: {in_channels} | Metaprograms (K): {optimal_k} | Priors: {'Loaded' if init_components is not None else 'None'}")
    print("=" * 95)

    # 1. Instantiate Model with Validated Priors
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

    # 2. Build Structured Training Cache with Required Patient Metadata
    training_cache = []
    for f in chunk_files:
        patient_name = f.stem.split("_chunk_")[0] if "_chunk_" in f.stem else f.stem
        training_cache.append({
            "patient_name": patient_name,
            "chunk_file": f,
        })

    meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
    total_steps_per_epoch = len(meta_batches)
    alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9))
    ema_mean = None

    telemetry = {
        "l_rec": [], "l_anc": [], "l_ortho": [], "l_im": [],
        "g_anchors_raw": [], "g_dynamic_logits": [], "g_topic_gene_logits": [],
        "g_spatial_bridge": [], "g_gnn_total": [], "g_gnn_rms": [], "g_total_model": [],
        "ste_jacobian_loss_ratio": [],
        "clip_scale_base": [], "clip_scale_anchor": [], "g_post_clip_anchor": [],
        "adamw_true_grad_force": [], "adamw_true_wd_force": [], "wd_to_grad_force_ratio": [],
        "cos_sim_grad_vs_wd": [],
    }

    print("\n[➤] Simulating 1 Full Epoch with Decoupled Gradient Clipping & Exact AdamW Dynamics...\n")

    for step_idx, meta_meta in enumerate(meta_batches):
        optimizer.zero_grad(set_to_none=True)
        batch_chunks = [torch.load(b["chunk_file"], map_location="cpu", weights_only=False) for b in meta_meta]
        n_chunks_in_batch = len(batch_chunks)

        batch_g_raw = []
        batch_g_dyn = []
        batch_g_bridge = []

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
            # FORWARD PASS
            # -------------------------------------------------------------
            h_id = model.id_enc(x)
            h_0 = model.lin_appnp(model.ctx_enc(x))

            macro_ctx = h_0.mean(dim=0)
            dict_shift = torch.tanh(model.spatial_bridge(macro_ctx)) * 2.0
            dict_shift.retain_grad()

            dynamic_logits = model.topic_gene_logits + dict_shift.view(model.n_metaprograms, -1)
            dynamic_logits.retain_grad()

            soft_anchors = F.softmax(dynamic_logits, dim=-1)
            raw_temp = getattr(model, "dict_temp", torch.tensor(cfg.dict_temp, device=device))
            safe_temp = torch.clamp(raw_temp, min=0.25, max=1.0)
            sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)

            anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()
            anchors_raw.retain_grad()

            # Message Passing
            N = h_0.size(0)
            if len(src) > 0:
                with torch.no_grad():
                    bio_h = torch.mm(x, anchors_raw.detach().t())
                    diff = bio_h[src] - bio_h[dst]
                    dist = (diff * diff).sum(dim=1)
                decay = torch.exp(-F.softplus(model.gamma) * dist)
            else:
                decay = torch.ones_like(weights)

            W_bil = weights * decay
            alpha = torch.sigmoid(model.alpha_proj(h_0)) * 0.85 + 0.10
            inv_alpha = 1.0 - alpha
            h_0_scaled = h_0 * alpha

            h_ctx = h_0
            for _ in range(model.k_hops):
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
                agg = F.silu(model.mp_update(out))
                h_ctx = agg * inv_alpha + h_0_scaled

            Q = model.q_proj(h_id)
            K = model.k_proj(h_ctx)
            V = model.v_proj(h_ctx)

            idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x.device.type == "mps" else torch.int64)
            self_loops = torch.arange(N, dtype=idx_dtype, device=x.device)
            src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
            dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

            cross_scores = (Q[dst_with_self] * K[src_with_self]).sum(dim=-1) / (model.hidden_dim ** 0.5)
            cross_att = scatter_softmax(cross_scores, dst_with_self, N)
            ctx_pulled = torch.zeros_like(Q)
            ctx_pulled.index_add_(0, dst_with_self, (V[src_with_self] * cross_att.unsqueeze(1)).contiguous())

            h_final = h_id + model.context_gate(ctx_pulled)
            h_norm = F.normalize(model.sp_norm(h_final), p=2, dim=-1)

            t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
            x_norm = F.normalize(x, p=2, dim=-1)
            bio_sim = torch.mm(x_norm, t_proj_weights.t())

            gnn_shift_raw = model.topic_proj(h_norm)
            gnn_shift_norm = F.normalize(gnn_shift_raw, p=2, dim=-1)
            base_logits = bio_sim + (cfg.gnn_shift_weight * gnn_shift_norm)

            noise = torch.randn_like(base_logits) * cfg.train_noise
            base_logits = base_logits + noise
            logits = base_logits * model.current_scale

            sparse_prob = entmax_bisect(logits, alpha=model.current_alpha, dim=1)
            smooth_prob = F.softmax(logits / model.current_temp, dim=1)
            smooth_weight = 0.50 - (0.45 * squeeze_progress)
            sparse_weight = 1.0 - smooth_weight
            prob = (sparse_weight * sparse_prob) + (smooth_weight * smooth_prob)

            fracs = prob * x.sum(dim=1, keepdim=True)
            f_train = fracs[train_idx]
            x_train = x[train_idx]

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

            loss, _ = model.calc_loss(
                recon, x_train, anchors_raw, None,
                ep=0, total_epochs=cfg.epochs,
                f_train=f_train, target_f_dist=target_f_dist,
                kl_weight=dynamic_kl_w
            )

            scaled_loss = loss / n_chunks_in_batch
            scaled_loss.backward()

            batch_g_raw.append(anchors_raw.grad.norm().item() if anchors_raw.grad is not None else 0.0)
            batch_g_dyn.append(dynamic_logits.grad.norm().item() if dynamic_logits.grad is not None else 0.0)
            batch_g_bridge.append(dict_shift.grad.norm().item() if dict_shift.grad is not None else 0.0)

            telemetry["l_rec"].append(model._last_losses["rec"])
            telemetry["l_anc"].append(model._last_losses["anc"])
            telemetry["l_ortho"].append(model._last_losses["ort"])
            telemetry["l_im"].append(model._last_losses["im"])

        # -------------------------------------------------------------
        # ACCUMULATED GRADIENT TELEMETRY (PRE-CLIP)
        # -------------------------------------------------------------
        mean_g_raw = float(np.mean(batch_g_raw))
        mean_g_dyn = float(np.mean(batch_g_dyn))
        mean_g_bridge = float(np.mean(batch_g_bridge))

        g_anchor_param = model.topic_gene_logits.grad.norm().item() if model.topic_gene_logits.grad is not None else 0.0

        # Parameter group segregation
        base_named_params = [
            (n, p) for n, p in model.named_parameters() 
            if "topic_gene_logits" not in n and p.grad is not None
        ]
        anchor_named_params = [
            (n, p) for n, p in model.named_parameters() 
            if "topic_gene_logits" in n and p.grad is not None
        ]

        # Calculate Norms & Independent Clipping Scales
        base_grad_list = [p.grad.detach() for _, p in base_named_params]
        anchor_grad_list = [p.grad.detach() for _, p in anchor_named_params]

        base_grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in base_grad_list]), 2).item() if base_grad_list else 0.0
        anchor_grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in anchor_grad_list]), 2).item() if anchor_grad_list else 0.0
        total_model_norm = math.sqrt(base_grad_norm**2 + anchor_grad_norm**2)

        clip_scale_base = min(1.0, cfg.grad_clip / (base_grad_norm + 1e-6))
        clip_scale_anchor = min(1.0, cfg.grad_clip / (anchor_grad_norm + 1e-6))
        post_clip_anchor_grad = g_anchor_param * clip_scale_anchor

        # GNN Backbone Component Breakdown
        gnn_grads = [
            p.grad for n, p in base_named_params
            if any(k in n for k in ["gat_", "ctx_enc", "id_enc", "q_proj", "k_proj", "v_proj", "topic_proj", "spatial_bridge"])
        ]
        if gnn_grads:
            gnn_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in gnn_grads]), 2).item()
            total_elements = sum(g.numel() for g in gnn_grads)
            gnn_rms_norm = math.sqrt(sum((g.detach() ** 2).sum().item() for g in gnn_grads) / max(1, total_elements))
        else:
            gnn_total_norm, gnn_rms_norm = 0.0, 0.0

        ste_ratio = mean_g_dyn / (mean_g_raw + 1e-9)

        # -------------------------------------------------------------
        # INJECTED FIX: DECOUPLED INDEPENDENT GRADIENT CLIPPING
        # -------------------------------------------------------------
        # 1. Clip GNN backbone and projection heads
        base_params = [p for _, p in base_named_params]
        if base_params:
            torch.nn.utils.clip_grad_norm_(base_params, max_norm=cfg.grad_clip)

        # 2. Clip anchor dictionary independently so it retains its full update budget
        anchor_params = [p for _, p in anchor_named_params]
        if anchor_params:
            torch.nn.utils.clip_grad_norm_(anchor_params, max_norm=cfg.grad_clip)

        # -------------------------------------------------------------
        # EXACT ADAMW SECOND-MOMENT & WEIGHT DECAY FORCE DECOMPOSITION
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
        adam_grad_step_vec = -lr * (exp_avg / bias_correction1) / denom
        adam_wd_step_vec = -lr * wd * p_tensor.detach()

        norm_grad_force = adam_grad_step_vec.norm().item()
        norm_wd_force = adam_wd_step_vec.norm().item()
        wd_to_grad_ratio = norm_wd_force / (norm_grad_force + 1e-12)
        cos_sim_grad_wd = F.cosine_similarity(
            adam_grad_step_vec.view(1, -1), adam_wd_step_vec.view(1, -1)
        ).item()

        optimizer.step()

        telemetry["g_anchors_raw"].append(mean_g_raw)
        telemetry["g_dynamic_logits"].append(mean_g_dyn)
        telemetry["g_topic_gene_logits"].append(g_anchor_param)
        telemetry["g_spatial_bridge"].append(mean_g_bridge)
        telemetry["g_gnn_total"].append(gnn_total_norm)
        telemetry["g_gnn_rms"].append(gnn_rms_norm)
        telemetry["g_total_model"].append(total_model_norm)
        telemetry["ste_jacobian_loss_ratio"].append(ste_ratio)
        telemetry["clip_scale_base"].append(clip_scale_base)
        telemetry["clip_scale_anchor"].append(clip_scale_anchor)
        telemetry["g_post_clip_anchor"].append(post_clip_anchor_grad)
        telemetry["adamw_true_grad_force"].append(norm_grad_force)
        telemetry["adamw_true_wd_force"].append(norm_wd_force)
        telemetry["wd_to_grad_force_ratio"].append(wd_to_grad_ratio)
        telemetry["cos_sim_grad_vs_wd"].append(cos_sim_grad_wd)

        if (step_idx + 1) % max(1, len(meta_batches) // 5) == 0 or step_idx == len(meta_batches) - 1:
            print(
                f"  [Meta-Batch {step_idx+1:02d}/{len(meta_batches):02d}] "
                f"||∇ Anchors||: {mean_g_raw:.2e} ➔ "
                f"Clip (Base: {clip_scale_base:.3f}, Anchor: {clip_scale_anchor:.3f}) ➔ "
                f"AdamW Force: (Grad={norm_grad_force:.2e}, WD={norm_wd_force:.2e}, WD/Grad={wd_to_grad_ratio:.2f}x)"
            )

    # -------------------------------------------------------------
    # SYNTHESIS & AUDIT REPORT
    # -------------------------------------------------------------
    delta_logits = (model.topic_gene_logits.detach() - initial_logits).norm().item()
    max_logit_delta = (model.topic_gene_logits.detach() - initial_logits).abs().max().item()

    ste_loss_mean = float(np.mean(telemetry["ste_jacobian_loss_ratio"]))
    clip_base_mean = float(np.mean(telemetry["clip_scale_base"]))
    clip_anchor_mean = float(np.mean(telemetry["clip_scale_anchor"]))
    total_model_norm_mean = float(np.mean(telemetry["g_total_model"]))
    wd_ratio_mean = float(np.mean(telemetry["wd_to_grad_force_ratio"]))
    cos_wd_mean = float(np.mean(telemetry["cos_sim_grad_vs_wd"]))

    print("\n" + "=" * 95)
    print("📊 LIBELLA AUDIT RESULTS (DECOUPLED CLIPPING VERIFIED)")
    print("=" * 95)

    print("\n1. BACKPROPAGATION CHAIN & GRADIENT SCALING:")
    print(f"   ① ||∂L / ∂(anchors_raw)||          = {np.mean(telemetry['g_anchors_raw']):.6e}")
    print(f"   ② ||∂L / ∂(dynamic_logits)||        = {np.mean(telemetry['g_dynamic_logits']):.6e}")
    print(f"   ③ ||∂L / ∂(topic_gene_logits)||     = {np.mean(telemetry['g_topic_gene_logits']):.6e}")
    print(f"   ④ ||∂L / ∂(dict_shift)||            = {np.mean(telemetry['g_spatial_bridge']):.6e}")
    print(f"   ⑤ GNN Backbone Total Norm (L2)      = {np.mean(telemetry['g_gnn_total']):.6e} (RMS: {np.mean(telemetry['g_gnn_rms']):.6e})")

    print("\n2. CHOKEPOINT EVALUATION:")
    print(f"   • Chokepoint A (Softmax STE Attenuation Ratio): {ste_loss_mean:.6e}")
    if ste_loss_mean < 1e-3:
        print(f"     ↳ 🔴 SEVERE: Softmax Jacobian is attenuating raw gradients by {1.0/ste_loss_mean:.0f}x.")
    else:
        print("     ↳ 🟢 HEALTHY: Jacobian transmission within expected operating bounds.")

    print(f"   • Chokepoint B (Decoupled Clipping Isolation):")
    print(f"     - Base Backbone Clip Scale:   {clip_base_mean:.4f}")
    print(f"     - Anchor Logits Clip Scale:  {clip_anchor_mean:.4f} (Budget Retention: {clip_anchor_mean*100:.1f}%)")
    if clip_anchor_mean < 0.1:
        print(f"     ↳ 🔴 SEVERE: Anchor parameter updates are being clipped heavily on their own.")
    else:
        print(f"     ↳ 🟢 HEALTHY: Anchor dictionary retains update capacity independent of backbone gradient spikes.")

    print(f"   • Chokepoint C (True AdamW Step Force Ratio): WD/Grad = {wd_ratio_mean:.3f}x (Cosine Similarity = {cos_wd_mean:+.3f})")
    if wd_ratio_mean > 1.0:
        print("     ↳ 🔴 SEVERE: Weight decay displacement strictly exceeds normalized data gradient displacement.")
    else:
        print(f"     ↳ 🟢 HEALTHY: Gradient updates dominate weight decay ({1.0/max(1e-6, wd_ratio_mean):.1f}x gradient margin).")

    print("\n3. TOTAL LOGIT DISPLACEMENT OVER 1 FULL EPOCH:")
    print(f"   • Cumulative Matrix L2 Drift: ||ΔW||_2 = {delta_logits:.6e}")
    print(f"   • Max Single Logit Shift:     max|Δw|  = {max_logit_delta:.6e}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Libella Exact Autograd & Decoupled Clipping Diagnostic")
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
    run_chokepoint_diagnostics(args.chunk_dir, args.common_genes, args.priors)