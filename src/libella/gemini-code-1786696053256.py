#!/usr/bin/env python3
"""Libella Forensic Autograd Diagnostic:
Pins down the exact mathematical chokepoints in gradient flow,
Softmax Jacobian attenuation, clipping starvation, and AdamW weight decay.
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from entmax import entmax_bisect

# --- Import pipeline components ---
from libella.config import cfg
from libella.model import LibellaGNN
from libella.utils import get_device, scatter_softmax
from libella.data import pad_mps_shapes


def run_chokepoint_diagnostics(chunk_dir_path: str):
    device = get_device()
    chunk_dir = Path(chunk_dir_path)
    chunk_files = sorted(list(chunk_dir.glob("*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found at: {chunk_dir}")

    print("=" * 90)
    print(f"🔬 LIBELLA FORENSIC GRADIENT FLOW & CHOKEPOINT AUDIT")
    print(f"   Target Device: {device} | Found {len(chunk_files)} cache chunks")
    print("=" * 90)

    # 1. Infer dimensions from first chunk
    first_chunk = torch.load(chunk_files[0], map_location="cpu", weights_only=False)
    in_channels = first_chunk["x"].shape[1]
    n_metaprograms = getattr(cfg, "k_components", 38)

    # 2. Instantiate Model
    model = LibellaGNN(in_channels=in_channels, n_metaprograms=n_metaprograms).to(device)
    model.train()

    base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
    anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
        {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor}
    ])

    initial_logits = model.topic_gene_logits.detach().clone()

    # Telemetry storage across all chunks
    chokepoint_data = {
        "l_rec": [], "l_anc": [], "l_ortho": [], "l_ent": [],
        "g_anchors_raw": [],
        "g_dynamic_logits": [],
        "g_topic_gene_logits": [],
        "g_spatial_bridge": [],
        "g_gnn_backbone": [],
        "g_total_model": [],
        "ste_jacobian_loss_ratio": [],
        "clip_scale_factor": [],
        "g_post_clip_anchor": [],
        "adamw_grad_step_norm": [],
        "adamw_wd_step_norm": [],
        "wd_to_grad_force_ratio": [],
        "cos_sim_grad_vs_wd": [],
    }

    print("\n[➤] Simulating 1 Epoch with Sub-Layer Gradient Hooks...\n")

    for step, chunk_path in enumerate(chunk_files):
        batch = torch.load(chunk_path, map_location="cpu", weights_only=False)

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
        if device.type != 'mps':
            src = src.to(torch.int64)
            dst = dst.to(torch.int64)

        optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------------------
        # FORWARD PASS WITH INTERMEDIATE TENSOR HOOKS
        # -------------------------------------------------------------
        h_id = model.id_enc(x)
        h_0 = model.lin_appnp(model.ctx_enc(x))

        macro_ctx = h_0.mean(dim=0)
        dict_shift = torch.tanh(model.spatial_bridge(macro_ctx)) * 2.0
        dict_shift.retain_grad()

        dynamic_logits = model.topic_gene_logits + dict_shift.view(model.n_metaprograms, -1)
        dynamic_logits.retain_grad()

        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        safe_temp = torch.clamp(getattr(model, 'dict_temp', torch.tensor(0.30, device=device)), min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)

        # Straight-Through Estimator (STE)
        anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()
        anchors_raw.retain_grad()

        # Graph Message Passing
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

        idx_dtype = src.dtype if len(src) > 0 else (torch.int32 if x.device.type == 'mps' else torch.int64)
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
        logits = base_logits * getattr(model, 'current_scale', cfg.scale_start)

        current_alpha = getattr(model, 'current_alpha', cfg.alpha_start)
        current_temp = getattr(model, 'current_temp', cfg.temp_start)
        sparse_prob = entmax_bisect(logits, alpha=current_alpha, dim=1)
        smooth_prob = F.softmax(logits / current_temp, dim=1)
        prob = (0.5 * sparse_prob) + (0.5 * smooth_prob)

        fracs = prob * x.sum(dim=1, keepdim=True)
        f_train = fracs[train_idx]
        x_train = x[train_idx]
        recon = f_train @ anchors_raw

        # -------------------------------------------------------------
        # LOSS FORMULATION
        # -------------------------------------------------------------
        num_pos = torch.clamp((x_train > 0).float().sum(), min=1.0)
        current_dynamic_w = ((x_train == 0).float().sum() / num_pos).detach()
        is_non_zero = (x_train > 0)
        active_mask = (is_non_zero | (torch.rand_like(x_train) < 0.05)).to(x_train.dtype)
        masked_w = torch.where(is_non_zero, current_dynamic_w, 1.0) * active_mask

        raw_delta = recon - x_train
        asym = 1.0 + (is_non_zero.to(x_train.dtype) * 2.0) * (raw_delta < 0).float()
        l_recon = torch.sum(masked_w * torch.log(torch.cosh(torch.clamp(raw_delta * asym, -30.0, 30.0) + 1e-6))) / max(1, x_train.shape[0])

        anc_norm = F.normalize(anchors_raw, p=2, dim=1)
        with torch.no_grad():
            ref_probs = F.softmax(model.anchor_logits / safe_temp, dim=-1)
            ref_norm = F.normalize(ref_probs, p=2, dim=1)
        l_anc = (1.0 - (anc_norm * ref_norm).sum(dim=1).mean()) * (l_recon.item() * 0.1)

        peak_excess = F.relu(anchors_raw - cfg.anchor_peak_threshold)
        collapse_penalty = (peak_excess ** 2).sum(dim=1).mean()
        gene_entropy = -(anchors_raw * torch.log(anchors_raw + 1e-9)).sum(dim=1).mean()
        raw_t_norm = F.normalize(anchors_raw, p=2, dim=-1)
        latent_ortho = torch.mm(raw_t_norm, raw_t_norm.t()) * model.ortho_mask
        l_ortho = ((F.relu(latent_ortho.max(dim=1)[0] - cfg.ortho_overlap_threshold) ** 2).mean() + collapse_penalty) * cfg.ortho_weight
        scaled_gene_ent = gene_entropy * (l_recon.item() * 0.01)

        total_loss = l_recon + l_anc + l_ortho + scaled_gene_ent

        # Backward Pass
        total_loss.backward()

        # -------------------------------------------------------------
        # CHOKEPOINT 1: BACKWARD CHAIN TENSOR GRADIENTS
        # -------------------------------------------------------------
        g_raw = anchors_raw.grad.norm().item() if anchors_raw.grad is not None else 0.0
        g_dyn = dynamic_logits.grad.norm().item() if dynamic_logits.grad is not None else 0.0
        g_anchor_param = model.topic_gene_logits.grad.norm().item() if model.topic_gene_logits.grad is not None else 0.0
        g_bridge = dict_shift.grad.norm().item() if dict_shift.grad is not None else 0.0

        gnn_backbone_grads = [
            p.grad.norm().item() for n, p in model.named_parameters()
            if p.grad is not None and any(k in n for k in ["gat_", "ctx_enc", "id_enc", "q_proj", "k_proj", "v_proj", "topic_proj"])
        ]
        g_gnn = float(np.mean(gnn_backbone_grads)) if gnn_backbone_grads else 0.0

        # STE Jacobian Attenuation Ratio: ||∇ dynamic_logits|| / ||∇ anchors_raw||
        ste_ratio = g_dyn / (g_raw + 1e-9)

        # -------------------------------------------------------------
        # CHOKEPOINT 2: GRADIENT CLIPPING SQUEEZE
        # -------------------------------------------------------------
        all_grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in all_grads]), 2).item()
        clip_scale = min(1.0, cfg.grad_clip / (total_norm + 1e-6))
        post_clip_anchor_grad = g_anchor_param * clip_scale

        # -------------------------------------------------------------
        # CHOKEPOINT 3: ADAMW MECHANICS & WEIGHT DECAY FORCE
        # -------------------------------------------------------------
        # Calculate true AdamW step decomposition on topic_gene_logits
        lr = cfg.lr_anchor
        wd = cfg.wd_anchor
        param_data = model.topic_gene_logits.detach()
        grad_data = model.topic_gene_logits.grad.detach() * clip_scale

        # Gradient force vector: ~ lr * grad
        grad_step = -lr * grad_data
        # Weight decay force vector: ~ -lr * wd * weight
        wd_step = -lr * wd * param_data

        norm_grad_step = grad_step.norm().item()
        norm_wd_step = wd_step.norm().item()
        wd_to_grad_ratio = norm_wd_step / (norm_grad_step + 1e-9)

        cos_grad_wd = F.cosine_similarity(grad_step.view(1, -1), wd_step.view(1, -1)).item()

        # Record
        chokepoint_data["l_rec"].append(l_recon.item())
        chokepoint_data["l_anc"].append(l_anc.item())
        chokepoint_data["l_ortho"].append(l_ortho.item())
        chokepoint_data["l_ent"].append(scaled_gene_ent.item())
        chokepoint_data["g_anchors_raw"].append(g_raw)
        chokepoint_data["g_dynamic_logits"].append(g_dyn)
        chokepoint_data["g_topic_gene_logits"].append(g_anchor_param)
        chokepoint_data["g_spatial_bridge"].append(g_bridge)
        chokepoint_data["g_gnn_backbone"].append(g_gnn)
        chokepoint_data["g_total_model"].append(total_norm)
        chokepoint_data["ste_jacobian_loss_ratio"].append(ste_ratio)
        chokepoint_data["clip_scale_factor"].append(clip_scale)
        chokepoint_data["g_post_clip_anchor"].append(post_clip_anchor_grad)
        chokepoint_data["adamw_grad_step_norm"].append(norm_grad_step)
        chokepoint_data["adamw_wd_step_norm"].append(norm_wd_step)
        chokepoint_data["wd_to_grad_force_ratio"].append(wd_to_grad_ratio)
        chokepoint_data["cos_sim_grad_vs_wd"].append(cos_grad_wd)

        # Apply Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        if (step + 1) % max(1, len(chunk_files) // 5) == 0 or step == len(chunk_files) - 1:
            print(
                f"  [Chunk {step+1:02d}/{len(chunk_files):02d}] "
                f"||∇ Raw Anchors||: {g_raw:.2e} ➔ "
                f"||∇ Dyn Logits||: {g_dyn:.2e} ➔ "
                f"Clip Scale: {clip_scale:.4f} ➔ "
                f"WD/Grad Force: {wd_to_grad_ratio:.2f}x"
            )

    # -------------------------------------------------------------
    # STATISTICAL SYNTHESIS & AUDIT REPORT
    # -------------------------------------------------------------
    delta_logits = (model.topic_gene_logits.detach() - initial_logits).norm().item()
    max_logit_delta = (model.topic_gene_logits.detach() - initial_logits).abs().max().item()

    print("\n" + "=" * 90)
    print("📊 CHOKEPOINT AUDIT RESULTS ACROSS 1 FULL EPOCH")
    print("=" * 90)

    print("\n1. AUTOGRAD BACKPROPAGATION CHAIN (Gradient Flow across layers):")
    print(f"   ① Loss Gradient on Anchors:  ||∂L / ∂(anchors_raw)||      = {np.mean(chokepoint_data['g_anchors_raw']):.6e}")
    print(f"   ② Softmax STE Output:        ||∂L / ∂(dynamic_logits)||    = {np.mean(chokepoint_data['g_dynamic_logits']):.6e}")
    print(f"   ③ Dictionary Param Gradient: ||∂L / ∂(topic_gene_logits)|| = {np.mean(chokepoint_data['g_topic_gene_logits']):.6e}")
    print(f"   ④ Spatial Bridge Gradient:   ||∂L / ∂(dict_shift)||        = {np.mean(chokepoint_data['g_spatial_bridge']):.6e}")
    print(f"   ⑤ GNN Backbone Mean Grad:    ||∂L / ∂(GNN_Weights)||       = {np.mean(chokepoint_data['g_gnn_backbone']):.6e}")

    print("\n2. CHOKEPOINT ANALYSIS:")
    ste_loss = np.mean(chokepoint_data['ste_jacobian_loss_ratio'])
    clip_factor = np.mean(chokepoint_data['clip_scale_factor'])
    total_model_norm = np.mean(chokepoint_data['g_total_model'])
    wd_ratio = np.mean(chokepoint_data['wd_to_grad_force_ratio'])
    cos_wd = np.mean(chokepoint_data['cos_sim_grad_vs_wd'])

    print(f"   • Chokepoint A (Softmax STE Attenuation): {ste_loss:.6e}x")
    print(f"     ↳ {('🔴 SEVERE: Softmax Jacobian is shrinking gradients by ' + f'{1.0/ste_loss:.0f}x!') if ste_loss < 1e-2 else '🟢 HEALTHY'}")

    print(f"   • Chokepoint B (Global Clipping Starvation): Scale Factor = {clip_factor:.6e} (Total Model Norm = {total_model_norm:.2f})")
    print(f"     ↳ {('🔴 SEVERE: Large GNN/Bridge grads trigger clip_grad_norm, crushing anchor grad to ' + f'{clip_factor*100:.2f}% of original!') if clip_factor < 0.1 else '🟢 HEALTHY'}")

    print(f"   • Chokepoint C (AdamW Weight Decay Cannibalization): Force Ratio = {wd_ratio:.2f}x (Cos Sim = {cos_wd:+.3f})")
    print(f"     ↳ {('🔴 SEVERE: Weight decay force is larger than data gradient force! It is actively erasing marker gene logits to 0.0.') if wd_ratio > 1.0 else '🟢 HEALTHY'}")

    print("\n3. TOTAL LOGIT DISPLACEMENT OVER 1 EPOCH:")
    print(f"   • Total L2 Matrix Drift:    ||ΔW||_2 = {delta_logits:.6e}")
    print(f"   • Max Single Logit Shift:   max|Δw|  = {max_logit_delta:.6e}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    import sys
    default_path = "/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks"
    target_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    run_chokepoint_diagnostics(target_path)