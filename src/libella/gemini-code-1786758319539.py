#!/usr/bin/env python3
"""Diagnostic script for Libella GNN: Quantifies gradient flow, anchor lock,
and parameter plasticity over 1 epoch.
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from entmax import entmax_bisect

# --- Import from your pipeline package ---
from libella.config import cfg
from libella.model import LibellaGNN
from libella.utils import get_device
from libella.data import pad_mps_shapes


def run_diagnostic(chunk_dir_path: str):
    device = get_device()
    chunk_dir = Path(chunk_dir_path)
    chunk_files = sorted(list(chunk_dir.glob("*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No .pt chunk files found in {chunk_dir}")

    print("=" * 80)
    print(f"[*] LIBELLA 1-EPOCH GRADIENT & ANCHOR PLASTICITY DIAGNOSTIC")
    print(f"[*] Device: {device} | Found {len(chunk_files)} chunk files.")
    print("=" * 80)

    # 1. Inspect first chunk to infer dimensionality
    first_chunk = torch.load(chunk_files[0], map_location="cpu", weights_only=False)
    in_channels = first_chunk["x"].shape[1]
    n_metaprograms = getattr(cfg, "k_components", 38)

    print(f"  ↳ in_channels: {in_channels} | n_metaprograms: {n_metaprograms}")

    # 2. Instantiate Model and Optimizers (1-to-1 replication)
    model = LibellaGNN(in_channels=in_channels, n_metaprograms=n_metaprograms).to(device)
    model.train()

    base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
    anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
        {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor}
    ])

    # Snapshot Initial Weights for Drift Calculation
    initial_topic_logits = model.topic_gene_logits.detach().clone()
    initial_gnn_weights = {
        name: p.detach().clone() 
        for name, p in model.named_parameters() 
        if p.requires_grad and "topic_gene_logits" not in name
    }

    # Tracking metrics across the epoch
    telemetry = {
        "grad_norm_gnn": [],
        "grad_norm_bridge": [],
        "grad_norm_topic_proj": [],
        "grad_norm_anchors": [],
        "grad_recon_norm": [],
        "grad_lock_norm": [],
        "grad_ortho_norm": [],
        "grad_ent_norm": [],
        "cos_sim_recon_vs_lock": [],
        "ste_attenuation_ratio": [],
        "effective_lr_anchors": [],
    }

    # Set active hyperparameters for Epoch 0
    model.current_scale = cfg.scale_start
    model.current_alpha = cfg.alpha_start
    model.current_temp = cfg.temp_start
    model.current_progress = 0.0

    print("\n[➤] Running 1-Epoch Forward, Gradient Decomposition & Backward Simulation...\n")

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
        # Forward Pass (Hook intermediate tensor for STE analysis)
        # -------------------------------------------------------------
        logits, anchors_raw = model.encode(x, src, dst, weights)

        # Hook gradients on anchors_raw before it hits straight-through estimator
        anchors_raw.retain_grad()

        sparse_prob = entmax_bisect(logits, alpha=model.current_alpha, dim=1)
        smooth_prob = F.softmax(logits / model.current_temp, dim=1)
        prob = (0.5 * sparse_prob) + (0.5 * smooth_prob)

        mag = x.sum(dim=1, keepdim=True)
        fracs = prob * mag

        f_train = fracs[train_idx]
        x_train = x[train_idx]
        recon = f_train @ anchors_raw

        # -------------------------------------------------------------
        # Decomposed Loss Calculation
        # -------------------------------------------------------------
        # 1. Reconstruction Loss
        num_pos = torch.clamp((x_train > 0).float().sum(), min=1.0)
        num_zeros = (x_train == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()
        is_non_zero = (x_train > 0)
        zero_mask = torch.rand_like(x_train) < 0.05
        active_mask = (is_non_zero | zero_mask).to(x_train.dtype)
        masked_w_mat = torch.where(is_non_zero, current_dynamic_w, 1.0) * active_mask

        raw_delta = recon - x_train
        asymmetry_factor = 1.0 + (is_non_zero.to(x_train.dtype) * 2.0) * (raw_delta < 0).float()
        scaled_delta = torch.clamp(raw_delta * asymmetry_factor, min=-30.0, max=30.0)
        l_recon = torch.sum(masked_w_mat * torch.log(torch.cosh(scaled_delta + 1e-6))) / max(1, x_train.shape[0])

        # 2. Anchor Lock Loss
        anc_norm = F.normalize(anchors_raw, p=2, dim=1)
        ref_probs = F.softmax(model.anchor_logits, dim=-1)
        ref_norm = F.normalize(ref_probs, p=2, dim=1)
        l_anc = 1.0 - (anc_norm * ref_norm).sum(dim=1).mean()
        recon_mag = l_recon.item()
        anc_scale = recon_mag * 0.1 * 1.0  # lock_weight=1.0 at epoch 0
        scaled_anc = l_anc * anc_scale

        # 3. Orthogonality & Entropy
        peak_excess = F.relu(anchors_raw - cfg.anchor_peak_threshold)
        collapse_penalty = (peak_excess ** 2).sum(dim=1).mean()
        gene_entropy = -(anchors_raw * torch.log(anchors_raw + 1e-9)).sum(dim=1).mean()
        raw_t_norm = F.normalize(anchors_raw, p=2, dim=-1)
        latent_ortho = torch.mm(raw_t_norm, raw_t_norm.t()) * model.ortho_mask
        l_ortho = (F.relu(latent_ortho.max(dim=1)[0] - cfg.ortho_overlap_threshold) ** 2).mean()
        scaled_ortho = (l_ortho + collapse_penalty) * cfg.ortho_weight
        scaled_gene_ent = gene_entropy * (recon_mag * 0.01)

        total_loss = l_recon + scaled_anc + scaled_ortho + scaled_gene_ent

        # -------------------------------------------------------------
        # AUTOGRAD GRADIENT DECOMPOSITION (For topic_gene_logits)
        # -------------------------------------------------------------
        g_recon = torch.autograd.grad(l_recon, model.topic_gene_logits, retain_graph=True, allow_unused=True)[0]
        g_lock = torch.autograd.grad(scaled_anc, model.topic_gene_logits, retain_graph=True, allow_unused=True)[0]
        g_ortho = torch.autograd.grad(scaled_ortho, model.topic_gene_logits, retain_graph=True, allow_unused=True)[0]
        g_ent = torch.autograd.grad(scaled_gene_ent, model.topic_gene_logits, retain_graph=True, allow_unused=True)[0]

        # Cosine similarity between reconstruction drive and anchor lock pull
        if g_recon is not None and g_lock is not None:
            flat_recon = g_recon.view(-1)
            flat_lock = g_lock.view(-1)
            cos_sim = F.cosine_similarity(flat_recon.unsqueeze(0), flat_lock.unsqueeze(0)).item()
        else:
            cos_sim = 0.0

        # Perform actual backward pass for optimizer
        total_loss.backward()

        # -------------------------------------------------------------
        # Gradient Flow Inspection Across Architecture
        # -------------------------------------------------------------
        gnn_grads = [
            p.grad.norm().item() for n, p in model.named_parameters() 
            if p.grad is not None and any(k in n for k in ["gat_", "ctx_enc", "id_enc", "q_proj", "k_proj", "v_proj"])
        ]
        bridge_grads = [
            p.grad.norm().item() for n, p in model.named_parameters() 
            if p.grad is not None and "spatial_bridge" in n
        ]
        topic_proj_grads = [
            p.grad.norm().item() for n, p in model.named_parameters() 
            if p.grad is not None and "topic_proj" in n
        ]

        # STE Attenuation: Norm at anchors_raw vs Norm at topic_gene_logits
        norm_raw = anchors_raw.grad.norm().item() if anchors_raw.grad is not None else 0.0
        norm_logits = model.topic_gene_logits.grad.norm().item() if model.topic_gene_logits.grad is not None else 0.0
        ste_ratio = (norm_logits / (norm_raw + 1e-9))

        # Record telemetry
        telemetry["grad_norm_gnn"].append(float(np.mean(gnn_grads)) if gnn_grads else 0.0)
        telemetry["grad_norm_bridge"].append(float(np.mean(bridge_grads)) if bridge_grads else 0.0)
        telemetry["grad_norm_topic_proj"].append(float(np.mean(topic_proj_grads)) if topic_proj_grads else 0.0)
        telemetry["grad_norm_anchors"].append(norm_logits)
        telemetry["grad_recon_norm"].append(g_recon.norm().item() if g_recon is not None else 0.0)
        telemetry["grad_lock_norm"].append(g_lock.norm().item() if g_lock is not None else 0.0)
        telemetry["grad_ortho_norm"].append(g_ortho.norm().item() if g_ortho is not None else 0.0)
        telemetry["grad_ent_norm"].append(g_ent.norm().item() if g_ent is not None else 0.0)
        telemetry["cos_sim_recon_vs_lock"].append(cos_sim)
        telemetry["ste_attenuation_ratio"].append(ste_ratio)

        # Apply Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        if (step + 1) % max(1, len(chunk_files) // 5) == 0 or step == len(chunk_files) - 1:
            print(
                f"  [Chunk {step+1:02d}/{len(chunk_files):02d}] "
                f"L_Recon: {l_recon.item():.4f} | "
                f"||∇_GNN||: {telemetry['grad_norm_gnn'][-1]:.4e} | "
                f"||∇_Anchors||: {telemetry['grad_norm_anchors'][-1]:.4e} | "
                f"Lock/Recon Ratio: {(telemetry['grad_lock_norm'][-1] / (telemetry['grad_recon_norm'][-1] + 1e-9)):.2f} | "
                f"Cos(g_rec, g_lock): {cos_sim:+.3f}"
            )

    # -----------------------------------------------------------------
    # Post-Epoch Parameter Drift & Rank-Order Audit
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[✓] 1-EPOCH DIAGNOSTIC SUMMARY & ROOT-CAUSE ANALYSIS")
    print("=" * 80)

    # 1. Parameter movement (L2 delta)
    delta_anchors = (model.topic_gene_logits.detach() - initial_topic_logits).norm().item()
    max_delta_anchor_elem = (model.topic_gene_logits.detach() - initial_topic_logits).abs().max().item()

    delta_gnn_list = [
        (p.detach() - initial_gnn_weights[name]).norm().item() 
        for name, p in model.named_parameters() 
        if name in initial_gnn_weights
    ]
    mean_gnn_delta = float(np.mean(delta_gnn_list)) if delta_gnn_list else 0.0

    # 2. Top-20 Gene Jaccard Index per topic
    k_init_top20 = torch.topk(initial_topic_logits, k=min(20, in_channels), dim=-1).indices.cpu().numpy()
    k_post_top20 = torch.topk(model.topic_gene_logits.detach(), k=min(20, in_channels), dim=-1).indices.cpu().numpy()

    jaccards = []
    for k in range(n_metaprograms):
        set_init = set(k_init_top20[k])
        set_post = set(k_post_top20[k])
        jaccards.append(len(set_init.intersection(set_post)) / len(set_init.union(set_post)))
    mean_jaccard = float(np.mean(jaccards))

    print(f"\n1. PARAMETER DISPLACEMENT (PLASTICITY):")
    print(f"   • Mean GNN Weights ||ΔW||_2:            {mean_gnn_delta:.6e}")
    print(f"   • Anchor Dictionary ||ΔLogits||_2:     {delta_anchors:.6e}")
    print(f"   • Max Single Logit Shift |Δw_ij|:      {max_delta_anchor_elem:.6e}")
    print(f"   • Top-20 Gene Retention (Jaccard):     {mean_jaccard * 100.0:.2f}% (100% = Completely Frozen)")

    print(f"\n2. GRADIENT FLOW MAGNITUDES (Mean over Epoch):")
    print(f"   • GNN Backbone (GAT/Encs):             {np.mean(telemetry['grad_norm_gnn']):.6e}")
    print(f"   • Spatial Bridge:                      {np.mean(telemetry['grad_norm_bridge']):.6e}")
    print(f"   • Topic Projection:                    {np.mean(telemetry['grad_norm_topic_proj']):.6e}")
    print(f"   • Anchor Logits (Net):                 {np.mean(telemetry['grad_norm_anchors']):.6e}")

    print(f"\n3. ANCHOR LOSS DECOMPOSITION & LOCK FORCES:")
    mean_g_rec = np.mean(telemetry["grad_recon_norm"])
    mean_g_lock = np.mean(telemetry["grad_lock_norm"])
    mean_g_ortho = np.mean(telemetry["grad_ortho_norm"])
    mean_cos_sim = np.mean(telemetry["cos_sim_recon_vs_lock"])
    mean_ste = np.mean(telemetry["ste_attenuation_ratio"])

    print(f"   • ||∇ L_recon||:                       {mean_g_rec:.6e}")
    print(f"   • ||∇ L_anchor_lock||:                 {mean_g_lock:.6e}")
    print(f"   • ||∇ L_ortho||:                       {mean_g_ortho:.6e}")
    print(f"   • Lock-to-Recon Gradient Ratio:        {(mean_g_lock / (mean_g_rec + 1e-9)):.2f}x")
    print(f"   • Cosine Sim (g_recon vs g_lock):      {mean_cos_sim:+.4f} (Negative = Direct Cancellation)")
    print(f"   • Softmax/STE Gradient Attenuation:    {mean_ste:.6e}x (Output-to-Input gradient ratio)")

    # 4. Actionable Diagnostic Verdict
    print(f"\n4. VERDICT & EXACT CULPRIT:")
    if mean_jaccard >= 0.999 or delta_anchors < 1e-4:
        print("   ❌ ANCHOR DICTIONARY IS SEVERELY FROZEN.")
        if mean_ste < 1e-3:
            print("   ↳ CULPRIT 1 (Softmax Jacobian Vanishing): `dynamic_logits` Softmax over large gene dimension is scaling down gradients by ~1e-4.")
        if (mean_g_lock / (mean_g_rec + 1e-9)) > 1.5 and mean_cos_sim < -0.5:
            print("   ↳ CULPRIT 2 (Anchor Lock Cancellation): `scaled_anc` has a higher gradient than `l_recon` and directly opposes reconstruction updates.")
        if cfg.lr_anchor < 1e-4:
            print(f"   ↳ CULPRIT 3 (Tiny lr_anchor): `cfg.lr_anchor` ({cfg.lr_anchor}) is too low relative to gradient magnitude.")
    else:
        print("   ✅ ANCHORS ARE ACTIVELY LEARNING. Significant gene displacement observed.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    default_path = "/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks"
    target_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    run_diagnostic(target_path)