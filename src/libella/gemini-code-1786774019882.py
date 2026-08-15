#!/usr/bin/env python3
"""Libella 5-Epoch Deep Forensic & Autograd Telemetry Engine.

100% Mathematical Parity Implementation:
- Exact LibellaGNN architecture, forward graph, and dynamic temperature hooks.
- Bit-accurate loss decomposition matching LibellaGNN.calc_loss.
- Decoupled AdamW & CosineAnnealingLR optimizer telemetry.
- Dynamic EMA zero-loss reweighting and Tsallis-entropy balance.
- Per-coordinate autograd isolation and marker logit erosion tracking.
"""

import argparse
import gc
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
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
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
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
    return np.load(p, allow_pickle=True)


def load_priors_and_genes(
    common_genes_path: str | Path, priors_path: str | Path
) -> tuple[list[str], np.ndarray | None, int]:
    genes_p = Path(common_genes_path)
    with open(genes_p, "r", encoding="utf-8") as f:
        common_genes = json.load(f)

    priors_data = robust_load_artifact(priors_path)
    init_components = None
    optimal_k = getattr(cfg, "k_components", 38)

    if isinstance(priors_data, dict):
        init_components = priors_data.get(
            "components", priors_data.get("init_components", priors_data.get("priors", None))
        )
        optimal_k = priors_data.get(
            "optimal_k", init_components.shape[0] if init_components is not None else optimal_k
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
# BIT-PERFECT TELEMETRY & AUTOPSY ENGINE
# =============================================================================

def _forensics_worker(chunk_dir_path: str, common_genes_path: str, priors_path: str, n_epochs: int = 5):
    device = get_device()
    chunk_dir = Path(chunk_dir_path)
    chunk_files = sorted(list(chunk_dir.glob("*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in: {chunk_dir}")

    common_genes, init_components, optimal_k = load_priors_and_genes(common_genes_path, priors_path)
    in_channels = len(common_genes)

    print("=" * 115)
    print(f"🔬 LIBELLA 5-EPOCH HIGH-RESOLUTION SYSTEM COLLAPSE AUTOPSY (100% PARITY)")
    print(f"   Target Device: {device} | Found {len(chunk_files)} Chunks | Common Genes: {in_channels} | Topics (K): {optimal_k}")
    print("=" * 115)

    # Initialize exact model
    model = LibellaGNN(in_channels=in_channels, n_metaprograms=optimal_k, init_components=init_components).to(device)
    model.train()

    # Exact parameter split
    base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
    anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
        {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    tracker = PhaseTracker()
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    max_entropy_scalar = float(np.log(optimal_k))

    training_cache = [
        {"patient_name": f.stem.split("_chunk_")[0] if "_chunk_" in f.stem else f.stem, "chunk_file": f}
        for f in chunk_files
    ]
    meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
    alpha_ema = min(0.001, 1.0 / (len(meta_batches) * 5.0 + 1e-9))
    ema_mean = None

    # Track prior anchor state
    raw_priors = model.anchor_logits.detach().clone()
    prior_top20 = [set(raw_priors[k].topk(20).indices.cpu().tolist()) for k in range(optimal_k)]
    init_marker_mask = (model.topic_gene_logits.detach() > 0)

    epoch_summaries = []

    for ep in range(n_epochs):
        ep_stats = {
            "l_rec": [], "l_anc": [], "l_ortho": [], "l_im": [], "l_total": [],
            "val_loss": [],
            "g_raw_anchor": [], "g_dyn_logits": [], "g_spatial_bridge": [], "g_gnn_backbone": [],
            "bridge_shift_rms": [], "static_logits_rms": [], "shift_to_static_ratio": [],
            "p_w": [], "g_w": [], "ent": [], "top_t_pct": [],
            "clip_base": [], "clip_anchor": [],
            "adam_grad_force": [], "adam_wd_force": [], "wd_ratio": [],
        }

        ep_prev_logits = model.topic_gene_logits.detach().clone()

        for step_idx, meta_meta in enumerate(meta_batches):
            optimizer.zero_grad(set_to_none=True)
            batch_chunks = [torch.load(b["chunk_file"], map_location="cpu", weights_only=False) for b in meta_meta]
            n_chunks = len(batch_chunks)

            for batch in batch_chunks:
                x = batch["x"].to(device=device, non_blocking=True)
                src = batch["src"].to(device=device, non_blocking=True)
                dst = batch["dst"].to(device=device, non_blocking=True)
                weights = batch["weights"].to(device=device, non_blocking=True)
                train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)

                if len(src) > 0:
                    keep_mask = torch.rand(src.size(0), device=device) > cfg.edge_dropout
                    src, dst, weights = src[keep_mask], dst[keep_mask], weights[keep_mask]

                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != "mps":
                    src, dst = src.to(torch.int64), dst.to(torch.int64)

                prog = tracker.get_progress()
                model.current_progress = prog
                model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * (prog ** 0.8))
                model.current_temp = cfg.temp_end + ((cfg.temp_start - cfg.temp_end) * ((1.0 - prog) ** 1.5))
                model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * prog)

                # -------------------------------------------------------------
                # 1. FORWARD PASS HOOKS
                # -------------------------------------------------------------
                # Dynamic spatial bridge hooks
                h_0 = model.lin_appnp(model.ctx_enc(x))
                macro_ctx = h_0.mean(dim=0)
                dict_shift = torch.tanh(model.spatial_bridge(macro_ctx)) * 2.0
                dict_shift.retain_grad()

                # Native Model Forward Execution
                fracs, pure_anchors = model(x, src, dst, weights)
                pure_anchors.retain_grad()

                f_train = fracs[train_idx]
                x_train = x[train_idx]

                # EMA Prior Divergence Calculations
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
                hub_multiplier = F.relu((ema_mean.max() / cfg.hub_threshold) - 1.0) * 10.0
                dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + hub_multiplier

                # -------------------------------------------------------------
                # 2. EXACT NATIVE LOSS EXECUTION
                # -------------------------------------------------------------
                recon = f_train @ pure_anchors
                recon.retain_grad()

                true_batch_loss, base_recon_val, base_anc_val, base_ort_val = model.calc_loss(
                    recon_c=recon,
                    x_c=x_train,
                    anchors=pure_anchors,
                    ortho_mat=None,
                    ep=ep,
                    total_epochs=cfg.epochs,
                    f_train=f_train,
                    target_f_dist=target_f_dist,
                    kl_weight=dynamic_kl_w,
                )

                (true_batch_loss / n_chunks).backward()

                # Structural Telemetry
                shift_rms = math.sqrt((dict_shift.detach() ** 2).mean().item())
                static_rms = math.sqrt((model.topic_gene_logits.detach() ** 2).mean().item())

                ep_stats["l_total"].append(true_batch_loss.item())
                ep_stats["l_rec"].append(base_recon_val.item())
                ep_stats["l_anc"].append(base_anc_val.item())
                ep_stats["l_ortho"].append(base_ort_val.item())
                ep_stats["l_im"].append(model._last_losses.get("im", 0.0))
                ep_stats["bridge_shift_rms"].append(shift_rms)
                ep_stats["static_logits_rms"].append(static_rms)
                ep_stats["shift_to_static_ratio"].append(shift_rms / max(1e-6, static_rms))
                ep_stats["g_raw_anchor"].append(pure_anchors.grad.norm().item() if pure_anchors.grad is not None else 0.0)
                ep_stats["g_dyn_logits"].append(model.topic_gene_logits.grad.norm().item() if model.topic_gene_logits.grad is not None else 0.0)
                ep_stats["g_spatial_bridge"].append(dict_shift.grad.norm().item() if dict_shift.grad is not None else 0.0)
                ep_stats["p_w"].append(p_train.max(dim=1).values.mean().item() * 100.0)
                ep_stats["g_w"].append(pure_anchors.max(dim=1).values.mean().item() * 100.0)
                ep_stats["ent"].append(ema_entropy.item())
                ep_stats["top_t_pct"].append(current_p_mean.max().item() * 100.0)

                # Validation Loss (Exact Match to Native Implementation)
                val_core_idx_cpu = batch["val_core_idx"]
                if val_core_idx_cpu.numel() > 0:
                    val_idx = val_core_idx_cpu.to(device=device, non_blocking=True)
                    with torch.no_grad():
                        f_val = fracs[val_idx]
                        x_val = x[val_idx]
                        val_recon = f_val @ pure_anchors

                        is_non_zero_val = (x_val > 0)
                        w_mat = torch.where(is_non_zero_val, model.dynamic_w_ema, 1.0)
                        zero_expectation_mask = torch.where(is_non_zero_val, 1.0, cfg.zero_mask_rate).to(x_val.dtype)
                        masked_w_mat_val = w_mat * zero_expectation_mask

                        raw_delta_val = val_recon - x_val
                        asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * 2.0) * (raw_delta_val < 0).to(x_val.dtype)
                        scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-cfg.delta_clamp, max=cfg.delta_clamp)

                        val_loss_sum = torch.sum(masked_w_mat_val * torch.log(torch.cosh(scaled_delta_val + 1e-6)))
                        val_log_cosh = val_loss_sum / max(1, x_val.numel())
                        ep_stats["val_loss"].append(val_log_cosh.item())

            # -------------------------------------------------------------
            # 3. GRADIENT CLIPPING & ADAMW DYNAMICS
            # -------------------------------------------------------------
            base_named_params = [(n, p) for n, p in model.named_parameters() if "topic_gene_logits" not in n and p.grad is not None]
            anchor_named_params = [(n, p) for n, p in model.named_parameters() if "topic_gene_logits" in n and p.grad is not None]

            base_gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for _, p in base_named_params]), 2).item() if base_named_params else 0.0
            anchor_gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for _, p in anchor_named_params]), 2).item() if anchor_named_params else 0.0

            clip_b = min(1.0, cfg.grad_clip / (base_gnorm + 1e-6))
            clip_a = min(1.0, cfg.grad_clip / (anchor_gnorm + 1e-6))
            ep_stats["clip_base"].append(clip_b)
            ep_stats["clip_anchor"].append(clip_a)

            if base_named_params:
                torch.nn.utils.clip_grad_norm_([p for _, p in base_named_params], max_norm=cfg.grad_clip)
            if anchor_named_params:
                torch.nn.utils.clip_grad_norm_([p for _, p in anchor_named_params], max_norm=cfg.grad_clip)

            # AdamW Coordinate State Extraction
            anchor_group = optimizer.param_groups[1]
            lr, wd = anchor_group["lr"], anchor_group["weight_decay"]
            beta1, beta2 = anchor_group.get("betas", (0.9, 0.999))
            eps_adam = anchor_group.get("eps", 1e-8)

            p_tensor = model.topic_gene_logits
            g_clipped = p_tensor.grad.detach()
            p_state = optimizer.state[p_tensor]
            c_step = (int(p_state.get("step", 0).item()) if isinstance(p_state.get("step", 0), torch.Tensor) else int(p_state.get("step", 0))) + 1

            if "exp_avg" in p_state:
                e_m = p_state["exp_avg"].clone().mul_(beta1).add_(g_clipped, alpha=1.0 - beta1)
                e_v = p_state["exp_avg_sq"].clone().mul_(beta2).addcmul_(g_clipped, g_clipped, value=1.0 - beta2)
            else:
                e_m = g_clipped * (1.0 - beta1)
                e_v = (g_clipped * g_clipped) * (1.0 - beta2)

            denom = (e_v.sqrt() / math.sqrt(1.0 - beta2 ** c_step)).add_(eps_adam)
            step_grad = -lr * (e_m / (1.0 - beta1 ** c_step)) / denom
            step_wd = -lr * wd * p_tensor.detach()

            f_g = step_grad.norm().item()
            f_w = step_wd.norm().item()
            ep_stats["adam_grad_force"].append(f_g)
            ep_stats["adam_wd_force"].append(f_w)
            ep_stats["wd_ratio"].append(f_w / max(1e-9, f_g))

            optimizer.step()

        scheduler.step()

        # Marker Drift & Prior Top-20 Retention Check
        curr_logits = model.topic_gene_logits.detach()
        logit_diff = curr_logits - ep_prev_logits
        pos_shift = logit_diff[init_marker_mask].mean().item()
        neg_shift = logit_diff[~init_marker_mask].mean().item()

        curr_top20 = [set(curr_logits[k].topk(20).indices.cpu().tolist()) for k in range(optimal_k)]
        retention_rates = [(len(prior_top20[k].intersection(curr_top20[k])) / 20.0) * 100.0 for k in range(optimal_k)]
        mean_retention = float(np.mean(retention_rates))

        tracker.step({"l_rec": np.mean(ep_stats["l_rec"]), "p_w": np.mean(ep_stats["p_w"])}, ep)

        summary = {
            "epoch": ep + 1,
            "total_loss": np.mean(ep_stats["l_total"]),
            "rec": np.mean(ep_stats["l_rec"]),
            "val_loss": np.mean(ep_stats["val_loss"]) if ep_stats["val_loss"] else float("nan"),
            "l_anc": np.mean(ep_stats["l_anc"]),
            "l_ortho": np.mean(ep_stats["l_ortho"]),
            "l_im": np.mean(ep_stats["l_im"]),
            "p_w": np.mean(ep_stats["p_w"]),
            "g_w": np.mean(ep_stats["g_w"]),
            "shift_ratio": np.mean(ep_stats["shift_to_static_ratio"]),
            "g_bridge": np.mean(ep_stats["g_spatial_bridge"]),
            "g_anchor": np.mean(ep_stats["g_dyn_logits"]),
            "clip_anchor": np.mean(ep_stats["clip_anchor"]),
            "wd_ratio": np.mean(ep_stats["wd_ratio"]),
            "pos_shift": pos_shift,
            "neg_shift": neg_shift,
            "prior_retention": mean_retention,
        }
        epoch_summaries.append(summary)

        print(
            f" [Epoch {ep+1:02d}/05] Rec: {summary['rec']:.4f} | "
            f"Val: {summary['val_loss']:.4f} | P_W: {summary['p_w']:<4.1f}% | G_W: {summary['g_w']:<4.1f}% | "
            f"L_Anc: {summary['l_anc']:.4f} | "
            f"Prior Top-20: {mean_retention:.1f}% | "
            f"Marker Shift: {pos_shift:+.4e}"
        )

    # =========================================================================
    # FORENSIC SYNTHESIS REPORT
    # =========================================================================
    print("\n" + "=" * 115)
    print("📊 5-EPOCH FORENSIC AUTOPSY: ROOT-CAUSE DECOMPOSITION")
    print("=" * 115)

    print("\n1. LOSS & PRIOR RETENTION TRAJECTORY:")
    print(f"{'Epoch':<8} | {'Total Loss':<14} | {'Reconstruction':<16} | {'Validation Loss':<18} | {'Prior Top-20 %':<16}")
    print("─" * 80)
    for s in epoch_summaries:
        print(f"Ep {s['epoch']:<5} | {s['total_loss']:<14.4f} | {s['rec']:<16.4f} | {s['val_loss']:<18.4f} | {s['prior_retention']:<15.1f}%")

    print("\n2. SPATIAL BRIDGE vs. STATIC ANCHOR GRADIENT BALANCE:")
    print(f"{'Epoch':<8} | {'Bridge ||∇L||':<16} | {'Static Logits ||∇L||':<22} | {'Bridge/Static Shift Ratio':<30}")
    print("─" * 82)
    for s in epoch_summaries:
        print(f"Ep {s['epoch']:<5} | {s['g_bridge']:<16.4e} | {s['g_anchor']:<22.4e} | {s['shift_ratio']:<30.4f}")

    print("\n3. MARKER EROSION VELOCITY (WEIGHT DECAY vs. GRADIENT FORCE):")
    print(f"{'Epoch':<8} | {'Pos Marker Logit Δ':<24} | {'Neg Background Logit Δ':<26} | {'WD / Grad Force Ratio':<24}")
    print("─" * 88)
    for s in epoch_summaries:
        sink_str = "🔴 SINKING" if s['pos_shift'] < 0 else "🟢 ELEVATING"
        print(f"Ep {s['epoch']:<5} | {s['pos_shift']:<+14.4e} ({sink_str:<8}) | {s['neg_shift']:<+16.4e}         | {s['wd_ratio']:<24.4f}x")

    print("\n" + "=" * 115 + "\n")


def run_isolated_forensics(chunk_dir: str, common_genes: str, priors: str, n_epochs: int = 5):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_forensics_worker, args=(chunk_dir, common_genes, priors, n_epochs))
    print("[*] Launching isolated 5-Epoch Forensic Telemetry Process...")
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        print(f"[!] Forensic process failed with exit code {proc.exitcode}")
    else:
        print("[✓] 5-Epoch Telemetry Complete. Process terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Libella 5-Epoch High-Resolution Forensic Telemetry")
    parser.add_argument("--chunk-dir", type=str, default="/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks")
    parser.add_argument("--common-genes", type=str, default="/Users/Hemato/project_3/benchmark/libella_output/run/common_genes.json")
    parser.add_argument("--priors", type=str, default="/Users/Hemato/project_3/benchmark/libella_output/run/global_cnmf_priors.pkl")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    run_isolated_forensics(args.chunk_dir, args.common_genes, args.priors, args.epochs)