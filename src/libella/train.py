"""Model training loops and orchestrators for the Spatial Ecotype GNN."""
from __future__ import annotations

import gc
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import cfg, paths
from .data import (
    SpatialBatcher,
    make_meta_batches,
    pad_mps_shapes,
    pt_to_scipy_csr,
)
from .model import LibellaGNN
from .utils import PhaseTracker, UnifiedLogger, export_latents_from_graphs, get_device



def _prep_ssd_chunks(graph_paths: list[Path]) -> list[dict[str, Any]]:
    """Slice patient graphs into SSD chunks for OOM-safe training."""
    print("  ↳ Slicing patient graphs to SSD chunks...")
    out_dir = paths.make_dirs(cfg.suffix)["out"]

    tmp_chunk_dir = out_dir / "temp_training_chunks"
    tmp_chunk_dir.mkdir(parents=True, exist_ok=True)
    training_cache: list[dict[str, Any]] = [] 

    for path in tqdm(graph_paths, desc="Slicing Sub-Graphs", leave=False):
        patient_name = path.stem.replace("_graph", "")
        
        existing_chunks = list(tmp_chunk_dir.glob(f"{patient_name}_chunk_*.pt"))
        if len(existing_chunks) > 0:
            for chunk_file in existing_chunks:
                training_cache.append({
                    "patient_name": patient_name, 
                    "chunk_file": chunk_file
                })
            continue 
            
        data = torch.load(path, map_location="cpu", weights_only=False)
        X_sp = pt_to_scipy_csr(data, "x_in")
        N_cells = X_sp.shape[0]
        
        e_attr = data.edge_attr.numpy()
        e_row = data.edge_index[0].numpy()
        e_col = data.edge_index[1].numpy()
        
        delattr(data, "edge_attr")
        delattr(data, "edge_index")
        
        adj_sp = sp.csr_matrix((e_attr, (e_row, e_col)), shape=(N_cells, N_cells))
        del e_attr, e_row, e_col
        
        batcher = SpatialBatcher(
            X=X_sp, adj=adj_sp, coords=data.pos.numpy(),
            train_mask=data.train_mask.numpy(), val_mask=data.val_mask.numpy(), 
            batch_size=cfg.batch_size, k_hops=cfg.k_hops, shuffle=True
        )
        
        for chunk_idx, core_idx in enumerate(batcher.chunks):
            if data.train_mask.numpy()[core_idx].sum() > 0:
                chunk_data = batcher.get_chunk(chunk_idx)
                
                # Pre-convert dense X
                if hasattr(chunk_data["x"], "toarray"):
                    chunk_x = torch.from_numpy(chunk_data["x"].toarray()).to(torch.float32)
                elif not isinstance(chunk_data["x"], torch.Tensor):
                    chunk_x = torch.tensor(chunk_data["x"], dtype=torch.float32)
                else:
                    chunk_x = chunk_data["x"].to(torch.float32)

                # Pre-convert Adjacency COO
                adj_coo = chunk_data["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32)

                # Pre-slice Core Node Indices (Eliminates runtime GPU mask filtering & syncs)
                local_core = chunk_data["local_core_idx"]
                train_core_idx = torch.from_numpy(local_core[chunk_data["train_mask"][local_core]]).to(torch.int64)
                val_core_idx = torch.from_numpy(local_core[chunk_data["val_mask"][local_core]]).to(torch.int64)

                packaged_chunk = {
                    "x": chunk_x,
                    "src": src,
                    "dst": dst,
                    "weights": weights,
                    "train_core_idx": train_core_idx,
                    "val_core_idx": val_core_idx,
                    "patient_name": patient_name
                }
                
                chunk_file = tmp_chunk_dir / f"{data.patient_name}_chunk_{chunk_idx}.pt"
                torch.save(packaged_chunk, chunk_file)
                
                training_cache.append({
                    "patient_name": data.patient_name, 
                    "chunk_file": chunk_file
                })
                
        del data, X_sp, adj_sp, batcher
        gc.collect()
        
    return training_cache


def prefetch_batches(
    meta_batches: list[list[dict[str, Any]]]
) -> Iterator[list[dict[str, Any]]]:
    """Direct synchronous generator: yields meta-batch chunk descriptors without loading tensors."""
    if not meta_batches:
        return

    for meta_meta in meta_batches:
        yield meta_meta



@torch.no_grad()
def init_decoder_bias_from_data(
    model: LibellaGNN,
    training_cache: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 4,
) -> None:
    """Initializes decoder_bias directly to the empirical cell-normalized mean of real core cells (CPU-resident)."""
    model.eval()
    total_normed = torch.zeros(model.in_channels, dtype=torch.float32, device="cpu")
    total_samples = 0
    meta_batches = make_meta_batches(training_cache, meta_batch_size=batch_size)

    for meta_meta in prefetch_batches(meta_batches):
        # Support both tuple/list yields and direct meta_meta generator
        batch_refs = meta_meta[0] if isinstance(meta_meta, tuple) else meta_meta

        for batch_ref in batch_refs:
            batch = torch.load(batch_ref["chunk_file"], map_location="cpu", weights_only=False)
            x_cpu = batch["x"].detach().to(device="cpu", dtype=torch.float32)
            
            # Extract core indices safely on CPU with explicit int64 casting
            train_idx = batch["train_core_idx"].detach().to(device="cpu", dtype=torch.int64)
            val_idx = batch.get("val_core_idx")
            if val_idx is not None and val_idx.numel() > 0:
                core_idx = torch.cat([train_idx, val_idx.detach().to(device="cpu", dtype=torch.int64)])
            else:
                core_idx = train_idx

            # Filter valid index range within the chunk size
            valid_mask = (core_idx >= 0) & (core_idx < x_cpu.size(0))
            core_idx = core_idx[valid_mask]

            if core_idx.numel() > 0:
                x_core = x_cpu[core_idx]
                cell_mass = torch.clamp(x_core.norm(p=2, dim=-1, keepdim=True), min=1e-5)
                total_normed += (x_core / cell_mass).sum(dim=0)
                total_samples += x_core.size(0)

            del batch, x_cpu, core_idx, train_idx

    if total_samples > 0:
        empirical_mean = (total_normed / total_samples).to(device=device)
        model.decoder_bias.data.copy_(empirical_mean)
        print(f"  ↳ Initialized decoder_bias to empirical mean (L2 norm: {model.decoder_bias.norm(2).item():.4f}, across {total_samples} core cells)")
    model.train()

def _init_model(
    common_genes: list[str],
    n_latents: int,
    checkpoint_path: Path | None = None,
) -> tuple[
    LibellaGNN,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    float,
    dict[str, Any] | None,
    dict[str, list],
    int,
]:
    """Initialize GNN model, optimizers, and load state if available."""
    device = get_device()
    model = LibellaGNN(
        in_channels=len(common_genes),
        n_metaprograms=n_latents,
    ).to(device)

    # 1. Parameter grouping with dedicated baseline & decoder learning rates
    bias_params = [
        p for n, p in model.named_parameters()
        if "decoder_bias" in n
    ]
    decoder_weight_params = [
        p for n, p in model.named_parameters()
        if "decoder_weight" in n or "encoder_weight" in n
    ]
    temp_routing_params = [
        p for n, p in model.named_parameters()
        if any(k in n for k in [
            "sign_tau", "ac_delta", "spatial_gain", "listen_gate", 
            "broadcast_gate", "spatial_gate_head", "b_scale", "b_enc", 
            "pade_gate", "k_predictor", "lambda_", "qwen_gate", "diff_norm"
        ])
    ]
    special_ids = {id(p) for p in bias_params + decoder_weight_params + temp_routing_params}
    base_params = [
        p for p in model.parameters()
        if id(p) not in special_ids
    ]

    lr_base = getattr(cfg, "lr_base", 1e-3)
    optimizer = torch.optim.Adam([
        {"params": base_params, "lr": lr_base * 2.0, "weight_decay": getattr(cfg, "wd_base", 1e-4)},
        {"params": decoder_weight_params, "lr": getattr(cfg, "lr_decoder", lr_base * 0.5), "weight_decay": 0.0},
        {"params": bias_params, "lr": getattr(cfg, "lr_decoder_bias", 1e-4), "weight_decay": 0.0},
        {"params": temp_routing_params, "lr": lr_base * 2.0, "weight_decay": 0.0},
    ], betas=(0.0, 0.999), eps=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=getattr(cfg, "epochs", 100), eta_min=getattr(cfg, "lr_min", 1e-6)
    )

    best_composite_score = float("inf")
    tracker_state = None
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "autopsy_metrics": []}
    start_epoch = 0

    out_dirs = paths.make_dirs(getattr(cfg, "suffix", "default"))
    resume_path = out_dirs["out"] / "resume_latest.pt"
    target_ckpt = resume_path if resume_path.exists() else checkpoint_path

    if target_ckpt and Path(target_ckpt).exists():
        try:
            print(f"  ↳ Loading state from: {Path(target_ckpt).name}")
            ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            best_composite_score = ckpt.get(
                "best_composite_score", ckpt.get("best_val_loss", float("inf"))
            )
            tracker_state = ckpt.get("tracker_state", None)
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"  ↳ Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"  ↳ [!] Failed to load checkpoint: {e}. Raising error to prevent accidental overwrite.")
            raise e

    return model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch


def _train_loop(
    model: LibellaGNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_cache: list[dict[str, Any]],
    start_epoch: int,
    best_composite_score: float,
    tracker_state: dict[str, Any] | None,
    history: dict[str, list],
) -> tuple[LibellaGNN, dict[str, list]]:
    print("\n-> Spatial Distillation (Top-K SAE)...")
    device = get_device()
    out_dirs = paths.make_dirs(getattr(cfg, "suffix", "default"))
    out_dir = out_dirs["out"]
    checkpoint_path = out_dirs["checkpoint"]

    logger = UnifiedLogger(
        backend=getattr(cfg, "logger_backend", "tensorboard"),
        run_name=f"run_{getattr(cfg, 'suffix', 'default')}",
        log_dir=str(out_dir),
    )
    global_step = 0
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    total_epochs = getattr(cfg, "epochs", 100)

    tracker = PhaseTracker(
        total_epochs=total_epochs,
        surge_tolerance=getattr(cfg, "surge_tolerance", 0.50),
        divergence_threshold=getattr(cfg, "divergence_threshold", 0.25),
        ramp_divergence_slack=getattr(cfg, "ramp_divergence_slack", 1.00),
    )
    if tracker_state is not None:
        tracker.__dict__.update(tracker_state)
        print(
            f"  ↳ Restored PhaseTracker state (Phase {tracker.phase}, "
            f"Pressure: {tracker.pressure:.2f}, Squeeze Progress: {tracker.get_squeeze_progress():.2f})"
        )

    # Pre-Training Initialization: MUST execute before entering the epoch loop and before any optimizer.step()
    if start_epoch == 0:
        init_decoder_bias_from_data(model, training_cache, device, batch_size=accumulation_steps)

    tqdm.write("\n[*] Training Loop Initialized...")

    for epoch in tqdm(range(start_epoch, total_epochs), desc="Training", leave=False):
        model.train()
        train_steps, val_steps = 0, 0
        train_chunk_count = 0

        train_loss_acc = 0.0
        val_loss_acc = 0.0

        epoch_telemetry_acc = {
            "l_rec": 0.0,
            "l_ort": 0.0,
            "l_budget": 0.0,
            "l_aux": 0.0,
            "l_align": 0.0,
            "l_gate_sparse": 0.0,
            "l_spatial": 0.0,
            "k_pred_mean": 0.0,
            "a_ij_mean": 0.0,
            "a_ij_density": 0.0,
            "shift_mag": 0.0,
            "csnn_listen": 0.0,
            "csnn_broadcast": 0.0,
            "l0_avg": 0.0,
            "dead_cnt": 0.0,
            "max_act": 0.0,
            "dyn_w": 0.0,
            "z_mag_mean": 0.0,
            "cell_mass_mean": 0.0,
        }

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps_per_epoch = len(meta_batches)
        alpha_ema = min(
            getattr(cfg, "alpha_ema_max", 0.05),
            1.0 / (total_steps_per_epoch * getattr(cfg, "alpha_ema_step_multiplier", 1.0) + 1e-9),
        )
        ema_latent_freq = None
        nan_detected = False

        for step, meta_meta in enumerate(prefetch_batches(meta_batches)):
            # Robust to both direct list or tuple yields
            batch_refs = meta_meta[0] if isinstance(meta_meta, tuple) else meta_meta

            optimizer.zero_grad(set_to_none=True)
            current_step_loss = 0.0
            last_r_pos = None
            last_dead_mask = None
            num_accum = len(batch_refs)

            for chunk_idx, batch_ref in enumerate(batch_refs):
                batch = torch.load(batch_ref["chunk_file"], map_location="cpu", weights_only=False)

                x = batch["x"].to(device=device, non_blocking=True)
                src = batch["src"].to(device=device, non_blocking=True)
                dst = batch["dst"].to(device=device, non_blocking=True)
                weights = batch["weights"].to(device=device, non_blocking=True)

                if model.training and len(src) > 0:
                    edge_drop = getattr(cfg, "edge_dropout", 0.0)
                    if edge_drop > 0.0:
                        keep_mask = torch.rand(src.size(0), device=device) > edge_drop
                        src = src[keep_mask]
                        dst = dst[keep_mask]
                        weights = weights[keep_mask]

                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)

                if device.type != "mps":
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

                # Continuous Multi-Scale Progress from PhaseTracker
                step_fraction = float(step) / max(1.0, float(total_steps_per_epoch))
                schedules = tracker.get_schedules(epoch, step_fraction)

                model.current_progress = schedules["squeeze_progress"]
                model.current_global_progress = schedules["global_progress"]
                model.current_spatial_progress = schedules["spatial_progress"]
                model.current_gamma_progress = schedules["gamma_progress"]
                model.target_k = float(getattr(cfg, "target_k", getattr(cfg, "topk_k", 38.0)))

                # 1. Forward Pass
                (
                    recon,
                    z,
                    w_dec_norm,
                    aux_recon,
                    r_norm,
                    cell_mass,
                    r_pos,
                    dead_mask,
                    spatial_context,
                    A_ij,
                    k_i_float,
                    delta_h,
                    z_canonical,
                    routed_scores,
                    diff_sim,
                ) = model(x, src, dst, weights)

                last_r_pos = r_pos
                last_dead_mask = dead_mask

                train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)
                x_train = x[train_idx]
                recon_train = recon[train_idx]
                z_train = z[train_idx]
                aux_recon_train = aux_recon[train_idx] if aux_recon is not None else None
                r_norm_train = r_norm[train_idx] if r_norm is not None else None
                k_i_train = k_i_float[train_idx] if k_i_float is not None else None
                routed_scores_train = routed_scores[train_idx] if routed_scores is not None else None

                # Filter edges incident to training core cells
                if len(src) > 0:
                    core_mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
                    core_mask[train_idx] = True
                    edge_mask = core_mask[src] | core_mask[dst]
                    src_loss = src[edge_mask]
                    dst_loss = dst[edge_mask]
                    A_ij_loss = A_ij[edge_mask] if A_ij is not None else None
                    diff_sim_loss = diff_sim[edge_mask] if diff_sim is not None else None
                else:
                    src_loss = src
                    dst_loss = dst
                    A_ij_loss = A_ij
                    diff_sim_loss = diff_sim

                # 2. Loss Calculation
                loss_res = model.calc_loss(
                    recon_train,
                    x_train,
                    z_train,
                    w_dec_norm,
                    routed_scores=routed_scores_train,
                    k_i_float=k_i_train,
                    aux_recon=aux_recon_train,
                    r_norm=r_norm_train,
                    progress=schedules["global_progress"],
                    spatial_shift=spatial_context,
                    src=src_loss,
                    dst=dst_loss,
                    z_full=z,
                    A_ij=A_ij_loss,
                    x_full=x,
                )
                base_sae_loss = loss_res[0]
                base_recon_val = loss_res[1]
                base_ort_val = loss_res[2]
                base_budget_val = loss_res[3]
                base_aux_val = loss_res[4]
                base_align_val = loss_res[5]
                base_gate_sparse_val = loss_res[6]

                # 3. Dedicated Contrastive Spatial Relation Loss with Warm-up Gate
                spatial_progress = schedules.get("spatial_progress", 0.0)

                if len(src_loss) > 0 and delta_h is not None and spatial_progress > 0.0:
                    x_norm = F.normalize(x, p=2, dim=-1)
                    l_spatial_rel = model.calc_spatial_loss(
                        delta_h, diff_sim_loss, src_loss, dst_loss, x_norm
                    )
                    spatial_loss_weight = getattr(cfg, "spatial_loss_weight", 15.0) * spatial_progress
                    spatial_loss_val = spatial_loss_weight * l_spatial_rel
                else:
                    l_spatial_rel = torch.tensor(0.0, device=device)
                    spatial_loss_val = torch.tensor(0.0, device=device)

                true_batch_loss = base_sae_loss + spatial_loss_val

                if torch.isnan(true_batch_loss) or torch.isinf(true_batch_loss):
                    nan_detected = True
                    break

                (true_batch_loss / num_accum).backward()

                current_step_loss += true_batch_loss.detach().item() / num_accum
                train_loss_acc += true_batch_loss.detach().item()
                train_steps += 1

                with torch.no_grad():
                    z_det = z_train.detach()
                    batch_active = (z_det > 0.01).float()
                    current_freq = batch_active.mean(dim=0)

                    if ema_latent_freq is None:
                        ema_latent_freq = current_freq.clone()
                    else:
                        ema_latent_freq.lerp_(current_freq, weight=alpha_ema)

                    dead_count_val = (
                        float((model.steps_since_active >= model.dead_step_threshold).sum().item())
                        if hasattr(model, "steps_since_active")
                        else 0.0
                    )

                    if len(src) > 0 and A_ij is not None:
                        a_det = A_ij.detach()
                        epoch_telemetry_acc["a_ij_mean"] += float(a_det.mean().item())
                        epoch_telemetry_acc["a_ij_density"] += float((a_det.abs() > 0.01).float().mean().item())
                        del a_det

                    if spatial_context is not None:
                        epoch_telemetry_acc["shift_mag"] += float(spatial_context.detach().abs().mean().item())

                    if model.last_listen_prob is not None and model.last_broadcast_prob is not None:
                        epoch_telemetry_acc["csnn_listen"] += float(model.last_listen_prob[train_idx].detach().mean().item())
                        epoch_telemetry_acc["csnn_broadcast"] += float(model.last_broadcast_prob[train_idx].detach().mean().item())

                    epoch_telemetry_acc["l_rec"] += float(base_recon_val.item())
                    epoch_telemetry_acc["l_ort"] += float(base_ort_val.item())
                    epoch_telemetry_acc["l_budget"] += float(base_budget_val.item())
                    epoch_telemetry_acc["l_aux"] += float(base_aux_val.item())
                    epoch_telemetry_acc["l_align"] += float(base_align_val.item())
                    epoch_telemetry_acc["l_gate_sparse"] += float(base_gate_sparse_val.item())
                    epoch_telemetry_acc["l_spatial"] += float(l_spatial_rel.item())
                    if k_i_train is not None:
                        epoch_telemetry_acc["k_pred_mean"] += float(k_i_train.detach().mean().item())
                    epoch_telemetry_acc["l0_avg"] += float(batch_active.sum(dim=-1).mean().item())
                    epoch_telemetry_acc["dead_cnt"] += dead_count_val
                    epoch_telemetry_acc["max_act"] += float(z_det.max().item())
                    epoch_telemetry_acc["dyn_w"] += float(model.dynamic_w_ema.item())
                    epoch_telemetry_acc["z_mag_mean"] += float(z_det.mean().item())
                    if cell_mass is not None:
                        epoch_telemetry_acc["cell_mass_mean"] += float(cell_mass[train_idx].detach().mean().item())

                    del z_det, batch_active, current_freq

                train_chunk_count += 1
                if len(src) > 0:
                    del core_mask, edge_mask
                if len(src_loss) > 0 and delta_h is not None and spatial_progress > 0.0:
                    del x_norm
                del train_idx, x_train, recon_train, z_train, aux_recon_train, r_norm_train, k_i_train, routed_scores_train, base_sae_loss, base_align_val, true_batch_loss, src_loss, dst_loss, A_ij_loss, diff_sim_loss, l_spatial_rel, spatial_loss_val

                # Release live autograd graph pointers held on model instance
                model.last_listen_prob = None
                model.last_broadcast_prob = None

                # Validation Evaluation
                val_core_idx_cpu = batch.get("val_core_idx")
                if val_core_idx_cpu is not None and val_core_idx_cpu.numel() > 0:
                    val_idx = val_core_idx_cpu.to(device=device, non_blocking=True)

                    with torch.no_grad():
                        val_recon = recon[val_idx]
                        x_val = x[val_idx]

                        is_non_zero_val = x_val > 0
                        dynamic_w = getattr(model, "dynamic_w_ema", torch.tensor(1.0, device=device))
                        w_mat = torch.where(is_non_zero_val, dynamic_w, 1.0)

                        variance_weight_val = w_mat * (1.0 + torch.log1p(x_val))
                        variance_weight_val = variance_weight_val / torch.clamp(variance_weight_val.mean(), min=1e-5)

                        raw_delta_val = val_recon - x_val
                        asym_penalty = getattr(cfg, "asym_penalty_weight", 0.5)
                        asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * asym_penalty) * (raw_delta_val < 0).to(x_val.dtype)
                        delta_clamp = getattr(cfg, "delta_clamp", 100.0)
                        scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-delta_clamp, max=delta_clamp)

                        # Numerically stable log-cosh + 1.5 power law for validation evaluation
                        abs_delta_val = torch.abs(scaled_delta_val)
                        stable_log_cosh_val = abs_delta_val + torch.log1p(torch.exp(-2.0 * abs_delta_val)) - math.log(2.0)
                        peak_penalty_val = (abs_delta_val + 1e-6).pow(1.5) * 0.05
                        per_cell_loss_val = torch.sum(variance_weight_val * (stable_log_cosh_val + peak_penalty_val), dim=-1)
                        val_recon_loss = torch.mean(per_cell_loss_val) / math.sqrt(x_val.shape[-1])

                        val_loss_acc += val_recon_loss.detach().item()
                        val_steps += 1

                    del val_idx, val_recon, x_val, w_mat, raw_delta_val, asym_val, scaled_delta_val, abs_delta_val, stable_log_cosh_val, peak_penalty_val, per_cell_loss_val, val_recon_loss

                del batch, src, dst, weights, x, recon, z, w_dec_norm, aux_recon, r_norm, cell_mass, spatial_context, A_ij, k_i_float, delta_h, z_canonical, routed_scores, diff_sim

            if nan_detected:
                optimizer.zero_grad(set_to_none=True)
                break

            # 1. Dual-Group Gradient Clipping & Grad Telemetry Extraction
            recon_keys = ("decoder_bias", "decoder_weight", "encoder_weight")
            recon_params = [
                p for n, p in model.named_parameters()
                if any(k in n for k in recon_keys) and p.grad is not None
            ]
            spatial_params = [
                p for n, p in model.named_parameters()
                if not any(k in n for k in recon_keys) and p.grad is not None
            ]

            if recon_params:
                torch.nn.utils.clip_grad_norm_(
                    recon_params, max_norm=getattr(cfg, "grad_clip_recon", 1000.0)
                )
            if spatial_params:
                torch.nn.utils.clip_grad_norm_(
                    spatial_params, max_norm=getattr(cfg, "grad_clip_spatial", 15.0)
                )

            # Extract active gradient norms BEFORE zero_grad / optimizer step clears them
            last_grad_stats = {}
            for p_name, p_val in model.named_parameters():
                if p_val.grad is not None:
                    p_clean = p_name.replace(".", "_")
                    last_grad_stats[f"g_{p_clean}"] = float(p_val.grad.detach().norm(2).item())
                    last_grad_stats[f"z_{p_clean}_pct"] = float((p_val.grad == 0).float().mean().item() * 100.0)

            # 2. Optimizer Step & Unit Sphere Normalization
            optimizer.step()

            with torch.no_grad():
                if hasattr(model, "normalize_decoder"):
                    model.normalize_decoder()
                elif hasattr(model, "decoder_weight"):
                    model.decoder_weight.data = F.normalize(model.decoder_weight.data, p=2, dim=-1)

                if last_dead_mask is not None and last_dead_mask.any() and last_r_pos is not None:
                    model.resample_dead_latents(last_r_pos, last_dead_mask, optimizer=optimizer)

            global_step += 1
            step_freq = getattr(cfg, "telemetry_step_freq", 0)
            if step_freq > 0 and (global_step % step_freq == 0):
                step_metrics = {
                    "step/batch_loss": current_step_loss,
                    "step/lr": optimizer.param_groups[0]["lr"],
                    **logger.get_memory_metrics(device),
                }
                logger.log_metrics(global_step, step_metrics)

            if device.type == "mps":
                torch.mps.empty_cache()

        if nan_detected:
            print(f"\n  ↳ [!] NaN gradient detected at Epoch {epoch}. Halting training.")
            break

        # Epoch Synchronization
        history["train_loss"].append(train_loss_acc / (train_steps + 1e-9))
        history["val_loss"].append(val_loss_acc / (val_steps + 1e-9))

        scheduler.step()
        gc.collect()

        # Telemetry Resolution (Zero-Division Safe)
        divisor = max(1, train_chunk_count)
        epoch_telemetry = {k: val / divisor for k, val in epoch_telemetry_acc.items()}

        current_l0_val = epoch_telemetry.get("l0_avg", float(model.n_latents))
        denom_latents = max(1.0, float(model.n_latents))
        epoch_telemetry["p_w"] = (1.0 - (current_l0_val / denom_latents)) * 100.0

        if ema_latent_freq is not None and ema_latent_freq.sum() > 0:
            p_norm = ema_latent_freq / torch.clamp(ema_latent_freq.sum(), min=1e-6)
            epoch_telemetry["ent"] = -(p_norm * torch.log(p_norm + 1e-9)).sum().item()
        else:
            epoch_telemetry["ent"] = 0.0

        epoch_schedules = tracker.get_schedules(epoch)

        current_lr = round(optimizer.param_groups[0]["lr"], 6)
        current_rec = epoch_telemetry.get("l_rec", float("inf"))
        current_l0 = epoch_telemetry.get("l0_avg", 0.0)
        current_dead = int(epoch_telemetry.get("dead_cnt", 0))

        # Harvest deep telemetry and merge into epoch_telemetry for PhaseTracker PID governor
        deep_stats = model.get_deep_telemetry()
        epoch_telemetry.update(deep_stats)

        # --- Multimodal Robust Pareto Composite Score ---
        # 1. Base Generalization Anchor (Validation Loss)
        val_recon = history["val_loss"][-1]

        # 2. Strict Target Sparsity Compliance (Linear Penalty)
        # Replaces the weak squared penalty. A strong linear penalty prevents the network 
        # from "buying" a lower val_loss by inflating L0 early in Phase 2.
        target_k = float(getattr(model, "target_k", 38.0))
        k_error_ratio = abs(current_l0 - target_k) / max(1.0, target_k)
        phi_budget = 1.0 + 5.0 * k_error_ratio

        # 3. Schedule Completion Gate (Governor Alignment)
        # Heavily penalizes checkpoints taken before max squeeze and spatial warmups are finished.
        # This mathematically forces the "best" model to be selected from the soak phase.
        squeeze_deficit = max(0.0, 1.0 - epoch_schedules["squeeze_progress"])
        spatial_deficit = max(0.0, 1.0 - epoch_schedules["spatial_progress"])
        phi_schedule = 1.0 + 10.0 * (squeeze_deficit + spatial_deficit)

        # 4. Dictionary Health (Dead Latents)
        dead_ratio = float(current_dead) / float(model.n_latents)
        phi_dict = 1.0 + 2.0 * dead_ratio

        # Unified Multiplicative Pareto Composite Score
        composite_score = val_recon * phi_budget * phi_schedule * phi_dict

        epoch_metrics = {
            "epoch": epoch,
            "phase": tracker.phase,
            "train_loss": round(history["train_loss"][-1], 4),
            "val_loss": round(history["val_loss"][-1], 4),
            "lr": current_lr,
            "composite_score": round(composite_score, 4),
            "composite_factors": {
                "phi_budget": round(phi_budget, 4),
                "phi_schedule": round(phi_schedule, 4),
                "phi_dict": round(phi_dict, 4),
            },
            "loss_components": {
                "rec": round(current_rec, 4),
                "ort": round(epoch_telemetry.get("l_ort", 0.0), 4),
                "budget": round(epoch_telemetry.get("l_budget", 0.0), 4),
                "aux": round(epoch_telemetry.get("l_aux", 0.0), 4),
                "spatial": round(epoch_telemetry.get("l_spatial", 0.0), 4),
                "align": round(epoch_telemetry.get("l_align", 0.0), 4),
                "gate_sparse": round(epoch_telemetry.get("l_gate_sparse", 0.0), 4),
                "dynamic_w_ema": round(epoch_telemetry.get("dyn_w", 1.0), 4),
            },
            "k_pred_mean": round(epoch_telemetry.get("k_pred_mean", target_k), 2),
            "pade_diagnostics": {
                "rank_inversion_pct": round(deep_stats.get("pade/rank_inversion_pct", 0.0), 2),
                "deriv_min": round(deep_stats.get("pade/derivative_min", 0.0), 4),
                "output_min": round(deep_stats.get("pade/output_min", 0.0), 4),
                "output_max": round(deep_stats.get("pade/output_max", 0.0), 4),
            },
            "spatial_topology": {
                "a_ij_mean": round(epoch_telemetry.get("a_ij_mean", 0.0), 4),
                "delta_ratio": round(deep_stats.get("spatial/delta_ratio", 0.0), 4),
                "active_edge_pct": round(epoch_telemetry.get("a_ij_density", 0.0) * 100.0, 2),
                "diff_lambda": round(deep_stats.get("diff_attn/lambda_effective", 0.8), 4),
                "shift_magnitude": round(epoch_telemetry.get("shift_mag", 0.0), 4),
                "csnn_listen_mean": round(epoch_telemetry.get("csnn_listen", 0.0), 4),
                "csnn_broadcast_mean": round(epoch_telemetry.get("csnn_broadcast", 0.0), 4),
            },
            "dead_latents": current_dead,
            "entropy": round(epoch_telemetry.get("ent", 0.0), 4),
            "l0_avg": round(current_l0, 2),
            "p_w_sparsity_pct": round(epoch_telemetry.get("p_w", 0.0), 2),
            "max_activation": round(epoch_telemetry.get("max_act", 0.0), 2),
            "z_mag_mean": round(epoch_telemetry.get("z_mag_mean", 0.0), 4),
            "cell_mass_mean": round(epoch_telemetry.get("cell_mass_mean", 0.0), 4),
            "laprune_gamma_effective": round(float(getattr(model, "laprune_gamma", 0.99)) * epoch_schedules["gamma_progress"], 4),
            "tracker": {
                "global_progress": round(epoch_schedules["global_progress"], 4),
                "spatial_progress": round(epoch_schedules["spatial_progress"], 4),
                "squeeze_progress": round(epoch_schedules["squeeze_progress"], 4),
                "pressure": round(getattr(tracker, "pressure", 0.0), 4),
                "target_k": target_k,
            },
        }
        # Keep only the last 5 epochs in RAM history to prevent long-run heap inflation
        history.setdefault("autopsy_metrics", []).append(epoch_metrics)
        if len(history["autopsy_metrics"]) > 5:
            history["autopsy_metrics"].pop(0)

        # Flush logger backend to disk
        if hasattr(logger, "flush"):
            logger.flush()

        # Single Flat Metric Map for Every Epoch
        unified_epoch_log = {
            "ep_train_loss": history["train_loss"][-1],
            "ep_val_loss": history["val_loss"][-1],
            "ep_composite_score": composite_score,
            "composite_phi_budget": phi_budget,
            "composite_phi_dict": phi_dict,
            "composite_phi_schedule": phi_schedule,
            "l_recon": epoch_telemetry.get("l_rec", 0.0),
            "l_ort": epoch_telemetry.get("l_ort", 0.0),
            "l_budget": epoch_telemetry.get("l_budget", 0.0),
            "l_aux": epoch_telemetry.get("l_aux", 0.0),
            "l_spatial": epoch_telemetry.get("l_spatial", 0.0),
            "l_align": epoch_telemetry.get("l_align", 0.0),
            "l_gate_sparse": epoch_telemetry.get("l_gate_sparse", 0.0),
            "l_dynamic_w_ema": epoch_telemetry.get("dyn_w", 1.0),
            "sae_k_pred_mean": epoch_telemetry.get("k_pred_mean", 0.0),
            "sae_sparsity_pct": epoch_telemetry.get("p_w", 0.0),
            "sae_entropy": epoch_telemetry.get("ent", 0.0),
            "graph_a_ij_mean": epoch_telemetry.get("a_ij_mean", 0.0),
            "graph_active_edge_pct": epoch_telemetry.get("a_ij_density", 0.0) * 100.0,
            "spatial_shift_mag": epoch_telemetry.get("shift_mag", 0.0),
            "csnn_listen_prob_mean": epoch_telemetry.get("csnn_listen", 0.0),
            "csnn_broadcast_prob_mean": epoch_telemetry.get("csnn_broadcast", 0.0),
            "sae_l0_total": current_l0,
            "sae_dead_latents": current_dead,
            "sae_z_mag_mean": epoch_telemetry.get("z_mag_mean", 0.0),
            "sae_cell_mass_mean": epoch_telemetry.get("cell_mass_mean", 0.0),
            "tr_global_progress": epoch_schedules["global_progress"],
            "tr_spatial_progress": epoch_schedules["spatial_progress"],
            "tr_squeeze_progress": epoch_schedules["squeeze_progress"],
            "tr_pressure": getattr(tracker, "pressure", 0.0),
            **deep_stats,
        }

        # Log flat metrics once per epoch
        logger.log_metrics(epoch, unified_epoch_log)

        # Only capture Pareto best checkpoints during the Soak Phase (Max Squeeze) or final epoch
        if (epoch_schedules["squeeze_progress"] >= 1.0 or epoch == total_epochs - 1) and composite_score < best_composite_score and not nan_detected:
            best_composite_score = composite_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_composite_score": best_composite_score,
                    "metrics": epoch_metrics,
                    "history": history,
                },
                checkpoint_path,
            )

        ckpt_freq = getattr(cfg, "checkpoint_freq", 10)
        if ((epoch + 1) % ckpt_freq == 0 or epoch == total_epochs - 1) and not nan_detected:
            autopsy_dir = out_dir / "autopsy_checkpoints"
            autopsy_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = autopsy_dir / f"epoch_{(epoch+1):03d}.pt"
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": epoch_metrics},
                ckpt_path,
            )

            logger.log_checkpoint_autopsy(epoch, str(ckpt_path))

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_composite_score": best_composite_score,
                    "tracker_state": tracker.__dict__,
                    "history": history,
                },
                out_dir / "resume_latest.pt",
            )

            with torch.no_grad():
                l0_val = epoch_telemetry.get("l0_avg", 0.0)
                l0_pct = (l0_val / model.n_latents) * 100.0
                pade_inv = deep_stats.get("pade/rank_inversion_pct", 0.0)
                d_ratio = deep_stats.get("spatial/delta_ratio", 0.0)
                tqdm.write(
                    f" [Ep {(epoch+1):03d}] Score:{composite_score:<6.2f} "
                    f"Rec:{epoch_telemetry.get('l_rec', 0.0):<5.3f} "
                    f"V_Loss:{history['val_loss'][-1]:<5.3f} | "
                    f"L0:{l0_val:<4.1f}/{model.n_latents} ({l0_pct:<4.1f}%) "
                    f"K_Pred:{epoch_telemetry.get('k_pred_mean', 0.0):<4.1f} "
                    f"Dead:{int(epoch_telemetry.get('dead_cnt', 0)):<3d} | "
                    f"Δ_ratio:{d_ratio:<4.3f} "
                    f"λ:{deep_stats.get('diff_attn/lambda_effective', 0.8):<4.2f} "
                    f"Spat_Loss:{epoch_telemetry.get('l_spatial', 0.0):<5.3f} "
                    f"Padé_Inv:{pade_inv:<4.1f}% "
                    f"Shift:{epoch_telemetry.get('shift_mag', 0.0):<4.2f}"
                )

        epochs_remaining = total_epochs - epoch - 1
        force_window = getattr(cfg, "phase2_force_window", 10)
        if tracker.phase == 1 and epochs_remaining <= force_window:
            tqdm.write(f"\n[!] Approaching max epochs ({total_epochs}). Forcing Phase 2.")
            tracker.force_phase2(epoch, epoch_telemetry.get("l_rec", 0.0))

        was_phase_1 = tracker.phase == 1
        current_val_loss = history["val_loss"][-1]
        is_done = tracker.step(epoch_telemetry, epoch, val_loss=current_val_loss)

        if was_phase_1 and tracker.phase == 2:
            best_composite_score = float("inf")  # Reset so Phase 1 soft metrics do not block Phase 2 checkpoints
            tqdm.write(
                f"\n[↳] Phase 1 Complete at Epoch {epoch} (Baseline Rec: {tracker.p1_baseline_rec:.2f}). "
                f"\n    Engaging Adaptive Loss-Gated Sparsification across {tracker.total_epochs - tracker.p2_start_epoch} epochs..."
            )

        if is_done:
            tqdm.write(
                f"\n[✓] Pareto Convergence Reached at Epoch {(epoch+1)}/{total_epochs} "
                f"(Val Loss: {current_val_loss:.4f}, Squeeze Progress: {tracker.get_squeeze_progress():.2%}). Terminating gracefully."
            )
            break

    logger.close()

    if checkpoint_path.exists() and best_composite_score < float("inf"):
        print(f"  ↳ Restoring in-memory model to best Pareto Phase 2 checkpoint ({checkpoint_path.name})...")
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
    else:
        print(f"  ↳ Retaining final epoch in-memory state (P_W = {epoch_telemetry.get('p_w', 0.0):.1f}%)...")

    return model, history


def train_gnn(
    graph_paths: list[Path],
    common_genes: list[str],
) -> tuple[LibellaGNN, dict[str, list], int]:
    """Master orchestrator for GNN training phase."""
    out_dirs = paths.make_dirs(getattr(cfg, "suffix", "default"))
    checkpoint_path = out_dirs["checkpoint"]
    out_dir = out_dirs["out"]
    device = get_device()

    n_latents = getattr(cfg, "n_latents", getattr(cfg, "n_metaprograms", 512))
    print(f"[*] Initializing Native Top-K SAE Latent Space (M = {n_latents}, Top-K = {getattr(cfg, 'topk_k', 3)})...")

    model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch = _init_model(
        common_genes, n_latents, checkpoint_path
    )
    gc.collect()

    training_cache = _prep_ssd_chunks(graph_paths)
    gc.collect()

    if start_epoch >= getattr(cfg, "epochs", 100):
        print(f"-> Training already reached target epoch ({start_epoch}/{getattr(cfg, 'epochs', 100)}). Skipping loop.")
        master_latent_path = out_dir / "libella_latent.npz"
        if not master_latent_path.exists():
            export_latents_from_graphs(model, graph_paths, out_dirs["out"], device)
        return model, history, n_latents

    model, history = _train_loop(
        model,
        optimizer,
        scheduler,
        training_cache,
        start_epoch,
        best_composite_score,
        tracker_state,
        history,
    )
    gc.collect()

    export_latents_from_graphs(model, graph_paths, out_dirs["out"], device)

    return model, history, n_latents

