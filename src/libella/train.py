"""Model training loops and orchestrators for the Spatial Ecotype GNN."""

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
from .utils import PhaseTracker, get_device, UnifiedLogger


def _init_model(
    common_genes: list[str], 
    n_latents: int, 
    checkpoint_path: Path | None = None
) -> tuple[LibellaGNN, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict[str, Any] | None, dict[str, list], int]:
    """Initialize GNN model, optimizers, and load state if available."""
    device = get_device()
    model = LibellaGNN(
        in_channels=len(common_genes), 
        n_metaprograms=n_latents
    ).to(device)
    
    # 1. Isolate decoder, jump threshold, and backbone parameters
    decoder_params = [p for n, p in model.named_parameters() if "decoder_" in n]
    threshold_params = [p for n, p in model.named_parameters() if "jump_threshold" in n]
    base_params = [
        p for n, p in model.named_parameters() 
        if "decoder_" not in n and "jump_threshold" not in n
    ]

    # 2. Decoder parameters receive zero weight decay (preserved via Oblique Retraction)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
        {"params": decoder_params, "lr": getattr(cfg, "lr_decoder", cfg.lr_base), "weight_decay": 0.0},
        {"params": threshold_params, "lr": getattr(cfg, "lr_threshold", cfg.lr_base * 0.5), "weight_decay": 0.0}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )
    
    best_composite_score = float("inf")
    tracker_state = None
    history = {"train_loss": [], "val_loss": [], "autopsy_metrics": []}
    start_epoch = 0

    out_dirs = paths.make_dirs(cfg.suffix)
    resume_path = out_dirs["out"] / "resume_latest.pt"
    target_ckpt = resume_path if resume_path.exists() else checkpoint_path

    if target_ckpt and Path(target_ckpt).exists():
        try:
            print(f"  ↳ Loading state from: {target_ckpt.name}")
            ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                
            best_composite_score = ckpt.get("best_composite_score", ckpt.get("best_val_loss", float("inf")))
            tracker_state = ckpt.get("tracker_state", None)
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"  ↳ Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"  ↳ [!] Failed to load checkpoint: {e}. Raising error to prevent accidental overwrite.")
            raise e 

    return model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch


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
) -> Iterator[tuple[list[dict[str, Any]], list[Any]]]:
    """Direct synchronous loader: 0 threads, 0 locks, 0 deadlocks on Apple Silicon."""
    if not meta_batches:
        return

    for meta_meta in meta_batches:
        chunks = []
        for b in meta_meta:
            chunk = torch.load(b["chunk_file"], map_location="cpu", weights_only=False)
            chunks.append(chunk)
        yield meta_meta, chunks



def _train_loop(
    model: LibellaGNN, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    training_cache: list[dict[str, Any]], 
    start_epoch: int, 
    best_composite_score: float, 
    tracker_state: dict[str, Any] | None,
    history: dict[str, list]
) -> tuple[LibellaGNN, dict[str, list]]:
    print("\n-> Spatial Distillation...")
    device = get_device()
    out_dirs = paths.make_dirs(cfg.suffix)
    out_dir = out_dirs["out"]
    checkpoint_path = out_dirs["checkpoint"]
    
    logger = UnifiedLogger(
        backend=getattr(cfg, "logger_backend", "tensorboard"),
        run_name=f"run_{cfg.suffix}",
        log_dir=str(out_dir)
    )
    global_step = 0
    
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)  

    tracker = PhaseTracker()
    if tracker_state is not None:
        tracker.__dict__.update(tracker_state)
        print(f"  ↳ Restored PhaseTracker state (Phase {tracker.phase}, Pressure: {tracker.pressure:.2f}, Progress: {tracker.get_progress():.2f})")
        
    tqdm.write("\n[*] Adaptive Scheduler Initialized...")

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training", leave=False):
        model.train()
        train_steps, val_steps = 0, 0
        train_chunk_count = 0

        # GPU-resident accumulator buffers
        train_loss_acc = torch.tensor(0.0, device=device)
        val_loss_acc = torch.tensor(0.0, device=device)
        
        gpu_telemetry = {
            'l_rec': torch.tensor(0.0, device=device),
            'l_ort': torch.tensor(0.0, device=device),
            'l_sparse': torch.tensor(0.0, device=device),
            'l_aux': torch.tensor(0.0, device=device),
            'l0_avg': torch.tensor(0.0, device=device),
            'dead_cnt': torch.tensor(0.0, device=device),
            'max_act': torch.tensor(0.0, device=device)
        }

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps_per_epoch = len(meta_batches)
        alpha_ema = min(0.005, 1.0 / (total_steps_per_epoch * 2.0 + 1e-9))
        ema_latent_freq = None
        nan_detected = False

        for step, (meta_meta, chunk_iter) in enumerate(prefetch_batches(meta_batches)):
            optimizer.zero_grad(set_to_none=True)
            for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):

                x = batch["x"].to(device=device, non_blocking=True)
                src = batch["src"].to(device=device, non_blocking=True)
                dst = batch["dst"].to(device=device, non_blocking=True)
                weights = batch["weights"].to(device=device, non_blocking=True)
                
                if model.training and len(src) > 0:
                    keep_mask = torch.rand(src.size(0), device=device) > cfg.edge_dropout
                    src = src[keep_mask]
                    dst = dst[keep_mask]
                    weights = weights[keep_mask]
                
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)

                if device.type != 'mps':
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

                prog = tracker.get_progress()
                model.current_progress = prog

                # 1. Defensive Forward Execution
                forward_res = model(x, src, dst, weights)
                recon, z, w_dec_norm = forward_res[0], forward_res[1], forward_res[2]
                aux_recon = forward_res[3] if len(forward_res) > 3 else None
                r_norm = forward_res[4] if len(forward_res) > 4 else None
                
                train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)
                x_train = x[train_idx]
                recon_train = recon[train_idx]
                z_train = z[train_idx]
                aux_recon_train = aux_recon[train_idx] if aux_recon is not None else None
                r_norm_train = r_norm[train_idx] if r_norm is not None else None

                # 2. Defensive Loss Calculation
                loss_res = model.calc_loss(
                    recon_train, x_train, z_train, w_dec_norm,
                    aux_recon=aux_recon_train, r_norm=r_norm_train,
                    progress=prog
                )
                true_batch_loss = loss_res[0]
                base_recon_val = loss_res[1]
                base_ort_val = loss_res[2]
                base_sparse_val = loss_res[3]
                base_aux_val = loss_res[4] if len(loss_res) > 4 else torch.tensor(0.0, device=device)

                if torch.isnan(true_batch_loss) or torch.isinf(true_batch_loss):
                    nan_detected = True
                    break

                (true_batch_loss / len(meta_meta)).backward()
                
                train_loss_acc += true_batch_loss.detach()
                train_steps += 1

                # 3. GPU Telemetry Tracking
                with torch.no_grad():
                    batch_active = (z_train > 0).float()
                    current_freq = batch_active.mean(dim=0)
                    
                    if ema_latent_freq is None:
                        ema_latent_freq = current_freq.clone()
                    else:
                        ema_latent_freq.lerp_(current_freq, weight=alpha_ema)

                    dead_count_val = (
                        (model.steps_since_active >= model.dead_step_threshold).float().sum() 
                        if hasattr(model, 'steps_since_active') else torch.tensor(0.0, device=device)
                    )

                    gpu_telemetry['l_rec'] += base_recon_val
                    gpu_telemetry['l_ort'] += base_ort_val
                    gpu_telemetry['l_sparse'] += base_sparse_val
                    gpu_telemetry['l_aux'] += base_aux_val
                    gpu_telemetry['l0_avg'] += batch_active.sum(dim=-1).mean()
                    gpu_telemetry['dead_cnt'] += dead_count_val
                    gpu_telemetry['max_act'] += z_train.max()

                train_chunk_count += 1

                del train_idx, x_train, recon_train, z_train, aux_recon_train, r_norm_train, true_batch_loss

                # Validation Evaluation
                val_core_idx_cpu = batch["val_core_idx"]
                if val_core_idx_cpu.numel() > 0:
                    val_idx = val_core_idx_cpu.to(device=device, non_blocking=True)
                    
                    with torch.no_grad():
                        val_recon = recon[val_idx]
                        x_val = x[val_idx]
                        
                        is_non_zero_val = (x_val > 0)
                        dynamic_w = getattr(model, 'dynamic_w_ema', torch.tensor(1.0, device=device))
                        w_mat = torch.where(is_non_zero_val, dynamic_w, 1.0)
                        zero_expectation_mask = torch.where(is_non_zero_val, 1.0, cfg.zero_mask_rate).to(x_val.dtype)
                        masked_w_mat_val = w_mat * zero_expectation_mask
                        
                        raw_delta_val = val_recon - x_val
                        asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * 2.0) * (raw_delta_val < 0).to(x_val.dtype)
                        scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-cfg.delta_clamp, max=cfg.delta_clamp)
                        
                        val_loss_sum = torch.sum(masked_w_mat_val * torch.log(torch.cosh(scaled_delta_val + 1e-6)))
                        val_log_cosh = val_loss_sum / max(1, x_val.numel())
                    
                        val_loss_acc += val_log_cosh.detach()
                        val_steps += 1
                        
                    del val_idx, val_recon, x_val, w_mat, raw_delta_val, asym_val, scaled_delta_val, val_loss_sum, val_log_cosh

                del batch, src, dst, weights, x, recon, z, w_dec_norm, aux_recon, r_norm 

            if nan_detected:
                optimizer.zero_grad(set_to_none=True)
                break

            # 1. Global Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

            # 2. Tangent-Space Projection
            with torch.no_grad():
                if hasattr(model, 'decoder_weight') and model.decoder_weight.grad is not None:
                    w = F.normalize(model.decoder_weight, p=2, dim=1)
                    grad = model.decoder_weight.grad
                    proj_grad = grad - (grad * w).sum(dim=1, keepdim=True) * w
                    model.decoder_weight.grad.copy_(proj_grad)

            optimizer.step()

            # 3. Strict Oblique Retraction
            with torch.no_grad():
                if hasattr(model, 'decoder_weight'):
                    model.decoder_weight.copy_(F.normalize(model.decoder_weight, p=2, dim=1))

            

        if nan_detected:
            print(f"\n  ↳ [!] NaN gradient detected at Epoch {epoch}. Halting training.")
            break

        # Epoch Synchronization
        history['train_loss'].append((train_loss_acc / (train_steps + 1e-9)).item())
        history['val_loss'].append((val_loss_acc / (val_steps + 1e-9)).item())

        scheduler.step()
        gc.collect()

        # Telemetry Resolution
        epoch_telemetry = {}
        if train_chunk_count > 0:
            for k, v in gpu_telemetry.items():
                epoch_telemetry[k] = (v / train_chunk_count).item()
            if ema_latent_freq is not None:
                p_norm = ema_latent_freq / torch.clamp(ema_latent_freq.sum(), min=1e-6)
                epoch_telemetry['ent'] = -(p_norm * torch.log(p_norm + 1e-9)).sum().item()
            else:
                epoch_telemetry['ent'] = 0.0

        # 1. Telemetry metric resolution
        current_lr = round(optimizer.param_groups[0]['lr'], 6)
        current_rec = epoch_telemetry.get("l_rec", float("inf"))
        current_l0 = epoch_telemetry.get("l0_avg", 0.0)
        current_dead = int(epoch_telemetry.get("dead_cnt", 0))

        epoch_metrics = {
            'epoch': epoch,
            'phase': tracker.phase,
            'train_loss': round(history['train_loss'][-1], 4),
            'val_loss': round(history['val_loss'][-1], 4),
            'lr': current_lr,
            'loss_components': {
                'rec': round(current_rec, 4),
                'ort': round(epoch_telemetry.get('l_ort', 0.0), 4),
                'sparse': round(epoch_telemetry.get('l_sparse', 0.0), 4),
                'aux': round(epoch_telemetry.get('l_aux', 0.0), 4)
            },
            'entropy': round(epoch_telemetry.get('ent', 0.0), 4),
            'l0_avg': round(current_l0, 2),
            'dead_latents': current_dead,
            'max_activation': round(epoch_telemetry.get('max_act', 0.0), 2)
        }
        history.setdefault('autopsy_metrics', []).append(epoch_metrics)

        composite_score = current_rec * math.sqrt(1.0 + (current_l0 / float(model.n_latents)))

        epoch_log = {
            "epoch/train_loss": history['train_loss'][-1],
            "epoch/val_loss": history['val_loss'][-1],
            "epoch/composite_score": composite_score,
            "loss/recon": epoch_telemetry.get('l_rec', 0.0),
            "loss/sparse": epoch_telemetry.get('l_sparse', 0.0),
            "sae/l0_avg": current_l0,
            "sae/dead_latents": current_dead,
        }
        logger.log_metrics(epoch, epoch_log)

        if composite_score < best_composite_score and not nan_detected:
            best_composite_score = composite_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_composite_score": best_composite_score,
                "metrics": epoch_metrics,
                "history": history
            }, checkpoint_path)

        if ((epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1) and not nan_detected:
            autopsy_dir = out_dir / "autopsy_checkpoints"
            autopsy_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": epoch_metrics}, autopsy_dir / f"epoch_{(epoch+1):03d}.pt")
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_composite_score": best_composite_score,
                "tracker_state": tracker.__dict__,
                "history": history
            }, out_dir / "resume_latest.pt")

            with torch.no_grad():
                l0_val = epoch_telemetry.get('l0_avg', 0.0)
                l0_pct = (l0_val / model.n_latents) * 100.0
                tqdm.write(
                    f" [Ep {(epoch+1):03d}] Rec:{epoch_telemetry.get('l_rec', 0.0):<5.3f} "
                    f"V_Loss:{history['val_loss'][-1]:<5.3f} | "
                    f"L0:{l0_val:<4.1f}/{model.n_latents} ({l0_pct:<4.1f}%) "
                    f"Dead:{int(epoch_telemetry.get('dead_cnt', 0)):<3d} | "
                    f"L_Ort:{epoch_telemetry.get('l_ort', 0.0):<5.3f} "
                    f"L_Sp:{epoch_telemetry.get('l_sparse', 0.0):<5.3f} "
                    f"L_Aux:{epoch_telemetry.get('l_aux', 0.0):<5.3f}"
                )

        epochs_remaining = cfg.epochs - epoch - 1
        if tracker.phase == 1 and epochs_remaining <= 20:
            tqdm.write(f"\n[!] Approaching max epochs ({cfg.epochs}). Forcing Phase 2.")
            tracker.force_phase2(epoch, epoch_telemetry.get('l_rec', 0.0))

        was_phase_1 = (tracker.phase == 1)
        is_done = tracker.step(epoch_telemetry, epoch)
        
        if was_phase_1 and tracker.phase == 2:
            tqdm.write(
                f"\n[↳] Phase 1 Complete at Epoch {epoch} (Baseline Rec: {tracker.p1_baseline_rec:.2f}). "
                f"\n    Engaging Adaptive Loss-Gated Sparsification..."
            )
            
        if is_done:
            final_pw = epoch_telemetry.get('p_w', 0.0)
            tqdm.write(f"\n[✓] Topic Sharpness (P_W) saturated at {final_pw:.2f}%. Terminating gracefully at Epoch {(epoch+1)}.")
            break
    
    logger.close()

    if checkpoint_path.exists() and best_composite_score < float("inf"):
        print(f"  ↳ Restoring in-memory model to best Pareto Phase 2 checkpoint ({checkpoint_path.name})...")
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
    else:
        print(f"  ↳ Retaining final epoch in-memory state (P_W = {epoch_telemetry.get('p_w', 0.0):.1f}%)...")

    # Post-Training: Export Master & Patient/Sample Latents
    print("\n-> Extracting & Exporting Libella Latent Representations...")
    model.eval()
    latent_records = []
    latent_cols = [f"MP_{i+1}" for i in range(model.n_latents)]
    
    with torch.no_grad():
        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        for meta_meta, chunk_iter in prefetch_batches(meta_batches):
            for batch_ref, batch in zip(meta_meta, chunk_iter):
                x = batch["x"].to(device=device, non_blocking=True)
                src = batch["src"].to(device=device, non_blocking=True)
                dst = batch["dst"].to(device=device, non_blocking=True)
                weights = batch["weights"].to(device=device, non_blocking=True)
                
                n_cells = x.size(0)
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                
                if device.type != 'mps':
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

                forward_eval = model(x, src, dst, weights)
                z = forward_eval[1]
                z_np = z[:n_cells].detach().cpu().numpy()
                
                patient_name = batch_ref.get("patient_name") or batch.get("patient_name")
                if not patient_name:
                    chunk_file = batch_ref.get("chunk_file")
                    patient_name = Path(chunk_file).stem.split("_chunk_")[0] if chunk_file else "sample_1"
                
                patient_name = str(patient_name)
                cell_ids = batch.get("barcodes") or batch.get("cell_ids")
                if cell_ids is None:
                    cell_ids = [f"{patient_name}_cell_{i}" for i in range(n_cells)]

                chunk_df = pd.DataFrame(z_np, index=cell_ids, columns=latent_cols)
                chunk_df["patient_name"] = patient_name
                latent_records.append(chunk_df)

    if latent_records:
        full_latents_df = pd.concat(latent_records, axis=0)

        master_latent_path = out_dir / "libella_latent.csv"
        full_latents_df.to_csv(master_latent_path, index_label="cell_id")
        print(f"  ↳ Master latents saved -> {master_latent_path}")

        sample_out_dir = out_dir / "sample_latents"
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        
        for p_name, sub_df in full_latents_df.groupby("patient_name"):
            clean_sub_df = sub_df.drop(columns=["patient_name"])
            
            root_sample_file = out_dir / f"libella_latent_{p_name}.csv"
            clean_sub_df.to_csv(root_sample_file, index_label="cell_id")
            
            nested_sample_file = sample_out_dir / f"{p_name}_latent.csv"
            clean_sub_df.to_csv(nested_sample_file, index_label="cell_id")
            
            print(f"  ↳ Patient latent saved -> {root_sample_file}")

    return model, history


def train_gnn(
    graph_paths: list[Path], common_genes: list[str]
) -> tuple[LibellaGNN, dict[str, list], int]:
    """Master orchestrator for GNN training phase."""
    out_dirs = paths.make_dirs(cfg.suffix)
    checkpoint_path = out_dirs["checkpoint"]

    n_latents = getattr(cfg, "n_latents", getattr(cfg, "n_metaprograms", 512))
    print(f"[*] Initializing Native SAE Latent Space (M = {n_latents} features on unit sphere)...")

    model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch = _init_model(
        common_genes, n_latents, checkpoint_path
    )
    gc.collect()

    if start_epoch >= cfg.epochs:
        print(f"-> Training already reached target epoch ({start_epoch}/{cfg.epochs}). Skipping loop.")
        return model, history, n_latents

    training_cache = _prep_ssd_chunks(graph_paths)
    gc.collect()  

    model, history = _train_loop(
        model, optimizer, scheduler, training_cache, start_epoch, best_composite_score, tracker_state, history
    )
    gc.collect()
    
    return model, history, n_latents