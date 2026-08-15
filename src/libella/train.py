"""Model training loops and orchestrators for the Spatial Ecotype GNN."""

import gc
import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm
import queue
from threading import Thread, Event

from .config import cfg, paths
from .data import (
    SpatialBatcher,
    make_meta_batches,
    pad_mps_shapes,
    pt_to_scipy_csr,
)
from .model import LibellaGNN
from .prior import get_priors
from .utils import get_device, PhaseTracker



def _init_model(
    common_genes: list[str], 
    optimal_k: int, 
    init_components: np.ndarray | None, 
    checkpoint_path: Path | None = None
) -> tuple[LibellaGNN, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict[str, Any] | None, dict[str, list], int]:
    """Initialize GNN model, optimizers, and load state if available."""
    device = get_device()
    model = LibellaGNN(
        in_channels=len(common_genes), 
        n_metaprograms=optimal_k, 
        init_components=init_components
    ).to(device)
    
    base_params = [p for n, p in model.named_parameters() if "topic_gene_logits" not in n]
    anchor_params = [p for n, p in model.named_parameters() if "topic_gene_logits" in n]

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": cfg.lr_base, "weight_decay": cfg.wd_base},
        {"params": anchor_params, "lr": cfg.lr_anchor, "weight_decay": cfg.wd_anchor} 
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
    
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)  
    ema_mean = None

    tracker = PhaseTracker()
    if tracker_state is not None:
        tracker.__dict__.update(tracker_state)
        print(f"  ↳ Restored PhaseTracker state (Phase {tracker.phase}, Progress: {tracker.internal_progress:.2f})")
        
    tqdm.write("\n[*] Adaptive Scheduler Initialized...")
    
    optimal_k = model.n_metaprograms
    max_entropy_scalar = float(np.log(optimal_k))

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training", leave=False):
        model.train()
        train_steps, val_steps = 0, 0
        train_chunk_count = 0

        # GPU-resident accumulator buffers (Zero CPU-GPU sync stalls during loop)
        train_loss_acc = torch.tensor(0.0, device=device)
        val_loss_acc = torch.tensor(0.0, device=device)
        epoch_p_mean_sum = torch.zeros(optimal_k, device=device)
        
        gpu_telemetry = {
            'ent': torch.tensor(0.0, device=device),
            'col_r': torch.tensor(0.0, device=device),
            'kl_w': torch.tensor(0.0, device=device),
            'g_w': torch.tensor(0.0, device=device),
            'p_w': torch.tensor(0.0, device=device),
            'l_rec': torch.tensor(0.0, device=device),
            'l_anc': torch.tensor(0.0, device=device),
            'l_ort': torch.tensor(0.0, device=device)
        }

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps_per_epoch = len(meta_batches)
        alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9))
        nan_detected = False

        for step, (meta_meta, chunk_iter) in enumerate(prefetch_batches(meta_batches)):
            optimizer.zero_grad(set_to_none=True)
            for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):

                # Direct non-blocking transfers from pre-tensorized SSD chunks
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
                model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * (prog ** 0.8))
                model.current_temp = cfg.temp_end + ((cfg.temp_start - cfg.temp_end) * ((1.0 - prog) ** 1.5))
                model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * prog)

                fracs, pure_anchors = model(x, src, dst, weights)
                
                
                train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)

                f_train = fracs[train_idx]
                x_train = x[train_idx]

                p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                current_p_mean = p_train.mean(dim=0)
                
                # Pure GPU EMA Target Calculation (0 CPU-GPU syncs)
                uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
                if ema_mean is None:
                    ema_mean = current_p_mean.detach()
                else:
                    ema_mean = alpha_ema * current_p_mean.detach() + (1 - alpha_ema) * ema_mean
                
                ideal_c = uniform_prior * 2.0 - ema_mean
                ideal_c = torch.clamp(ideal_c, min=1e-5) 
                target_f_dist = ideal_c / ideal_c.sum()
                    
                ema_entropy = -torch.sum(ema_mean * torch.log(ema_mean + 1e-9))
                collapse_ratio = torch.clamp(1.0 - (ema_entropy / max_entropy_scalar), min=0.0, max=1.0)

                peak_p = ema_mean.max()
                hub_multiplier = F.relu((peak_p / cfg.hub_threshold) - 1.0) * 10.0 

                dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + hub_multiplier 

                recon = f_train @ pure_anchors
                
                true_batch_loss, base_recon_val, base_anc_val, base_ort_val = model.calc_loss(
                    recon, x_train, pure_anchors, None, epoch, cfg.epochs, 
                    f_train=f_train, target_f_dist=target_f_dist, kl_weight=dynamic_kl_w
                )

                if torch.isnan(true_batch_loss) or torch.isinf(true_batch_loss):
                    nan_detected = True
                    break

                (true_batch_loss / len(meta_meta)).backward()
                
                train_loss_acc += true_batch_loss.detach()
                train_steps += 1

                # 3. Complete GPU Telemetry Accumulation
                gpu_telemetry['g_w'] += pure_anchors.max(dim=1).values.mean().detach() * 100.0
                gpu_telemetry['p_w'] += p_train.max(dim=1).values.mean().detach() * 100.0
                gpu_telemetry['ent'] += ema_entropy.detach()
                gpu_telemetry['col_r'] += collapse_ratio.detach()
                gpu_telemetry['kl_w'] += dynamic_kl_w.detach()
                
                # Accumulate all sub-losses cleanly
                gpu_telemetry['l_rec'] += base_recon_val.detach()
                gpu_telemetry['l_anc'] += base_anc_val.detach()
                gpu_telemetry['l_ort'] += base_ort_val.detach()
                
                epoch_p_mean_sum += current_p_mean.detach()
                train_chunk_count += 1

                del train_idx, f_train, x_train, p_train, current_p_mean, uniform_prior, target_f_dist, recon, true_batch_loss, base_recon_val

                # Validation Evaluation (Zero-sync check using CPU-resident numel)
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
                    
                        val_loss_acc += val_log_cosh.detach()
                        val_steps += 1
                        
                    del val_idx, f_val, x_val, val_recon, w_mat, raw_delta_val, asym_val, scaled_delta_val, val_loss_sum, val_log_cosh

                del batch, src, dst, weights, x, fracs, pure_anchors
                model.current_f_prob = None  

            if nan_detected:
                optimizer.zero_grad(set_to_none=True)
                break

            # 1. Clip GNN backbone and projection heads
            base_params = [
                p for n, p in model.named_parameters() 
                if "topic_gene_logits" not in n and p.grad is not None
            ]
            if base_params:
                torch.nn.utils.clip_grad_norm_(base_params, max_norm=cfg.grad_clip)

            # 2. Clip anchor dictionary independently so it retains its full update budget
            anchor_params = [
                p for n, p in model.named_parameters() 
                if "topic_gene_logits" in n and p.grad is not None
            ]
            if anchor_params:
                torch.nn.utils.clip_grad_norm_(anchor_params, max_norm=cfg.grad_clip)

            optimizer.step()

        if nan_detected:
            print(f"\n  ↳ [!] NaN gradient detected at Epoch {epoch}. Halting training.")
            break

        # Single CPU-GPU Synchronization at Epoch Boundary
        history['train_loss'].append((train_loss_acc / (train_steps + 1e-9)).item())
        history['val_loss'].append((val_loss_acc / (val_steps + 1e-9)).item())

        scheduler.step()
        gc.collect()

        # Telemetry Resolution (Executed once per epoch)
        epoch_telemetry = {}
        if train_chunk_count > 0:
            for k, v in gpu_telemetry.items():
                epoch_telemetry[k] = (v / train_chunk_count).item()
            
            epoch_p_mean = (epoch_p_mean_sum / train_chunk_count).cpu()
            top_topic_val, top_topic_idx = epoch_p_mean.max(dim=0)
            epoch_telemetry['top_t_pct'] = top_topic_val.item() * 100.0
            epoch_telemetry['top_t_id'] = top_topic_idx.item()

        current_lr = round(optimizer.param_groups[0]['lr'], 6)
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': round(history['train_loss'][-1], 4),
            'val_loss': round(history['val_loss'][-1], 4),
            'lr': current_lr,
            'loss_components': {
                'rec': round(epoch_telemetry.get('l_rec', 0.0), 4),
                'anc': round(epoch_telemetry.get('l_anc', 0.0), 4),
                'ort': round(epoch_telemetry.get('l_ort', 0.0), 4)
            },
            'entropy': round(epoch_telemetry.get('ent', 0.0), 4),
            'collapse_ratio': round(epoch_telemetry.get('col_r', 0.0), 4),
            'kl_weight': round(epoch_telemetry.get('kl_w', 0.0), 4),
            'g_w': round(epoch_telemetry.get('g_w', 0.0), 2),
            'p_w': round(epoch_telemetry.get('p_w', 0.0), 2),
            'top_topic_id': epoch_telemetry.get('top_t_id', 0),
            'top_topic_pct': round(epoch_telemetry.get('top_t_pct', 0.0), 2)
        }
        history['autopsy_metrics'].append(epoch_metrics)

        # -------------------------------------------------------------
        # 1. Pareto Composite Quality Checkpointing (Strict Score Trigger)
        # -------------------------------------------------------------
        current_rec = epoch_telemetry.get("l_rec", float("inf"))
        current_pw = epoch_telemetry.get("p_w", 0.0)
        composite_score = current_rec / max(1.0, math.sqrt(current_pw / 100.0))

        # Saves checkpoint ONLY when a new true Pareto peak is achieved
        if composite_score < best_composite_score and not nan_detected:
            best_composite_score = composite_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_composite_score": best_composite_score,
                "metrics": epoch_metrics,
                "history": history
            }, checkpoint_path)

        # -------------------------------------------------------------
        # 2. Periodic Resume State Saving (Every 5 Epochs)
        # -------------------------------------------------------------
        if ((epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1) and not nan_detected:
            autopsy_dir = out_dir / "autopsy_checkpoints"
            autopsy_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": epoch_metrics}, autopsy_dir / f"epoch_{(epoch+1):03d}.pt")
            
            # Serialize complete tracker brain with zero tensor dependencies
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
                tqdm.write(
                    f" [Ep {(epoch+1):03d}] Pure_Rec:{epoch_telemetry.get('l_rec', 0.0):<5.3f} "
                    f"V_Loss:{history['val_loss'][-1]:<5.3f} (Tot_Loss:{history['train_loss'][-1]:<5.3f}) | "
                    f"G_W:{epoch_telemetry.get('g_w', 0.0):<4.1f}% P_W:{epoch_telemetry.get('p_w', 0.0):<4.1f}% "
                    f"TopT:{epoch_telemetry.get('top_t_id', 0)}({epoch_telemetry.get('top_t_pct', 0.0):<4.1f}%) "
                    f"Ent:{epoch_telemetry.get('ent', 0.0):<4.2f} | "
                    f"KL_W:{epoch_telemetry.get('kl_w', 0.0):<4.2f} "
                    f"L_Anc:{epoch_telemetry.get('l_anc', 0.0):<5.3f} "
                    f"L_Ort:{epoch_telemetry.get('l_ort', 0.0):<5.3f}"
                )

        # -------------------------------------------------------------
        # 2. EVERY EPOCH: Tracker Math & Failsafes
        # -------------------------------------------------------------
        epochs_remaining = cfg.epochs - epoch - 1
        if tracker.phase == 1 and epochs_remaining <= 20:
            tqdm.write(f"\n[!] Approaching max epochs ({cfg.epochs}). Forcing Phase 2.")
            tracker.force_phase2(epoch, epoch_telemetry.get('l_rec', 0.0))

        # Track previous phase state so we can print the transition cleanly
        was_phase_1 = (tracker.phase == 1)

        # Step the tracker with the full telemetry dictionary
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

    if checkpoint_path.exists():
        print(f"  ↳ Restoring in-memory model to best Pareto checkpoint ({checkpoint_path.name})...")
        best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    return model, history

def train_gnn(
    graph_paths: list[Path], common_genes: list[str]
) -> tuple[LibellaGNN, dict[str, list], int]:
    """Master orchestrator for GNN training phase."""
    out_dirs = paths.make_dirs(cfg.suffix)
    checkpoint_path = out_dirs["checkpoint"]

    # 1. Get priors/slots as normal
    init_components, optimal_k, _ = get_priors(graph_paths)
    
    if cfg.phase == "EXTRACT_PRIORS":
        print("\n[✓] Phase 'EXTRACT_PRIORS' complete. Exiting before training.")
        import sys; sys.exit(0)

    n_extra_slots = cfg.extra_topics
    if n_extra_slots > 0:
        print(f"  ↳ Appending {n_extra_slots} extra randomized slots to the prior.")
        if init_components is not None:
            extra_slots = np.zeros((n_extra_slots, init_components.shape[1]))
            init_components = np.vstack([init_components, extra_slots])
            
        optimal_k += n_extra_slots
    gc.collect()

    # 2. _init_model will automatically check resume_latest.pt or checkpoint_path
    model, optimizer, scheduler, best_composite_score, tracker_state, history, start_epoch = _init_model(
        common_genes, optimal_k, init_components, checkpoint_path
    )
    del init_components
    gc.collect()

    # 3. Only skip if all epochs were actually completed
    if start_epoch >= cfg.epochs:
        print(f"-> Training already reached target epoch ({start_epoch}/{cfg.epochs}). Skipping loop.")
        return model, history, optimal_k

    training_cache = _prep_ssd_chunks(graph_paths)
    gc.collect()  

    # 4. Resume training loop from start_epoch
    model, history = _train_loop(
        model, optimizer, scheduler, training_cache, start_epoch, best_composite_score, tracker_state, history
    )
    gc.collect()
    
    return model, history, optimal_k