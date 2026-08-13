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
from tqdm import tqdm

from .config import cfg, paths
from .data import (
    SpatialBatcher,
    make_meta_batches,
    pad_mps_shapes,
    pt_to_scipy_csr,
)
from .model import LibellaGNN
from .prior import get_priors
from .utils import get_device



def _init_model(
    common_genes: list[str], 
    optimal_k: int, 
    init_components: np.ndarray | None, 
    checkpoint_path: Path | None = None
) -> tuple[LibellaGNN, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, float, dict[str, list], int]:
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
    
    best_val_loss = float("inf")
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
                
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"  ↳ Successfully resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"  ↳ [!] Failed to load checkpoint: {e}. Raising error to prevent accidental overwrite.")
            raise e 

    return model, optimizer, scheduler, best_val_loss, history, start_epoch

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
                chunk_file = tmp_chunk_dir / f"{data.patient_name}_chunk_{chunk_idx}.pt"
                torch.save(chunk_data, chunk_file)
                
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
    """Async fetch of SSD chunks parallel to GPU compute."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        if not meta_batches:
            return
            
        # Kick off the very first SSD read
        futures = [executor.submit(torch.load, b['chunk_file'], map_location='cpu', weights_only=False) for b in meta_batches[0]]
        
        for i in range(len(meta_batches)):
            # Wait for current batch to finish loading
            loaded_chunks = [f.result() for f in futures]
            
            # 🚨 Kick off the reads for the NEXT batch BEFORE yielding to the GPU!
            if i + 1 < len(meta_batches):
                futures = [executor.submit(torch.load, b['chunk_file'], map_location='cpu', weights_only=False) for b in meta_batches[i+1]]
                
            yield meta_batches[i], loaded_chunks
            
def _train_loop(
    model: LibellaGNN, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LRScheduler, 
    training_cache: list[dict[str, Any]], 
    start_epoch: int, 
    best_val_loss: float, 
    history: dict[str, list]
) -> tuple[LibellaGNN, dict[str, list]]:
    print("\n-> Spatial Distillation...")
    device = get_device()
    out_dirs = paths.make_dirs(cfg.suffix)
    out_dir = out_dirs["out"]
    checkpoint_path = out_dirs["checkpoint"]
    
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)  
    ema_mean = None

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training", leave=False):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        train_steps, val_steps = 0, 0

        
        epoch_telemetry = {
            'ent': 0.0, 'col_r': 0.0, 'kl_w': 0.0, 'g_w': 0.0, 'p_w': 0.0, 
            'l_rec': 0.0, 'l_anc': 0.0, 'l_ort': 0.0
        }
        epoch_p_mean_sum = 0
        train_chunk_count = 0
        
        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        optimizer.zero_grad(set_to_none=True)
        
        total_steps_per_epoch = len(meta_batches)
        alpha_ema = min(0.001, 1.0 / (total_steps_per_epoch * 5.0 + 1e-9)) 
        nan_detected = False

        for step, (meta_meta, loaded_chunks) in enumerate(prefetch_batches(meta_batches)):
            optimizer.zero_grad(set_to_none=True)
            for chunk_idx, batch_ref in enumerate(meta_meta):
                batch = loaded_chunks[chunk_idx]
                
                x_dense_np = batch["x"].toarray()
                x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)
                
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32)
                del adj_coo
                
                if model.training:
                    keep_mask = torch.rand(src.size(0)) > 0.40
                    src = src[keep_mask]
                    dst = dst[keep_mask]
                    weights = weights[keep_mask]
                
                src = src.to(device)
                dst = dst.to(device)
                weights = weights.to(device)
                
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                
                # 🚨 Minimal Fix: Only Windows/CUDA get int64. Mac stays int32.
                if device.type != 'mps':
                    src = src.to(torch.int64)
                    dst = dst.to(torch.int64)

                linear_progress = epoch / max(1, cfg.epochs - 1)
                adjusted_progress = max(0.0, (linear_progress - 0.25) / 0.75)
                progress = 0.5 * (1.0 - math.cos(math.pi * adjusted_progress))

                model.current_scale = cfg.scale_start + ((cfg.scale_end - cfg.scale_start) * progress)    
                model.current_alpha = cfg.alpha_start + ((cfg.alpha_end - cfg.alpha_start) * progress)     
                model.current_temp = cfg.temp_start - ((cfg.temp_start - cfg.temp_end) * progress)    

                fracs, pure_anchors = model(x, src, dst, weights)
                
                local_core = batch["local_core_idx"]
                core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    fracs, pure_anchors = model(x, src, dst, weights)
                
                t_mask_np = batch["train_mask"][local_core]
                if t_mask_np.sum() > 0:
                    t_mask_gpu = torch.from_numpy(t_mask_np).to(dtype=torch.bool, device=device)
                    train_idx = core_gpu[t_mask_gpu]

                    f_train = fracs[train_idx]
                    x_train = x[train_idx]
                    

                    p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                    current_p_mean = p_train.mean(dim=0)
                    
                    # 🚨 Glacier EMA Target Calculation
                    uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
                    if ema_mean is None:
                        ema_mean = current_p_mean.detach()
                    else:
                        ema_mean = alpha_ema * current_p_mean.detach() + (1 - alpha_ema) * ema_mean
                    
                    ideal_c = uniform_prior * 2.0 - ema_mean
                    ideal_c = torch.clamp(ideal_c, min=1e-5) 
                    target_f_dist = ideal_c / ideal_c.sum()
                        
                    ema_entropy = -torch.sum(ema_mean * torch.log(ema_mean + 1e-9))
                    max_entropy = np.log(pure_anchors.shape[0])
                    collapse_ratio = torch.clamp(1.0 - (ema_entropy / max_entropy), min=0.0, max=1.0).item()


                    ideal_p = 1.0 / pure_anchors.shape[0]
                    peak_p = ema_mean.max().item()
                    
                    hub_multiplier = max(0.0, (peak_p / 0.15) - 1.0) * 10.0 

                    dynamic_kl_w = cfg.kl_base + (collapse_ratio * cfg.kl_collapse_weight) + (hub_multiplier) 

                    recon = f_train @ pure_anchors
                    true_batch_loss, base_recon_val = model.calc_loss(
                        recon, x_train, pure_anchors, None, epoch, cfg.epochs, 
                        f_train=f_train, target_f_dist=target_f_dist, kl_weight=dynamic_kl_w
                    )

                    if torch.isnan(true_batch_loss) or torch.isinf(true_batch_loss):
                        nan_detected = True
                        break

                    (true_batch_loss / len(meta_meta)).backward()
                    
                    train_loss += true_batch_loss.item() 
                    train_steps += 1

                    g_w_val = pure_anchors.max(dim=1).values.mean().item() * 100.0
                    p_w_val = p_train.max(dim=1).values.mean().item() * 100.0

                    ent_val = ema_entropy.detach().cpu().item() if isinstance(ema_entropy, torch.Tensor) else ema_entropy
                    epoch_telemetry['ent'] += ent_val
                    epoch_telemetry['col_r'] += collapse_ratio
                    epoch_telemetry['kl_w'] += dynamic_kl_w
                    epoch_telemetry['g_w'] += g_w_val
                    epoch_telemetry['p_w'] += p_w_val
                    
  
                    chunk_losses = getattr(model, '_last_losses', {})
                    epoch_telemetry['l_rec'] += chunk_losses.get('rec', 0.0)
                    epoch_telemetry['l_anc'] += chunk_losses.get('anc', 0.0)
                    epoch_telemetry['l_ort'] += chunk_losses.get('ort', 0.0)
                    
                    # Accumulate chunk's p_mean on CPU to find true epoch dominant topic
                    epoch_p_mean_sum += current_p_mean.detach().cpu()
                    train_chunk_count += 1

                    
                    del t_mask_gpu, train_idx, f_train, x_train, p_train, current_p_mean, uniform_prior, target_f_dist, recon, true_batch_loss, base_recon_val

                v_mask_np = batch["val_mask"][local_core]
                if v_mask_np.sum() > 0:
                    v_mask_gpu = torch.from_numpy(v_mask_np).to(dtype=torch.bool, device=device)
                    val_idx = core_gpu[v_mask_gpu]

                    model.eval() 
                    with torch.no_grad():
                        clean_fracs, clean_anchors = model(x, src, dst, weights)

                        f_val = fracs[val_idx]
                        x_val = x[val_idx]
                        val_recon = f_val @ pure_anchors
                        
                        w_mat = 1.0 + (x_val > 0).float() * (model.dynamic_w_ema - 1.0)
                        is_non_zero_val = (x_val > 0)
                        
                        raw_delta_val = val_recon - x_val
                        asym_val = 1.0 + (is_non_zero_val.float() * 2.0) * (raw_delta_val < 0).float()
                        
                        scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-cfg.delta_clamp, max=cfg.delta_clamp)
                        
                    
                        zero_expectation_mask = is_non_zero_val.float() + (~is_non_zero_val).float() * cfg.zero_mask_rate
                        
                        masked_w_mat_val = w_mat * zero_expectation_mask
                        
                        val_loss_sum = torch.sum(masked_w_mat_val * torch.log(torch.cosh(scaled_delta_val + 1e-6)))
                        
                        
                        N_cells_val = torch.clamp(torch.tensor(x_val.shape[0], dtype=torch.float32, device=device), min=1.0)
                        # N_genes_val = x_val.shape[1]
                        val_log_cosh = val_loss_sum / N_cells_val
                    
                        val_loss += val_log_cosh.item()
                        val_steps += 1

                    model.train()     
                    del v_mask_gpu, val_idx, clean_fracs, clean_anchors, f_val, x_val, val_recon, w_mat, raw_delta_val, asym_val, scaled_delta_val, val_loss_sum, val_log_cosh
                    

                del batch, src, dst, weights, x, fracs, pure_anchors, core_gpu, local_core, t_mask_np, v_mask_np
                model.current_f_prob = None  
                

            # A. OPTIMIZER STEP (Runs ONCE after accumulating all chunks in the meta_batch)
            if nan_detected:
                optimizer.zero_grad(set_to_none=True) # Flush bad accumulation
                break
            

            base_params = [p for n, p in model.named_parameters() if 'topic_gene_logits' not in n]
            anchor_params = [p for n, p in model.named_parameters() if 'topic_gene_logits' in n]

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            del loaded_chunks
            
        if nan_detected:
            print(f"\n  ↳ [!] NaN gradient detected at Epoch {epoch}. Halting training.")
            break
                    
        history['train_loss'].append(train_loss / (train_steps + 1e-9))
        history['val_loss'].append(val_loss / (val_steps + 1e-9))

        scheduler.step()
        gc.collect()

        # 🚨 FAST AVERAGE: Calculate true epoch-level metrics
        if train_chunk_count > 0:
            for k in epoch_telemetry:
                epoch_telemetry[k] /= train_chunk_count
            
            # Calculate true dominant topic over the whole epoch
            epoch_p_mean = epoch_p_mean_sum / train_chunk_count
            top_topic_val, top_topic_idx = epoch_p_mean.max(dim=0)
            epoch_telemetry['top_t_pct'] = top_topic_val.item() * 100.0
            epoch_telemetry['top_t_id'] = top_topic_idx.item()

        # (Delete the old ls = {...} snapshot line)
        current_lr = round(optimizer.param_groups[0]['lr'], 6)
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': round(history['train_loss'][-1], 4), # Blended Total
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

        if ((epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1) and not nan_detected:
            autopsy_dir = out_dir / "autopsy_checkpoints"
            autopsy_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": epoch_metrics}, autopsy_dir / f"epoch_{(epoch+1):03d}.pt")
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history
            }, out_dir / "resume_latest.pt")

            if history["val_loss"][-1] < best_val_loss:
                best_val_loss = history["val_loss"][-1]
                torch.save({
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss, 
                    "history": history
                }, checkpoint_path)

            with torch.no_grad():
                g_w = epoch_telemetry.get('g_w', 0.0)
                p_w = epoch_telemetry.get('p_w', 0.0)
                top_id = epoch_telemetry.get('top_t_id', 0)
                top_pct = epoch_telemetry.get('top_t_pct', 0.0)
                ent_val = epoch_telemetry.get('ent', 0.0)
                kl_w = epoch_telemetry.get('kl_w', 0.0)
                
                # Retrieve pure averaged losses
                l_rec = epoch_telemetry.get('l_rec', 0.0)
                l_anc = epoch_telemetry.get('l_anc', 0.0)
                l_ort = epoch_telemetry.get('l_ort', 0.0)
                
                # 🚨 LOG FIX: 'Pure_Rec' is now the headline training convergence metric!
                tqdm.write(
                    f" [Ep {epoch:03d}] Pure_Rec:{l_rec:<5.3f} V_Loss:{history['val_loss'][-1]:<5.3f} (Tot_Loss:{history['train_loss'][-1]:<5.3f}) | "
                    f"G_W:{g_w:<4.1f}% P_W:{p_w:<4.1f}% TopT:{top_id}({top_pct:<4.1f}%) Ent:{ent_val:<4.2f} | "
                    f"KL_W:{kl_w:<4.2f} L_Anc:{l_anc:<4.2f} L_Ort:{l_ort:<4.2f}"
                )


    torch.save({
        "epoch": cfg.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history
    }, checkpoint_path)
        
    return model, history

def train_gnn(
    graph_paths: list[Path], common_genes: list[str]
) -> tuple[LibellaGNN, dict[str, list], int]:
    """Master orchestrator for GNN training phase."""
    out_dirs = paths.make_dirs(cfg.suffix)
    final_path = out_dirs["checkpoint"]

    if final_path.exists():
        print(f"-> Completed model checkpoint found at {final_path}. Loading final state...")
        checkpoint = torch.load(final_path, map_location=get_device(), weights_only=False)
        
        _, optimal_k, _ = get_priors(graph_paths)
        
        model, optimizer, scheduler, _, history, _ = _init_model(
            common_genes, optimal_k, None, checkpoint
        )
        return model, history, optimal_k

    init_components, optimal_k, checkpoint = get_priors(graph_paths)
    
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

    model, optimizer, scheduler, best_val_loss, history, start_epoch = _init_model(
        common_genes, optimal_k, init_components, checkpoint
    )
    del checkpoint, init_components
    gc.collect()

    if start_epoch >= cfg.epochs:
        print(f"-> Training already reached target epoch ({start_epoch}/{cfg.epochs}). Skipping loop.")
        return model, history, optimal_k

    training_cache = _prep_ssd_chunks(graph_paths)
    gc.collect()  

    model, history = _train_loop(
        model, optimizer, scheduler, training_cache, start_epoch, best_val_loss, history
    )
    gc.collect()
    
    return model, history, optimal_k
    
