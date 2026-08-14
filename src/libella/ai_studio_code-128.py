import sys
import os
from pathlib import Path

# --- Auto-resolve libella package path ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_SRC = CURRENT_DIR.parent / "src" if (CURRENT_DIR.parent / "src").exists() else CURRENT_DIR
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import gc
import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from libella.config import cfg, paths, RunConfig
from libella.utils import get_device
from libella.train import _init_model, make_meta_batches, prefetch_batches, pad_mps_shapes


def power_law(x, a, b):
    """Deep Learning Scaling Law: Steps = a * (Dataset_Size)^b"""
    return a * np.power(x, b)


def fit_scaling_curve(x_data, y_data):
    """Fits power law with a robust log-space fallback if non-linear fit diverges."""
    x_arr = np.array(x_data, dtype=np.float64)
    y_arr = np.array(y_data, dtype=np.float64)
    
    try:
        # p0: reasonable initial guesses for a (intercept) and b (scaling exponent ~0.3 - 0.7)
        popt, _ = curve_fit(
            power_law, 
            x_arr, 
            y_arr, 
            p0=[10.0, 0.5], 
            bounds=(0, [np.inf, 2.0]), 
            maxfev=5000
        )
        return popt[0], popt[1]
    except Exception:
        # Fallback: Log-linear regression log(y) = log(a) + b*log(x)
        log_x = np.log(x_arr)
        log_y = np.log(y_arr)
        slope, intercept = np.polyfit(log_x, log_y, deg=1)
        a = np.exp(intercept)
        b = max(0.01, min(slope, 1.5))
        return a, b


def run_pilot(subset_chunks, optimal_k, common_genes):
    """Runs a Phase 1-only loop to find the exact overfitting point."""
    device = get_device()
    model, optimizer, _, _, _, _ = _init_model(common_genes, optimal_k, None, None)
    model.train()
    
    # Lock Phase 1 parameters
    model.current_scale = getattr(cfg, "scale_start", 1.0)
    model.current_alpha = getattr(cfg, "alpha_start", 0.0)
    model.current_temp = getattr(cfg, "temp_start", 1.0)
    
    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    meta_batches = make_meta_batches(subset_chunks, meta_batch_size=accumulation_steps)
    
    best_val = float('inf')
    best_step = 1
    patience = 5
    counter = 0
    total_steps = 0
    
    max_epochs = 150
    
    for epoch in range(max_epochs):
        val_loss = 0.0
        val_steps = 0
        
        for step, (meta_meta, loaded_chunks) in enumerate(prefetch_batches(meta_batches)):
            optimizer.zero_grad(set_to_none=True)
            for chunk_idx, batch_ref in enumerate(meta_meta):
                batch = loaded_chunks[chunk_idx]
                
                # Setup Tensors
                x = torch.from_numpy(batch["x"].toarray()).to(dtype=torch.float32, device=device)
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32).to(device)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32).to(device)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32).to(device)
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                
                if device.type != 'mps':
                    src, dst = src.to(torch.int64), dst.to(torch.int64)

                fracs, pure_anchors = model(x, src, dst, weights)
                
                # --- Quick Val Loss ---
                local_core = batch["local_core_idx"]
                core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
                v_mask_np = batch["val_mask"][local_core]
                
                if v_mask_np.sum() > 0:
                    val_idx = core_gpu[torch.from_numpy(v_mask_np).to(dtype=torch.bool, device=device)]
                    f_val = fracs[val_idx]
                    x_val = x[val_idx]
                    val_recon = f_val @ pure_anchors
                    
                    is_non_zero_val = (x_val > 0)
                    w_mat = torch.where(is_non_zero_val, 1.0, 1.0)
                    zero_mask = torch.where(is_non_zero_val, 1.0, getattr(cfg, "zero_mask_rate", 0.1)).to(x_val.dtype)
                    
                    raw_delta = val_recon - x_val
                    asym = 1.0 + (is_non_zero_val.to(x_val.dtype) * 2.0) * (raw_delta < 0).to(x_val.dtype)
                    scaled_delta = torch.clamp(raw_delta * asym, min=-getattr(cfg, "delta_clamp", 15.0), max=getattr(cfg, "delta_clamp", 15.0))
                    
                    v_loss = torch.sum((w_mat * zero_mask) * torch.log(torch.cosh(scaled_delta + 1e-6)))
                    val_loss += (v_loss / max(1.0, float(x_val.shape[0]))).item()
                    val_steps += 1
                
                # --- Mock Training Step to advance weights ---
                t_mask_np = batch["train_mask"][local_core]
                if t_mask_np.sum() > 0:
                    t_idx = core_gpu[torch.from_numpy(t_mask_np).to(dtype=torch.bool, device=device)]
                    f_train, x_train = fracs[t_idx], x[t_idx]
                    recon = f_train @ pure_anchors
                    
                    # Pass f_train here instead of None
                    t_loss, _ = model.calc_loss(recon, x_train, f_train, pure_anchors, epoch, max_epochs)
                    (t_loss / len(meta_meta)).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, "grad_clip", 1.0))
            optimizer.step()
            total_steps += 1
            
        # Check Overfitting
        avg_val = val_loss / max(1, val_steps)
        if avg_val < best_val:
            best_val = avg_val
            best_step = total_steps
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            break

    # Clean up GPU allocation
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()

    return max(1, best_step)


def calibrate_scaling_laws():
    print("\n🚀 INITIATING EMPIRICAL SCALING CALIBRATION 🚀")
    out_dirs = paths.make_dirs(cfg.suffix)
    chunk_dir = out_dirs["out"] / "temp_training_chunks"
    
    all_chunks = sorted(list(chunk_dir.glob("*_chunk_*.pt")))
    if not all_chunks:
        print(f"[!] No pre-sliced chunks found in {chunk_dir}.")
        print("    Run the main pipeline up to 'BUILD_GRAPHS' first.")
        return
        
    # Dynamically extract actual gene dimension and K from chunks/cfg
    first_chunk = torch.load(all_chunks[0], map_location="cpu", weights_only=False)
    actual_num_genes = first_chunk["x"].shape[1]
    optimal_k = getattr(cfg, "k", 100)
    common_genes = [f"Gene_{i}" for i in range(actual_num_genes)]
    
    print(f"[*] Detected dataset structure: {len(all_chunks)} chunks | {actual_num_genes} genes | k={optimal_k}")

    batch_size = getattr(cfg, "batch_size", first_chunk["x"].shape[0])
    total_cells = len(all_chunks) * batch_size
    fractions = [0.05, 0.15, 0.25, 0.5, 1]
    
    results_cells = []
    results_steps = []
    
    for frac in fractions:
        n_chunks = max(1, int(len(all_chunks) * frac))
        subset = [
            {
                "chunk_file": c,
                "patient_name": c.stem.rsplit("_chunk_", 1)[0] if "_chunk_" in c.stem else "sample_0"
            }
            for c in random.sample(all_chunks, n_chunks)
        ]
        subset_cells = n_chunks * batch_size
        
        print(f"\n[➤] Running Pilot: {frac*100:.0f}% of Data ({subset_cells} cells, {n_chunks} chunks)...")
        opt_steps = run_pilot(subset, optimal_k, common_genes)
        
        print(f"    ↳ Overfit Boundary Detected at {opt_steps} optimization steps.")
        results_cells.append(subset_cells)
        results_steps.append(opt_steps)
        
    # --- Fit Scaling Law ---
    print("\n📊 Fitting Deep Learning Power Law (Steps = a * Cells^b)...")
    a_param, b_param = fit_scaling_curve(results_cells, results_steps)
    
    target_steps = power_law(total_cells, a_param, b_param)
    
    # Convert Steps back to Epochs for the full dataset
    chunks_per_epoch = max(1.0, len(all_chunks) / getattr(cfg, "meta_batch_size", 4))
    optimal_epochs = max(1, int(target_steps / chunks_per_epoch))
    p2_epochs = max(10, int(optimal_epochs * 0.3))
    
    print("\n" + "="*55)
    print(f"🎯 CALIBRATION COMPLETE FOR {total_cells} CELLS")
    print("="*55)
    print(f"Fitted Law: Steps = {a_param:.4f} * Cells^{b_param:.4f}")
    print(f"Predicted Optimal Steps: {int(target_steps)}")
    print(f"Chunks per Epoch:        {chunks_per_epoch:.1f}")
    print(f"\n✅ RECOMMENDED SCHEDULE:")
    print(f"   Phase 1 (Representation): {optimal_epochs} Epochs")
    print(f"   Phase 2 (Sparsification): {p2_epochs} Epochs")
    print(f"   Total cfg.epochs to set:  {optimal_epochs + p2_epochs}")
    print("="*55)
    
    # Plot curve
    plot_path = out_dirs["out"] / "scaling_law_calibration.png"
    x_line = np.linspace(min(results_cells), total_cells, 100)
    y_line = power_law(x_line, a_param, b_param)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_line, y_line, '--', color='gray', label="Fitted Scaling Law")
    plt.scatter(results_cells, results_steps, color='blue', s=100, label="Pilot Runs (Actual)")
    plt.scatter([total_cells], [target_steps], color='red', marker='*', s=200, label="Full Dataset Prediction")
    
    plt.title(f"Empirical Scaling Law ($Steps = {a_param:.2f} \\times Cells^{{{b_param:.2f}}}$)")
    plt.xlabel("Dataset Size (Cells)")
    plt.ylabel("Optimal Gradient Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"📉 Saved visualization to '{plot_path}'")


if __name__ == "__main__":
    calibrate_scaling_laws()