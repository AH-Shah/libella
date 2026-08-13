import time
import torch
import gc
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np

# --- Import your project modules ---
from libella.config import cfg, paths
from libella.model import LibellaGNN
from libella.data import pad_mps_shapes

warnings.filterwarnings("ignore")

# --- Mocking your exact batching logic ---
def make_meta_batches(training_cache, meta_batch_size=4):
    """Simple chunk grouper for telemetry."""
    meta_batches = []
    for i in range(0, len(training_cache), meta_batch_size):
        meta_batches.append(training_cache[i:i+meta_batch_size])
    return meta_batches

def prefetch_batches(meta_batches):
    """Your exact async fetcher."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        if not meta_batches: return
            
        futures = [executor.submit(torch.load, b['chunk_file'], map_location='cpu', weights_only=False) for b in meta_batches[0]]
        for i in range(len(meta_batches)):
            loaded_chunks = [f.result() for f in futures]
            if i + 1 < len(meta_batches):
                futures = [executor.submit(torch.load, b['chunk_file'], map_location='cpu', weights_only=False) for b in meta_batches[i+1]]
            yield meta_batches[i], loaded_chunks

def run_epoch_telemetry():
    device = torch.device("mps")
    torch.mps.empty_cache()
    
    print("="*80)
    print(" 🚀 PIPELINE BOTTLENECK TELEMETRY 🚀")
    print("="*80)

    # 1. Build Training Cache from your directory
    chunk_dir = Path("/Users/Hemato/project_3/benchmark/results_profile_50ep/run/temp_training_chunks/")
    chunk_files = list(chunk_dir.glob("*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No .pt files found in {chunk_dir}")
        
    print(f"[+] Found {len(chunk_files)} chunks. Simulating 1 Epoch...")
    training_cache = [{"chunk_file": f} for f in chunk_files]
    
    model = LibellaGNN(in_channels=2000, n_metaprograms=30).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    accumulation_steps = getattr(cfg, "meta_batch_size", 4)
    meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)

    # Timing metrics
    total_wait_time = 0.0
    total_prep_time = 0.0
    total_gpu_time = 0.0
    total_epoch_time = 0.0

    epoch_start = time.perf_counter()
    fetcher = prefetch_batches(meta_batches)

    # We manually iterate to measure the wait time EXACTLY
    while True:
        torch.mps.synchronize()
        wait_start = time.perf_counter()
        
        try:
            meta_meta, loaded_chunks = next(fetcher)
        except StopIteration:
            break
            
        torch.mps.synchronize()
        wait_time = (time.perf_counter() - wait_start) * 1000
        total_wait_time += wait_time

        for chunk_idx, batch_ref in enumerate(meta_meta):
            batch = loaded_chunks[chunk_idx]
            
            # --- MEASURE DATA PREP TIME ---
            torch.mps.synchronize()
            prep_start = time.perf_counter()
            
            # (Assuming you kept the SciPy method based on our last conversation, 
            # if not, swap this back to PyTorch COO dense creation)
            x_dense_np = batch["x"].toarray()
            x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)
            
            adj_coo = batch["adj"].tocoo()
            src = torch.from_numpy(adj_coo.row).to(torch.int32).to(device)
            dst = torch.from_numpy(adj_coo.col).to(torch.int32).to(device)
            weights = torch.from_numpy(adj_coo.data).to(torch.float32).to(device)
            
            x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
            if device.type != 'mps':
                src = src.to(torch.int64)
                dst = dst.to(torch.int64)
                
            local_core = batch["local_core_idx"]
            core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
            t_mask_gpu = torch.from_numpy(batch["train_mask"][local_core]).to(dtype=torch.bool, device=device)
            train_idx = core_gpu[t_mask_gpu]

            torch.mps.synchronize()
            prep_time = (time.perf_counter() - prep_start) * 1000
            total_prep_time += prep_time

            # --- MEASURE GPU COMPUTE TIME ---
            torch.mps.synchronize()
            gpu_start = time.perf_counter()

            fracs, pure_anchors = model(x, src, dst, weights)
            
            f_train = fracs[train_idx]
            x_train = x[train_idx]

            p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
            current_p_mean = p_train.mean(dim=0)
            uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
            target_f_dist = torch.clamp(uniform_prior * 2.0 - current_p_mean.detach(), min=1e-5)
            target_f_dist = target_f_dist / target_f_dist.sum()
            recon = f_train @ pure_anchors

            loss, _ = model.calc_loss(
                recon, x_train, pure_anchors, None, ep=1, total_epochs=30, 
                f_train=f_train, target_f_dist=target_f_dist, kl_weight=5.0
            )

            (loss / len(meta_meta)).backward()

            torch.mps.synchronize()
            gpu_time = (time.perf_counter() - gpu_start) * 1000
            total_gpu_time += gpu_time

        # Optimizer runs once per meta-batch
        torch.mps.synchronize()
        opt_start = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        total_gpu_time += (time.perf_counter() - opt_start) * 1000

    total_epoch_time = time.perf_counter() - epoch_start

    print("\n" + "="*80)
    print(" 📊 EPOCH TIMELINE RESULTS 📊")
    print("="*80)
    print(f"Total Chunks Processed:  {len(chunk_files)}")
    print(f"Real Epoch Time (Wall):  {total_epoch_time:.2f} seconds\n")
    
    print(f"1. SSD/Thread Wait Time: {total_wait_time / 1000:.2f} seconds (Time spent frozen waiting for I/O)")
    print(f"2. Data Prep Time:       {total_prep_time / 1000:.2f} seconds (CPU densification, Padding, GPU Transfer)")
    print(f"3. GPU Compute Time:     {total_gpu_time / 1000:.2f} seconds (Forward + Loss + Backward + Step)")
    print("="*80)
    
    if (total_wait_time + total_prep_time) > total_gpu_time:
        print("🚨 DIAGNOSIS: YOU ARE DATALOADER BOTTLENECKED.")
        print("Your GPU is starving for data. The CPU/SSD cannot feed it fast enough.")
    else:
        print("✅ DIAGNOSIS: YOU ARE GPU BOTTLENECKED.")
        print("The data pipeline is keeping up smoothly.")

if __name__ == "__main__":
    run_epoch_telemetry()