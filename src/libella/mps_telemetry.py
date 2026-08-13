import time
import torch
import warnings
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- Import your project modules ---
from libella.config import cfg, paths
from libella.model import LibellaGNN
from libella.data import pad_mps_shapes

warnings.filterwarnings("ignore")

# --- Background VRAM Poller ---
# This tracks memory in the background without EVER blocking the GPU
class VRAMMonitor:
    def __init__(self):
        self.keep_running = True
        self.peak_vram = 0.0
        self.current_vram = 0.0
        self.history = []

    def poll(self):
        while self.keep_running:
            mem = torch.mps.current_allocated_memory() / (1024 ** 2)
            self.current_vram = mem
            if mem > self.peak_vram:
                self.peak_vram = mem
            self.history.append(mem)
            time.sleep(0.005) # Poll every 5 milliseconds

# --- Mocking your exact batching logic ---
def make_meta_batches(training_cache, meta_batch_size=4):
    return [training_cache[i:i+meta_batch_size] for i in range(0, len(training_cache), meta_batch_size)]

def prefetch_batches(meta_batches):
    with ThreadPoolExecutor(max_workers=8) as executor:
        if not meta_batches: return
        futures = [executor.submit(torch.load, b['chunk_file'], map_location='cpu', weights_only=False) for b in meta_batches[0]]
        for i in range(len(meta_batches)):
            loaded_chunks = [f.result() for f in futures]
            if i + 1 < len(meta_batches):
                futures = [executor.submit(torch.load, b['chunk_file'], map_location='cpu', weights_only=False) for b in meta_batches[i+1]]
            yield meta_batches[i], loaded_chunks

def run_true_async_epoch():
    device = torch.device("mps")
    torch.mps.empty_cache()
    
    chunk_dir = Path("/Users/Hemato/project_3/benchmark/results_profile_50ep/run/temp_training_chunks/")
    chunk_files = list(chunk_dir.glob("*.pt"))
    
    print(f"\n[+] Executing TRUE ASYNC Telemetry on {len(chunk_files)} chunks...")
    training_cache = [{"chunk_file": f} for f in chunk_files]
    
    model = LibellaGNN(in_channels=2000, n_metaprograms=30).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    meta_batches = make_meta_batches(training_cache, meta_batch_size=getattr(cfg, "meta_batch_size", 4))
    
    # Start the background VRAM thread
    monitor = VRAMMonitor()
    monitor_thread = threading.Thread(target=monitor.poll)
    monitor_thread.start()

    epoch_start_time = time.time()
    fetcher = prefetch_batches(meta_batches)

    total_chunks = 0
    t_data_prep = 0.0
    t_forward_loss = 0.0
    t_backward_opt = 0.0

    print("Running pipeline at full speed... (No Syncs)")

    while True:
        try:
            meta_meta, loaded_chunks = next(fetcher)
        except StopIteration:
            break

        for chunk_idx, batch_ref in enumerate(meta_meta):
            batch = loaded_chunks[chunk_idx]
            total_chunks += 1
            
            # --- MEASURE 1: DATA PREP (CPU -> GPU) ---
            t0 = time.time()
            x_dense_np = batch["x"].toarray()
            x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)
            
            adj_coo = batch["adj"].tocoo()
            src = torch.from_numpy(adj_coo.row).to(torch.int32).to(device)
            dst = torch.from_numpy(adj_coo.col).to(torch.int32).to(device)
            weights = torch.from_numpy(adj_coo.data).to(torch.float32).to(device)
            
            x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
            if device.type != 'mps':
                src, dst = src.to(torch.int64), dst.to(torch.int64)

            local_core = batch["local_core_idx"]
            core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
            t_mask_gpu = torch.from_numpy(batch["train_mask"][local_core]).to(dtype=torch.bool, device=device)
            train_idx = core_gpu[t_mask_gpu]
            
            # 🚨 Single sync just to bound the data prep time precisely
            torch.mps.synchronize()
            t_data_prep += (time.time() - t0)

            # --- MEASURE 2: FORWARD + LOSS (Async Dispatch) ---
            t1 = time.time()
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
            t_forward_loss += (time.time() - t1)

            # --- MEASURE 3: BACKWARD (Async Dispatch) ---
            t2 = time.time()
            (loss / len(meta_meta)).backward()
            t_backward_opt += (time.time() - t2)

        # Optimizer Step
        t3 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        t_backward_opt += (time.time() - t3)

    # Final sync to wait for the GPU to finish the very last chunk's math
    torch.mps.synchronize()
    total_time = time.time() - epoch_start_time

    # Stop monitor
    monitor.keep_running = False
    monitor_thread.join()

    print("\n" + "="*80)
    print(f" 🚀 TRUE ASYNC PIPELINE RESULTS ({total_chunks} Chunks) 🚀")
    print("="*80)
    print(f"Total Wall Clock Time:  {total_time:.2f} seconds")
    print(f"True Peak VRAM Spike:   {monitor.peak_vram:.2f} MB")
    print("-" * 80)
    print(f"Avg Data Prep / Chunk:  {(t_data_prep / total_chunks) * 1000:.2f} ms")
    print(f"Avg CPU Fwd/Loss Disp.: {(t_forward_loss / total_chunks) * 1000:.2f} ms")
    print(f"Avg CPU Bwd/Opt Disp.:  {(t_backward_opt / total_chunks) * 1000:.2f} ms")
    print("="*80)
    
    # Check RAM volatility
    ram_swings = max(monitor.history) - min(monitor.history)
    print(f"VRAM Volatility (Max swing): {ram_swings:.2f} MB")
    if ram_swings > 2000:
        print("-> RAM is highly jerky. The garbage collector is struggling.")
    else:
        print("-> RAM is stable.")

if __name__ == "__main__":
    run_true_async_epoch()