import time
import torch
import gc
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# --- Import your project modules ---
from libella.config import cfg, paths
from libella.model import LibellaGNN
from libella.data import pad_mps_shapes

warnings.filterwarnings("ignore")

# --- Global Aggregators ---
phase_metrics = defaultdict(lambda: {"time": 0.0, "vram_delta": 0.0, "peak_vram": 0.0, "count": 0})
module_metrics = defaultdict(lambda: {"time": 0.0, "vram_delta": 0.0, "count": 0})

class TrackMPS:
    def __init__(self, step_name: str, group_by_chunk=True):
        self.step_name = step_name
        self.group_by_chunk = group_by_chunk

    def __enter__(self):
        torch.mps.synchronize()
        self.t0 = time.perf_counter()
        self.m0 = torch.mps.current_allocated_memory() / (1024 ** 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.mps.synchronize()
        self.t1 = time.perf_counter()
        self.m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        
        dt_ms = (self.t1 - self.t0) * 1000
        dm_mb = self.m1 - self.m0
        
        stats = phase_metrics[self.step_name]
        stats["time"] += dt_ms
        stats["vram_delta"] += dm_mb
        stats["peak_vram"] = max(stats["peak_vram"], self.m1)
        stats["count"] += 1

def attach_telemetry_hooks(model: torch.nn.Module):
    def pre_hook_fn(module, input, name):
        torch.mps.synchronize()
        module._telemetry_start_time = time.perf_counter()
        module._telemetry_start_mem = torch.mps.current_allocated_memory() / (1024 ** 2)

    def hook_fn(module, input, output, name):
        torch.mps.synchronize()
        t1 = time.perf_counter()
        m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        
        dt_ms = (t1 - module._telemetry_start_time) * 1000
        dm_mb = m1 - module._telemetry_start_mem
        
        stats = module_metrics[name]
        stats["time"] += dt_ms
        stats["vram_delta"] += dm_mb
        stats["count"] += 1

    for name, layer in model.named_children():
        layer.register_forward_pre_hook(lambda m, i, n=name: pre_hook_fn(m, i, n))
        layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

# --- Mocking your exact batching logic ---
def make_meta_batches(training_cache, meta_batch_size=4):
    meta_batches = []
    for i in range(0, len(training_cache), meta_batch_size):
        meta_batches.append(training_cache[i:i+meta_batch_size])
    return meta_batches

def prefetch_batches(meta_batches):
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
    
    chunk_dir = Path("/Users/Hemato/project_3/benchmark/results_profile_50ep/run/temp_training_chunks/")
    chunk_files = list(chunk_dir.glob("*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No .pt files found in {chunk_dir}")
        
    print(f"\n[+] Executing High-Res Telemetry on {len(chunk_files)} chunks...")
    training_cache = [{"chunk_file": f} for f in chunk_files]
    
    model = LibellaGNN(in_channels=2000, n_metaprograms=30).to(device)
    model.train()
    attach_telemetry_hooks(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    meta_batches = make_meta_batches(training_cache, meta_batch_size=getattr(cfg, "meta_batch_size", 4))
    fetcher = prefetch_batches(meta_batches)

    # Note: Because we inject synchronize() everywhere, the total epoch will take slightly longer
    # than 16s. This is normal (profiling overhead). The ratios remain 100% accurate.
    
    total_chunks = 0
    total_meta_steps = 0

    while True:
        with TrackMPS("[0] Async Dataloader Wait"):
            try:
                meta_meta, loaded_chunks = next(fetcher)
            except StopIteration:
                break

        for chunk_idx, batch_ref in enumerate(meta_meta):
            batch = loaded_chunks[chunk_idx]
            total_chunks += 1
            
            with TrackMPS("[1] CPU Densify & Extract (SciPy)"):
                x_dense_np = batch["x"].toarray()
                x = torch.from_numpy(x_dense_np).to(dtype=torch.float32)
                
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32)

            with TrackMPS("[1] Transfer to GPU"):
                x, src, dst, weights = x.to(device), src.to(device), dst.to(device), weights.to(device)
                
            with TrackMPS("[1] Pad MPS Shapes"):
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != 'mps':
                    src, dst = src.to(torch.int64), dst.to(torch.int64)
                    
            with TrackMPS("[2] TOTAL FORWARD PASS"):
                fracs, pure_anchors = model(x, src, dst, weights)
            
            with TrackMPS("[3] Masking & Targets Setup"):
                local_core = batch["local_core_idx"]
                core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
                t_mask_gpu = torch.from_numpy(batch["train_mask"][local_core]).to(dtype=torch.bool, device=device)
                train_idx = core_gpu[t_mask_gpu]

                f_train = fracs[train_idx]
                x_train = x[train_idx]

                p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                current_p_mean = p_train.mean(dim=0)
                uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
                target_f_dist = torch.clamp(uniform_prior * 2.0 - current_p_mean.detach(), min=1e-5)
                target_f_dist = target_f_dist / target_f_dist.sum()
                recon = f_train @ pure_anchors

            with TrackMPS("[4] Loss Function"):
                loss, _ = model.calc_loss(
                    recon, x_train, pure_anchors, None, ep=1, total_epochs=30, 
                    f_train=f_train, target_f_dist=target_f_dist, kl_weight=5.0
                )

            with TrackMPS("[5] Backward Pass"):
                (loss / len(meta_meta)).backward()

        total_meta_steps += 1
        with TrackMPS("[6] Optimizer (Clip + Step + Zero)"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    print("\n" + "="*90)
    print(f" 🚀 FULL RESOLUTION AVERAGE (Across {total_chunks} Chunks) 🚀")
    print("="*90)
    
    # 1. Print Phase Metrics
    for step_name, stats in sorted(phase_metrics.items()):
        if stats["count"] == 0: continue
        
        # Optimizer runs per meta-batch, everything else runs per chunk
        divisor = total_meta_steps if "[6]" in step_name or "[0]" in step_name else total_chunks
        
        avg_t = stats["time"] / divisor
        avg_v = stats["vram_delta"] / divisor
        peak = stats["peak_vram"]
        
        color = "\033[92m" if avg_v <= 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{step_name:<36} | Avg Time: {avg_t:>7.2f} ms | {color}Avg VRAM Δ: {avg_v:>7.2f} MB{reset} | Peak: {peak:>7.2f} MB")
        
        # If Forward Pass, print Submodules
        if "[2]" in step_name:
            for m_name, m_stats in module_metrics.items():
                if m_stats["count"] == 0: continue
                m_avg_t = m_stats["time"] / total_chunks
                m_avg_v = m_stats["vram_delta"] / total_chunks
                print(f"    ↳ [Sub] {m_name:<26} | Avg Time: {m_avg_t:>7.2f} ms | Avg VRAM Δ: {m_avg_v:>7.2f} MB")
                
    print("="*90)

if __name__ == "__main__":
    run_epoch_telemetry()