import time
import torch
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# --- Import your project modules ---
from libella.config import cfg 
from libella.model import LibellaGNN
from libella.data import pad_mps_shapes

warnings.filterwarnings("ignore")

# --- Async Event Trackers ---
phase_records = []
module_records = []

class TrackMPSAsync:
    """Drops async timestamps into the GPU queue without halting it."""
    def __init__(self, step_name: str):
        self.name = step_name
        self.start_evt = torch.mps.Event(enable_timing=True)
        self.end_evt = torch.mps.Event(enable_timing=True)

    def __enter__(self):
        self.m0 = torch.mps.current_allocated_memory() / (1024 ** 2)
        self.start_evt.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_evt.record()
        self.m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        phase_records.append((self.name, self.start_evt, self.end_evt, self.m0, self.m1))

def attach_async_hooks(model: torch.nn.Module):
    def pre_hook_fn(module, input, name):
        module._async_start_evt = torch.mps.Event(enable_timing=True)
        module._async_start_evt.record()
        module._async_m0 = torch.mps.current_allocated_memory() / (1024 ** 2)

    def hook_fn(module, input, output, name):
        end_evt = torch.mps.Event(enable_timing=True)
        end_evt.record()
        m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        module_records.append((name, module._async_start_evt, end_evt, module._async_m0, m1))

    for name, layer in model.named_children():
        layer.register_forward_pre_hook(lambda m, i, n=name: pre_hook_fn(m, i, n))
        layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

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

def run_async_telemetry():
    device = torch.device("mps")
    torch.mps.empty_cache()
    
    chunk_dir = Path("/Users/Hemato/project_3/benchmark/results_profile_50ep/run/temp_training_chunks/")
    chunk_files = list(chunk_dir.glob("*.pt"))
    
    print(f"\n[+] Executing TRUE ASYNC HIGH-RES Telemetry on {len(chunk_files)} chunks...")
    training_cache = [{"chunk_file": f} for f in chunk_files]
    
    model = LibellaGNN(in_channels=2000, n_metaprograms=30).to(device)
    model.train()
    attach_async_hooks(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    meta_batches = make_meta_batches(training_cache, meta_batch_size=getattr(cfg, "meta_batch_size", 4))
    fetcher = prefetch_batches(meta_batches)
    total_chunks = 0

    print("Running pipeline at full speed... (No Syncing during Math!)")
    wall_start = time.time()

    # THE HOT LOOP (Runs completely untouched)
    while True:
        try:
            meta_meta, loaded_chunks = next(fetcher)
        except StopIteration:
            break

        for chunk_idx, batch_ref in enumerate(meta_meta):
            batch = loaded_chunks[chunk_idx]
            total_chunks += 1
            
            with TrackMPSAsync("[1] CPU Densify & Extract (SciPy)"):
                x_dense_np = batch["x"].toarray()
                x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)
                
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).to(torch.int32).to(device)
                dst = torch.from_numpy(adj_coo.col).to(torch.int32).to(device)
                weights = torch.from_numpy(adj_coo.data).to(torch.float32).to(device)
                
                x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
                if device.type != 'mps':
                    src, dst = src.to(torch.int64), dst.to(torch.int64)

            with TrackMPSAsync("[2] TOTAL FORWARD PASS"):
                fracs, pure_anchors = model(x, src, dst, weights)
            
            with TrackMPSAsync("[3] Masking & Targets Setup"):
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

            with TrackMPSAsync("[4] Loss Function"):
                loss, _ = model.calc_loss(
                    recon, x_train, pure_anchors, None, ep=1, total_epochs=30, 
                    f_train=f_train, target_f_dist=target_f_dist, kl_weight=5.0
                )

            with TrackMPSAsync("[5] Backward Pass"):
                (loss / len(meta_meta)).backward()

        with TrackMPSAsync("[6] Optimizer (Clip + Step + Zero)"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # 🚨 ONLY SYNC ONCE AT THE VERY END
    torch.mps.synchronize()
    wall_time = time.time() - wall_start
    print("Done. Compiling Event Timers...")

    # --- AGGREGATE RESULTS ---
    phase_stats = defaultdict(lambda: {"time": 0.0, "vram_delta": 0.0, "peak_vram": 0.0, "count": 0})
    mod_stats = defaultdict(lambda: {"time": 0.0, "vram_delta": 0.0, "count": 0})

    for name, s_evt, e_evt, m0, m1 in phase_records:
        try:
            t = s_evt.elapsed_time(e_evt)
            phase_stats[name]["time"] += t
            phase_stats[name]["vram_delta"] += (m1 - m0)
            phase_stats[name]["peak_vram"] = max(phase_stats[name]["peak_vram"], m1)
            phase_stats[name]["count"] += 1
        except RuntimeError:
            pass # Metal fused this operation to 0.0ms; ignore and continue

    for name, s_evt, e_evt, m0, m1 in module_records:
        try:
            t = s_evt.elapsed_time(e_evt)
            mod_stats[name]["time"] += t
            mod_stats[name]["vram_delta"] += (m1 - m0)
            mod_stats[name]["count"] += 1
        except RuntimeError:
            pass # Metal fused this operation to 0.0ms; ignore and continue

    print("\n" + "="*85)
    print(f" 🚀 TRUE ASYNC HIGH-RES RESULTS ({total_chunks} Chunks) 🚀")
    print(f" Total Wall-Clock Time: {wall_time:.2f} seconds")
    print("="*85)
    
    for step_name, stats in sorted(phase_stats.items()):
        if stats["count"] == 0: continue
        avg_t = stats["time"] / stats["count"]
        avg_v = stats["vram_delta"] / stats["count"]
        peak = stats["peak_vram"]
        
        color = "\033[92m" if avg_v <= 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{step_name:<36} | Avg Time: {avg_t:>7.2f} ms | {color}Avg VRAM Δ: {avg_v:>7.2f} MB{reset} | Peak: {peak:>7.2f} MB")
        
        if "[2]" in step_name:
            for m_name, m_stats in mod_stats.items():
                if m_stats["count"] == 0: continue
                m_avg_t = m_stats["time"] / m_stats["count"]
                m_avg_v = m_stats["vram_delta"] / m_stats["count"]
                print(f"    ↳ [Sub] {m_name:<26} | Avg Time: {m_avg_t:>7.2f} ms | Avg VRAM Δ: {m_avg_v:>7.2f} MB")
    print("="*85)

    # Clean up to prevent PyTorch shutdown crash
    phase_records.clear()
    module_records.clear()
    import gc
    gc.collect()

if __name__ == "__main__":
    run_async_telemetry()