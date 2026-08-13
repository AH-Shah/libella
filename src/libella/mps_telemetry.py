import time
import torch
import gc
import warnings
from pathlib import Path
from collections import defaultdict

# --- Import your actual project modules here ---
from libella.config import cfg, paths
from libella.model import LibellaGNN
from libella.data import pad_mps_shapes

warnings.filterwarnings("ignore")

# --- Global Aggregators ---
is_warmup = True
phase_metrics = defaultdict(lambda: {"time": 0.0, "vram_delta": 0.0, "peak_vram": 0.0, "count": 0, "level": 0})
module_metrics = defaultdict(lambda: {"time": 0.0, "vram_delta": 0.0, "count": 0})

# 1. Averaging Context Manager
class TrackMPS:
    def __init__(self, step_name: str, level: int = 1):
        self.step_name = step_name
        self.level = level

    def __enter__(self):
        torch.mps.synchronize()
        self.t0 = time.perf_counter()
        self.m0 = torch.mps.current_allocated_memory() / (1024 ** 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.mps.synchronize()
        self.t1 = time.perf_counter()
        self.m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        
        if not is_warmup:
            dt_ms = (self.t1 - self.t0) * 1000
            dm_mb = self.m1 - self.m0
            
            stats = phase_metrics[self.step_name]
            stats["time"] += dt_ms
            stats["vram_delta"] += dm_mb
            stats["peak_vram"] = max(stats["peak_vram"], self.m1)
            stats["count"] += 1
            stats["level"] = self.level

# 2. Averaging Submodule Hooks
def attach_telemetry_hooks(model: torch.nn.Module):
    def pre_hook_fn(module, input, name):
        torch.mps.synchronize()
        module._telemetry_start_time = time.perf_counter()
        module._telemetry_start_mem = torch.mps.current_allocated_memory() / (1024 ** 2)

    def hook_fn(module, input, output, name):
        torch.mps.synchronize()
        t1 = time.perf_counter()
        m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        
        if not is_warmup:
            dt_ms = (t1 - module._telemetry_start_time) * 1000
            dm_mb = m1 - module._telemetry_start_mem
            
            stats = module_metrics[name]
            stats["time"] += dt_ms
            stats["vram_delta"] += dm_mb
            stats["count"] += 1

    for name, layer in model.named_children():
        layer.register_forward_pre_hook(lambda m, i, n=name: pre_hook_fn(m, i, n))
        layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

# 3. The Core Execution Block (Runs repeatedly)
def run_single_iteration(target_pt: Path, model, optimizer, device):
    # PHASE 1: DATA PIPELINE
    with TrackMPS("[1] Disk Read (torch.load)"):
        batch = torch.load(target_pt, map_location='cpu', weights_only=False)
        
    with TrackMPS("[1] Extract & Densify X (SciPy toarray)"):
        x_dense_np = batch["x"].toarray()
        x = torch.from_numpy(x_dense_np).to(dtype=torch.float32, device=device)

    with TrackMPS("[1] Extract Adj"):
        adj_coo = batch["adj"].tocoo()
        src = torch.from_numpy(adj_coo.row).to(torch.int32).to(device)
        dst = torch.from_numpy(adj_coo.col).to(torch.int32).to(device)
        weights = torch.from_numpy(adj_coo.data).to(torch.float32).to(device)
        del adj_coo

    with TrackMPS("[1] Execute pad_mps_shapes"):
        x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
        if device.type != 'mps':
            src = src.to(torch.int64)
            dst = dst.to(torch.int64)

    # PHASE 3: FORWARD PASS
    with TrackMPS("[3] TOTAL FORWARD PASS TIME"):
        fracs, pure_anchors = model(x, src, dst, weights)

    with TrackMPS("[3] Prepare Targets / EMA Prior"):
        local_core = batch["local_core_idx"]
        core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
        t_mask_gpu = torch.from_numpy(batch["train_mask"][local_core]).to(dtype=torch.bool, device=device)
        train_idx = core_gpu[t_mask_gpu]

        f_train = fracs[train_idx]
        x_train = x[train_idx]

        p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
        current_p_mean = p_train.mean(dim=0)
        uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
        ema_mean = current_p_mean.detach()
        ideal_c = torch.clamp(uniform_prior * 2.0 - ema_mean, min=1e-5)
        target_f_dist = ideal_c / ideal_c.sum()
        recon = f_train @ pure_anchors

    # PHASE 4: BACKWARD PASS
    with TrackMPS("[4] Loss Calculation (calc_loss)"):
        true_batch_loss, _ = model.calc_loss(
            recon, x_train, pure_anchors, None, ep=1, total_epochs=30, 
            f_train=f_train, target_f_dist=target_f_dist, kl_weight=5.0
        )

    with TrackMPS("[4] Backward Pass (Gradient Calc)"):
        true_batch_loss.backward()

    with TrackMPS("[4] Gradient Clipping"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

    with TrackMPS("[4] Optimizer Step"):
        optimizer.step()
        
    with TrackMPS("[4] Optimizer Zero Grad"):
        optimizer.zero_grad(set_to_none=True)

    # Clean up to prevent OOM across loops
    del batch, x, src, dst, weights, fracs, pure_anchors
    del f_train, x_train, p_train, recon, true_batch_loss
    gc.collect()

# 4. Master Orchestrator
def run_telemetry():
    global is_warmup
    device = torch.device("mps")
    torch.mps.empty_cache()
    
    target_pt = Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_0.pt")
    if not target_pt.exists():
        raise FileNotFoundError(f"Cannot find chunk at {target_pt}")

    # Initialize Model Once
    print("\n[+] Initializing Model...")
    # (Assuming x.shape[1] is 2000 for your data, hardcoded here for init)
    model = LibellaGNN(in_channels=2000, n_metaprograms=30).to(device)
    model.train()
    attach_telemetry_hooks(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("[+] Waking up GPU (2 Warmup Iterations)...")
    is_warmup = True
    for _ in range(2):
        run_single_iteration(target_pt, model, optimizer, device)

    print("[+] Executing 10-Second Averaging Loop...")
    is_warmup = False
    start_time = time.time()
    iters = 0
    
    # Run aggressively until 10 seconds elapse
    while time.time() - start_time < 10.0:
        run_single_iteration(target_pt, model, optimizer, device)
        iters += 1

    print("\n" + "="*80)
    print(f" 🚀 AVERAGED TELEMETRY RESULTS ({iters} ITERATIONS) 🚀")
    print("="*80)

    # Print nicely formatted averages
    for step_name, stats in phase_metrics.items():
        if stats["count"] == 0: continue
        
        avg_t = stats["time"] / stats["count"]
        avg_v = stats["vram_delta"] / stats["count"]
        peak = stats["peak_vram"]
        
        color = "\033[92m" if avg_v <= 0 else "\033[91m"
        reset = "\033[0m"
        
        print(f"  ↳ {step_name:<40} | Avg Time: {avg_t:>7.2f} ms | {color}Avg VRAM Δ: {avg_v:>7.2f} MB{reset} | Peak: {peak:>7.2f} MB")
        
        # If this is the forward pass, print the submodule breakdown beneath it
        if "[3] TOTAL FORWARD" in step_name:
            for m_name, m_stats in module_metrics.items():
                if m_stats["count"] == 0: continue
                m_avg_t = m_stats["time"] / m_stats["count"]
                m_avg_v = m_stats["vram_delta"] / m_stats["count"]
                print(f"      [Sub] {m_name:<34} | Avg Time: {m_avg_t:>7.2f} ms | Avg VRAM Δ: {m_avg_v:>7.2f} MB")
    
    print("="*80)

if __name__ == "__main__":
    run_telemetry()