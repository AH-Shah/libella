import time
import torch
import gc
import warnings
from pathlib import Path

# --- Import your actual project modules here ---
# Adjust these imports based on your file structure.
from libella.config import cfg, paths
from libella.model import LibellaGNN
from libella.data import pad_mps_shapes

warnings.filterwarnings("ignore")

# 1. Ultra-Precise MPS Hardware Profiler
class TrackMPS:
    """Invasive context manager that forces GPU syncs to get true nanosecond timings and byte-exact VRAM."""
    def __init__(self, step_name: str, level: int = 0):
        self.step_name = step_name
        self.level = level
        self.indent = "  " * level

    def __enter__(self):
        torch.mps.synchronize()
        self.t0 = time.perf_counter()
        self.m0 = torch.mps.current_allocated_memory() / (1024 ** 2) # Convert bytes to MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.mps.synchronize()
        self.t1 = time.perf_counter()
        self.m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        
        dt_ms = (self.t1 - self.t0) * 1000
        dm_mb = self.m1 - self.m0
        
        # Color coding for terminal visibility
        color = "\033[92m" if dm_mb <= 0 else "\033[91m"
        reset = "\033[0m"
        
        print(f"{self.indent}↳ {self.step_name:<35} | {dt_ms:>8.2f} ms | {color}VRAM Change: {dm_mb:>7.2f} MB{reset} | Peak VRAM: {self.m1:>7.2f} MB")

# 2. Model Forward Hook Injector
def attach_telemetry_hooks(model: torch.nn.Module):
    """Automatically attaches timing hooks to every sub-module inside your GNN."""
    def hook_fn(module, input, output, name):
        torch.mps.synchronize()
        t1 = time.perf_counter()
        m1 = torch.mps.current_allocated_memory() / (1024 ** 2)
        dt_ms = (t1 - module._telemetry_start_time) * 1000
        dm_mb = m1 - module._telemetry_start_mem
        
        print(f"      [SubModule] {name:<23} | {dt_ms:>8.2f} ms | VRAM Delta: {dm_mb:>7.2f} MB")

    def pre_hook_fn(module, input, name):
        torch.mps.synchronize()
        module._telemetry_start_time = time.perf_counter()
        module._telemetry_start_mem = torch.mps.current_allocated_memory() / (1024 ** 2)

    print("\n[+] Attaching surgical telemetry hooks to LibellaGNN submodules...")
    for name, layer in model.named_children():
        layer.register_forward_pre_hook(lambda m, i, n=name: pre_hook_fn(m, i, n))
        layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))


def run_telemetry():
    device = torch.device("mps")
    torch.mps.empty_cache()
    
    print("="*80)
    print(" 🚀 LIBELLA MPS TELEMETRY HARNESS 🚀")
    print("="*80)
    
    # ---------------------------------------------------------
    # TARGET FILE (As ordered)
    # ---------------------------------------------------------
    target_pt = Path("/Users/Hemato/project_3/benchmark/libella_output/run/temp_training_chunks/benchmark_data_chunk_0.pt")
    if not target_pt.exists():
        raise FileNotFoundError(f"Cannot find chunk at {target_pt}")

    # ---------------------------------------------------------
    # PHASE 1: Data I/O & Preprocessing
    # ---------------------------------------------------------
    print("\n--- PHASE 1: DATA PIPELINE ---")
    with TrackMPS("1. Disk Read (torch.load)", level=1):
        batch = torch.load(target_pt, map_location='cpu', weights_only=False)
        
    with TrackMPS("2. Extract & Densify X (SciPy COO -> Torch Dense)", level=1):
        x_coo = batch["x"].tocoo()
        row = torch.from_numpy(x_coo.row).to(torch.int32).to(device)
        col = torch.from_numpy(x_coo.col).to(torch.int32).to(device)
        val = torch.from_numpy(x_coo.data).to(torch.float32).to(device)
        indices = torch.stack([row, col], dim=0)
        x = torch.sparse_coo_tensor(indices, val, size=x_coo.shape, device=device).to_dense()
        del x_coo, row, col, val, indices

    with TrackMPS("3. Extract Adj (SciPy COO -> Torch Nodes/Edges)", level=1):
        adj_coo = batch["adj"].tocoo()
        src = torch.from_numpy(adj_coo.row).to(torch.int32).to(device)
        dst = torch.from_numpy(adj_coo.col).to(torch.int32).to(device)
        weights = torch.from_numpy(adj_coo.data).to(torch.float32).to(device)
        del adj_coo

    with TrackMPS("4. Execute pad_mps_shapes", level=1):
        x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)
        
    # Apply your device-specific casting fix
    if device.type != 'mps':
        src = src.to(torch.int64)
        dst = dst.to(torch.int64)

    # ---------------------------------------------------------
    # PHASE 2: Model Initialization
    # ---------------------------------------------------------
    print("\n--- PHASE 2: MODEL INITIALIZATION ---")
    in_channels = x.shape[1] # Usually 2000
    n_metaprograms = 30      # Assuming default from your config
    
    with TrackMPS("1. Instantiate LibellaGNN & move to MPS", level=1):
        model = LibellaGNN(in_channels=in_channels, n_metaprograms=n_metaprograms).to(device)
        model.train()
        
    attach_telemetry_hooks(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # ---------------------------------------------------------
    # PHASE 3: Forward Pass (The Math)
    # ---------------------------------------------------------
    print("\n--- PHASE 3: FORWARD PASS ---")
    with TrackMPS("1. TOTAL FORWARD PASS TIME", level=1):
        fracs, pure_anchors = model(x, src, dst, weights)

    # Setup targets for loss calculation just like the _train_loop
    local_core = batch["local_core_idx"]
    core_gpu = torch.from_numpy(local_core).to(dtype=torch.int64, device=device)
    t_mask_gpu = torch.from_numpy(batch["train_mask"][local_core]).to(dtype=torch.bool, device=device)
    train_idx = core_gpu[t_mask_gpu]

    f_train = fracs[train_idx]
    x_train = x[train_idx]

    # Calculate targets for loss
    with TrackMPS("2. Prepare Targets / EMA Prior", level=1):
        p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
        current_p_mean = p_train.mean(dim=0)
        uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
        ema_mean = current_p_mean.detach()
        ideal_c = torch.clamp(uniform_prior * 2.0 - ema_mean, min=1e-5)
        target_f_dist = ideal_c / ideal_c.sum()
        recon = f_train @ pure_anchors

    # ---------------------------------------------------------
    # PHASE 4: Loss & Backward Pass
    # ---------------------------------------------------------
    print("\n--- PHASE 4: AUTOGRAD & OPTIMIZER ---")
    with TrackMPS("1. Loss Calculation (calc_loss)", level=1):
        true_batch_loss, base_recon_val = model.calc_loss(
            recon, x_train, pure_anchors, None, ep=1, total_epochs=30, 
            f_train=f_train, target_f_dist=target_f_dist, kl_weight=5.0
        )

    with TrackMPS("2. Backward Pass (Gradient Calculation)", level=1):
        true_batch_loss.backward()

    with TrackMPS("3. Gradient Clipping", level=1):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)

    with TrackMPS("4. Optimizer Step", level=1):
        optimizer.step()
        
    with TrackMPS("5. Optimizer Zero Grad", level=1):
        optimizer.zero_grad(set_to_none=True)

    print("\n" + "="*80)
    print(" ✅ TELEMETRY COMPLETE. REVIEW VRAM DELTAS AND TIMINGS ABOVE.")
    print("="*80)

if __name__ == "__main__":
    run_telemetry()