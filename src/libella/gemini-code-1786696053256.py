import copy
import ctypes
import ctypes.util
import gc
import json
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------
# ENVIRONMENT INITIALIZATION
# ---------------------------------------------------------------------
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["OMP_NUM_THREADS"] = "1"

if not torch.backends.mps.is_available():
    raise RuntimeError("This audit script requires Apple Silicon MPS backend.")

device = torch.device("mps")
process = psutil.Process(os.getpid())

# =====================================================================
# 1. DARWIN MACH KERNEL METRIC (Exact /usr/bin/time -l phys_footprint)
# =====================================================================
class TaskVMInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_uint32),
        ("page_size", ctypes.c_int32),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("device_peak", ctypes.c_uint64),
        ("internal", ctypes.c_uint64),
        ("internal_peak", ctypes.c_uint64),
        ("external", ctypes.c_uint64),
        ("external_peak", ctypes.c_uint64),
        ("reusable", ctypes.c_uint64),
        ("reusable_peak", ctypes.c_uint64),
        ("purgeable_volatile_pmap", ctypes.c_uint64),
        ("purgeable_volatile_resident", ctypes.c_uint64),
        ("purgeable_volatile_virtual", ctypes.c_uint64),
        ("compressed", ctypes.c_uint64),
        ("compressed_peak", ctypes.c_uint64),
        ("compressed_lifetime", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("min_address", ctypes.c_uint64),
        ("max_address", ctypes.c_uint64),
    ]

_LIBC = ctypes.CDLL(ctypes.util.find_library("c")) if sys.platform == "darwin" else None
_TASK_VM_INFO = 22
_TASK_VM_INFO_COUNT = ctypes.sizeof(TaskVMInfo) // ctypes.sizeof(ctypes.c_uint32)

def get_phys_footprint_mb() -> float:
    if _LIBC is not None:
        try:
            info = TaskVMInfo()
            count = ctypes.c_uint32(_TASK_VM_INFO_COUNT)
            if _LIBC.task_info(_LIBC.mach_task_self(), _TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)) == 0:
                return info.phys_footprint / (1024 ** 2)
        except Exception:
            pass
    return process.memory_info().rss / (1024 ** 2)

def get_vram_mb() -> float:
    return torch.mps.current_allocated_memory() / (1024 ** 2) if hasattr(torch.mps, "current_allocated_memory") else 0.0


# =====================================================================
# 2. ZERO-OVERHEAD NON-BLOCKING AUDITOR
# =====================================================================
class MemoryAuditor:
    records: List[Dict[str, Any]] = []
    _lock = threading.Lock()

    def __init__(self, step_name: str, meta_info: str = ""):
        self.step_name = step_name
        self.meta_info = meta_info

    def __enter__(self):
        self.start_footprint = get_phys_footprint_mb()
        self.start_rss = process.memory_info().rss / (1024 ** 2)
        self.start_vram = get_vram_mb()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.end_footprint = get_phys_footprint_mb()
        self.end_rss = process.memory_info().rss / (1024 ** 2)
        self.end_vram = get_vram_mb()

        duration = (self.end_time - self.start_time) * 1000
        delta_footprint = self.end_footprint - self.start_footprint
        delta_rss = self.end_rss - self.start_rss
        delta_vram = self.end_vram - self.start_vram

        record = {
            "step": self.step_name,
            "info": self.meta_info,
            "end_footprint_mb": self.end_footprint,
            "delta_footprint_mb": delta_footprint,
            "end_rss_mb": self.end_rss,
            "delta_rss_mb": delta_rss,
            "end_vram_mb": self.end_vram,
            "delta_vram_mb": delta_vram,
            "duration_ms": duration,
        }

        with MemoryAuditor._lock:
            self.records.append(record)

        if abs(delta_footprint) > 100 or abs(delta_rss) > 50:
            print(
                f"  ⚡ [SPIKE] {self.step_name:<28} | "
                f"Footprint: {self.end_footprint:7.1f} MB (Δ {delta_footprint:+6.1f} MB) | "
                f"RSS: {self.end_rss:6.1f} MB | VRAM: {self.end_vram:6.1f} MB | "
                f"{duration:5.1f}ms"
            )

    @classmethod
    def print_diagnostic_table(cls):
        print("\n" + "=" * 120)
        print(" " * 42 + "📊 COMPLETE PIPELINE MEMORY AUDIT 📊")
        print("=" * 120)

        if not cls.records:
            print("No profiling events recorded.")
            return

        df = pd.DataFrame(cls.records)
        top_spikes = df.sort_values(by="delta_footprint_mb", ascending=False).head(15)

        display_df = pd.DataFrame({
            "Subprocess / Step": top_spikes["step"],
            "Context / Chunk": top_spikes["info"],
            "Phys Footprint": top_spikes["end_footprint_mb"].map(lambda x: f"{x:.1f} MB"),
            "Δ Footprint": top_spikes["delta_footprint_mb"].map(lambda x: f"{x:+.1f} MB"),
            "CPU RSS": top_spikes["end_rss_mb"].map(lambda x: f"{x:.1f} MB"),
            "Active VRAM": top_spikes["end_vram_mb"].map(lambda x: f"{x:.1f} MB"),
            "Duration": top_spikes["duration_ms"].map(lambda x: f"{x:.1f} ms"),
        })

        print("\nTop 15 Physical Unified Memory Spikes (by Δ Kernel Footprint):")
        print(display_df.to_string(index=False))

        print("\n" + "-" * 120)
        print("Aggregated Summary by Step Type:")
        print("-" * 120)
        agg_df = df.groupby("step").agg({
            "delta_footprint_mb": ["sum", "max"],
            "delta_rss_mb": ["sum", "max"],
            "delta_vram_mb": ["sum", "max"],
            "duration_ms": ["sum", "count"]
        })
        agg_df.columns = ["Sum ΔFootprint", "Max ΔFootprint", "Sum ΔRSS", "Max ΔRSS", "Sum ΔVRAM", "Max ΔVRAM", "Total Time (ms)", "Calls"]
        agg_df = agg_df.sort_values(by="Max ΔFootprint", ascending=False)
        print(agg_df.to_string())
        print("=" * 120 + "\n")


# =====================================================================
# 3. FAST SHAPE PADDING & HELPERS
# =====================================================================
def pad_mps_shapes(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor, batch_size: int = 10000):
    n_max = batch_size * 4
    e_max = n_max * 10
    node_bucket = max(1024, int(round((n_max * 0.20) / 1024) * 1024))
    edge_bucket = max(4096, int(round((e_max * 0.20) / 4096) * 4096))

    N = x.size(0)
    E = src.size(0)

    N_pad = ((N + node_bucket - 1) // node_bucket) * node_bucket
    E_pad = ((E + edge_bucket - 1) // edge_bucket) * edge_bucket

    if E_pad > E and N_pad == N:
        N_pad += node_bucket

    if N_pad > N:
        x_dummy = torch.zeros(N_pad - N, x.size(1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, x_dummy], dim=0)

    if E_pad > E:
        dummy_idx = torch.full((E_pad - E,), N, dtype=src.dtype, device=src.device)
        dummy_w = torch.zeros(E_pad - E, dtype=weights.dtype, device=weights.device)
        src = torch.cat([src, dummy_idx], dim=0)
        dst = torch.cat([dst, dummy_idx], dim=0)
        weights = torch.cat([weights, dummy_w], dim=0)

    return x, src, dst, weights

def scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src_safe = torch.clamp(src, min=-60.0, max=60.0)
    exp_val = torch.exp(src_safe)
    sum_val = torch.zeros(num_nodes, dtype=src.dtype, device=src.device).scatter_add(0, index, exp_val)
    return exp_val / (sum_val[index] + 1e-9)


# =====================================================================
# 4. UPDATED LIBELLA GNN (With Persistent ortho_mask)
# =====================================================================
class LibellaGNN(nn.Module):
    def __init__(self, in_channels: int, n_metaprograms: int, init_components: np.ndarray | None = None, hidden_dim: int = 128, k_hops: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_hops = k_hops
        self.n_metaprograms = n_metaprograms
        self.in_channels = in_channels

        self.ctx_enc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.lin_appnp = nn.Linear(hidden_dim, hidden_dim)

        self.id_enc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(hidden_dim)
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.sp_norm = nn.LayerNorm(hidden_dim)

        self.topic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_metaprograms)
        )

        self.spatial_bridge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_metaprograms * in_channels)
        )

        self.dict_temp = nn.Parameter(torch.tensor(0.30))

        if init_components is not None:
            active_mask = (init_components > 0)
            base_logits = np.where(active_mask, 2.0, -2.0)
            noise = np.random.randn(*base_logits.shape) * 0.1
            init_logits = base_logits + noise
            self.topic_gene_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
            self.register_buffer('anchor_logits', torch.tensor(init_logits, dtype=torch.float32).clone())
        else:
            self.topic_gene_logits = nn.Parameter(torch.randn(n_metaprograms, in_channels))
            self.register_buffer('anchor_logits', torch.ones(n_metaprograms, in_channels))

        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.alpha_proj = nn.Linear(hidden_dim, 1)
        self.register_buffer('dynamic_w_ema', torch.tensor(1.0, dtype=torch.float32))

        # Persistent GPU Buffer for Orthogonality Loss
        self.register_buffer('ortho_mask', 1.0 - torch.eye(n_metaprograms, dtype=torch.float32))

        self.gat_w_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_w_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_w_edge = nn.Linear(1, hidden_dim, bias=True)
        self.gat_a = nn.Linear(hidden_dim, 1, bias=False)
        self.att_temp = nn.Parameter(torch.tensor(-2.25))
        self.mp_update = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x_dense: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weights: torch.Tensor):
        if len(src) > 0:
            src = src.contiguous()
            dst = dst.contiguous()

        h_id = self.id_enc(x_dense)
        h_0 = self.lin_appnp(self.ctx_enc(x_dense))

        macro_ctx = h_0.mean(dim=0)
        dict_shift = torch.tanh(self.spatial_bridge(macro_ctx)) * 2.0
        dynamic_logits = self.topic_gene_logits + dict_shift.view(self.n_metaprograms, -1)

        soft_anchors = F.softmax(dynamic_logits, dim=-1)
        safe_temp = torch.clamp(self.dict_temp, min=0.25, max=1.0)
        sharp_anchors = F.softmax(dynamic_logits / safe_temp, dim=-1)
        anchors_raw = sharp_anchors.detach() + soft_anchors - soft_anchors.detach()

        N = h_0.size(0)

        if len(src) > 0:
            with torch.no_grad():
                bio_h = torch.mm(x_dense, anchors_raw.detach().t())
                diff = bio_h[src] - bio_h[dst]
                dist = (diff * diff).sum(dim=1)
            decay = torch.exp(-F.softplus(self.gamma) * dist)
        else:
            decay = torch.ones_like(edge_weights)

        W_bil = edge_weights * decay
        alpha = torch.sigmoid(self.alpha_proj(h_0)) * 0.85 + 0.10
        inv_alpha = 1.0 - alpha
        h_0_scaled = h_0 * alpha

        h_ctx = h_0
        for _ in range(self.k_hops):
            out = torch.zeros_like(h_ctx)
            if len(src) > 0:
                h_src_proj = self.gat_w_src(h_ctx)
                h_dst_proj = self.gat_w_dst(h_ctx)
                edge_proj = self.gat_w_edge(W_bil.unsqueeze(1))
                h_edge = h_src_proj[src] + h_dst_proj[dst] + edge_proj

                e_raw = self.gat_a(F.leaky_relu(h_edge)).squeeze(-1)
                tau = torch.clamp(F.softplus(self.att_temp), min=0.05)
                e_scaled = e_raw / tau

                alpha_att = scatter_softmax(e_scaled, dst, N)
                msg = h_ctx[src] * alpha_att.unsqueeze(1)
                out.index_add_(0, dst, msg)

            agg = F.silu(self.mp_update(out))
            h_ctx = agg * inv_alpha + h_0_scaled

        Q = self.q_proj(h_id)
        K = self.k_proj(h_ctx)
        V = self.v_proj(h_ctx)

        self_loops = torch.arange(N, dtype=src.dtype if len(src) > 0 else torch.int32, device=x_dense.device)
        src_with_self = torch.cat([src, self_loops]) if len(src) > 0 else self_loops
        dst_with_self = torch.cat([dst, self_loops]) if len(src) > 0 else self_loops

        q_dst = Q[dst_with_self]
        k_src = K[src_with_self]
        v_src = V[src_with_self]

        cross_scores = (q_dst * k_src).sum(dim=-1) / (self.hidden_dim ** 0.5)
        cross_att = scatter_softmax(cross_scores, dst_with_self, N)

        pulled_msg = (v_src * cross_att.unsqueeze(1)).contiguous()
        ctx_pulled = torch.zeros_like(Q)
        ctx_pulled.index_add_(0, dst_with_self, pulled_msg)

        h_final = h_id + self.context_gate(ctx_pulled)
        h_norm = F.normalize(self.sp_norm(h_final), p=2, dim=-1)

        t_proj_weights = F.normalize(anchors_raw, p=2, dim=-1)
        x_norm = F.normalize(x_dense, p=2, dim=-1)
        bio_sim = torch.mm(x_norm, t_proj_weights.t())

        gnn_shift_raw = self.topic_proj(h_norm)
        gnn_shift_norm = F.normalize(gnn_shift_raw, p=2, dim=-1)

        base_logits = bio_sim + (0.5 * gnn_shift_norm)
        if self.training:
            base_logits = base_logits + (torch.randn_like(base_logits) * 0.05)

        return base_logits * 15.0, anchors_raw

    def forward(self, x_dense: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_weights: torch.Tensor):
        logits, anchors_raw = self.encode(x_dense, src, dst, edge_weights)
        prob = F.softmax(logits / 1.5, dim=1)
        mag = x_dense.sum(dim=1, keepdim=True)
        return prob * mag, anchors_raw

    def calc_loss(self, recon_c: torch.Tensor, x_c: torch.Tensor, anchors: torch.Tensor, f_train: torch.Tensor | None = None, target_f_dist: torch.Tensor | None = None, kl_weight: float | torch.Tensor = 0.5):
        is_non_zero = (x_c > 0)
        num_pos = torch.clamp(is_non_zero.float().sum(), min=1.0)
        num_zeros = (x_c == 0).float().sum()
        current_dynamic_w = (num_zeros / num_pos).detach()

        if self.training:
            self.dynamic_w_ema.lerp_(current_dynamic_w, weight=0.1)

        zero_mask = torch.rand_like(x_c) < 0.05
        active_mask = (is_non_zero | zero_mask).to(x_c.dtype)
        masked_w_mat = torch.where(is_non_zero, current_dynamic_w, 1.0) * active_mask

        raw_delta = recon_c - x_c
        asymmetry_factor = 1.0 + (is_non_zero.to(x_c.dtype) * 2.0) * (raw_delta < 0).float()
        scaled_delta = torch.clamp(raw_delta * asymmetry_factor, min=-30.0, max=30.0)

        l_recon_sum = torch.sum(masked_w_mat * torch.log(torch.cosh(scaled_delta + 1e-6)))
        l_recon = l_recon_sum / max(1, x_c.shape[0])

        anc_norm = F.normalize(anchors, p=2, dim=1)
        ref_norm = F.normalize(F.softmax(self.anchor_logits, dim=-1), p=2, dim=1)
        l_anc = 1.0 - (anc_norm * ref_norm).sum(dim=1).mean()

        peak_excess = F.relu(anchors - 0.80)
        collapse_penalty = (peak_excess ** 2).sum(dim=1).mean()
        gene_entropy = -(anchors * torch.log(anchors + 1e-9)).sum(dim=1).mean()

        raw_t_norm = F.normalize(anchors, p=2, dim=-1)
        latent_ortho = torch.mm(raw_t_norm, raw_t_norm.t()) * self.ortho_mask
        l_ortho = (F.relu(latent_ortho.max(dim=1)[0] - 0.25) ** 2).mean()

        im_loss = torch.tensor(0.0, device=x_c.device)
        if f_train is not None:
            f_norm = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
            p_mean = torch.clamp(f_norm.mean(dim=0), min=1e-7)
            if target_f_dist is not None:
                kl_marginal = (p_mean * (torch.log(p_mean) - torch.log(target_f_dist + 1e-9))).sum()
            else:
                uniform = torch.ones_like(p_mean) / anchors.shape[0]
                kl_marginal = (p_mean * (torch.log(p_mean) - torch.log(uniform))).sum()
            im_loss = kl_weight * kl_marginal * (l_recon.detach() * 0.05)

        total_loss = l_recon + (l_anc * 0.1 * l_recon.detach()) + (l_ortho * 10.0 * l_recon.detach()) + (gene_entropy * 0.01 * l_recon.detach()) + im_loss
        return total_loss, l_recon.detach()


# =====================================================================
# 5. PRE-FLIGHT CHUNK FORMAT VALIDATOR
# =====================================================================
def validate_chunk_format(chunks_dir: Path) -> List[Path]:
    chunk_files = sorted(list(chunks_dir.glob("*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No .pt chunks found in {chunks_dir}")

    print(f"\n[🔍 PRE-FLIGHT] Inspecting chunk directory: {chunks_dir.name}")
    print(f"  ↳ Found {len(chunk_files)} total SSD chunk files.")

    sample_file = chunk_files[0]
    sample_chunk = torch.load(sample_file, map_location="cpu", weights_only=False)

    print(f"\n[📦 SAMPLE CHUNK INSPECTION] '{sample_file.name}':")
    expected_keys = ["x", "src", "dst", "weights", "train_core_idx", "val_core_idx"]

    all_keys_present = True
    for key in expected_keys:
        if key in sample_chunk:
            val = sample_chunk[key]
            dtype_str = str(val.dtype) if hasattr(val, "dtype") else type(val).__name__
            shape_str = str(tuple(val.shape)) if hasattr(val, "shape") else str(len(val))
            print(f"  • {key:<16}: Present | Type: {type(val).__name__:<12} | Dtype: {dtype_str:<14} | Shape: {shape_str}")
        else:
            print(f"  ❌ {key:<16}: MISSING from chunk!")
            all_keys_present = False

    if not all_keys_present:
        print("\n⚠️ WARNING: Some pre-tensorized keys are missing. Please re-run _prep_ssd_chunks() with the latest train.py.")
    else:
        print("\n✅ PRE-FLIGHT PASSED: Chunk schema matches the zero-sync pre-tensorized standard.")

    return chunk_files


# =====================================================================
# 6. ASYNC ZERO-OVERHEAD PREFETCHER
# =====================================================================
def make_meta_batches(training_cache: List[Dict[str, Any]], meta_batch_size: int = 5):
    from collections import defaultdict
    import random
    patient_bins = defaultdict(list)
    for b in training_cache:
        patient_bins[b['patient_name']].append(b)
    for p in patient_bins:
        random.shuffle(patient_bins[p])
    meta_batches = []
    active_patients = list(patient_bins.keys())
    while active_patients:
        current_meta = []
        random.shuffle(active_patients)
        selected = active_patients[:meta_batch_size]
        for p in selected:
            current_meta.append(patient_bins[p].pop())
            if not patient_bins[p]:
                active_patients.remove(p)
        while len(current_meta) < meta_batch_size and active_patients:
            p = random.choice(active_patients)
            current_meta.append(patient_bins[p].pop())
            if not patient_bins[p]:
                active_patients.remove(p)
        meta_batches.append(current_meta)
    return meta_batches

def prefetch_batches_native(meta_batches: List[List[Dict[str, Any]]]):
    for meta_meta in meta_batches:
        chunk_queue = queue.Queue(maxsize=1)
        stop_event = Event()

        def safe_put(item: Any) -> bool:
            while not stop_event.is_set():
                try:
                    chunk_queue.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def worker():
            try:
                for b in meta_meta:
                    if stop_event.is_set():
                        break
                    chunk = torch.load(b['chunk_file'], map_location='cpu', weights_only=False)
                    if not safe_put(chunk):
                        break
            except Exception as e:
                safe_put(e)
            finally:
                safe_put(None)

        t = Thread(target=worker, daemon=True)
        t.start()

        def chunk_iterator():
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        yield meta_meta, chunk_iterator(), stop_event, t


# =====================================================================
# 7. TELEMETRY AUDIT RUNNER
# =====================================================================
def run_telemetry_audit(genes_file: Path, priors_file: Path, chunks_dir: Path, epochs: int = 1):
    chunk_files = validate_chunk_format(chunks_dir)

    with open(genes_file, "r") as f:
        common_genes = json.load(f)

    # Load priors
    raw_obj = None
    try:
        import joblib
        raw_obj = joblib.load(priors_file)
    except Exception:
        raw_obj = torch.load(priors_file, map_location="cpu", weights_only=False)

    init_components = None
    optimal_k = 38
    if isinstance(raw_obj, dict):
        for k in ["init_components", "priors", "components"]:
            if k in raw_obj:
                init_components = raw_obj[k]
                break
        optimal_k = raw_obj.get("optimal_k", init_components.shape[0] if init_components is not None else optimal_k)
    elif isinstance(raw_obj, (np.ndarray, torch.Tensor)):
        init_components = raw_obj
        optimal_k = init_components.shape[0]

    if isinstance(init_components, torch.Tensor):
        init_components = init_components.detach().cpu().numpy()

    print(f"  ↳ Loaded {len(common_genes)} consensus genes | Optimal K: {optimal_k}")

    training_cache = []
    for f in chunk_files:
        p_name = f.stem.split("_chunk_")[0]
        training_cache.append({"patient_name": p_name, "chunk_file": f})

    # Model & Optimizers
    with MemoryAuditor("Model & Optimizer Init"):
        model = LibellaGNN(in_channels=len(common_genes), n_metaprograms=optimal_k, init_components=init_components).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    accumulation_steps = 5
    max_entropy_scalar = float(np.log(optimal_k))
    alpha_ema = 0.001
    ema_mean = None

    print(f"\n[➤] RUNNING TELEMETRY PROFILING ON 500K CHUNKS ({len(training_cache)} sub-graphs)...")

    for epoch in range(epochs):
        model.train()
        train_steps, val_steps = 0, 0
        train_chunk_count = 0

        train_loss_acc = torch.tensor(0.0, device=device)
        val_loss_acc = torch.tensor(0.0, device=device)
        epoch_p_mean_sum = torch.zeros(optimal_k, device=device)

        gpu_telemetry = {
            'ent': torch.tensor(0.0, device=device),
            'col_r': torch.tensor(0.0, device=device),
            'kl_w': torch.tensor(0.0, device=device),
            'g_w': torch.tensor(0.0, device=device),
            'p_w': torch.tensor(0.0, device=device),
            'l_rec': torch.tensor(0.0, device=device)
        }

        meta_batches = make_meta_batches(training_cache, meta_batch_size=accumulation_steps)
        total_steps = len(meta_batches)

        pbar = tqdm(prefetch_batches_native(meta_batches), total=total_steps, desc=f"Audit Epoch {epoch+1}/{epochs}")

        for step_idx, (meta_meta, chunk_iter, stop_ev, worker_thread) in enumerate(pbar):
            with MemoryAuditor("optimizer.zero_grad", f"S{step_idx}"):
                optimizer.zero_grad(set_to_none=True)

            for chunk_idx, (batch_ref, batch) in enumerate(zip(meta_meta, chunk_iter)):
                tag = f"S{step_idx}_C{chunk_idx}"

                with MemoryAuditor("1. Direct GPU Tensor Transfer", tag):
                    x = batch["x"].to(device=device, non_blocking=True)
                    src = batch["src"].to(device=device, non_blocking=True)
                    dst = batch["dst"].to(device=device, non_blocking=True)
                    weights = batch["weights"].to(device=device, non_blocking=True)

                with MemoryAuditor("2. MPS Shape Padding", tag):
                    if model.training and len(src) > 0:
                        keep_mask = torch.rand(src.size(0), device=device) > 0.40
                        src = src[keep_mask]
                        dst = dst[keep_mask]
                        weights = weights[keep_mask]
                    x, src, dst, weights = pad_mps_shapes(x, src, dst, weights)

                with MemoryAuditor("3. GNN Forward Pass", tag):
                    fracs, pure_anchors = model(x, src, dst, weights)

                with MemoryAuditor("4. Training Reconstruction & Loss", tag):
                    train_idx = batch["train_core_idx"].to(device=device, non_blocking=True)
                    f_train = fracs[train_idx]
                    x_train = x[train_idx]

                    p_train = f_train / (f_train.sum(dim=1, keepdim=True) + 1e-9)
                    current_p_mean = p_train.mean(dim=0)

                    uniform_prior = torch.ones_like(current_p_mean) / pure_anchors.shape[0]
                    if ema_mean is None:
                        ema_mean = current_p_mean.detach()
                    else:
                        ema_mean = alpha_ema * current_p_mean.detach() + (1 - alpha_ema) * ema_mean

                    ideal_c = torch.clamp(uniform_prior * 2.0 - ema_mean, min=1e-5)
                    target_f_dist = ideal_c / ideal_c.sum()

                    ema_entropy = -torch.sum(ema_mean * torch.log(ema_mean + 1e-9))
                    collapse_ratio = torch.clamp(1.0 - (ema_entropy / max_entropy_scalar), min=0.0, max=1.0)
                    peak_p = ema_mean.max()
                    hub_multiplier = F.relu((peak_p / 0.15) - 1.0) * 10.0
                    dynamic_kl_w = 0.10 + (collapse_ratio * 3.0) + hub_multiplier

                    recon = f_train @ pure_anchors
                    true_batch_loss, base_recon_val = model.calc_loss(
                        recon, x_train, pure_anchors, f_train=f_train, target_f_dist=target_f_dist, kl_weight=dynamic_kl_w
                    )

                with MemoryAuditor("5. Autograd Backward", tag):
                    (true_batch_loss / len(meta_meta)).backward()

                    train_loss_acc += true_batch_loss.detach()
                    train_steps += 1

                    gpu_telemetry['g_w'] += pure_anchors.max(dim=1).values.mean().detach() * 100.0
                    gpu_telemetry['p_w'] += p_train.max(dim=1).values.mean().detach() * 100.0
                    gpu_telemetry['ent'] += ema_entropy.detach()
                    gpu_telemetry['col_r'] += collapse_ratio.detach()
                    gpu_telemetry['kl_w'] += dynamic_kl_w.detach()
                    gpu_telemetry['l_rec'] += base_recon_val.detach()

                    epoch_p_mean_sum += current_p_mean.detach()
                    train_chunk_count += 1

                    del train_idx, f_train, x_train, p_train, current_p_mean, uniform_prior, target_f_dist, recon, true_batch_loss, base_recon_val

                # Validation Slice
                val_core_idx_cpu = batch["val_core_idx"]
                if val_core_idx_cpu.numel() > 0:
                    with MemoryAuditor("6. Validation Log-Cosh", tag):
                        val_idx = val_core_idx_cpu.to(device=device, non_blocking=True)
                        with torch.no_grad():
                            f_val = fracs[val_idx]
                            x_val = x[val_idx]
                            val_recon = f_val @ pure_anchors

                            is_non_zero_val = (x_val > 0)
                            w_mat = torch.where(is_non_zero_val, model.dynamic_w_ema, 1.0)
                            zero_expectation_mask = torch.where(is_non_zero_val, 1.0, 0.05).to(x_val.dtype)
                            masked_w_mat_val = w_mat * zero_expectation_mask

                            raw_delta_val = val_recon - x_val
                            asym_val = 1.0 + (is_non_zero_val.to(x_val.dtype) * 2.0) * (raw_delta_val < 0).to(x_val.dtype)
                            scaled_delta_val = torch.clamp(raw_delta_val * asym_val, min=-30.0, max=30.0)

                            val_loss_sum = torch.sum(masked_w_mat_val * torch.log(torch.cosh(scaled_delta_val + 1e-6)))
                            val_log_cosh = val_loss_sum / max(1, x_val.shape[0])

                            val_loss_acc += val_log_cosh.detach()
                            val_steps += 1

                        del val_idx, f_val, x_val, val_recon, w_mat, raw_delta_val, asym_val, scaled_delta_val, val_loss_sum, val_log_cosh

                del batch, src, dst, weights, x, fracs, pure_anchors

            stop_ev.set()
            worker_thread.join(timeout=0.2)

            with MemoryAuditor("7. clip_grad & optimizer.step", f"S{step_idx}"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                optimizer.step()

        # Telemetry Aggregation (1 single sync per epoch)
        epoch_telemetry = {k: (v / max(1, train_chunk_count)).item() for k, v in gpu_telemetry.items()}
        epoch_p_mean = (epoch_p_mean_sum / max(1, train_chunk_count)).cpu()
        top_topic_val, top_topic_idx = epoch_p_mean.max(dim=0)

        mean_train_loss = (train_loss_acc / (train_steps + 1e-9)).item()
        mean_val_loss = (val_loss_acc / (val_steps + 1e-9)).item()

        print("\n" + "=" * 110)
        print(
            f"[Ep {(epoch+1):03d}] Pure_Rec:{epoch_telemetry.get('l_rec', 0.0):<5.3f} "
            f"V_Loss:{mean_val_loss:<5.3f} (Tot_Loss:{mean_train_loss:<5.3f}) | "
            f"G_W:{epoch_telemetry.get('g_w', 0.0):<4.1f}% P_W:{epoch_telemetry.get('p_w', 0.0):<4.1f}% "
            f"TopT:{top_topic_idx.item()}({top_topic_val.item()*100:<4.1f}%) "
            f"Ent:{epoch_telemetry.get('ent', 0.0):<4.2f} | "
            f"KL_W:{epoch_telemetry.get('kl_w', 0.0):<4.2f}"
        )
        print("=" * 110 + "\n")


# =====================================================================
# 8. SCRIPT ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    BASE_DIR = Path("/Users/Hemato/project_3/benchmark/libella_output/run")
    CHUNKS_DIR = BASE_DIR / "temp_training_chunks"
    GENES_FILE = BASE_DIR / "common_genes.json"
    PRIORS_FILE = BASE_DIR / "global_cnmf_priors.pkl"

    try:
        run_telemetry_audit(
            genes_file=GENES_FILE,
            priors_file=PRIORS_FILE,
            chunks_dir=CHUNKS_DIR,
            epochs=1
        )
    finally:
        MemoryAuditor.print_diagnostic_table()