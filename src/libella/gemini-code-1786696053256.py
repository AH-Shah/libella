import argparse
import gc
import json
import os
import sys
import time
import threading
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import psutil
import torch
from tqdm import tqdm

# Import your pipeline modules
from libella.config import RunConfig, cfg, paths, init_env
from libella.data import get_consensus_genes, build_graph_safe
from libella.inference import (
    get_ecotypes,
    make_domains,
    plot_curves,
    process_pt,
    run_meta,
)
from libella.train import train_gnn
from libella.utils import get_device, get_whitelist


# ==========================================
# 1. HIGH-PRECISION MEMORY TRACKER
# ==========================================
class MemoryProfileReport:
    records: List[Dict] = []

    @classmethod
    def print_summary(cls):
        print("\n" + "=" * 90)
        print(" " * 28 + "🏁 MEMORY PROFILING REPORT 🏁")
        print("=" * 90)
        
        df = pd.DataFrame(cls.records)
        if df.empty:
            print("No memory records collected.")
            return

        # Sort by peak RSS memory to surface the heaviest bottleneck immediately
        df = df.sort_values(by="peak_rss_bytes", ascending=False)

        print(f"\n🏆 TOP PEAK MEMORY CONSUMERS (Total System Peak: {df['peak_rss_bytes'].max() / (1024**3):.2f} GB):")
        print("-" * 90)
        
        display_df = pd.DataFrame({
            "Stage / Subprocess": df["stage"],
            "Target / Details": df["detail"],
            "Start RAM": df["start_rss_gb"].map(lambda x: f"{x:.2f} GB"),
            "Peak RAM": df["peak_rss_gb"].map(lambda x: f"{x:.2f} GB"),
            "Delta (RAM)": df["delta_rss_gb"].map(lambda x: f"{x:+.2f} GB"),
            "Peak RAM (Bytes)": df["peak_rss_bytes"].map(lambda x: f"{x:,}"),
            "Duration": df["duration_sec"].map(lambda x: f"{x:.2f}s")
        })
        print(display_df.to_string(index=False))
        print("=" * 90 + "\n")


class TrackMemory:
    """Context manager that samples process memory at 10ms intervals to catch transient spikes."""
    
    def __init__(self, stage: str, detail: str = ""):
        self.stage = stage
        self.detail = detail
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self._stop_event = threading.Event()
        self._thread = None
        self.start_rss = 0
        self.start_time = 0

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                current_rss = self.process.memory_info().rss
                if current_rss > self.peak_rss:
                    self.peak_rss = current_rss
            except Exception:
                pass
            time.sleep(0.01)  # 10ms sampling interval

    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self.start_time = time.perf_counter()

        # Start sampling thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()

        end_rss = self.process.memory_info().rss
        duration = time.perf_counter() - self.start_time
        
        # Check one last time
        if end_rss > self.peak_rss:
            self.peak_rss = end_rss

        delta_rss = end_rss - self.start_rss

        MemoryProfileReport.records.append({
            "stage": self.stage,
            "detail": self.detail,
            "start_rss_gb": self.start_rss / (1024 ** 3),
            "peak_rss_gb": self.peak_rss / (1024 ** 3),
            "peak_rss_bytes": self.peak_rss,
            "delta_rss_gb": delta_rss / (1024 ** 3),
            "duration_sec": duration
        })

        print(
            f"  ↳ [MEM] {self.stage} ({self.detail}) | "
            f"Peak: {self.peak_rss / (1024**3):.2f} GB ({self.peak_rss:,} bytes) | "
            f"Delta: {delta_rss / (1024**3):+.2f} GB | Time: {duration:.2f}s"
        )


# ==========================================
# 2. INSTRUMENTED PIPELINE
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Libella Pipeline Memory Footprint")
    parser.add_argument("manifest", type=str, help="Path to manifest CSV.")
    parser.add_argument("--out-dir", type=str, default="./libella_output", help="Output directory.")
    parser.add_argument("--mode", type=str, choices=["DEV", "DISCOVERY", "PUBLISH"], default="PUBLISH")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining.")
    
    dummy_cfg = RunConfig()
    for field in dataclasses.fields(RunConfig):
        if field.name not in ["mode", "force_retrain", "suffix"]:
            arg_name = f"--{field.name.replace('_', '-')}"
            f_type = type(getattr(dummy_cfg, field.name)) if getattr(dummy_cfg, field.name) is not None else float
            if f_type == bool:
                parser.add_argument(arg_name, type=lambda x: (str(x).lower() == 'true'), default=None)
            else:
                parser.add_argument(arg_name, type=f_type, default=None)

    return parser.parse_args()


def setup_config_from_args(args: argparse.Namespace) -> None:
    new_cfg = RunConfig.from_mode(args.mode)
    new_cfg.force_retrain = args.force_retrain
    
    for field in dataclasses.fields(RunConfig):
        if field.name not in ["mode", "force_retrain", "suffix"]:
            val = getattr(args, field.name, None)
            if val is not None:
                setattr(new_cfg, field.name, val)
    
    for key, value in vars(new_cfg).items():
        setattr(cfg, key, value)
        
    paths.out_base = Path(args.out_dir).resolve()


def run_pipeline_instrumented(manifest_path: Path) -> None:
    init_env()
    out_dirs = paths.make_dirs(cfg.suffix)
    
    with TrackMemory("Manifest Ingestion", manifest_path.name):
        manifest = pd.read_csv(manifest_path)
        manifest.columns = manifest.columns.str.strip()
        for col in manifest.columns:
            if manifest[col].dtype == 'object':
                manifest[col] = manifest[col].astype(str).str.strip()

        manifest["discovery"] = manifest["discovery"].astype(str).str.lower().isin(["true", "1", "yes", "t"])
        manifest["projection"] = manifest["projection"].astype(str).str.lower().isin(["true", "1", "yes", "t"])

        if "patient_id" not in manifest.columns:
            manifest["patient_id"] = manifest["filepath"].apply(lambda x: Path(x).stem)
        if "dataset_id" not in manifest.columns:
            manifest["dataset_id"] = "Unknown"
            
        file_metadata = {
            Path(row["filepath"]): {"pt_id": str(row["patient_id"]), "ds_id": str(row["dataset_id"])}
            for _, row in manifest.iterrows()
        }
            
        discovery_files = [Path(f) for f in manifest[manifest["discovery"]]["filepath"]]
        projection_files = [Path(f) for f in manifest[manifest["projection"]]["filepath"]]
        
        if cfg.mode == "DEV":
            discovery_files = discovery_files[:2]
            projection_files = projection_files[:2]

    nmf_model_path = out_dirs["nmf_model"]
    genes_path = out_dirs["genes"]
    names_path = out_dirs["names"]
    out_dir = out_dirs["out"]

    if nmf_model_path.exists() and genes_path.exists() and names_path.exists() and not cfg.force_retrain:
        with TrackMemory("Load Pretrained Model", "Disk to Memory"):
            _ = torch.load(nmf_model_path, map_location="cpu", weights_only=False)
            with open(genes_path, "r") as f: common_genes = json.load(f)
            with open(names_path, "r") as f: name_data = json.load(f)
            meta_names = name_data["names"]
            used_topics = np.array(name_data["used_indices"])
    else:
        # --- PHASE 1a: Graph Building ---
        if cfg.phase in ["ALL", "BUILD_GRAPHS", "EXTRACT_PRIORS", "TRAIN"]:
            clean_whitelist = set() if cfg.unsupervised else get_whitelist(paths.sig_csv)
            
            # Pre-flight Check
            if not cfg.unsupervised and len(discovery_files) > 0:
                with TrackMemory("Pre-flight Scanpy Check", discovery_files[0].name):
                    import scanpy as sc
                    _tmp_adata = sc.read_h5ad(discovery_files[0], backed='r')
                    _upper_genes = set(str(g).upper() for g in _tmp_adata.var_names)
                    _overlap = len(_upper_genes.intersection(clean_whitelist))

            # Consensus Gene Extraction
            consensus_cache = out_dir / f"consensus_genes_cache_{cfg.top_n_genes}.json"
            if consensus_cache.exists() and not cfg.force_retrain:
                with open(consensus_cache, "r") as f: common_genes = json.load(f)
            else:
                with TrackMemory("Consensus Genes Calculation", f"{len(discovery_files)} files"):
                    common_genes = get_consensus_genes(discovery_files, clean_whitelist, top_n=cfg.top_n_genes)
                with open(consensus_cache, "w") as f: json.dump(common_genes, f)
                with open(genes_path, "w") as f: json.dump([str(g) for g in common_genes], f)

            # Per-File Spatial Graph Construction
            g_paths = []
            for f in tqdm(discovery_files, desc="Building Spatial Graphs", leave=False):
                with TrackMemory("build_graph_safe", f.name):
                    if res := build_graph_safe(f, common_genes):
                        g_paths.append(res)
                gc.collect()

        # --- PHASE 1b: Training ---
        if cfg.phase in ["ALL", "EXTRACT_PRIORS", "TRAIN"]:
            if 'g_paths' not in locals():
                g_paths = list(out_dirs["graphs"].glob("*_graph.pt"))
                with open(genes_path, "r") as f: common_genes = json.load(f)

            with TrackMemory("train_gnn", f"{len(g_paths)} graphs"):
                model, hist, optimal_k = train_gnn(g_paths, common_genes)
                plot_curves(hist)

            with TrackMemory("get_ecotypes", f"{len(g_paths)} graphs"):
                common_genes, meta_names, used_topics = get_ecotypes(model, g_paths, common_genes)

            del model
            gc.collect()

    # --- PHASE 2: Inference & Domain Smoothing ---
    if cfg.phase in ["ALL", "INFERENCE"]:
        all_prebuilt_graphs = list(out_dirs["graphs"].glob("*_graph.pt"))
        
        if 'common_genes' not in locals():
            with open(genes_path, "r") as f: common_genes = json.load(f)
        if 'meta_names' not in locals():
            with open(names_path, "r") as f: 
                name_data = json.load(f)
                meta_names = name_data["names"]
                used_topics = np.array(name_data["used_indices"])

        with TrackMemory("make_domains", f"{len(all_prebuilt_graphs)} graphs"):
            make_domains(
                graph_paths=all_prebuilt_graphs, 
                common_genes=common_genes,
                model_path=nmf_model_path,
                out_dir=out_dir
            )
        
        # --- PHASE 3: Per-Patient Projection ---
        results = []
        for f in tqdm(projection_files, desc="Projecting Patient Ecologies"):
            pt_id = file_metadata[f]["pt_id"]
            ds_id = file_metadata[f]["ds_id"]
            with TrackMemory("process_pt", f"Patient: {pt_id}"):
                if res := process_pt(f, common_genes, meta_names, used_topics, pt_id, ds_id):
                    results.append(res)
            gc.collect()
                
        if results:
            with TrackMemory("run_meta", f"{len(results)} patient results"):
                run_meta(results)


def main():
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    setup_config_from_args(args)
    
    try:
        run_pipeline_instrumented(Path(args.manifest.strip()))
    finally:
        # Ensure the report prints even if an OOM/Exception occurs mid-run
        MemoryProfileReport.print_summary()


if __name__ == "__main__":
    main()