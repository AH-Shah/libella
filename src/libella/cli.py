"""Command-line interface and main orchestration for Libella."""

import argparse
import gc
import json
import multiprocessing
from pathlib import Path
import pandas as pd

import numpy as np
import torch
from tqdm import tqdm

from .config import RunConfig, cfg, paths, init_env
from .data import get_consensus_genes, build_graph_safe
from .inference import (
    get_ecotypes,
    make_domains,
    plot_curves,
    process_pt,
    run_meta,
)
from .train import train_gnn
from .utils import get_device

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Libella Spatial Transcriptomics Pipeline")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["DEV", "DISCOVERY", "PUBLISH"], 
        default="PUBLISH",
        help="Pipeline execution mode."
    )
    parser.add_argument(
        "--force-retrain", 
        action="store_true",
        help="Force the GNN to retrain even if checkpoints exist."
    )
    return parser.parse_args()

def setup_config_from_args(args: argparse.Namespace) -> None:
    """Update the global config instance dynamically based on CLI args."""
    new_cfg = RunConfig.from_mode(args.mode)
    new_cfg.force_retrain = args.force_retrain
    
    # Update the existing global `cfg` object in-place so module references hold
    for key, value in vars(new_cfg).items():
        setattr(cfg, key, value)

    
def run_pipeline() -> None:
    """Execute the full Libella pipeline."""
    init_env()
    
    out_dirs = paths.make_dirs(cfg.suffix)
    
    all_h5ad_files = list(paths.data_dir.glob("*.h5ad"))
    wt_files = [f for f in all_h5ad_files if "wt" in f.name.lower() or "gse0" in f.name.lower()]
    discovery_files = wt_files
    
    if cfg.mode == "DEV":
        print("\n[!] DEV MODE: Slicing cohort to 2 files.")
        discovery_files = discovery_files[:2]
        all_h5ad_files = all_h5ad_files[:2]
        
    if not discovery_files:
        raise ValueError(f"No discovery files found in {paths.data_dir}!")

    print(f"\n[i] INITIATING Libella ({cfg.mode} MODE)")



    nmf_model_path = out_dirs["nmf_model"]
    genes_path = out_dirs["genes"]
    names_path = out_dirs["names"]
    out_dir = out_dirs["out"]

    if nmf_model_path.exists() and genes_path.exists() and names_path.exists() and not cfg.force_retrain:
        try:
            _ = torch.load(nmf_model_path, map_location="cpu", weights_only=False)
            with open(genes_path, "r") as f: 
                common_genes = json.load(f)
            with open(names_path, "r") as f: 
                name_data = json.load(f)
                
            meta_names = name_data["names"]
            used_topics = np.array(name_data["used_indices"])
            print(f"[✓] Loaded pre-trained Libella model ({len(meta_names)} Ecotypes)")
        except Exception as e:
            print(f"  ↳ [!] Pre-trained file verification failed ({e}). Forcing retrain sequence.")
            cfg.force_retrain = True
    else:
        print(f"\n[➤] PHASE 1: Discovering MetaPrograms ({len(discovery_files)} WT samples)")
        consensus_cache = out_dir / f"consensus_genes_cache_{cfg.top_n_genes}.json"
        
        from .utils import get_whitelist
        clean_whitelist = get_whitelist(paths.sig_csv)
        
        if consensus_cache.exists() and not cfg.force_retrain:
            with open(consensus_cache, "r") as f:
                common_genes = json.load(f)
        else:
            common_genes = get_consensus_genes(discovery_files, clean_whitelist, top_n=cfg.top_n_genes)
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            with open(consensus_cache, "w") as f: json.dump(common_genes, f)
            with open(genes_path, "w") as f: json.dump([str(g) for g in common_genes], f)
        
        g_paths = []
        for f in tqdm(discovery_files, desc="Building Spatial Graphs", leave=False):
            if res := build_graph_safe(f, common_genes):
                g_paths.append(res)
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
        
        model, hist, optimal_k = train_gnn(g_paths, common_genes)
        plot_curves(hist)
        
        common_genes, meta_names, used_topics = get_ecotypes(model, g_paths, common_genes)
        del model; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    print("\n[➤] PHASE 2: Spatial Projection & Macro-Domain Smoothing")
    all_prebuilt_graphs = list(out_dirs["graphs"].glob("*_graph.pt"))

    make_domains(
        graph_paths=all_prebuilt_graphs, 
        common_genes=common_genes,
        model_path=nmf_model_path,
        out_dir=out_dir
    )
    
    print("\n[➤] PHASE 3: Spatial Topology & Interface Mapping")
    results = []
    for f in tqdm(all_h5ad_files, desc="Projecting Patient Ecologies"):
        if res := process_pt(f, common_genes, meta_names, used_topics):
            results.append(res)
            
    if results:
        print("  ↳ Running Final Meta-Analysis on Cohort Interfaces...")
        run_meta(results)
        
    print(f"\n[✓] PIPELINE COMPLETE. All outputs saved to: {out_dir}\n")

def main() -> None:
    """Entry point for the command line interface."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    setup_config_from_args(args)
    run_pipeline(Path(args.manifest))

if __name__ == "__main__":
    main()