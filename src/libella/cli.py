import argparse
import gc
import json
import multiprocessing
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import dataclasses

from .config import RunConfig, cfg, paths, init_env
from .data import get_consensus_genes, build_graph_safe
from .inference import (
    get_ecotypes,
    make_domains,
    plot_curves,
    process_pt,
    run_meta,
)
from .model import LibellaGNN              
from .train import train_gnn
from .utils import get_device, get_whitelist # Moved import here

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Libella Spatial Transcriptomics Pipeline")
    
    # Core Arguments
    parser.add_argument("manifest", type=str, help="Path to manifest CSV.")
    parser.add_argument("--out-dir", type=str, default="./libella_output", help="Output directory.")
    parser.add_argument("--mode", type=str, choices=["DEV", "DISCOVERY", "PUBLISH"], default="PUBLISH")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining.")
    
    # Dynamically expose ALL RunConfig fields as CLI arguments
    dummy_cfg = RunConfig()
    for field in dataclasses.fields(RunConfig):
        if field.name not in ["mode", "force_retrain", "suffix"]:
            arg_name = f"--{field.name.replace('_', '-')}"
            f_type = type(getattr(dummy_cfg, field.name)) if getattr(dummy_cfg, field.name) is not None else float
            
            # Handle boolean toggles appropriately
            if f_type == bool:
                parser.add_argument(arg_name, type=lambda x: (str(x).lower() == 'true'), default=None, help=f"Override {field.name}")
            else:
                parser.add_argument(arg_name, type=f_type, default=None, help=f"Override {field.name} (default: {getattr(dummy_cfg, field.name)})")

    return parser.parse_args()

def setup_config_from_args(args: argparse.Namespace) -> None:
    """Update the global config instance dynamically based on CLI args."""
    new_cfg = RunConfig.from_mode(args.mode)
    new_cfg.force_retrain = args.force_retrain
    
    # Override defaults with any explicit CLI arguments provided by the user
    for field in dataclasses.fields(RunConfig):
        if field.name not in ["mode", "force_retrain", "suffix"]:
            val = getattr(args, field.name, None)
            if val is not None:
                setattr(new_cfg, field.name, val)
    
    # Update the global `cfg` object in-place
    for key, value in vars(new_cfg).items():
        setattr(cfg, key, value)
        
    paths.out_base = Path(args.out_dir).resolve()

def run_pipeline(manifest_path: Path) -> None:
    """Execute the full Libella pipeline."""
    init_env()
    out_dirs = paths.make_dirs(cfg.suffix)
    
    # Read Manifest
    manifest = pd.read_csv(manifest_path)
    manifest.columns = manifest.columns.str.strip()
    
    for col in manifest.columns:
        if manifest[col].dtype == 'object':
            manifest[col] = manifest[col].astype(str).str.strip()

    if "filepath" not in manifest.columns:
        raise ValueError("Manifest CSV must contain a 'filepath' column.")
        
    if "discovery" not in manifest.columns or "projection" not in manifest.columns:
        raise ValueError("Manifest CSV must contain 'discovery' and 'projection' columns (e.g., True/False).")
    
    manifest["discovery"] = manifest["discovery"].astype(str).str.lower().isin(["true", "1", "yes", "t"])
    manifest["projection"] = manifest["projection"].astype(str).str.lower().isin(["true", "1", "yes", "t"])
    

    if "patient_id" not in manifest.columns:
        manifest["patient_id"] = manifest["filepath"].apply(lambda x: Path(x).stem)
    if "dataset_id" not in manifest.columns:
        manifest["dataset_id"] = "Unknown"
        
    # Build a metadata lookup dictionary
    file_metadata = {
        Path(row["filepath"]): {"pt_id": str(row["patient_id"]), "ds_id": str(row["dataset_id"])}
        for _, row in manifest.iterrows()
    }
        
    discovery_files = [Path(f) for f in manifest[manifest["discovery"]]["filepath"]]
    projection_files = [Path(f) for f in manifest[manifest["projection"]]["filepath"]]
    
    if cfg.mode == "DEV":
        print("\n[!] DEV MODE: Slicing cohorts to 2 files maximum.")
        discovery_files = discovery_files[:2]
        projection_files = projection_files[:2]
        
    if not discovery_files:
        raise ValueError(f"No discovery files found in {manifest_path}!")

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
        # --- GRAPH BUILDING BLOCK ---
        if cfg.phase in ["ALL", "BUILD_GRAPHS", "EXTRACT_PRIORS", "TRAIN"]:
            print(f"\n[➤] PHASE 1a: Building Spatial Graphs ({len(discovery_files)} WT samples)")
            consensus_cache = out_dir / f"consensus_genes_cache_{cfg.top_n_genes}.json"
            
            clean_whitelist = set() if cfg.unsupervised else get_whitelist(paths.sig_csv)
            
            # 🚨 ADDED: Pre-flight Gene Name Check
            if not cfg.unsupervised and len(discovery_files) > 0:
                import scanpy as sc
                _tmp_adata = sc.read_h5ad(discovery_files[0], backed='r')
                _upper_genes = set(str(g).upper() for g in _tmp_adata.var_names)
                _overlap = len(_upper_genes.intersection(clean_whitelist))
                if _overlap < 50:
                    raise ValueError(
                        f"\n[!] CRITICAL DATA ERROR: Only {_overlap} genes in {discovery_files[0].name} matched the signature dictionary.\n"
                        f"    Libella requires standard Human HGNC gene symbols (e.g., 'CD8A', 'SOX2') in adata.var_names.\n"
                        f"    If your data uses Ensembl IDs (ENSG...) or alternative aliases, please map them to HGNC symbols before running Libella."
                    )
            
            if consensus_cache.exists() and not cfg.force_retrain:
                with open(consensus_cache, "r") as f:
                    common_genes = json.load(f)
            else:
                common_genes = get_consensus_genes(discovery_files, clean_whitelist, top_n=cfg.top_n_genes)
                gc.collect()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                with open(consensus_cache, "w") as f: json.dump(common_genes, f)
                with open(genes_path, "w") as f: json.dump([str(g) for g in common_genes], f)
            
            g_paths = []
            for f in tqdm(discovery_files, desc="Building Spatial Graphs", leave=False):
                if res := build_graph_safe(f, common_genes):
                    g_paths.append(res)
                gc.collect()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            if cfg.phase == "BUILD_GRAPHS":
                print("\n[✓] Phase 'BUILD_GRAPHS' complete. Exiting.")
                return

        # --- TRAINING & PRIOR BLOCK ---
        if cfg.phase in ["ALL", "EXTRACT_PRIORS", "TRAIN"]:
            print(f"\n[➤] PHASE 1b: GNN Training & MetaProgram Extraction")
            # If we bypassed graph building, load preexisting ones
            if 'g_paths' not in locals():
                g_paths = list(out_dirs["graphs"].glob("*_graph.pt"))
                with open(genes_path, "r") as f: common_genes = json.load(f)

            model, hist, optimal_k = train_gnn(g_paths, common_genes)
            plot_curves(hist)
            
            del model; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            if cfg.phase == "TRAIN":
                print("\n[✓] Phase 'TRAIN' complete. Exiting.")
                return

    # --- INFERENCE & TOPOLOGY BLOCK ---
    if cfg.phase in ["ALL", "INFERENCE"]:
        print("\n[➤] PHASE 2: Spatial Projection & Macro-Domain Smoothing")
        all_prebuilt_graphs = list(out_dirs["graphs"].glob("*_graph.pt"))
        
        # Ensure dependencies exist if we launched straight into inference
        if 'common_genes' not in locals():
            with open(genes_path, "r") as f: common_genes = json.load(f)
        if 'meta_names' not in locals():
            with open(names_path, "r") as f: 
                name_data = json.load(f)
                meta_names = name_data["names"]
                used_topics = np.array(name_data["used_indices"])

        make_domains(
            graph_paths=all_prebuilt_graphs, 
            common_genes=common_genes,
            model_path=nmf_model_path,
            out_dir=out_dir
        )
        
        print("\n[➤] PHASE 3: Spatial Topology & Interface Mapping")
        results = []
        for f in tqdm(projection_files, desc="Projecting Patient Ecologies"):
            pt_id = file_metadata[f]["pt_id"]
            ds_id = file_metadata[f]["ds_id"]
            if res := process_pt(f, common_genes, meta_names, used_topics, pt_id, ds_id):
                results.append(res)
                
        if results:
            print("  ↳ Running Final Meta-Analysis on Cohort Interfaces...")
            run_meta(results)
            
        print(f"\n[✓] PIPELINE COMPLETE. All outputs saved to: {out_dir}\n")

def main() -> None:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    setup_config_from_args(args)
    run_pipeline(Path(args.manifest.strip()))

if __name__ == "__main__":
    main()