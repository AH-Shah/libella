"""Inference, topology mapping, and spatial domain generation."""

import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import t
from tqdm import tqdm

from .config import cfg, paths
from .data import (
    SpatialBatcher,
    build_spatial_graph,
    pad_mps_shapes,
    pt_to_scipy_csr,
    geo_sketch,
)
from .model import LibellaGNN
from .topology_3d import generate_3d_topology_network
from .utils import get_device

def plot_curves(history: dict[str, list]) -> None:
    """Plot GNN convergence history."""
    if not history or len(history.get("train_loss", [])) == 0: 
        return
    
    out_dir = paths.make_dirs(cfg.suffix)["out"]
    
    t_loss = [x.item() if torch.is_tensor(x) else x for x in history["train_loss"]]
    v_loss = [x.item() if torch.is_tensor(x) else x for x in history["val_loss"]]
    
    safe_history = {"train_loss": t_loss, "val_loss": v_loss}
    pd.DataFrame(safe_history).to_csv(out_dir / "GNN_Training_History.csv", index_label="Epoch")
    
    plt.figure(figsize=(6, 4))
    plt.plot(t_loss, label="Train Loss", color="#4B8BBE", lw=2)
    plt.plot(v_loss, label="Val Loss", color="#E66100", lw=2)
    plt.title("GNN Convergence Curve", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("log(cosh) Loss")
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_dir / "Fig_S1_GNN_Learning_Curve.pdf", transparent=True)
    plt.close()

def validate_panel(
    present_genes: list[str], common_genes: list[str], model_path: Path
) -> list[int]:
    """Validate gene panel signal enrichment."""

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        logits = checkpoint["model_state_dict"]["topic_gene_logits"]
        optimal_k = checkpoint.get("optimal_k", logits.shape[0]) 
        
        signatures = F.softplus(logits).detach().cpu().numpy()
        signatures = signatures / (signatures.sum(axis=1, keepdims=True) + 1e-9)
    except Exception as e:
        print(f"Warning: Failed loading model checkpoint in validation: {e}", file=sys.stderr)
        return []
    
    gene_to_idx = {g: i for i, g in enumerate(common_genes)}
    present_idx = np.array([gene_to_idx[g] for g in present_genes if g in gene_to_idx])
    
    M_panel = len(present_idx)
    N_total = max(1, len(common_genes))
    
    if M_panel == 0: return []
    if (M_panel / N_total) >= cfg.panel_overlap_thresh: return list(range(optimal_k))
        
    random_expectation_floor = M_panel / N_total
    p_sample = np.ones(N_total) / N_total
    n_perms = min(1000, cfg.n_perms_entropy)
    
    u_noise = np.random.rand(n_perms, N_total)
    gumbel_noise = -np.log(-np.log(u_noise + 1e-9))
    weighted_scores = np.log(p_sample + 1e-9) + gumbel_noise
    random_panels = np.argsort(weighted_scores, axis=1)[:, -M_panel:]
    
    valid_indices = []
    for i in range(optimal_k):
        if i >= signatures.shape[0]: break
        observed_mass = signatures[i, present_idx].sum()
        has_signal_enrichment = observed_mass > random_expectation_floor
        null_dist = signatures[i, random_panels].sum(axis=1)
        p_val = (np.sum(null_dist >= observed_mass) + 1.0) / (n_perms + 1.0)
        top_50_idx = np.argsort(signatures[i])[-50:]
        overlap_count = len(np.intersect1d(top_50_idx, present_idx))
        if (p_val < 0.05 and has_signal_enrichment) or (overlap_count >= 3):
            valid_indices.append(i)
    return valid_indices

def _calc_I_perm(
    F_mat: np.ndarray, 
    A_sp: sp.csr_matrix, 
    N: int, 
    starts: np.ndarray, 
    ends: np.ndarray, 
    s_idx: np.ndarray
) -> np.ndarray:
    """Fast localized topological permutations."""
    perm_idx = np.empty(N, dtype=np.int32)
    for st, ed in zip(starts, ends):
        if ed - st > 1:
            perm_idx[s_idx[st:ed]] = np.random.permutation(s_idx[st:ed])
        else:
            perm_idx[s_idx[st:ed]] = s_idx[st:ed]
    return F_mat[perm_idx].T @ (A_sp.dot(F_mat[perm_idx]))

def _load_inf_data(
    h5ad_path: Path, common_genes: list[str]
) -> tuple[sp.csr_matrix, np.ndarray, list[str]]:
    """Load, sketch, and normalize inference matrices."""

    adata_backed = sc.read_h5ad(h5ad_path, backed="r")
    
    gene_to_idx = {g: i for i, g in enumerate(adata_backed.var_names)}
    present_genes = [g for g in common_genes if g in gene_to_idx]
    col_indices = [gene_to_idx[g] for g in present_genes]
    
    adata_mem = adata_backed[:, col_indices].to_memory()
    X_raw_sub = adata_mem.X.tocsr()
    coords = adata_mem.obsm["spatial"].copy()
    
    del adata_backed, adata_mem; gc.collect()

    n_cells = X_raw_sub.shape[0]
    if cfg.max_cells_per_sample and n_cells > cfg.max_cells_per_sample:
        if cfg.use_sketching:
            adata_tmp = ad.AnnData(X=X_raw_sub)
            adata_tmp.var_names = present_genes
            adata_tmp.obsm["spatial"] = coords
            idx = geo_sketch(adata_tmp, cfg.max_cells_per_sample, common_genes)
            del adata_tmp
        else:
            idx = np.sort(np.random.choice(n_cells, cfg.max_cells_per_sample, replace=False))
            
        X_raw_sub = X_raw_sub[idx]
        coords = coords[idx]
        n_cells = X_raw_sub.shape[0]
    
    n_genes = len(common_genes)
    target_col_map = np.array([common_genes.index(g) for g in present_genes])
    
    X_raw_coo = X_raw_sub.tocoo()
    new_cols = target_col_map[X_raw_coo.col]
    X_sparse = sp.csr_matrix((X_raw_coo.data, (X_raw_coo.row, new_cols)), 
                             shape=(n_cells, n_genes), dtype=np.float32)
    del X_raw_sub, X_raw_coo; gc.collect()
    
    X_sparse.data = np.clip(X_sparse.data, 0, None)
    
    cell_sums = np.array(X_sparse.sum(axis=1)).flatten()
    cell_sums[cell_sums == 0] = 1.0 
    scaling_factors = 1e4 / cell_sums
    X_sparse = sp.diags(scaling_factors) @ X_sparse
    X_sparse.data = np.log1p(X_sparse.data)
    
    return X_sparse, coords, present_genes

def _run_inf(
    X_in_sparse: sp.csr_matrix, 
    coords: np.ndarray, 
    common_genes: list[str], 
    used_indices: np.ndarray
) -> np.ndarray:
    """Run GC-safe PyTorch graph inference."""
    device = get_device()
    nmf_model_path = paths.make_dirs(cfg.suffix)["nmf_model"]
    adj_sym = build_spatial_graph(coords, X_in_sparse)
    
    model_checkpoint = torch.load(nmf_model_path, map_location=device, weights_only=False)
    saved_k = model_checkpoint["model_state_dict"]["topic_gene_logits"].shape[0]
    model = LibellaGNN(in_channels=len(common_genes), n_metaprograms=saved_k).to(device)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.eval()
    
    N_cells = X_in_sparse.shape[0]
    batcher = SpatialBatcher(
        X=X_in_sparse, adj=adj_sym, coords=coords, 
        train_mask=np.zeros(N_cells), val_mask=np.zeros(N_cells), 
        batch_size=cfg.batch_size, k_hops=cfg.k_hops, shuffle=False 
    )
    
    W_global = np.zeros((N_cells, saved_k), dtype=np.float32)
    with torch.no_grad():
        for batch in batcher:
            x_dense = torch.from_numpy(batch["x"].toarray()).float().to(device)
            adj_coo = batch["adj"].tocoo()
            src = torch.from_numpy(adj_coo.row).long().to(device)
            dst = torch.from_numpy(adj_coo.col).long().to(device)
            weights = torch.from_numpy(adj_coo.data).float().to(device)
            x_dense, src, dst, weights = pad_mps_shapes(x_dense, src, dst, weights)
            fracs, _ = model(x_dense, src, dst, weights)
            fracs_cpu = fracs.cpu().numpy()
            W_global[batch["orig_core_idx"]] = fracs_cpu[batch["local_core_idx"]]
            
            del x_dense, adj_coo, src, dst, weights, fracs, batch, fracs_cpu
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
    del model, batcher, adj_sym; gc.collect()
    
    W_global = W_global / (W_global.sum(axis=1, keepdims=True) + 1e-9)
    return W_global[:, used_indices]

def _calc_topo(
    W: np.ndarray, 
    coords: np.ndarray, 
    active_meta_names: list[str], 
    sid: str, 
    did: str
) -> pd.DataFrame:
    """Extract spatial adjacency significance tests."""

    tree = cKDTree(coords)
    N_cells = W.shape[0]
    K_niches = len(active_meta_names)
    
    median_nn_dist = np.median(tree.query(coords, k=2)[0][:, 1])
    radius = median_nn_dist * cfg.radius_multiplier

    pairs_arr = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs_arr) > 0:
        row = np.concatenate([pairs_arr[:, 0], pairs_arr[:, 1]])
        col = np.concatenate([pairs_arr[:, 1], pairs_arr[:, 0]])
        data_ones = np.ones(len(row), dtype=np.float32)
        A_sparse = sp.csr_matrix((data_ones, (row, col)), shape=(N_cells, N_cells))
    else:
        A_sparse = sp.csr_matrix((N_cells, N_cells), dtype=np.float32)
        
    A_sparse.setdiag(0) 
    A_sparse.eliminate_zeros()
    
    I_matrix = W.T @ (A_sparse.dot(W))
    
    n_blocks = max(2, N_cells // 200)
    n_bins = max(2, int(np.ceil(np.sqrt(n_blocks))))
    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_bins + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), n_bins + 1)
    s_blks = (np.digitize(coords[:, 0], x_bins) - 1) * (n_bins + 1) + (np.digitize(coords[:, 1], y_bins) - 1)
    
    sort_idx = np.argsort(s_blks)
    s_blks_sorted = s_blks[sort_idx]
    _, block_starts = np.unique(s_blks_sorted, return_index=True)
    block_ends = np.append(block_starts[1:], N_cells)

    n_perms = cfg.n_perms_topo
    expected_I_stack = np.array(Parallel(n_jobs=-1, backend="loky")(
        delayed(_calc_I_perm)(W, A_sparse, N_cells, block_starts, block_ends, sort_idx) for _ in range(n_perms)
    ))
        
    E_mean = expected_I_stack.mean(axis=0)
    E_std = expected_I_stack.std(axis=0) + 1e-9
    
    I_sym = I_matrix + I_matrix.T
    E_mean_sym = E_mean + E_mean.T
    E_std_sym = np.sqrt(E_std**2 + E_std.T**2) + 1e-9 
    
    idx_i, idx_j = np.triu_indices(K_niches, k=1)
    obs_all = I_sym[idx_i, idx_j]
    exp_m_all = E_mean_sym[idx_i, idx_j]
    exp_s_all = E_std_sym[idx_i, idx_j]
    
    z_scores = (obs_all - exp_m_all) / exp_s_all
    log2_enrich = np.log2((obs_all + 1e-9) / (exp_m_all + 1e-9))
    
    topo_res = []
    mass_prob = W.sum(axis=0) / N_cells
    
    for idx in range(len(idx_i)):
        i, j = idx_i[idx], idx_j[idx]
        if (mass_prob[i] > cfg.min_topic_mass) and (mass_prob[j] > cfg.min_topic_mass):
            topo_res.append({
                "Sample": sid, 
                "Dataset": did, 
                "Interface": f"{active_meta_names[i]}<->{active_meta_names[j]}",
                "Continuous_Observed": obs_all[idx],
                "Continuous_Expected_Mean": exp_m_all[idx],
                "Spatial_Z_Score": z_scores[idx],
                "Log2_Enrichment": log2_enrich[idx]
            })
                
    return pd.DataFrame(topo_res)

class SpatialAnalyzer:
    """OOM-safe spatial inference orchestrator."""
    def __init__(
        self, 
        h5ad_path: Path, 
        common_genes: list[str], 
        meta_names: list[str], 
        used_indices: np.ndarray,
        pt_id: str,
        ds_id: str
    ) -> None:
        self.h5ad_path = h5ad_path
        self.common_genes = common_genes
        self.meta_names = meta_names
        self.used_indices_global = used_indices
        self.sid = pt_id
        self.did = ds_id
        self.results: dict[str, pd.DataFrame] = {}

    def run(self) -> None:
        X_in_sparse, coords, present_genes = _load_inf_data(self.h5ad_path, self.common_genes)
        
        if len(coords) < 50:
            return

        nmf_model_path = paths.make_dirs(cfg.suffix)["nmf_model"]
        rel_indices = validate_panel(present_genes, self.common_genes, nmf_model_path)
        final_valid_indices = [idx for idx in rel_indices if idx in self.used_indices_global]
        used_indices = np.array(final_valid_indices, dtype=np.int32)
        
        used_list = list(self.used_indices_global)
        active_meta_names = [self.meta_names[used_list.index(idx)] for idx in final_valid_indices]
        
        if len(used_indices) == 0:
            return

        W = _run_inf(X_in_sparse, coords, self.common_genes, used_indices)
        del X_in_sparse; gc.collect()

        topo_df = _calc_topo(W, coords, active_meta_names, self.sid, self.did)
        del W, coords; gc.collect()

        self.results["Topology"] = topo_df
        
        indiv_out = paths.make_dirs(cfg.suffix)["indiv"]
        for k, v in self.results.items():
            if not v.empty: 
                out_path = indiv_out / self.sid
                out_path.mkdir(parents=True, exist_ok=True)
                v.to_csv(out_path / f"{k}.csv", index=False)

def process_pt(
    f: Path, common_genes: list[str], meta_names: list[str], used_indices: np.ndarray,
    pt_id: str, ds_id: str
) -> dict[str, pd.DataFrame] | None:
    """Process single patient topology."""
    out_dirs = paths.make_dirs(cfg.suffix)
    indiv_out = out_dirs["indiv"]
    out_dir = out_dirs["out"]
    
    s_dir = indiv_out / pt_id
    
    if (s_dir / "Topology.csv").exists(): 
        return {
            "Topology": pd.read_csv(s_dir / "Topology.csv")
        }
        
    try:
        analyzer = SpatialAnalyzer(f, common_genes, meta_names, used_indices, pt_id, ds_id)
        analyzer.run()
        return analyzer.results
    except Exception as e:
        with open(out_dir / "error_log.txt", "a") as err_f: 
            err_f.write(f"[{f.stem}] FAILED: {e}\n")
        return None

def run_meta(all_results: list[dict[str, pd.DataFrame] | None]) -> None:
    """Run cross-cohort topological meta-analysis."""
    out_dir = paths.make_dirs(cfg.suffix)["out"]
    
    def fdr_bh(pvals: list[float] | np.ndarray) -> np.ndarray:
        pvals = np.asarray(pvals)
        n = len(pvals)
        if n == 0: return np.array([])
        sorted_indices = np.argsort(pvals)
        sorted_pvals = pvals[sorted_indices]
        qvals = np.zeros(n)
        min_q = 1.0
        for i in range(n - 1, -1, -1):
            q = (sorted_pvals[i] * n) / (i + 1)
            min_q = min(min_q, q)
            qvals[sorted_indices[i]] = min_q
        return np.minimum(qvals, 1.0)
    
    all_programs_set: set[str] = set()
    for r in all_results:
        if r is None: continue
        if "Topology" in r and not r["Topology"].empty:
            for interface in r["Topology"]["Interface"].unique():
                all_programs_set.update(interface.split("<->"))
                
    def sort_key(x: str) -> int:
        match = re.search(r"ECO(\d+)", x)
        return int(match.group(1)) if match else 999
        
    master_programs_sorted = sorted(list(all_programs_set), key=sort_key)


    topo_list = [r["Topology"] for r in all_results if r is not None and "Topology" in r and not r["Topology"].empty]
    if topo_list:
        df = pd.concat(topo_list).assign(
            Patient_ID=lambda x: x["Sample"]
        ).groupby(["Patient_ID", "Interface"]).mean(numeric_only=True).reset_index()
        
        res = []
        for i in df["Interface"].unique():
            sub = df[df["Interface"] == i].dropna(subset=["Log2_Enrichment"])
            pts = sub["Patient_ID"].unique()
            k_pts = len(pts)
            
            if k_pts > 1:
                log2_vals = sub["Log2_Enrichment"].values
                meta_eff = np.mean(log2_vals)
                meta_se = np.std(log2_vals, ddof=1) / (np.sqrt(k_pts) + 1e-9)
                meta_t = meta_eff / (meta_se + 1e-9)
                p_val = t.sf(np.abs(meta_t), df=k_pts-1) * 2.0
                
                lopo = 0
                for pt in pts:
                    s = sub[sub["Patient_ID"] != pt]["Log2_Enrichment"].values
                    if len(s) > 1:
                        loo_eff = np.mean(s)
                        loo_se = np.std(s, ddof=1) / (np.sqrt(len(s)) + 1e-9)
                        loo_t = loo_eff / (loo_se + 1e-9)
                        loo_p = t.sf(np.abs(loo_t), df=len(s)-1) * 2.0
                        
                        if (loo_eff * meta_eff > 0) and (loo_p < 0.05):
                            lopo += 1
                            
                res.append({
                    "Interface": i, 
                    "Pooled_Log2_Enrichment": round(float(meta_eff), 3), 
                    "Cross_Patient_SE": round(float(meta_se), 3),
                    "P_val_raw": p_val, 
                    "LOPO_Robustness": f"{lopo}/{k_pts}"
                })
                
        if res: 
            pvals = [r["P_val_raw"] for r in res]
            qvals = fdr_bh(pvals)
            for i, r in enumerate(res):
                r["FDR_q"] = round(qvals[i], 4)
                del r["P_val_raw"] 
            
            matrix_df = pd.DataFrame("", index=master_programs_sorted, columns=master_programs_sorted)
            
            for r in res:
                p1, p2 = r["Interface"].split("<->")
                idx1 = master_programs_sorted.index(p1)
                idx2 = master_programs_sorted.index(p2)
                
                cell_val = f"Eff:{r['Pooled_Log2_Enrichment']:.3f} | {r['LOPO_Robustness']} | {r['FDR_q']:.3f}"
                
                if idx1 > idx2:
                    matrix_df.loc[p1, p2] = cell_val
                elif idx2 > idx1:
                    matrix_df.loc[p2, p1] = cell_val

            matrix_df.to_csv(out_dir / "Global_Meta_Topology_Continuous_Matrix.csv")

            generate_3d_topology_network(
                res=res, 
                output_path=out_dir / "Global_Meta_Topology_3D.html", 
                fdr_threshold=cfg.fdr_threshold
            )


def get_ecotypes(
    model: LibellaGNN, graph_paths: list[Path], common_genes: list[str]
) -> tuple[list[str], list[str], np.ndarray]:
    """Execute GNN inference and discover tissue ecotypes."""
    model.eval()
    device = get_device()
    out_dirs = paths.make_dirs(cfg.suffix)
    
    model.current_progress = 1.0
    model.current_scale = cfg.inference_scale
    model.current_alpha = cfg.inference_alpha
    model.current_temp = cfg.inference_temp
    
    with torch.no_grad():
        safe_temp = torch.clamp(getattr(model, "dict_temp", torch.tensor(0.30)), min=0.25, max=1.0)
        components = F.softmax(model.topic_gene_logits / safe_temp, dim=-1).cpu().numpy()

    all_fractions = []
    patient_names = [] 
    
    with torch.no_grad():
        for path in tqdm(graph_paths, desc="Extracting Spatial Ecotypes", leave=False):
            data = torch.load(path, map_location="cpu", weights_only=False) 
            
            X_sp = pt_to_scipy_csr(data, "x_in")
            N_cells = X_sp.shape[0]
            
            e_attr = data.edge_attr.numpy()
            e_row = data.edge_index[0].numpy()
            e_col = data.edge_index[1].numpy()
            
            delattr(data, "edge_attr")
            delattr(data, "edge_index")
            
            adj_sp = sp.csr_matrix((e_attr, (e_row, e_col)), shape=(N_cells, N_cells))
            del e_attr, e_row, e_col
            
            batcher = SpatialBatcher(
                X=X_sp, adj=adj_sp, coords=data.pos.numpy(),
                train_mask=np.zeros(N_cells), val_mask=np.zeros(N_cells), 
                batch_size=cfg.inf_batch_size, k_hops=cfg.k_hops, shuffle=False
            )
            
            K_topics = model.topic_gene_logits.shape[0]
            patient_frac_matrix = np.zeros((N_cells, K_topics), dtype=np.float32)
            
            for batch in batcher:
                x_dense = torch.from_numpy(batch["x"].toarray()).float().to(device)
                adj_coo = batch["adj"].tocoo()
                src = torch.from_numpy(adj_coo.row).long().to(device)
                dst = torch.from_numpy(adj_coo.col).long().to(device)
                weights = torch.from_numpy(adj_coo.data).float().to(device)
                x_dense, src, dst, weights = pad_mps_shapes(x_dense, src, dst, weights)
                
                fracs, _ = model(x_dense, src, dst, weights)
                fracs_cpu = fracs.cpu().numpy()
                patient_frac_matrix[batch["orig_core_idx"]] = fracs_cpu[batch["local_core_idx"]]
                
                del x_dense, adj_coo, src, dst, weights, fracs, batch, fracs_cpu
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
            
            patient_frac_matrix = patient_frac_matrix / (patient_frac_matrix.sum(axis=1, keepdims=True) + 1e-9)
            
            all_fractions.append(patient_frac_matrix)
            patient_names.extend([data.patient_name] * N_cells)
            del data, X_sp, adj_sp, batcher, patient_frac_matrix; gc.collect()
            
    all_fractions = np.vstack(all_fractions)
    optimal_k = model.topic_gene_logits.shape[0]
    
    print(f"  ↳ Refining {optimal_k} Initial Programs...")
    used_topics = refine_outputs(model, all_fractions, patient_names, common_genes, components)
    
    topic_mass = all_fractions.mean(axis=0)
    merged_components = components[used_topics]
    merged_components = merged_components / (merged_components.sum(axis=1, keepdims=True) + 1e-9)
    
    print(f"\n[✓] Libella discovered {len(merged_components)} distinct Meso-Niches:")
    
    meta_names = []
    for i in range(len(merged_components)):
        top_indices = np.argsort(merged_components[i])[::-1]
        clean_top_genes = [common_genes[idx] for idx in top_indices][:3]
        top_genes_str = "_".join(clean_top_genes)
        meta_names.append(f"{i+1}_{top_genes_str}")

    torch.save({
        "model_state_dict": model.state_dict(), 
        "optimal_k": optimal_k
    }, out_dirs["nmf_model"])
    
    with open(out_dirs["genes"], "w") as f: json.dump(common_genes, f)
    with open(out_dirs["names"], "w") as f: json.dump({"names": meta_names, "used_indices": used_topics.tolist()}, f)
    
    for i, name in enumerate(meta_names):
        print(f"    * {name} (Mass: {topic_mass[used_topics[i]]:.3f})")
        
    del model; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
        
    return common_genes, meta_names, used_topics


def refine_outputs(
    model: LibellaGNN, 
    all_fractions: np.ndarray, 
    patient_names: list[str], 
    common_genes: list[str], 
    components: np.ndarray
) -> np.ndarray:
    """Prune artifact programs using Shannon Entropy thresholds."""
    K = all_fractions.shape[1]
    out_dir = paths.make_dirs(cfg.suffix)["out"]
    
    # 1. Shannon Entropy Pruning (Patient-level)
    frac_df = pd.DataFrame(all_fractions)
    frac_df["patient"] = patient_names
    patient_mass = frac_df.groupby("patient").sum().values.T 
    
    probs = patient_mass / (patient_mass.sum(axis=1, keepdims=True) + 1e-9)
    true_entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    
    null_entropies = []
    n_perms = cfg.n_perms_entropy
    for _ in range(n_perms):
        shuffled = np.random.permutation(patient_mass.flatten()).reshape(patient_mass.shape)
        sh_p = shuffled / (shuffled.sum(axis=1, keepdims=True) + 1e-9)
        null_entropies.append(-np.sum(sh_p * np.log(sh_p + 1e-9), axis=1))
    
    null_threshold = np.percentile(np.vstack(null_entropies), 1)
    valid_entropy = true_entropy > null_threshold
    
    # 2. Extract Valid Indices directly 
    final_idx = [i for i in range(K) if valid_entropy[i]]

    # 3. Generate Markdown Report
    w_prob = components
    
    def get_top_genes_str(idx: int) -> str:
        top_idx = np.argsort(w_prob[idx])[-10:][::-1]
        return ", ".join([f"{common_genes[i]}|{w_prob[idx, i]:.3f}" for i in top_idx])

    md_lines = [
        "# GNN Spatial Ecotype Refinement Report\n",
        f"**Initial Discovered K:** `{K}`",
        f"**Final Refined K:** `{len(final_idx)}`\n",
        "## 1. Initial Discovered Programs\n"
    ]
    
    for i in range(K):
        md_lines.append(f"- **Program {i}**: {get_top_genes_str(i)}")
        
    md_lines.extend([
        "\n## 2. Artifact Pruning (Shannon Entropy)",
        f"- **Calculated Null Threshold (1st Percentile):** `{null_threshold:.3f}`",
        "\n**Deleted Programs:**"
    ])
    
    deleted_count = 0
    for i in range(K):
        if not valid_entropy[i]:
            max_p = (np.max(patient_mass[i]) / (np.sum(patient_mass[i]) + 1e-9)) * 100
            md_lines.append(f"- **Program {i}**: Deleted. (`{max_p:.1f}%` mass in 1 patient)")
            deleted_count += 1
            
    if deleted_count == 0:
        md_lines.append("- *No programs were deleted during entropy pruning.*")

    md_lines.extend([
        f"\n## 3. Final Refined Ecotypes (K = {len(final_idx)})\n"
    ])
    for i in final_idx:
        md_lines.append(f"- **Program {i}**: {get_top_genes_str(i)}")

    report_path = out_dir / "GNN_Program_Refinement_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"  ↳ Refinement complete. Report saved to: {report_path.name}")
    return np.array(final_idx)

def make_domains(
    graph_paths: list[Path], common_genes: list[str], model_path: Path, out_dir: Path
) -> Path | None:
    """Discover macro tissue domains via Leiden clustering and anisotropic smoothing."""
    out_pq = out_dir / "Global_Smoothed_Macro_Domains.parquet"
    
    if out_pq.is_file():
        print(f"[✓] {out_pq.name} already exists. Skipping Macro-Domain generation.")
        return out_pq

    try:
        device = get_device()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        K_topics = ckpt["model_state_dict"]["topic_gene_logits"].shape[0]
        
        model = LibellaGNN(in_channels=len(common_genes), n_metaprograms=K_topics).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        W_list = []
        
        print("  ↳ Projecting Libella across pre-built graphs...")
        with torch.no_grad():
            for path in tqdm(graph_paths, desc="Extracting Micro-States", leave=False):
                data = torch.load(path, map_location="cpu", weights_only=False)
                
                x_sp = pt_to_scipy_csr(data, "x_in")
                N = x_sp.shape[0]
                
                s = data.edge_index[0].numpy()
                d = data.edge_index[1].numpy()
                w = data.edge_attr.numpy()
                
                delattr(data, "edge_index")
                delattr(data, "edge_attr")
                
                adj_sp = sp.csr_matrix((w, (s, d)), shape=(N, N))
                
                W_local = np.zeros((N, K_topics), dtype=np.float32)
                batcher = SpatialBatcher(
                    X=x_sp, adj=adj_sp, coords=data.pos.numpy(), 
                    train_mask=np.zeros(N), val_mask=np.zeros(N), 
                    batch_size=cfg.inf_batch_size, k_hops=cfg.k_hops, shuffle=False
                )

                for b in batcher:
                    x_d = torch.from_numpy(b["x"].toarray()).float().to(device)
                    adj_c = b["adj"].tocoo()
                    bs = torch.from_numpy(adj_c.row).long().to(device)
                    bd = torch.from_numpy(adj_c.col).long().to(device)
                    bw = torch.from_numpy(adj_c.data).float().to(device)
                    
                    x_d, bs, bd, bw = pad_mps_shapes(x_d, bs, bd, bw)
                    
                    fracs, _ = model(x_d, bs, bd, bw) 
                    probs = fracs / (fracs.sum(dim=1, keepdim=True) + 1e-9)
                    W_local[b["orig_core_idx"]] = probs.cpu().numpy()[b["local_core_idx"]]
                    
                W_list.append(W_local)
                del data, x_sp, adj_sp, batcher; gc.collect()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()

        W_global = np.vstack(W_list)
        
        print("  ↳ Discovering Macro-Compartments (Auto-K Leiden)...")
        corr_mat = np.corrcoef(W_global.T)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0) 
        np.fill_diagonal(corr_mat, 1.0)
        
        adata = sc.AnnData(X=np.clip(corr_mat, -1.0, 1.0))
        sc.pp.neighbors(adata, n_neighbors=5, metric="correlation")
        sc.tl.leiden(adata, resolution=cfg.leiden_res, random_state=42)
        leiden_mapping = adata.obs["leiden"].astype(int).values
        
        n_domains = len(np.unique(leiden_mapping))
        print(f"    * Discovered {n_domains} Macro Tissue Domains.")
        print("  ↳ Applying Native Anisotropic Graph Smoothing...")
        
        writer = None 
        offset = 0
        
        for path in tqdm(graph_paths, desc="Smoothing & Streaming to Parquet", leave=False):
            data = torch.load(path, map_location="cpu", weights_only=False)
            coords = data.pos.numpy()
            N = coords.shape[0]
            
            s = data.edge_index[0].numpy()
            d = data.edge_index[1].numpy()
            A_native = sp.csr_matrix((np.ones_like(s, dtype=np.float32), (s, d)), shape=(N, N))
            
            if hasattr(data, "x_in"): delattr(data, "x_in")
            if hasattr(data, "edge_index"): delattr(data, "edge_index")
            if hasattr(data, "edge_attr"): delattr(data, "edge_attr")
            del data; gc.collect()
            
            W_pt = W_global[offset:offset+N]
            offset += N
            
            W_macro = np.zeros((N, n_domains), dtype=np.float32)
            for c in range(n_domains):
                topics_in_macro = np.where(leiden_mapping == c)[0]
                if len(topics_in_macro) > 0: 
                    W_macro[:, c] = W_pt[:, topics_in_macro].sum(axis=1)
                
            raw_macro_labels = np.argmax(W_macro, axis=1).astype(np.int32)
            W_macro_onehot = np.eye(n_domains, dtype=np.float32)[raw_macro_labels]
            
            votes = A_native @ W_macro_onehot
            votes = votes + (W_macro_onehot * cfg.smoothing_self_weight) 
            smoothed_macro_labels = np.argmax(votes, axis=1).astype(np.int32)
            
            raw_topic_labels = np.argmax(W_pt, axis=1).astype(np.int32)
            clean_patient_name = path.stem.replace("_graph", "")
            
            df_chunk = pd.DataFrame({
                "Patient": pd.Series([clean_patient_name] * N, dtype="category"), 
                "X": coords[:, 0].astype(np.float32),
                "Y": coords[:, 1].astype(np.float32),
                "Raw_GNN_MicroState": raw_topic_labels,
                "Raw_Macro_Domain": raw_macro_labels,
                "Smoothed_Macro_Domain": smoothed_macro_labels
            })
            
            table = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(out_pq, table.schema, compression="snappy")
            
            writer.write_table(table)
            
            del A_native, W_pt, W_macro, W_macro_onehot, votes, df_chunk, table
            gc.collect()
            
        if writer is not None:
            writer.close()
            
        print(f"[✓] Master spatial mapping saved to: {out_pq.name}")
        return out_pq
        
    except Exception as e:
        print(f"\n  ↳ [!] Phase 3 Macro-Domains failed: {str(e)}")
        print("  ↳ [!] Safely bypassing to Topology phase...")
        return None