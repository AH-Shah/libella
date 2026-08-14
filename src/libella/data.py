"""Data processing, graph construction, and spatial sampling routines."""

import gc
import multiprocessing
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from joblib import delayed
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from tqdm import tqdm
from collections.abc import Iterator

from .config import cfg, paths
from .utils import get_device

def calc_moran(
    X_sparse: sp.csr_matrix, edge_index: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Compute spatial Moran's I using O(1) memory algebraic expansion."""
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        return np.zeros(X_sparse.shape[1], dtype=np.float32)
        
    # 1. Compute basic statistics directly from sparse
    mean = np.array(X_sparse.mean(axis=0)).flatten() 
    sum_x = np.array(X_sparse.sum(axis=0)).flatten()
    
    X_sq = X_sparse.copy()
    X_sq.data **= 2
    sum_x2 = np.array(X_sq.sum(axis=0)).flatten()
    del X_sq # Free temporary squared matrix instantly
    
    # Denominator: sum((X - mean)^2)
    denom = sum_x2 - 2 * mean * sum_x + num_nodes * (mean ** 2) + 1e-9
    
    # 2. Build fast Sparse Adjacency from edges
    A = sp.coo_matrix((np.ones(n_edges, dtype=np.float32), (edge_index[0], edge_index[1])), 
                      shape=(num_nodes, num_nodes)).tocsr()
                      
    # 3. Algebraic Numerator Expansion (O(1) memory, C-optimized speed)
    # Sum of (X_u * X_v) over all edges globally:
    sum_u_v = np.array(X_sparse.multiply(A.dot(X_sparse)).sum(axis=0)).flatten()
    
    # Sum of X_u and X_v weighted by edge degrees
    deg_u = np.array(A.sum(axis=1)).flatten()
    deg_v = np.array(A.sum(axis=0)).flatten()
    
    sum_u = X_sparse.T.dot(deg_u)  # Highly optimized Sparse Matrix-Vector product
    sum_v = X_sparse.T.dot(deg_v)
    
    # Exact algebraic expansion of sum_edge (X_u - mean) * (X_v - mean)
    numerator = sum_u_v - mean * sum_u - mean * sum_v + n_edges * (mean ** 2)
    
    moran = (num_nodes / (n_edges + 1e-9)) * (numerator / denom)
    return np.nan_to_num(moran)


def build_spatial_graph(
    coords: np.ndarray, X_sparse: sp.csr_matrix, k_neighbors: int = cfg.k_neighbors
) -> sp.csr_matrix:
    """Construct anisotropic spatial graph using physical KNN and transcriptomic cosine similarity."""
    N = coords.shape[0]
    
    # 1. Physical KNN
    tree = cKDTree(coords)
    dists, knn_idx = tree.query(coords, k=k_neighbors, workers=-1)
    
    # Dynamically scale sizes based on whatever K you choose
    k_minus_1 = k_neighbors - 1
    src = np.repeat(np.arange(N), k_minus_1)
    dst = knn_idx[:, 1:].flatten()
    phys_dists = dists[:, 1:].flatten()
    
    # Algebraic L2 Norms bypasses the memory-doubling sklearn normalize()
    l2_norms = np.sqrt(np.array(X_sparse.power(2).sum(axis=1)).flatten())
    l2_norms[l2_norms == 0] = 1e-9  
    
    # 2. Transcriptomic Cosine Similarity (Cell-Chunked 3D Einsum)
    cos_sim = np.zeros(len(src), dtype=np.float32)
    

    chunk_size = cfg.chunk_size 
    idx_pointer = 0
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        n_chunk = end - i
        
        # 1. Source cells (No redundancy) -> Shape: (n_chunk, G)
        X_s_dense = X_sparse[i:end].toarray().astype(np.float32, copy=False)
        
        # 2. Dest cells (Neighbors) -> Shape: (n_chunk * k_minus_1, G)
        d_chunk_idx = knn_idx[i:end, 1:].flatten()
        X_d_dense = X_sparse[d_chunk_idx].toarray().astype(np.float32, copy=False)
        
        # 3. Reshape dest to match sources -> Shape: (n_chunk, k_minus_1, G)
        X_d_dense = X_d_dense.reshape(n_chunk, k_minus_1, -1)
        
        # 4. 🚨 FAST 3D EINSUM: Calculates exact dot products without duplicating source genes
        # 'ig' (Sources), 'ikg' (Destinations) -> 'ik' (Dot product per edge)
        dot_prods = np.einsum('ig,ikg->ik', X_s_dense, X_d_dense).flatten()
        
        # 5. Apply L2 normalization
        s_norms = l2_norms[i:end].repeat(k_minus_1)
        d_norms = l2_norms[d_chunk_idx]
        
        chunk_sims = dot_prods / (s_norms * d_norms)
        
        # Assign back to flat array
        edges_in_chunk = n_chunk * k_minus_1
        cos_sim[idx_pointer : idx_pointer + edges_in_chunk] = chunk_sims
        idx_pointer += edges_in_chunk
        
    # Free chunk memory instantly
    try: del X_s_dense, X_d_dense, dot_prods, chunk_sims, s_norms, d_norms
    except UnboundLocalError: pass
    gc.collect()
        
    # 3. Z-Score Pruning Per-Cell (NumPy Vectorized - Now Dynamic!)
    sim_matrix = cos_sim.reshape(N, k_minus_1)
    means = sim_matrix.mean(axis=1)
    stds = sim_matrix.std(axis=1) + 1e-9
    z_scores_flat = ((sim_matrix - means[:, None]) / stds[:, None]).flatten()
    
    mask = z_scores_flat > -0.5
    
    src_m = src[mask]
    dst_m = dst[mask]
    dist_m = phys_dists[mask]
    sim_m = cos_sim[mask]
    
    # 4. Bilateral Exponential Decay
    median_dist = np.median(dist_m) + 1e-9
    spatial_decay = np.exp(-(dist_m**2) / (2 * median_dist**2))
    weights = spatial_decay * sim_m
    
# 5. Build Symmetric Laplacian
    if len(src_m) > 0:
        edges = np.sort(np.vstack([src_m, dst_m]).T, axis=1)
        final_edges, unique_idx = np.unique(edges, axis=0, return_index=True)
        final_weights = weights[unique_idx]
        
        edge_index_np = np.vstack([final_edges[:, 0], final_edges[:, 1]])
        return get_sym_laplacian(edge_index_np, final_weights, N)    
    else:
        return sp.eye(N).tocsr()
    
def get_sym_laplacian(
    edge_index: np.ndarray, edge_weights: np.ndarray, num_nodes: int
) -> sp.csr_matrix:
    """Compute normalized symmetric graph Laplacian."""
    adj = sp.coo_matrix((edge_weights, (edge_index[0], edge_index[1])), 
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T
    adj.setdiag(1.0)
    
    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, out=d_inv_sqrt, where=deg>0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return D_inv_sqrt @ adj @ D_inv_sqrt

def geo_subsample(
    coords: np.ndarray, 
    target_cells: int, 
    global_bounds: tuple[float, float, float, float], 
    global_n_bins: int
) -> np.ndarray:
    """Perform vectorized geometric subsampling on a universal grid."""
    if target_cells <= 0:
        return np.array([], dtype=int)
        
    n_cells_total = coords.shape[0]
    if n_cells_total <= target_cells:
        return np.arange(n_cells_total)

    xmin, xmax, ymin, ymax = global_bounds
        
    x_bins = np.linspace(xmin, xmax, global_n_bins + 1)
    y_bins = np.linspace(ymin, ymax, global_n_bins + 1)
    
    # Assign cells to the universal grids
    x_idx = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, global_n_bins - 1)
    y_idx = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, global_n_bins - 1)
    grid_ids = x_idx * global_n_bins + y_idx 

    # Instant vectorized shuffling inside spatial blocks
    noise = np.random.rand(n_cells_total)
    order = np.lexsort((noise, grid_ids))
    grid_ids_sorted = grid_ids[order]
    
    _, block_starts = np.unique(grid_ids_sorted, return_index=True)
    block_ends = np.append(block_starts[1:], n_cells_total)
    
    cells_per_grid = max(1, target_cells // len(block_starts))
    
    sampled_indices = []
    leftover_indices = []
    
    for start, end in zip(block_starts, block_ends):
        chunk = order[start:end]
        sampled_indices.append(chunk[:cells_per_grid])
        if len(chunk) > cells_per_grid:
            leftover_indices.append(chunk[cells_per_grid:])
            
    sampled_indices = np.concatenate(sampled_indices)
    
    # Fill any deficit using random leftovers to hit the exact target
    deficit = target_cells - len(sampled_indices)
    if deficit > 0 and leftover_indices:
        leftovers = np.concatenate(leftover_indices)
        recovery = np.random.choice(leftovers, size=min(deficit, len(leftovers)), replace=False)
        sampled_indices = np.concatenate([sampled_indices, recovery])
        
    return np.sort(np.random.choice(sampled_indices, target_cells, replace=False) if len(sampled_indices) > target_cells else sampled_indices)


def _get_sketch_clusters(
    adata_backed: ad.AnnData, present_genes: list[str], n_clusters: int
) -> np.ndarray:
    """Generate PCA and K-Means clusters while isolating RAM usage."""
    # 1. Pull matrix and immediately force CSR for speed
    X_expr = adata_backed[:, present_genes].X.copy()
    if sp.issparse(X_expr) and not isinstance(X_expr, sp.csr_matrix):
        X_expr = X_expr.tocsr()
        
    # 2. IN-PLACE Log1p Transformation (Saves 4+ GB)
    if sp.issparse(X_expr):
        X_expr.data = np.log1p(X_expr.data)
    else:
        np.log1p(X_expr, out=X_expr) 
        
    # 3. Normalize and completely destroy the raw matrix
    X_expr_norm = normalize(X_expr, norm="l2", axis=1)
    del X_expr 
    
    # 4. PCA and completely destroy the normalized matrix
    n_components = min(50, len(present_genes) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_pca = svd.fit_transform(X_expr_norm)
    del X_expr_norm 

    # 5. High-resolution clustering to isolate rare niches
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        batch_size=cfg.batch_size, 
        random_state=42, 
        n_init=3
    )
    cluster_labels = kmeans.fit_predict(X_pca)
    
    return cluster_labels


def geo_sketch(
    adata_backed: ad.AnnData, target_cells: int, common_genes: list[str]
) -> np.ndarray:
    """Perform transcriptome-stratified geometric sketching."""

    n_cells_total = adata_backed.shape[0]
    if n_cells_total <= target_cells:
        return np.arange(n_cells_total)

    cg_set = set(common_genes)
    present = [g for g in adata_backed.var_names if g in cg_set]
    coords = adata_backed.obsm['spatial']
    
    n_clusters = min(500, max(2, n_cells_total // 20))
    
    # --- 1. Isolate the massive RAM spike completely ---
    cluster_labels = _get_sketch_clusters(adata_backed, present, n_clusters)
    gc.collect() # Force Python to tell OS the RAM is free
    
    # --- 2. Calculate Quotas ---
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)

    quotas = np.zeros(n_clusters, dtype=int)
    
    rare_threshold = max(20, target_cells // n_clusters)
    remaining_target = target_cells
    
    for c in range(n_clusters):
        if 0 < cluster_sizes[c] <= rare_threshold:
            take = min(cluster_sizes[c], max(0, remaining_target))
            quotas[c] = take
            remaining_target -= take
            
    abundant_clusters = [c for c in range(n_clusters) if cluster_sizes[c] > rare_threshold]
    total_abundant_cells = sum(cluster_sizes[c] for c in abundant_clusters)
    
    if total_abundant_cells > 0 and remaining_target > 0:
        for c in abundant_clusters:
            quotas[c] = int(remaining_target * (cluster_sizes[c] / total_abundant_cells))
            
        deficit = remaining_target - sum(quotas[c] for c in abundant_clusters)
        if deficit > 0 and abundant_clusters:
            largest = abundant_clusters[np.argmax([cluster_sizes[c] for c in abundant_clusters])]
            quotas[largest] += deficit

    # --- 3. Execute Topology-Preserving Spatial Sampling on Universal Grid ---
    global_bounds = (coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max())
    global_n_bins = int(np.sqrt(target_cells))
    
    final_indices = []
    
    for c in range(n_clusters):
        if quotas[c] == 0:
            continue
            
        c_indices = np.where(cluster_labels == c)[0]
        
        if cluster_sizes[c] == quotas[c]:
            final_indices.append(c_indices)  # Rare Cell Override (100% Kept)
        else:
            c_coords = coords[c_indices]
            sampled_relative_idx = geo_subsample(
                c_coords, quotas[c], global_bounds, global_n_bins
            )
            final_indices.append(c_indices[sampled_relative_idx])
            
    final_idx_array = np.sort(np.concatenate(final_indices))
    return final_idx_array

def make_meta_batches(
    training_cache: list[dict[str, Any]], meta_batch_size: int
) -> list[list[dict[str, Any]]]:
    """Group spatial chunks into highly diverse patient-stratified meta-batches."""
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
        
        # Pop one diverse chunk from each selected patient
        for p in selected:
            current_meta.append(patient_bins[p].pop())
            if not patient_bins[p]:
                active_patients.remove(p)
                
        # If we couldn't fill a meta batch, pad with whatever is left
        while len(current_meta) < meta_batch_size and active_patients:
            p = random.choice(active_patients)
            current_meta.append(patient_bins[p].pop())
            if not patient_bins[p]:
                active_patients.remove(p)
                
        meta_batches.append(current_meta)
        
    return meta_batches

def pt_to_scipy_csr(data_obj: Any, attr_name: str) -> sp.csr_matrix:
    """Convert PyTorch COO to SciPy CSR and instantly free the tensor."""
    pt_coo = getattr(data_obj, attr_name)
    if not pt_coo.is_coalesced():
        pt_coo = pt_coo.coalesce()
        
    row = pt_coo.indices()[0].numpy()
    col = pt_coo.indices()[1].numpy()
    val = pt_coo.values().numpy()
    shape = tuple(pt_coo.shape)
    
    # 🚨 THE FIX: Surgically destroy the PyTorch tensor from the parent 
    # BEFORE SciPy allocates its compressed arrays.
    delattr(data_obj, attr_name)
    del pt_coo
    
    mat = sp.csr_matrix((val, (row, col)), shape=shape)
    
    # Free the raw numpy arrays immediately (SciPy makes internal CSR copies)
    del row, col, val
    
    return mat


class SpatialBatcher:
    """OOM-Proof graph sampler utilizing continuous physical blocks."""

    def __init__(
        self, 
        X: sp.csr_matrix, 
        adj: sp.csr_matrix, 
        coords: np.ndarray, 
        train_mask: np.ndarray, 
        val_mask: np.ndarray, 
        batch_size: int | None = None, 
        k_hops: int | None = None, 
        shuffle: bool = True
    ):
        self.X = X
        self.adj = adj
        self.coords = coords
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.k_hops = k_hops if k_hops is not None else cfg.k_hops
        
        active_batch_size = batch_size if batch_size is not None else cfg.batch_size
        self.n_cells = X.shape[0]
        
        # Sort cells into physical spatial grids
        n_bins = int(np.ceil(np.sqrt(self.n_cells / active_batch_size)))
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        
        # Prevent division by zero if 1D structure
        if xmax == xmin: xmax += 1e-9
        if ymax == ymin: ymax += 1e-9
            
        x_bins = np.linspace(xmin, xmax, n_bins + 1)
        y_bins = np.linspace(ymin, ymax, n_bins + 1)
        
        x_idx = np.digitize(coords[:, 0], x_bins)
        y_idx = np.digitize(coords[:, 1], y_bins)
        grid_ids = x_idx * (n_bins + 1) + y_idx
        
        self.indices = np.argsort(grid_ids)
        self.chunks = [self.indices[i:i + active_batch_size] for i in range(0, self.n_cells, active_batch_size)]
        
        if shuffle:
            np.random.shuffle(self.chunks)
            
    def __len__(self) -> int:
        return len(self.chunks)
        
    def get_chunk(self, chunk_idx: int) -> dict[str, Any]:
        """Dynamically expands the K-Hop halo ONLY when requested. OOM Proof."""
        core_idx = self.chunks[chunk_idx]
        active_mask = np.zeros(self.n_cells, dtype=np.float32)
        active_mask[core_idx] = 1.0
        
        for _ in range(self.k_hops):
            active_mask = (self.adj.dot(active_mask) > 0).astype(np.float32)
        
        subgraph_nodes = np.where(active_mask > 0)[0]
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(subgraph_nodes)}
        local_core_idx = np.array([global_to_local[idx] for idx in core_idx])
        
        return {
            'x': self.X[subgraph_nodes],
            'adj': self.adj[subgraph_nodes, :][:, subgraph_nodes],
            'coords': self.coords[subgraph_nodes], 
            'local_core_idx': local_core_idx,
            'orig_core_idx': core_idx, 
            'train_mask': self.train_mask[subgraph_nodes],
            'val_mask': self.val_mask[subgraph_nodes],
            'patient_name': getattr(self, 'patient_name', 'Unknown')
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self.chunks)):
            yield self.get_chunk(i)

def _load_h5ad(
    f: Path | str, clean_whitelist: set[str]
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Isolate AnnData loading, filtering, and normalization."""
    adata_backed = sc.read_h5ad(f, backed="r")
    n_cells_total = adata_backed.shape[0]
    FEATURE_CAP = cfg.feature_cap 
    
    if n_cells_total > FEATURE_CAP:
        idx = np.sort(np.random.choice(n_cells_total, FEATURE_CAP, replace=False))
        adata_mem = adata_backed[idx].to_memory()
    else:
        adata_mem = adata_backed.to_memory()
        
    
    upper_var_names = np.array([str(g).upper() for g in adata_mem.var_names])
    valid_genes_mask = np.array([g in clean_whitelist for g in upper_var_names])
    valid_genes = upper_var_names[valid_genes_mask]
    
    X_csc = adata_mem.X.tocsc()
    X_sparse = X_csc[:, valid_genes_mask].tocsr()
    coords = adata_mem.obsm["spatial"]
    
    del adata_backed, adata_mem, X_csc; gc.collect()
    
    cell_counts = np.array(X_sparse.sum(axis=1)).flatten()
    gene_counts = np.array((X_sparse > 0).sum(axis=0)).flatten()

    upper_limit = np.percentile(cell_counts, 99) 
    valid_cells_mask = (cell_counts >= 10) & (cell_counts <= upper_limit)
    min_cells = int(X_sparse.shape[0] * 0.001)
    max_cells = int(X_sparse.shape[0] * 0.95)
    final_gene_mask = (gene_counts >= min_cells) & (gene_counts <= max_cells)
    
    X_sparse = X_sparse[valid_cells_mask, :][:, final_gene_mask]
    coords = coords[valid_cells_mask]
    final_genes = valid_genes[final_gene_mask]
    
    # Apply normalization and log1p strictly IN-PLACE on the underlying C-arrays
    cell_sums = np.array(X_sparse.sum(axis=1)).flatten()
    cell_sums[cell_sums == 0] = 1.0 
    scaling_factors = 1e4 / cell_sums
    
    X_sparse = sp.diags(scaling_factors) @ X_sparse
    X_sparse.data = np.log1p(X_sparse.data)
    
    return X_sparse, coords, final_genes
    
def _calc_hvg(
    X_sparse: sp.csr_matrix, final_genes: np.ndarray
) -> dict[str, float]:
    """Wrap Scanpy HVG to isolate AnnData memory footprint."""
    adata_tmp = ad.AnnData(X=X_sparse)
    adata_tmp.var_names = final_genes
    
    sc.pp.highly_variable_genes(adata_tmp, flavor="seurat")
    hvg_dict = {
        g: val for g, val in zip(adata_tmp.var_names, adata_tmp.var["dispersions_norm"]) 
        if not np.isnan(val)
    }
    
    del adata_tmp; gc.collect()
    return hvg_dict

def _calc_moran_dict(
    X_sparse: sp.csr_matrix, coords: np.ndarray, final_genes: np.ndarray
) -> dict[str, float]:
    """Isolate KDTree and Spatial edge building for Moran's I."""
    tree = cKDTree(coords)
    _, knn_idx = tree.query(coords, k=cfg.moran_k)
    
    src = np.repeat(np.arange(coords.shape[0]), 6)
    dst = knn_idx[:, 1:].flatten()
    edges = np.unique(np.sort(np.vstack([src, dst]).T, axis=1), axis=0)
    
    moran_scores = calc_moran(X_sparse, edges.T, coords.shape[0])
    moran_dict = {
        g: val for g, val in zip(final_genes, moran_scores) 
        if not np.isnan(val)
    }
    
    del tree, knn_idx, src, dst, edges; gc.collect()
    return moran_dict


def _process_h5ad(
    f: Path, clean_whitelist: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Process single H5AD file for spatial feature ranking."""
    try:
        X_sparse, coords, final_genes = _load_h5ad(f, clean_whitelist)
        hvg_dict = _calc_hvg(X_sparse, final_genes)
        moran_dict = _calc_moran_dict(X_sparse, coords, final_genes)
        
        # Final cleanup before passing back to Joblib
        del X_sparse, coords, final_genes
        gc.collect()
        
        return hvg_dict, moran_dict
        
    except Exception as e:
        print(f"  ↳ [!] Skipping {f.stem} for feature selection: {e}")
        return {}, {}
      
def get_consensus_genes(
    h5ad_files: list[Path], clean_whitelist: set[str], top_n: int | None = None
) -> list[str]:
    top_n = top_n if top_n is not None else cfg.top_n_genes
    print("  ↳ Extracting Moran/HVG spatial features...")
    
    n_workers = min(4, multiprocessing.cpu_count())
    
    with joblib.Parallel(n_jobs=n_workers, backend="loky") as parallel:
        results = parallel(
            delayed(_process_h5ad)(f, clean_whitelist) 
            for f in tqdm(h5ad_files, desc="Extracting features", leave=False)
        )
    
    global_hvg: defaultdict[str, float] = defaultdict(float)
    global_moran: defaultdict[str, float] = defaultdict(float)
    for h_dict, m_dict in results:
        for g, val in h_dict.items(): global_hvg[g] += val
        for g, val in m_dict.items(): global_moran[g] += val
    
    # Get all unique genes that passed the whitelist and appeared in the data
    valid_genes = list(set(global_hvg.keys()) | set(global_moran.keys()))
    
    # If the whitelist already returned fewer genes than the cap, return them all
    if len(valid_genes) <= top_n:
        print(f"  ↳ Whitelisted Feature Space: {len(valid_genes)} genes ")
        return valid_genes
        
    # --- RANK AGGREGATION TO FIND TOP SPATIAL PREDICTORS ---
    # Convert to pandas Series for vectorized ranking, filling missing genes with 0
    hvg_series = pd.Series(global_hvg).reindex(valid_genes, fill_value=0)
    moran_series = pd.Series(global_moran).reindex(valid_genes, fill_value=0)
    
    # Rank descending (Rank 1 = highest metric)
    hvg_ranks = hvg_series.rank(ascending=False)
    moran_ranks = moran_series.rank(ascending=False)
    
    # Combined score: lower sum is better (e.g. Rank 1 + Rank 1 = 2)
    combined_score = hvg_ranks + moran_ranks
    
    # Sort and strictly cap to top_n
    top_genes = combined_score.sort_values(ascending=True).head(top_n).index.tolist()
    print(f"  ↳ Filtered {len(valid_genes)} candidate genes to Top {len(top_genes)} via joint HVG/Moran ranking")
    return [str(g) for g in top_genes]

def to_pt_sparse(mat: sp.csr_matrix | sp.coo_matrix) -> torch.Tensor:
    """Safely convert SciPy sparse to PyTorch COO."""
    mat_coo = mat.tocoo()
    
    # Use copy=False to prevent NumPy from unnecessarily duplicating the arrays
    row = mat_coo.row.astype(np.int64, copy=False)
    col = mat_coo.col.astype(np.int64, copy=False)
    data = mat_coo.data.astype(np.float32, copy=False)
    
    indices = torch.from_numpy(np.vstack((row, col)))
    values = torch.from_numpy(data)
    shape = mat_coo.shape
    
    # Annihilate intermediate SciPy structures BEFORE PyTorch allocation
    del mat_coo, row, col, data
    
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def _remap_and_norm(
    f: Path, common_genes: list[str]
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Consolidate loading, sketching, and mapping into a single GC-safe scope."""

    # 1. Load and Sketch (Auto-Uppercase)
    adata_backed = sc.read_h5ad(f, backed='r')
    upper_var_names = [str(g).upper() for g in adata_backed.var_names]
    gene_to_idx = {g: i for i, g in enumerate(upper_var_names)}
    present_genes = [g for g in common_genes if g in gene_to_idx]
    col_indices = [gene_to_idx[g] for g in present_genes]
    
    adata_mem = adata_backed[:, col_indices].to_memory()
    X_raw_sub = adata_mem.X.tocsr()
    coords = adata_mem.obsm['spatial'].copy()
    
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
        n_cells = len(idx)

    target_col_map = np.array([common_genes.index(g) for g in present_genes])
    X_raw_coo = X_raw_sub.tocoo()
    new_cols = target_col_map[X_raw_coo.col]
    
    X_raw_sparse = sp.csr_matrix((X_raw_coo.data, (X_raw_coo.row, new_cols)), 
                                 shape=(n_cells, len(common_genes)), dtype=np.float32)
                                 
    del X_raw_sub, X_raw_coo 
    gc.collect()
    
    X_raw_sparse.data = np.clip(X_raw_sparse.data, 0, None)
    
    # 3. Create Normalized Matrix
    cell_sums = np.array(X_raw_sparse.sum(axis=1)).flatten()
    cell_sums[cell_sums == 0] = 1.0 
    scaling_factors = 1e4 / cell_sums
    
    X_in_sparse = sp.diags(scaling_factors) @ X_raw_sparse 
    X_in_sparse.data = np.log1p(X_in_sparse.data)
    
    return X_raw_sparse, X_in_sparse, coords

def build_pt_graph(f: Path, common_genes: list[str]) -> Path | None:
    """Master orchestrator for PyTorch Geometric graph building."""
    try:
        X_raw_sparse, X_in_sparse, coords = _remap_and_norm(f, common_genes)
        adj_sym = build_spatial_graph(coords, X_in_sparse)
        
        n_cells = X_raw_sparse.shape[0]
        train_mask = torch.zeros(n_cells, dtype=torch.bool)
        val_mask = torch.zeros(n_cells, dtype=torch.bool)
        
        perm = np.random.permutation(n_cells)
        train_cutoff = int(n_cells * 0.85)
        train_mask[perm[:train_cutoff]] = True
        val_mask[perm[train_cutoff:]] = True
        
        data = Data(train_mask=train_mask, val_mask=val_mask)
        data.pos = torch.from_numpy(coords).float()
        data.patient_name = f.stem
        
        # Pack PyTorch matrices safely using the scope-isolated helper
        data.x_in = to_pt_sparse(X_in_sparse)
        data.y_raw = to_pt_sparse(X_raw_sparse)
        
        # Clear SciPy versions from RAM instantly
        del X_in_sparse, X_raw_sparse
        gc.collect()
        
        adj_sym_coo = adj_sym.tocoo()
        data.edge_index = torch.from_numpy(np.vstack((adj_sym_coo.row, adj_sym_coo.col)).astype(np.int32))
        data.edge_attr = torch.from_numpy(adj_sym_coo.data.astype(np.float32))

        out_dir = paths.make_dirs(cfg.suffix)["graphs"]
        out_path = out_dir / f"{f.stem}_graph.pt"
        torch.save(data, out_path)
        
        del data, adj_sym, adj_sym_coo; gc.collect()
        return out_path
    except Exception as e: 
        print(f"  ↳ [!] Graph construction failed for {f.stem}: {e}")
        return None
        
def pad_mps_shapes(
    x: torch.Tensor, 
    src: torch.Tensor, 
    dst: torch.Tensor, 
    weights: torch.Tensor, 
    node_bucket: int | None = None, 
    edge_bucket: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate optimal buckets and pad graph shapes for PyTorch MPS."""
    if node_bucket is None or edge_bucket is None:
        if cfg.k_hops <= 1:
            n_max = cfg.batch_size * 2
        elif cfg.k_hops == 2:
            n_max = cfg.batch_size * 4
        else:
            n_max = cfg.batch_size * 8
            
        e_max = n_max * 10
        node_bucket = max(1024, int(round((n_max * 0.20) / 1024) * 1024))
        edge_bucket = max(4096, int(round((e_max * 0.20) / 4096) * 4096))

    N = x.size(0)
    E = src.size(0)

    # Calculate padded sizes (rounding up to nearest bucket)
    N_pad = ((N + node_bucket - 1) // node_bucket) * node_bucket
    E_pad = ((E + edge_bucket - 1) // edge_bucket) * edge_bucket

    # If we need to pad edges, we MUST guarantee at least one dummy node exists 
    # for the dummy edges to safely point to.
    if E_pad > E and N_pad == N:
        N_pad += node_bucket 

    # 1. Pad Nodes (with zeros)
    if N_pad > N:
        x_dummy = torch.zeros(N_pad - N, x.size(1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, x_dummy], dim=0)

    # 2. Pad Edges (Dummy edges pointing from Dummy Node -> Dummy Node with weight 0)
    if E_pad > E:
        # N is the index of the FIRST dummy node. We route all fake math through it.
        dummy_idx = torch.full((E_pad - E,), N, dtype=src.dtype, device=src.device)
        dummy_w = torch.zeros(E_pad - E, dtype=weights.dtype, device=weights.device)

        src = torch.cat([src, dummy_idx], dim=0)
        dst = torch.cat([dst, dummy_idx], dim=0)
        weights = torch.cat([weights, dummy_w], dim=0)

    return x, src, dst, weights

def build_graph_safe(f: Path, c_genes: list[str]) -> Path | None:
    """Wrap graph builder with safety checks to avoid identical retrains."""
    clean_stem = f.stem.replace("_graph", "")
    out_dir = paths.make_dirs(cfg.suffix)["graphs"]
    expected_out = out_dir / f"{clean_stem}_graph.pt"
    
    if expected_out.exists() and not cfg.force_retrain:
        try:
            chk = torch.load(expected_out, map_location="cpu", weights_only=False)
            valid_genes = hasattr(chk, "x_in") and chk.x_in.shape[1] == len(c_genes)
            
            n_cells = chk.x_in.shape[0]
            valid_cells = True
            if cfg.max_cells_per_sample is not None:
                if n_cells > cfg.max_cells_per_sample * 1.5:
                    valid_cells = False
                    
            if valid_genes and valid_cells:
                return expected_out
        except Exception:
            pass
        
        expected_out.unlink(missing_ok=True)
        
    return build_pt_graph(f, c_genes)


