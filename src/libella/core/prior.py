"""Biological prior extraction and dictionary learning for spatial metaprograms."""

import ast
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .config import NOISE_REGEX, cfg, paths
from .data import pt_to_scipy_csr
from .utils import get_device


def _parse_nouns(
    genes_path: Path, sig_csv_path: Path
) -> tuple[np.ndarray, list[str], list[str]]:
    """Parse and prune biological signature CSV into a clean matrix."""
    with open(genes_path, "r") as f:
        target_genes: list[str] = json.load(f)

    gene2idx = {g: i for i, g in enumerate(target_genes)}
    n_target_genes = len(target_genes)

    sig_df = pd.read_csv(sig_csv_path)
    n_lineages = len(sig_df)
    lineage_names = sig_df["Cell_type"].tolist() if "Cell_type" in sig_df.columns else [f"Lineage_{i}" for i in range(n_lineages)]

    atlas_matrix = np.zeros((n_lineages, n_target_genes), dtype=np.float32)

    for i, (_, row) in enumerate(sig_df.iterrows()):
        raw_genes_str = row["Genes"]
        if pd.isna(raw_genes_str):
            continue

        try:
            genes_list = ast.literal_eval(raw_genes_str)
        except (ValueError, SyntaxError):
            genes_list = [g.strip() for g in str(raw_genes_str).split(",")]

        for g in genes_list:
            g_clean = str(g).strip()
            if NOISE_REGEX.match(g_clean):
                continue
            if g_clean in gene2idx:
                atlas_matrix[i, gene2idx[g_clean]] = 1.0

    return atlas_matrix, target_genes, lineage_names


def _compress_nouns(
    atlas_matrix: np.ndarray, 
    lineage_names: list[str], 
    target_genes: list[str], 
    n_clusters: int = 30
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compress signature hierarchy using Agglomerative Clustering."""
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters, metric="jaccard", linkage="average"
    )
    cluster_labels = clusterer.fit_predict(atlas_matrix)

    n_genes = atlas_matrix.shape[1]
    nouns = np.zeros((n_clusters, n_genes), dtype=np.float32)
    noun_reports = []

    for cluster_id in range(n_clusters):
        child_indices = np.where(cluster_labels == cluster_id)[0]
        child_matrix = atlas_matrix[child_indices]
        nouns[cluster_id] = np.max(child_matrix, axis=0)

        gene_freqs = np.sum(child_matrix, axis=0)
        top5_gene_idx = np.argsort(-gene_freqs)[:5]
        top5_genes = [target_genes[idx] for idx in top5_gene_idx if gene_freqs[idx] > 0]

        if len(top5_genes) < 5:
            noun_nonzero = np.where(nouns[cluster_id] > 0)[0]
            for idx in noun_nonzero:
                g = target_genes[idx]
                if g not in top5_genes:
                    top5_genes.append(g)
                if len(top5_genes) == 5:
                    break

        child_state_names = [lineage_names[i] for i in child_indices]
        noun_reports.append({
            "cluster_id": cluster_id,
            "n_child_states": len(child_indices),
            "child_states": child_state_names,
            "top_5_genes": top5_genes
        })

    return nouns, noun_reports


def _get_raw_adjs(graph_paths: list[Path]) -> np.ndarray:
    """Extract raw spatial adjacency dictionaries via MiniBatchDictionaryLearning."""
    def process_graph(path: Path) -> sp.csr_matrix:
        data = torch.load(path, map_location="cpu", weights_only=False)

        if hasattr(data, "y_raw") and getattr(data.y_raw, "is_sparse", False):
            y_local = pt_to_scipy_csr(data, "y_raw")
        else:
            y_local = data.y_raw.copy()
        del data

        n_cells = y_local.shape[0]
        if y_local.nnz > 0:
            cap = np.percentile(y_local.data, 95)
            np.clip(y_local.data, 0, cap, out=y_local.data)

        tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
        y_processed = tfidf.fit_transform(y_local).astype(np.float32, copy=False)

        max_priors_cells = cfg.prior_cells_per_sample

        if n_cells > max_priors_cells:
            idx = np.random.RandomState(42).choice(
                n_cells, max_priors_cells, replace=False
            )
            y_processed = y_processed[idx]

        return sp.csr_matrix(y_processed)

    reservoir_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(
            executor.map(process_graph, graph_paths),
            total=len(graph_paths),
            desc="Extracting Spatial Cohort Matrices",
            leave=False,
        ):
            reservoir_list.append(res)

    union_matrix_scaled = sp.vstack(reservoir_list)
    del reservoir_list
    gc.collect()

    n_total_cells = union_matrix_scaled.shape[0]
    if n_total_cells > 200000:
        X_csc = union_matrix_scaled.tocsc()
        top_indices = set()
        cells_per_gene = 2000

        for i in range(X_csc.shape[1]):
            start_idx, end_idx = X_csc.indptr[i], X_csc.indptr[i + 1]
            data = X_csc.data[start_idx:end_idx]
            cell_indices = X_csc.indices[start_idx:end_idx]

            if len(data) > cells_per_gene:
                top_idx = np.argpartition(data, -cells_per_gene)[-cells_per_gene:]
                top_indices.update(cell_indices[top_idx])
            else:
                top_indices.update(cell_indices)

        selected_cells = np.array(list(top_indices))
        union_matrix_scaled = union_matrix_scaled[selected_cells]
        del X_csc, selected_cells, top_indices
        gc.collect()

    X_dense = union_matrix_scaled.toarray()
    del union_matrix_scaled
    gc.collect()

    dict_learner = MiniBatchDictionaryLearning(
        n_components=100, alpha=2.0, random_state=42, max_iter=100
    )
    dict_learner.fit(X_dense)

    raw_spatial_topics = np.abs(dict_learner.components_)
    del X_dense
    gc.collect()

    return raw_spatial_topics


def _apply_sieve(
    raw_spatial_topics: np.ndarray, nouns: np.ndarray, target_genes: list[str]
) -> tuple[np.ndarray, dict[str, int], list[dict[str, Any]]]:
    """Sieve raw topics to extract orthogonal spatial adjectives."""
    sim_matrix = cosine_similarity(raw_spatial_topics, nouns)
    candidate_adjectives = []
    candidate_reports = []

    n_mapped = 0
    n_chimeras = 0
    n_other_discarded = 0

    for i in range(raw_spatial_topics.shape[0]):
        topic_sims = sim_matrix[i]
        max_sim = np.max(topic_sims)
        num_sim_above_15 = np.sum(topic_sims > 0.15)

        if max_sim > 0.10:
            n_mapped += 1
            continue

        if num_sim_above_15 >= 2:
            n_chimeras += 1
            continue

        if max_sim < 0.10:
            topic_weights = raw_spatial_topics[i]
            bin_topic = np.zeros_like(topic_weights, dtype=np.float32)

            top35_idx = np.argpartition(topic_weights, -35)[-35:]
            bin_topic[top35_idx] = 1.0

            candidate_adjectives.append(bin_topic)

            top10_idx = np.argsort(-topic_weights)[:10]
            top10_genes = [target_genes[idx] for idx in top10_idx]
            candidate_reports.append({
                "topic_id": i,
                "top_10_genes": top10_genes,
                "top10_set": set(top10_genes)
            })
        else:
            n_other_discarded += 1

    if len(candidate_adjectives) > 0:
        adj_matrix = np.array(candidate_adjectives, dtype=np.float32)
        n_candidates = adj_matrix.shape[0]
        keep_mask = np.ones(n_candidates, dtype=bool)

        for i in range(n_candidates):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_candidates):
                if not keep_mask[j]:
                    continue

                # 1. Check overlap on top 35 binarized genes
                overlap_35 = np.dot(adj_matrix[i], adj_matrix[j])  # Number of shared genes out of 35

                # 2. Check overlap on top 10 marker genes
                overlap_10 = len(candidate_reports[i]["top10_set"].intersection(candidate_reports[j]["top10_set"]))

                # Purge if >= 50% top-35 overlap (>=18 genes) OR >= 4 top-10 genes overlap
                if overlap_35 >= 10 or overlap_10 >= 2:
                    keep_mask[j] = False

        final_adjectives = adj_matrix[keep_mask]
        final_reports = [rep for rep, keep in zip(candidate_reports, keep_mask) if keep]
        n_redundant = int(np.sum(~keep_mask))
    else:
        final_adjectives = np.empty((0, nouns.shape[1]), dtype=np.float32)
        final_reports = []
        n_redundant = 0

    sieve_stats = {
        "n_mapped": n_mapped,
        "n_chimeras": n_chimeras,
        "n_other_discarded": n_other_discarded,
        "n_redundant": n_redundant,
        "n_kept": final_adjectives.shape[0]
    }

    return final_adjectives, sieve_stats, final_reports


def _write_report(
    noun_reports: list[dict[str, Any]], 
    sieve_stats: dict[str, int], 
    adjective_reports: list[dict[str, Any]], 
    report_path: Path
) -> None:
    """Write extraction logs and statistics to a report text file."""
    lines = ["[NOUNS]"]
    for nr in noun_reports:
        top_g = ",".join(nr["top_5_genes"])
        states = ",".join(nr["child_states"])
        lines.append(f"N{nr['cluster_id']:02d}|n={nr['n_child_states']:02d}|top=[{top_g}]|states=[{states}]")

    lines.append("\n[SIEVE_STATS]")
    lines.append(
        f"mapped={sieve_stats['n_mapped']}|chimera={sieve_stats['n_chimeras']}|"
        f"redundant={sieve_stats['n_redundant']}|intermediate={sieve_stats['n_other_discarded']}|"
        f"kept={sieve_stats['n_kept']}"
    )

    lines.append("\n[ADJECTIVES]")
    for idx, ar in enumerate(adjective_reports):
        top_g = ",".join(ar["top_10_genes"])
        lines.append(f"A{idx:02d}|raw_topic={ar['topic_id']:02d}|top10=[{top_g}]")

    report_path.write_text("\n".join(lines))


def get_priors(
    graph_paths: list[Path]
) -> tuple[np.ndarray | None, int | None, dict[str, Any] | None]:
    """Load or extract biological priors for the Libella model."""
    checkpoint = None
    optimal_k = None
    init_components = None
    
    out_dirs = paths.setup_output_dirs(cfg.suffix)
    checkpoint_path = out_dirs["checkpoint"]
    cnmf_priors_path = out_dirs["cnmf_priors"]
    genes_path = out_dirs["genes"]
    out_dir = out_dirs["out"]

    if checkpoint_path.exists() and not cfg.force_retrain:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=get_device(), weights_only=False)
            optimal_k = checkpoint["model_state_dict"]["topic_gene_logits"].shape[0]
        except Exception:
            checkpoint = None

    if cnmf_priors_path.exists():
        try:
            init_components = joblib.load(cnmf_priors_path)
            optimal_k = init_components.shape[0]
            print(f"[✓] Loaded Priors (K={optimal_k})")
        except Exception:
            init_components = None

    if init_components is None:
        print("  ↳ Extracting Biological Priors...")

        atlas_matrix, target_genes, lineage_names = _parse_nouns(
            genes_path, paths.sig_csv
        )
        nouns, noun_reports = _compress_nouns(
            atlas_matrix, lineage_names, target_genes, n_clusters=30
        )

        raw_spatial_topics = _get_raw_adjs(graph_paths)

        orthogonal_adjectives, sieve_stats, adjective_reports = _apply_sieve(
            raw_spatial_topics, nouns, target_genes
        )

        if len(orthogonal_adjectives) > 0:
            init_components = np.vstack([nouns, orthogonal_adjectives])
        else:
            init_components = nouns

        optimal_k = init_components.shape[0]

        x_pelka = nouns.shape[0]
        y_dict = orthogonal_adjectives.shape[0]
        print(f"  ↳ {x_pelka} pelka lineages + {y_dict} dict addition prior is sealed.")

        report_path = out_dir / "prior.txt"
        _write_report(noun_reports, sieve_stats, adjective_reports, report_path)

        try:
            joblib.dump(init_components, cnmf_priors_path)
        except Exception:
            pass

    return init_components, optimal_k, checkpoint
    
