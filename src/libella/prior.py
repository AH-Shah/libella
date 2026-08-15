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
from tqdm import tqdm

from .config import NOISE_REGEX, cfg, paths
from .data import pt_to_scipy_csr
from .utils import get_device


def _parse_nouns(
    genes_path: Path, sig_csv_path: Path
) -> tuple[np.ndarray, list[str], list[str]]:
    """Parse and prune biological signature CSV into a clean binary matrix."""
    with open(genes_path, "r") as f:
        genes_data = json.load(f)

    if isinstance(genes_data, list):
        target_genes = [str(g) for g in genes_data]
    elif isinstance(genes_data, dict):
        target_genes = [
            str(g) for g in genes_data.get("genes", list(genes_data.values()))
        ]
    else:
        raise ValueError(f"Unsupported JSON format in {genes_path}")

    gene2idx = {g: i for i, g in enumerate(target_genes)}
    n_target_genes = len(target_genes)

    if not sig_csv_path.exists():
        print(
            f"[!] Warning: Signature CSV not found at {sig_csv_path}. Proceeding with empty atlas."
        )
        return np.empty((0, n_target_genes), dtype=np.float32), target_genes, []

    sig_df = pd.read_csv(sig_csv_path)
    n_lineages = len(sig_df)
    lineage_names = (
        sig_df["Cell_type"].tolist()
        if "Cell_type" in sig_df.columns
        else [f"Lineage_{i}" for i in range(n_lineages)]
    )

    atlas_matrix = np.zeros((n_lineages, n_target_genes), dtype=np.float32)

    for i, (_, row) in enumerate(sig_df.iterrows()):
        raw_genes_str = row.get("Genes", "")
        if pd.isna(raw_genes_str) or not raw_genes_str:
            continue

        try:
            genes_list = ast.literal_eval(str(raw_genes_str))
        except (ValueError, SyntaxError, TypeError):
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
    n_clusters: int = 25,
    min_genes: int = 15,
    max_genes: int = 100,
    consensus_freq: float = 0.30,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compress signature hierarchy using Agglomerative Clustering with consensus trimming."""
    n_lineages, n_genes = atlas_matrix.shape
    if n_lineages == 0:
        return np.empty((0, n_genes), dtype=np.float32), []

    actual_k = min(n_clusters, n_lineages)
    if actual_k <= 1:
        cluster_labels = np.zeros(n_lineages, dtype=int)
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=actual_k, metric="jaccard", linkage="average"
        )
        cluster_labels = clusterer.fit_predict(atlas_matrix)

    valid_nouns = []
    noun_reports = []

    for cluster_id in range(actual_k):
        child_indices = np.where(cluster_labels == cluster_id)[0]
        if len(child_indices) == 0:
            continue

        child_matrix = atlas_matrix[child_indices]
        n_children = len(child_indices)
        gene_frequencies = np.sum(child_matrix, axis=0)
        raw_union_count = int(np.sum(gene_frequencies > 0))

        # 1. Consensus selection
        if n_children > 1:
            consensus_mask = (gene_frequencies / n_children) >= consensus_freq
            active_indices = np.where(consensus_mask)[0]
        else:
            active_indices = np.where(gene_frequencies > 0)[0]

        # 2. Strict size bounding
        if len(active_indices) > max_genes:
            top_k_idx = np.argsort(-gene_frequencies[active_indices])[:max_genes]
            active_indices = active_indices[top_k_idx]
        elif len(active_indices) < min_genes:
            if raw_union_count >= min_genes:
                active_indices = np.argsort(-gene_frequencies)[:min_genes]
            else:
                # Prune fragile micro-clusters (< min_genes total)
                continue

        noun_vec = np.zeros(n_genes, dtype=np.float32)
        noun_vec[active_indices] = 1.0
        valid_nouns.append(noun_vec)

        top5_gene_idx = np.argsort(-gene_frequencies[active_indices])[:5]
        top5_genes = [target_genes[active_indices[idx]] for idx in top5_gene_idx]

        child_state_names = [lineage_names[i] for i in child_indices]
        noun_reports.append(
            {
                "cluster_id": len(valid_nouns) - 1,
                "n_child_states": n_children,
                "child_states": child_state_names,
                "top_5_genes": top5_genes,
                "n_genes": len(active_indices),
            }
        )

    if valid_nouns:
        nouns = np.array(valid_nouns, dtype=np.float32)
    else:
        nouns = np.empty((0, n_genes), dtype=np.float32)

    return nouns, noun_reports


def _get_raw_adjs(graph_paths: list[Path]) -> np.ndarray:
    """Extract raw spatial adjacency dictionaries via MiniBatchDictionaryLearning."""
    def process_graph(path: Path) -> sp.csr_matrix:
        data = torch.load(path, map_location="cpu", weights_only=False)

        if hasattr(data, "y_raw") and getattr(data.y_raw, "is_sparse", False):
            y_local = pt_to_scipy_csr(data, "y_raw")
        elif isinstance(getattr(data, "y_raw", None), torch.Tensor):
            y_local = sp.csr_matrix(data.y_raw.detach().cpu().numpy())
        elif isinstance(getattr(data, "y_raw", None), np.ndarray):
            y_local = sp.csr_matrix(data.y_raw)
        elif sp.issparse(getattr(data, "y_raw", None)):
            y_local = data.y_raw.tocsr()
        else:
            raise TypeError(f"Unsupported data type for y_raw in {path}")
        del data

        if y_local.nnz > 0:
            cap = np.percentile(y_local.data, 95)
            np.clip(y_local.data, 0, cap, out=y_local.data)

        tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
        y_processed = tfidf.fit_transform(y_local).astype(np.float32, copy=False)

        max_priors_cells = getattr(cfg, "prior_cells_per_sample", 5000)
        n_cells = y_processed.shape[0]

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

    n_dict_comp = getattr(cfg, "n_dict_components", 12)
    dict_learner = MiniBatchDictionaryLearning(
        n_components=n_dict_comp,
        alpha=2.0,
        random_state=42,
        max_iter=150,
        batch_size=512,
    )
    dict_learner.fit(X_dense)

    raw_spatial_topics = np.abs(dict_learner.components_)
    del X_dense
    gc.collect()

    return raw_spatial_topics


def _apply_sieve(
    raw_spatial_topics: np.ndarray,
    nouns: np.ndarray,
    target_genes: list[str],
    top_k_genes: int = 35,
    max_noun_overlap: float = 0.35,
    max_internal_overlap: float = 0.30,
    max_total_adjectives: int = 8,
) -> tuple[np.ndarray, dict[str, int], list[dict[str, Any]]]:
    """Sieve raw topics to extract orthogonal spatial adjectives using overlap metrics."""
    n_genes = nouns.shape[1] if nouns.shape[0] > 0 else len(target_genes)
    noun_sets = [set(np.where(nouns[i] > 0)[0]) for i in range(nouns.shape[0])]
    candidate_adjectives = []
    n_mapped = 0

    # 1. Filter dictionary candidates against Lineage Nouns
    for i in range(raw_spatial_topics.shape[0]):
        weights = raw_spatial_topics[i]
        top_k = min(top_k_genes, len(weights))
        top_indices = set(np.argpartition(weights, -top_k)[-top_k:])

        # Overlap coefficient: |A ∩ N| / min(|A|, |N|)
        if noun_sets:
            overlaps = [
                len(top_indices.intersection(n_s)) / min(len(top_indices), len(n_s))
                for n_s in noun_sets
                if min(len(top_indices), len(n_s)) > 0
            ]
            max_overlap = max(overlaps) if overlaps else 0.0
        else:
            max_overlap = 0.0

        if max_overlap >= max_noun_overlap:
            n_mapped += 1
            continue

        bin_topic = np.zeros(n_genes, dtype=np.float32)
        bin_topic[list(top_indices)] = 1.0

        top10_idx = np.argsort(-weights)[:10]
        top10_genes = [
            target_genes[idx] for idx in top10_idx if idx < len(target_genes)
        ]

        candidate_adjectives.append(
            {
                "bin_topic": bin_topic,
                "top_indices": top_indices,
                "topic_id": i,
                "top_10_genes": top10_genes,
                "energy": float(np.sum(weights)),
            }
        )

    # Sort surviving candidates by total activation energy descending
    candidate_adjectives.sort(key=lambda x: x["energy"], reverse=True)

    # 2. Internal orthogonal deduplication with budget cap
    accepted_adjectives = []
    accepted_sets = []
    final_reports = []
    n_redundant = 0
    n_other_discarded = 0

    for cand in candidate_adjectives:
        if len(accepted_adjectives) >= max_total_adjectives:
            n_other_discarded += 1
            continue

        gene_set = cand["top_indices"]
        if not accepted_sets:
            is_redundant = False
        else:
            max_int_ov = max(
                len(gene_set.intersection(s)) / min(len(gene_set), len(s))
                for s in accepted_sets
                if min(len(gene_set), len(s)) > 0
            )
            is_redundant = max_int_ov >= max_internal_overlap

        if not is_redundant:
            accepted_adjectives.append(cand["bin_topic"])
            accepted_sets.append(gene_set)
            final_reports.append(
                {
                    "topic_id": cand["topic_id"],
                    "top_10_genes": cand["top_10_genes"],
                }
            )
        else:
            n_redundant += 1

    if accepted_adjectives:
        final_adjectives = np.array(accepted_adjectives, dtype=np.float32)
    else:
        final_adjectives = np.empty((0, n_genes), dtype=np.float32)

    sieve_stats = {
        "n_mapped": n_mapped,
        "n_chimeras": 0,
        "n_other_discarded": n_other_discarded,
        "n_redundant": n_redundant,
        "n_kept": final_adjectives.shape[0],
    }

    return final_adjectives, sieve_stats, final_reports


def _write_report(
    noun_reports: list[dict[str, Any]],
    sieve_stats: dict[str, int],
    adjective_reports: list[dict[str, Any]],
    report_path: Path,
) -> None:
    """Write extraction logs and statistics to a report text file."""
    lines = ["[NOUNS]"]
    for nr in noun_reports:
        top_g = ",".join(nr["top_5_genes"])
        states = ",".join(nr["child_states"])
        lines.append(
            f"N{nr['cluster_id']:02d}|n={nr['n_child_states']:02d}|top=[{top_g}]|states=[{states}]"
        )

    lines.append("\n[SIEVE_STATS]")
    lines.append(
        f"mapped={sieve_stats.get('n_mapped', 0)}|chimera={sieve_stats.get('n_chimeras', 0)}|"
        f"redundant={sieve_stats.get('n_redundant', 0)}|intermediate={sieve_stats.get('n_other_discarded', 0)}|"
        f"kept={sieve_stats.get('n_kept', 0)}"
    )

    lines.append("\n[ADJECTIVES]")
    for idx, ar in enumerate(adjective_reports):
        top_g = ",".join(ar["top_10_genes"])
        lines.append(f"A{idx:02d}|raw_topic={ar['topic_id']:02d}|top10=[{top_g}]")

    report_path.write_text("\n".join(lines))


def get_priors(
    graph_paths: list[Path],
) -> tuple[np.ndarray | None, int | None, dict[str, Any] | None]:
    """Load or extract biological priors for the Libella model."""
    checkpoint = None
    optimal_k = None
    init_components = None

    out_dirs = paths.make_dirs(cfg.suffix)
    checkpoint_path = out_dirs["checkpoint"]
    cnmf_priors_path = out_dirs["cnmf_priors"]
    genes_path = out_dirs["genes"]
    out_dir = out_dirs["out"]

    if checkpoint_path.exists() and not getattr(cfg, "force_retrain", False):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=get_device(), weights_only=False
            )
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
        raw_spatial_topics = _get_raw_adjs(graph_paths)

        if getattr(cfg, "unsupervised", False):
            print("  ↳ [!] UNSUPERVISED MODE: Bypassing signatures.csv.")
            init_components = raw_spatial_topics
            optimal_k = init_components.shape[0]
            print(
                f"  ↳ Extracted {optimal_k} unsupervised spatial dictionary components."
            )
        else:
            atlas_matrix, target_genes, lineage_names = _parse_nouns(
                genes_path, paths.sig_csv
            )
            nouns, noun_reports = _compress_nouns(
                atlas_matrix,
                lineage_names,
                target_genes,
                n_clusters=getattr(cfg, "n_prior_lineages", 25),
                min_genes=getattr(cfg, "min_genes_noun", 15),
                max_genes=getattr(cfg, "max_genes_noun", 100),
                consensus_freq=getattr(cfg, "consensus_freq", 0.30),
            )

            orthogonal_adjectives, sieve_stats, adjective_reports = _apply_sieve(
                raw_spatial_topics,
                nouns,
                target_genes,
                top_k_genes=getattr(cfg, "top_k_adjective_genes", 35),
                max_noun_overlap=getattr(cfg, "max_noun_overlap", 0.35),
                max_internal_overlap=getattr(cfg, "max_internal_overlap", 0.30),
                max_total_adjectives=getattr(cfg, "max_prior_adjectives", 8),
            )

            if len(orthogonal_adjectives) > 0 and len(nouns) > 0:
                init_components = np.vstack([nouns, orthogonal_adjectives])
            elif len(orthogonal_adjectives) > 0:
                init_components = orthogonal_adjectives
            else:
                init_components = nouns

            optimal_k = init_components.shape[0]
            x_nouns = nouns.shape[0]
            y_dict = orthogonal_adjectives.shape[0]
            print(
                f"  ↳ {x_nouns} compressed lineages + {y_dict} dict addition prior is sealed (Base K={optimal_k})."
            )

            report_path = out_dir / "prior.txt"
            _write_report(noun_reports, sieve_stats, adjective_reports, report_path)

        try:
            if init_components is not None:
                joblib.dump(init_components, cnmf_priors_path)
        except Exception:
            pass

    return init_components, optimal_k, checkpoint