import ast
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.text import TfidfTransformer

# ==============================================================================
# 1. Configuration & Paths (Adjust if needed)
# ==============================================================================
genes_path = Path("/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/common_genes.json")
h5ad_path = Path("/Users/Hemato/project_3/benchmark/benchmark_data.h5ad")

# Locate signatures.csv (adjust if in a different subfolder)
sig_csv_candidates = [
    Path("/Users/Hemato/project_3/benchmark/signatures.csv"),
    Path("/Users/Hemato/project_3/signatures.csv"),
    Path("/Users/Hemato/project_3/benchmark/benchmark_output/libella/run/signatures.csv"),
]
sig_csv_path = next((p for p in sig_csv_candidates if p.exists()), sig_csv_candidates[0])

# Parameters
N_PRIOR_LINEAGES = 25
MIN_GENES_NOUN = 15
MAX_GENES_NOUN = 100
CONSENSUS_FREQ = 0.30  # Gene must appear in >= 30% of merged child lineages
TOP_K_ADJECTIVES = 35
MAX_JACCARD_TO_NOUN = 0.25
MAX_JACCARD_INTERNAL = 0.35
N_DICT_COMPONENTS = 30

NOISE_REGEX = re.compile(r"^(MT-|RPS|RPL|HSP|MALAT1|NEAT1)", re.IGNORECASE)

# ==============================================================================
# 2. Step 1: Parse Raw Signatures
# ==============================================================================
print("=" * 110)
print("STEP 1: PARSING RAW SIGNATURES")
print("=" * 110)

with open(genes_path, "r") as f:
    genes_data = json.load(f)
target_genes = [str(g) for g in (genes_data if isinstance(genes_data, list) else genes_data.get("genes", list(genes_data.values())))]
gene2idx = {g: i for i, g in enumerate(target_genes)}
n_target_genes = len(target_genes)

if not sig_csv_path.exists():
    print(f"[!] Warning: Signature CSV not found at {sig_csv_path}. Please check the path.")
    sig_df = pd.DataFrame(columns=["Cell_type", "Genes"])
else:
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
    except (ValueError, SyntaxError):
        genes_list = [g.strip() for g in str(raw_genes_str).split(",")]

    for g in genes_list:
        g_clean = str(g).strip()
        if NOISE_REGEX.match(g_clean):
            continue
        if g_clean in gene2idx:
            atlas_matrix[i, gene2idx[g_clean]] = 1.0

print(f"Loaded {n_lineages} raw lineages across {n_target_genes} target genes.")

# ==============================================================================
# 3. Step 2: Consensus Compression of Nouns (Logging Merges & Trims)
# ==============================================================================
print("\n" + "=" * 110)
print(f"STEP 2: NOUN COMPRESSION AUDIT (Target Clusters: {N_PRIOR_LINEAGES})")
print("=" * 110)

clusterer = AgglomerativeClustering(
    n_clusters=min(N_PRIOR_LINEAGES, n_lineages), metric="jaccard", linkage="average"
)
cluster_labels = clusterer.fit_predict(atlas_matrix)

noun_records = []
valid_nouns = []

for cluster_id in range(clusterer.n_clusters):
    child_indices = np.where(cluster_labels == cluster_id)[0]
    child_matrix = atlas_matrix[child_indices]
    n_children = len(child_indices)
    child_names = [lineage_names[i] for i in child_indices]

    # Raw union count vs Consensus count
    union_indices = np.where(np.max(child_matrix, axis=0) > 0)[0]
    raw_union_count = len(union_indices)
    
    gene_frequencies = np.sum(child_matrix, axis=0)

    if n_children > 1:
        # Consensus: >= 30% recurrence
        consensus_mask = (gene_frequencies / n_children) >= CONSENSUS_FREQ
        active_indices = np.where(consensus_mask)[0]
    else:
        active_indices = np.where(gene_frequencies > 0)[0]

    # Size clamping
    status = "Merged & Trimmed" if n_children > 1 else "Single Lineage"
    trimmed_fringe = raw_union_count - len(active_indices)

    if len(active_indices) > MAX_GENES_NOUN:
        top_k_idx = np.argsort(-gene_frequencies[active_indices])[:MAX_GENES_NOUN]
        active_indices = active_indices[top_k_idx]
        status += f" (Capped at {MAX_GENES_NOUN})"
    elif len(active_indices) < MIN_GENES_NOUN:
        if raw_union_count >= MIN_GENES_NOUN:
            active_indices = np.argsort(-gene_frequencies)[:MIN_GENES_NOUN]
            status += f" (Expanded to min {MIN_GENES_NOUN})"
        else:
            status = f"PRUNED: Micro-cluster (< {MIN_GENES_NOUN} total genes)"
            noun_records.append({
                "Cluster": cluster_id,
                "N_Children": n_children,
                "Child_Lineages": ", ".join(child_names[:3]) + (f" (+{n_children-3})" if n_children > 3 else ""),
                "Raw_Union": raw_union_count,
                "Consensus_N": 0,
                "Trimmed_Fringe": raw_union_count,
                "Status": status,
                "Top_5_Genes": "None"
            })
            continue

    noun_vec = np.zeros(n_target_genes, dtype=np.float32)
    noun_vec[active_indices] = 1.0
    valid_nouns.append(noun_vec)

    top5_idx = np.argsort(-gene_frequencies[active_indices])[:5]
    top5_genes = [target_genes[active_indices[idx]] for idx in top5_idx]

    noun_records.append({
        "Cluster": cluster_id,
        "N_Children": n_children,
        "Child_Lineages": ", ".join(child_names[:2]) + (f" (+{n_children-2})" if n_children > 2 else ""),
        "Raw_Union": raw_union_count,
        "Consensus_N": len(active_indices),
        "Trimmed_Fringe": trimmed_fringe,
        "Status": status,
        "Top_5_Genes": ", ".join(top5_genes)
    })

nouns_matrix = np.array(valid_nouns, dtype=np.float32)
df_nouns = pd.DataFrame(noun_records)

print(df_nouns[["Cluster", "N_Children", "Raw_Union", "Consensus_N", "Trimmed_Fringe", "Status", "Top_5_Genes"]].to_string(index=False))
print(f"\nRetained Nouns: {nouns_matrix.shape[0]} / {clusterer.n_clusters} clusters (Mean genes/noun: {nouns_matrix.sum(axis=1).mean():.1f})")

# ==============================================================================
# 4. Step 3: Spatial Dictionary Learning & Orthogonal Sieve
# ==============================================================================
print("\n" + "=" * 110)
print(f"STEP 3: SPATIAL DICTIONARY LEARNING & ORTHOGONAL SIEVE")
print("=" * 110)

# Extract spatial topics from AnnData expression background
print("Extracting spatial dictionary components from dataset background...")
adata = sc.read_h5ad(h5ad_path, backed="r")
valid_h5_genes = [g for g in target_genes if g in adata.var_names]
sub_X = adata[:, valid_h5_genes].to_memory().X

if sp.issparse(sub_X):
    X_sample = sub_X[:50000].tocsr().astype(np.float32)
else:
    X_sample = sp.csr_matrix(sub_X[:50000], dtype=np.float32)

tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
X_tfidf = tfidf.fit_transform(X_sample)

dict_learner = MiniBatchDictionaryLearning(
    n_components=N_DICT_COMPONENTS, alpha=2.0, random_state=42, max_iter=100
)
dict_learner.fit(X_tfidf.toarray())
raw_spatial_topics = np.abs(dict_learner.components_)

# Align raw dictionary topics to target_genes index space
h5_to_target = [gene2idx[g] for g in valid_h5_genes]
aligned_topics = np.zeros((raw_spatial_topics.shape[0], n_target_genes), dtype=np.float32)
for local_idx, global_idx in enumerate(h5_to_target):
    aligned_topics[:, global_idx] = raw_spatial_topics[:, local_idx]

# Sieve execution
noun_sets = [set(np.where(nouns_matrix[i] > 0)[0]) for i in range(nouns_matrix.shape[0])]
candidate_adjectives = []

sieve_records = []

for i in range(aligned_topics.shape[0]):
    weights = aligned_topics[i]
    top_indices = set(np.argpartition(weights, -TOP_K_ADJECTIVES)[-TOP_K_ADJECTIVES:])
    
    top10_idx = np.argsort(-weights)[:5]
    top5_genes = [target_genes[idx] for idx in top10_idx]

    # 1. Jaccard against all Nouns
    jaccards_to_nouns = [
        (len(top_indices.intersection(n_s)) / len(top_indices.union(n_s)), n_idx)
        for n_idx, n_s in enumerate(noun_sets)
    ]
    max_noun_jaccard, matched_noun = max(jaccards_to_nouns, key=lambda x: x[0])

    if max_noun_jaccard >= MAX_JACCARD_TO_NOUN:
        sieve_records.append({
            "Topic": f"Topic_{i:02d}",
            "Max_Noun_Jaccard": f"{max_noun_jaccard:.2f} (Noun_{matched_noun:02d})",
            "Internal_Jaccard": "-",
            "Decision": f"DISCARDED: Redundant with Noun {matched_noun:02d}",
            "Top_5_Genes": ", ".join(top5_genes)
        })
        continue

    bin_topic = np.zeros(n_target_genes, dtype=np.float32)
    bin_topic[list(top_indices)] = 1.0
    candidate_adjectives.append((bin_topic, top_indices, i, top5_genes, max_noun_jaccard, matched_noun))

# 2. Internal Orthogonal Sieve among candidate adjectives
accepted_adjectives = []
accepted_sets = []

for bin_vec, gene_set, topic_id, top5_genes, max_n_jac, matched_n in candidate_adjectives:
    if not accepted_sets:
        accepted_adjectives.append(bin_vec)
        accepted_sets.append(gene_set)
        sieve_records.append({
            "Topic": f"Topic_{topic_id:02d}",
            "Max_Noun_Jaccard": f"{max_n_jac:.2f} (Noun_{matched_n:02d})",
            "Internal_Jaccard": "0.00 (First)",
            "Decision": "KEPT: Orthogonal Spatial Adjective",
            "Top_5_Genes": ", ".join(top5_genes)
        })
    else:
        internal_jaccards = [
            (len(gene_set.intersection(s)) / len(gene_set.union(s)), idx)
            for idx, s in enumerate(accepted_sets)
        ]
        max_internal_jaccard, matched_adj = max(internal_jaccards, key=lambda x: x[0])

        if max_internal_jaccard >= MAX_JACCARD_INTERNAL:
            sieve_records.append({
                "Topic": f"Topic_{topic_id:02d}",
                "Max_Noun_Jaccard": f"{max_n_jac:.2f} (Noun_{matched_n:02d})",
                "Internal_Jaccard": f"{max_internal_jaccard:.2f} (Adj_{matched_adj:02d})",
                "Decision": f"DISCARDED: Internal Duplicate of Adj_{matched_adj:02d}",
                "Top_5_Genes": ", ".join(top5_genes)
            })
        else:
            accepted_adjectives.append(bin_vec)
            accepted_sets.append(gene_set)
            sieve_records.append({
                "Topic": f"Topic_{topic_id:02d}",
                "Max_Noun_Jaccard": f"{max_n_jac:.2f} (Noun_{matched_n:02d})",
                "Internal_Jaccard": f"{max_internal_jaccard:.2f}",
                "Decision": "KEPT: Orthogonal Spatial Adjective",
                "Top_5_Genes": ", ".join(top5_genes)
            })

df_sieve = pd.DataFrame(sieve_records)
print(df_sieve.to_string(index=False))

# ==============================================================================
# 5. Final Assembly Inspection
# ==============================================================================
if accepted_adjectives:
    adjectives_matrix = np.array(accepted_adjectives, dtype=np.float32)
    final_priors = np.vstack([nouns_matrix, adjectives_matrix])
else:
    final_priors = nouns_matrix

print("\n" + "=" * 110)
print(f"FINAL GNN PRIOR ASSEMBLY SUMMARY (Total K = {final_priors.shape[0]})")
print("=" * 110)
print(f"• Bounded Nouns:       {nouns_matrix.shape[0]} modules (Gene sizes: min={nouns_matrix.sum(axis=1).min():.0f}, max={nouns_matrix.sum(axis=1).max():.0f})")
print(f"• Spatial Adjectives:  {len(accepted_adjectives)} modules (Fixed size: {TOP_K_ADJECTIVES})")
print(f"• Mega-Blobs (> 100):  {np.sum(final_priors.sum(axis=1) > 100)} (Target: 0)")
print(f"• Micro-Sets (< 15):   {np.sum(final_priors.sum(axis=1) < 15)} (Target: 0)")
print(f"• Prior Matrix Shape:  {final_priors.shape[0]} Topics x {final_priors.shape[1]} Target Genes")
print("=" * 110)