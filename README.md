# Libella

**Memory-Optimized Spatial Transcriptomics GNN Pipeline**

Libella is an end-to-end Graph Neural Network (GNN) framework designed to discover spatial ecotypes and map topological interfaces across spatial transcriptomics cohorts. By using contiguous spatial batching with $k$-hop preservation, it achieves $O(N)$ compute scalability and $O(1)$ peak memory footprint relative to dataset size. Natively supports NVIDIA CUDA and Apple Silicon (MPS).

---

## Quick Start

Execution is fully CLI-driven via a single `.csv` manifest file pointing to your `.h5ad` datasets.

```bash
# Standard discovery run
libella manifest.csv --out-dir ./results --mode DISCOVERY

# Fast development test (restricts epochs & cell counts)
libella manifest.csv --out-dir ./results_dev --mode DEV

```

---

## Installation

Requires **Python $\ge$ 3.9**.

```bash
pip install libella

```

---

## Data Preparation

Input `.h5ad` files must feature **Human HGNC gene symbols** (e.g., `CD8A`) in `adata.var_names`.

* **Mouse / Rat Data:** Automatically converted to uppercase (e.g., `Sox2` $\rightarrow$ `SOX2`) for cross-species compatibility.
* **Ensembl IDs:** Must be converted to HGNC symbols before execution.

```python
import mygene
import scanpy as sc

# Load data
adata = sc.read_h5ad("my_data.h5ad")

# Query Ensembl to HGNC mapping
mg = mygene.MyGeneInfo()
results = mg.querymany(
    adata.var_names, 
    scopes="ensembl.gene", 
    fields="symbol", 
    species="human"
)

# Remap gene symbols
symbol_map = {res["query"]: res.get("symbol", res["query"]) for res in results}
adata.var_names = [symbol_map.get(g, g) for g in adata.var_names]

adata.write_h5ad("my_data_mapped.h5ad")

```

---

## Manifest Schema

The `manifest.csv` defines the pipeline execution graph. It requires `filepath`, `discovery`, and `projection` fields.

```csv
filepath,dataset_id,patient_id,discovery,projection
/path/to/sample1.h5ad,Dataset_A,Patient_1,True,True
/path/to/sample2.h5ad,Dataset_A,Patient_2,False,True

```

| Flag | Value | Description |
| --- | --- | --- |
| **`discovery`** | `True` | Computes consensus genes, biological priors, and trains the core GNN. |
| **`projection`** | `True` | Projects the trained GNN model onto the sample to extract spatial topology. |

> **Note:** Samples marked with `discovery=False` and `projection=True` are evaluated as hold-out validation cohorts.

---

## Configuration & Hyperparameters

All pipeline parameters can be overridden at runtime via CLI arguments.

```bash
libella manifest.csv \
  --out-dir ./results \
  --epochs 100 \
  --batch-size 15000 \
  --lr-base 0.0005

```

### Core Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--epochs` | `30` | Number of GNN training iterations |
| `--batch-size` | `10000` | Spatial chunk size for memory bounding |
| `--top-n-genes` | `2000` | Number of spatial consensus genes to extract |
| `--k-neighbors` | `11` | Physical graph neighbors constructed per node |
| `--k-hops` | `2` | GNN message-passing neighborhood depth |
| `--dict-temp` | `0.3` | Softmax temperature for spatial dictionary learning |
| `--entropy-pruning` | `True` | Toggles batch-effect artifact pruning |

> Run `libella -h` to inspect the full list of CLI flags.

---

## Output Directory Structure

Executing the pipeline populates `--out-dir` with the following structure:

```text
results/
├── graphs/
│   └── *.pt                                  # Serialized PyTorch Geometric spatial graphs
├── individual_samples/
│   └── *.csv                                 # Sample-specific topological metrics
└── out/
    ├── final_gnn_model.pt                    # Trained GNN weights checkpoint
    ├── Global_Smoothed_Macro_Domains.parquet # Cell-level spatial domain assignments
    └── Global_Meta_Topology_Continuous_Matrix.csv # Cohort-wide topological matrix

```