# Libella
**Memory-Optimized Spatial Transcriptomics GNN Pipeline**

Libella is an end-to-end Graph Neural Network (GNN) pipeline for discovering spatial ecotypes and mapping topological interfaces across spatial transcriptomics cohorts. Utilizing contiguous spatial batching with k-hop preservation, it achieves $O(N)$ compute scalability and $O(1)$ peak memory usage relative to dataset size. It natively supports NVIDIA CUDA and Apple MPS execution.

## Installation
Requires Python >= 3.9.
```bash
pip install libella

Quick Start

Execution is CLI-driven, requiring only a .csv manifest pointing to .h5ad files.

# Standard discovery run
libella manifest.csv --out-dir ./results --mode DISCOVERY

# Fast development test (limits epochs and cell counts)
libella manifest.csv --out-dir ./results_dev --mode DEV

Data Requirements

Input .h5ad files must use Human HGNC gene symbols (e.g., CD8A) in
adata.var_names.

  - Mouse/Rat Data: Automatically uppercased (e.g., Sox2 → SOX2) for
    cross-species compatibility.
  - Ensembl IDs: Must be mapped to HGNC symbols prior to execution.

Example Ensembl to HGNC conversion (Python/Scanpy):

import scanpy as sc
import mygene

adata = sc.read_h5ad("my_data.h5ad")
mg = mygene.MyGeneInfo()
results = mg.querymany(adata.var_names, scopes='ensembl.gene', fields='symbol', species='human')

symbol_map = {res['query']: res.get('symbol', res['query']) for res in results}
adata.var_names = [symbol_map.get(g, g) for g in adata.var_names]
adata.write_h5ad("my_data_mapped.h5ad")

The Manifest File

The manifest.csv dictates the pipeline execution graph. It requires filepath,
discovery (boolean), and projection (boolean) columns.

filepath,dataset_id,patient_id,discovery,projection
/path/to/sample1.h5ad,Dataset_A,Patient_1,True,True
/path/to/sample2.h5ad,Dataset_A,Patient_2,False,True

  - Discovery (True): Computes consensus genes, biological priors, and trains
    the GNN.
  - Projection (True): Projects the trained model onto the sample to map spatial
    topology. (If Discovery is False but Projection is True, the file is treated
    purely as a hold-out validation sample).

Hyperparameter Control

All pipeline parameters can be overridden via CLI flags.

libella manifest.csv \
  --out-dir ./results \
  --epochs 100 \
  --batch-size 15000 \
  --lr-base 0.0005

Core Configuration Flags:

  - --epochs: Training iterations (Default: 30)
  - --batch-size: Spatial chunk size for memory bounding (Default: 10000)
  - --top-n-genes: Spatial consensus genes to extract (Default: 2000)
  - --k-neighbors: Physical graph neighbors per node (Default: 11)
  - --k-hops: GNN message passing depth (Default: 2)
  - --dict-temp: Softmax temperature for spatial dictionaries (Default: 0.3)
  - --entropy-pruning: Toggle batch-effect artifact pruning (Default: True)

(Run libella -h for the full parameter dictionary).

Output Structure

Upon completion, the target --out-dir will populate:

  - graphs/: Serialized PyTorch Geometric .pt spatial graphs.
  - individual_samples/: Patient-specific topology CSV metrics.
  - out/:
      - final_gnn_model.pt: Serialized model weights.
      - Global_Smoothed_Macro_Domains.parquet: Cell-level spatial mappings.
      - Global_Meta_Topology_Continuous_Matrix.csv: Extracted cross-cohort
        topology matrix.

