# Libella ⍙
**A Memory-Optimized Spatial Transcriptomics GNN Pipeline**

Libella is a fast, end-to-end Graph Neural Network (GNN) based pipeline for discovering orthogonal, reproducible, and single-cell resolution spatial ecotypes and mapping topological interfaces across spatial transcriptomics cohorts. It is heavily optimized for low-RAM footprints—utilizing a custom contigous spatial batching with k-hops preservation—achieving $O(N)$ compute scalability and $O(1)$ peak memory usage with respect to dataset size. It supports NVIDIA CUDA and Apple MPS execution.

## Installation

Libella can be installed directly via pip. Ensure you are using Python >= 3.9.

```bash
pip install libella
```

## Quick Start

Libella is designed to be executed entirely from the command line. All you need is a `.csv` manifest pointing to your `.h5ad` files.

```bash
# Run with standard discovery settings
libella manifest.csv --out-dir ./results --mode DISCOVERY

# Run a quick test (limits epochs and cell counts)
libella manifest.csv --out-dir ./results_dev --mode DEV
```

### The Manifest File
Your `manifest.csv` must contain a `filepath` and a `split` column. You can optionally include `patient_id` and `dataset_id`.

```csv
filepath,dataset_id,patient_id,split
/path/to/sample1.h5ad,Dataset_A,Patient_1,discovery
/path/to/sample2.h5ad,Dataset_A,Patient_2,validation
```
* **Discovery Split:** Used to learn consensus genes, extract biological priors, and train the GNN.
* **Validation Split:** Used strictly for mapping final spatial topology and evaluating generalization.

---

## Hyperparameter Control

Every setting inside the Libella pipeline can be dynamically overridden directly from the terminal without editing any Python code.

### Standard Execution Overrides
```bash
libella manifest.csv \
  --out-dir ./results \
  --epochs 100 \
  --batch-size 15000 \
  --lr-base 0.0005
```

### Key Configuration Flags
* `--epochs`: Training duration (Default: 30)
* `--batch-size`: GNN spatial batching chunk size (Default: 10000)
* `--top-n-genes`: Number of spatial consensus genes to extract (Default: 2000)
* `--k-neighbors`: Physical graph neighbors (Default: 11)
* `--k-hops`: GNN message passing depth (Default: 2)
* `--dict-temp`: Softmax temperature for spatial dictionaries (Default: 0.3)
* `--force-retrain`: Overwrite existing GNN models and retrain from scratch.

*(Run `libella -h` to see the full list of over 40+ configurable hyper-parameters).*

---

## Output Structure

Upon completion, your `--out-dir` will contain:

* `run_discovery/` (or `run_publish/`)
    * `graphs/`: Serialized PyTorch Geometric `.pt` graphs for each sample.
    * `individual_samples/`: Folders per patient containing localized Topology CSV metrics.
    * `out/`: 
        * `final_gnn_model.pt`: The trained Libella GNN weights.
        * `Global_Smoothed_Macro_Domains.parquet`: Cell-level mappings for downstream analysis.
        * `Global_Meta_Topology_Continuous_Matrix.csv`: Extracted cross-cohort topology matrix.
        * `GNN_Learning_Curve.pdf`: Training validation curves.
```