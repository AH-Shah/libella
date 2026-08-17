import multiprocessing
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Tuple

def init_env():
    """Set environment runtime flags."""
    warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["KMP_WARNINGS"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

@dataclass
class RunConfig:
    """Pipeline configuration schema."""
    mode: str = "PUBLISH"
    force_retrain: bool = False
    
    # Execution Phases & Setup
    phase: str = "ALL"
    unsupervised: bool = False
    
    # Native K Controllers
    n_prior_lineages: int = 30   
    n_dict_components: int = 100
    k_override: int | None = 1024
    
    hidden_dim: int = 128
    k_hops: int = 2
    extra_topics: int = 0
    
    batch_size: int = 5000
    meta_batch_size: int = 5
    epochs: int = 300

    # Graph Construction & Data
    k_neighbors: int = 11
    moran_k: int = 7
    chunk_size: int = 2000
    feature_cap: int = 50_000
    
    top_n_genes: int = 2000
    max_cells_per_sample: int = 1_000_000
    prior_cells_per_sample: int = 5_000_000
    
    # Model Physics & Schedules
    dict_temp: float = 0.25
    att_temp: float = -2.25
    gnn_shift_weight: float = 0.5
    train_noise: float = 0.05
    edge_dropout: float = 0.40
    scale_start: float = 8.0
    scale_end: float = 16.0
    alpha_start: float = 1.2
    alpha_end: float = 1.7
    temp_start: float = 1.5
    temp_end: float = 0.3

    # Loss & Regularization (Preserved + Native SAE Controls)
    kl_weight: float = 5
    kl_base: float = 0.25
    kl_collapse_weight: float = 3.0
    hub_threshold: float = 0.2
    anchor_peak_threshold: float = 0.50
    ortho_weight: float = 10.0
    ortho_overlap_threshold: float = 0.15
    ortho_margin: float = 0.10             # SAE Oblique Cosine Margin on Unit Sphere
    tsallis_alpha: float = 1.5
    delta_clamp: float = 30.0
    zero_mask_rate: float = 0.05
    
    # Native SAE Sparsity & Capacity Engine (Calibrated)
    l1_coeff: float = 1e-3                 # Base L1 proxy coefficient for Jump gate
    aux_weight: float = 0.05               # Gentle AuxK revival (prevents noise overfitting)
    aux_k: int = 4                         # Top-k dead atoms per batch
    dead_step_threshold: int = 200         # Latents must be idle for 200 steps before revival
    ortho_sample_size: int = 128           # Low memory Gram matrix slice

    # Optimizers
    lr_base: float = 0.001
    wd_base: float = 1e-4
    lr_decoder: float = 0.001              # Isolated learning rate for Unit-Norm Oblique Decoder
    lr_threshold: float = 0.0005           # Learning rate for Jump thresholds (θ)
    lr_anchor: float = 0.001
    wd_anchor: float = 0
    grad_clip: float = 5.0

    # Inference & Topology
    entropy_pruning: bool = True
    inference_scale: float = 16.0
    inference_alpha: float = 1.65
    inference_temp: float = 0.3
    inf_batch_size: int = 5000
    panel_overlap_thresh: float = 0.80
    radius_multiplier: float = 3.33
    min_topic_mass: float = 0.01
    leiden_res: float = 1.5
    smoothing_self_weight: float = 0.1
    fdr_threshold: float = 0.05
    
    n_perms_entropy: int = 10_000
    n_perms_topo: int = 1000
    
    n_jobs: int = field(default_factory=lambda: min(8, multiprocessing.cpu_count()))
    use_sketching: bool = True
    suffix: str = ""

    @classmethod
    def from_mode(cls, mode: str = "PUBLISH") -> "RunConfig":
        """Init config by mode."""
        base = cls(mode=mode)
        if mode == "DEV":
            base.epochs = 2
            base.max_cells_per_sample = 1_000
            base.prior_cells_per_sample = 500
            base.n_perms_entropy = 5
            base.n_perms_topo = 2
            base.n_jobs = -1
            base.use_sketching = False
            base.suffix = "_dev"
        elif mode == "DISCOVERY":
            base.epochs = 50
            base.max_cells_per_sample = 30_000
            base.n_perms_entropy = 100
            base.n_perms_topo = 25
            base.suffix = "_discovery"
        elif mode != "PUBLISH":
            raise ValueError(f"Unknown mode: {mode}")
        return base

        

@dataclass
class Paths:
    """Project path manager."""
    out_base: Path = Path("./libella_output")
    

    sig_csv: Path = Path(__file__).parent / "signatures.csv"
    
    def make_dirs(self, suffix: str = "") -> dict:
        out_dir = self.out_base / f"run{suffix}"
        dirs = {
            "out": out_dir,
            "indiv": out_dir / "individual_samples",
            "graphs": out_dir / "graphs",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # File paths
        dirs["cnmf_priors"] = out_dir / "global_cnmf_priors.pkl"
        dirs["nmf_model"] = out_dir / "final_gnn_model.pt"
        dirs["genes"] = out_dir / "common_genes.json"
        dirs["names"] = out_dir / "meta_names.json"
        dirs["checkpoint"] = out_dir / "gnn_checkpoint.pt"
        return dirs

NOISE_PATTERNS = [
    r"^M?RP[LSP]+[A-Z0-9]*\d+",   # Ribosomal proteins
    r"^(?:EEF|EIF)\d[A-Z0-9]*",   # Elongation/Initiation factors
    r"^PABP[CN]\d+.*",            # Poly-A binding
    r"^(?:FAU|SUB1|OAZ[12])$",    # Ubiquitin-ribosome fusions and antizymes
    r"^MT-.*",                    # Canonical mitochondrial
    r"^ATP\d+[A-Z0-9]*",          # ALL ATPases 
    r"^(?:NDUF|COX|UQCR)[A-Z0-9]+", # Respiration chains
    r"^CHCHD\d+.*",               # Mitochondrial coiled-coil
    r"^(?:TOMM|TIMM)\d+[A-Z0-9]*$", # Mitochondrial translocases (Outer/Inner)
    r"^(?:SLC25A[56]|VDAC1|CYCS)$", # Mitochondrial ADP/ATP carriers & Cytochrome C
    r"^(?:HSP|DNAJ)[A-Z0-9]+.*",  # Heatshock & co-chaperones
    r"^CCT\d+.*",                 # Chaperonin complex
    r"^TRIM\d+.*",                # Tripartite motif baseline
    r"^PSM[ABCDE]\d+.*",          # Proteasome subunits
    r"^(?:CALR|PTGES3)$",         # Calreticulin & HSP90 co-chaperones
    r"^ACT[BG]\d*.*",             # Beta/Gamma Actins (Protects ACTA2/Smooth muscle)
    r"^TUB[A-Z]+\d*.*",           # ALL Tubulins (Catches TUBB, TUBA1A, etc.)
    r"^MYL12[AB].*",              # Ubiquitous non-muscle myosin
    r"^(?:FLNA|CLT[AB])$",        # Filamin A & Clathrin
    r"^(?:ARPC\d[A-Z]?|ARF\d)$",  # Actin nucleation
    r"^(?:CAP[12G]|CAPZ[AB]\d?|ANXA\d+)$", # Actin capping & Annexins
    r"^(?:TMSB\d+[A-Z0-9]*|PTM[AS])$", # Thymosins/Prothymosins
    r"^(?:SRSF[1-9]|NPM1|NCL|NOP\d+|NHP2|NACA)$", # Spliceosome/Nucleolus
    r"^HNRNP.*",                  # Heterogeneous nuclear ribonucleoproteins
    r"^SNRP[A-Z0-9]+.*",          # Small nuclear ribonucleoproteins
    r"^DDX\d+.*",                 # DEAD-box RNA helicases
    r"^HIST[1-4].*",              # Historic histones
    r"^H(?:[1-7]|2A|2B)[-A-Z0-9]+", # Linker histones
    r"^HMG[ABN]\d+$",             # High mobility group chromatin
    r"^SEC[1236][1-9][A-Z0-9]*",  # ER Translocation
    r"^(?:SSR[1-4]|SPCS[1-3]|POMP)$", # Translocon receptors
    r"^RAN(?:BP\d+)?$",           # Nuclear transport
    r"^RAB\d+[A-Z]*$",            # Vesicle GTPases
    r"^RABAC1$",                  
    r"^(?:GAPDH|B2M|PPI[ABF]|PGK1|ENO1|ALDOC|UAP1|YWHA[ABHQZ])$", # Universal qPCR controls
    r"^(?:GDI|HINT|TPI|PNRC)\d+.*", # Basal enzymes
    r"^(?:PRDX|SOD)\d+$",         # Oxidative stress (Peroxiredoxins, SOD)
    r"^TXN(?:IP)?$",              # Thioredoxins
    r"^(?:SH3BGRL\d*|CDV3|MIF|CALM\d+)$", # Basal signaling/Calmodulins
    r"^UB[ABC]\d*.*",             # Ubiquitin background
    r"^UBE2[A-Z\d]+$",            # Ubiquitin-conjugating enzymes
    r"^MT[0-9]+[A-Z].*",
    r"^[A-Z]{2}\d{5,}\.\d+",      # BAC clones
    r"^(?:RP[1-9]\d*|CTD)-.*",    # Roswell/Caltech mapping clones
    r"^LINC\d+",                  # LncRNAs
    r"^C[0-9XY]+orf\d+.*",        # Open reading frames
    r".*-AS\d*$",                 # Antisense transcripts
    r"^MIR\d+.*",                 # Micro RNAs
    r"^(?:MALAT|NEAT|TSIX|XIST)\d*.*", # Massive structural lncRNAs
    r"^ZNF\d+",                   # Zinc fingers
    r"^(?:OR|TAS[12]R)\d+[A-Z]+\d+.*", # Olfactory/Taste receptors
    r"^(?:SMCO|KIR|PRSS|FAM)\d+[A-Z]*", # Uncharacterized/Polymorphic
    r"^(?:KIAA|NPIPB)\d+.*",      # Orphan/Pseudogene repeats
    r"^KRT(?:2[1-9]|[3-9]\d|[1-9]\d{2,}).*", # Hair/Skin Keratin contamination
    r"^HLA-[ABC].*",               # Ubiquitous on all nucleated cells
    # Added .* to DUSP, EGR, IER, and BTG so it catches DUSP4-1 or DUSP4P1
    r"^(?:FOS(?:B|L\d)?|JUN[BD]?|EGR\d+.*|IER\d+.*|BTG\d+.*|ZFP36.*|DUSP\d+.*)$",
    r"^(?:CIRBP|RACK1|SFPQ|RBM39|YBX\d+.*)$",
    r"^(?:CFL\d+.*|PFN\d+.*|MYL6|AHNAK.*)$"
]


NOISE_REGEX = re.compile("|".join(NOISE_PATTERNS), flags=re.IGNORECASE)

cfg = RunConfig.from_mode("PUBLISH")
paths = Paths()
