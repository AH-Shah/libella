#!/usr/bin/env python3
"""
Libella Trajectory Health & Architectural Autopsy
Comprehensive analysis of checkpoints, weights, attention dynamics, and dictionary health.
"""

import os
import re
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set plot aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


class TrajectoryAutopsy:
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")

        self.ckpt_files = self._discover_checkpoints()
        if not self.ckpt_files:
            raise ValueError(f"No valid .pt checkpoints found in {self.checkpoint_dir}")

        print(f"[i] Found {len(self.ckpt_files)} checkpoints in: {self.checkpoint_dir}")
        self.history: List[Dict[str, Any]] = []

    def _discover_checkpoints(self) -> List[Path]:
        """Discover and sort checkpoint files numerically by epoch."""
        files = list(self.checkpoint_dir.glob("epoch_*.pt")) + list(self.checkpoint_dir.glob("*.pt"))
        files = list(set(files))

        def extract_epoch(p: Path) -> int:
            match = re.search(r'epoch_?(\d+)', p.name)
            if match:
                return int(match.group(1))
            match_num = re.search(r'(\d+)', p.name)
            return int(match_num.group(1)) if match_num else 0

        # Filter out non-epoch files like resume_latest unless they have numeric names
        valid_files = [f for f in files if re.search(r'\d+', f.stem)]
        return sorted(valid_files, key=extract_epoch)

    @staticmethod
    def _matrix_effective_rank(mat: torch.Tensor) -> float:
        """Compute effective rank of a matrix via singular value entropy."""
        if mat.ndim != 2:
            return 0.0
        try:
            _, s, _ = torch.linalg.svd(mat.float(), full_matrices=False)
            s_sum = s.sum()
            if s_sum <= 1e-9:
                return 0.0
            p = s / s_sum
            p = p[p > 1e-9]
            entropy = -(p * torch.log(p)).sum().item()
            return math.exp(entropy)
        except Exception:
            return 0.0

    @staticmethod
    def _matrix_condition_number(mat: torch.Tensor) -> float:
        """Compute condition number (s_max / s_min) of a linear transformation."""
        try:
            s = torch.linalg.svdvals(mat.float())
            s_max = s[0].item()
            s_min = s[-1].item()
            return (s_max / (s_min + 1e-9)) if s_min > 1e-9 else 1e6
        except Exception:
            return 1.0

    def run_autopsy(self) -> pd.DataFrame:
        """Execute in-depth analytical parsing across all epochs."""
        print("[➤] Extracting trajectory metrics and tensor dynamics...")
        first_anchor_logits: Optional[torch.Tensor] = None

        for ckpt_path in self.ckpt_files:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            metrics = ckpt.get("metrics", {})
            epoch = ckpt.get("epoch", int(re.search(r'\d+', ckpt_path.stem).group(0)))

            entry: Dict[str, Any] = {"epoch": epoch, "file": ckpt_path.name}

            # -------------------------------------------------------------
            # 1. TELEMETRY & LOSS RESCUE
            # -------------------------------------------------------------
            entry["train_loss"] = metrics.get("train_loss", np.nan)
            entry["val_loss"] = metrics.get("val_loss", np.nan)
            loss_components = metrics.get("loss_components", {})
            entry["l_rec"] = loss_components.get("rec", np.nan)
            entry["l_anc"] = loss_components.get("anc", np.nan)
            entry["l_ort"] = loss_components.get("ort", np.nan)

            entry["entropy"] = metrics.get("entropy", np.nan)
            entry["collapse_ratio"] = metrics.get("collapse_ratio", np.nan)
            entry["kl_weight"] = metrics.get("kl_weight", np.nan)
            entry["g_w"] = metrics.get("g_w", np.nan)
            entry["p_w"] = metrics.get("p_w", np.nan)
            entry["top_topic_pct"] = metrics.get("top_topic_pct", np.nan)
            entry["top_topic_id"] = metrics.get("top_topic_id", np.nan)

            # -------------------------------------------------------------
            # 2. DICTIONARY & METAPROGRAM GEOMETRY
            # -------------------------------------------------------------
            if "topic_gene_logits" in state_dict:
                t_logits = state_dict["topic_gene_logits"].float()
                K, G = t_logits.shape
                entry["K_topics"] = K
                entry["G_genes"] = G

                # Dict temperature
                d_temp = state_dict.get("dict_temp", torch.tensor(0.30)).float().clamp(0.25, 1.0).item()
                entry["dict_temp"] = d_temp

                # Softmax Sharp Anchors
                anchors = F.softmax(t_logits / d_temp, dim=-1)

                # Peak weight ($G_W$) and per-topic gene entropy
                anchor_peaks = anchors.max(dim=1).values
                entry["anchor_peak_mean"] = anchor_peaks.mean().item() * 100.0
                entry["anchor_peak_min"] = anchor_peaks.min().item() * 100.0

                gene_ent = -(anchors * torch.log(anchors + 1e-9)).sum(dim=1)
                entry["gene_entropy_mean"] = gene_ent.mean().item()

                # Orthogonality & Gram Matrix Analysis
                anc_norm = F.normalize(anchors, p=2, dim=1)
                gram = torch.mm(anc_norm, anc_norm.t())
                eye_mask = 1.0 - torch.eye(K, dtype=torch.float32)
                off_diag_gram = gram * eye_mask

                entry["topic_max_overlap"] = off_diag_gram.max().item()
                entry["topic_mean_overlap"] = (off_diag_gram.sum() / max(1, K * (K - 1))).item()
                entry["topic_effective_rank"] = self._matrix_effective_rank(anc_norm)
                entry["topic_effective_rank_ratio"] = entry["topic_effective_rank"] / K

                # Drift from baseline epoch 0
                if first_anchor_logits is None:
                    first_anchor_logits = t_logits.clone()
                    entry["anchor_drift"] = 0.0
                else:
                    first_anc_norm = F.normalize(F.softmax(first_anchor_logits / d_temp, dim=-1), p=2, dim=1)
                    cos_drift = 1.0 - (anc_norm * first_anc_norm).sum(dim=1).mean().item()
                    entry["anchor_drift"] = cos_drift

            # Spatial Bridge Dynamics
            if "spatial_bridge.3.weight" in state_dict:
                entry["spatial_bridge_norm"] = torch.norm(state_dict["spatial_bridge.3.weight"].float()).item()
            elif "spatial_bridge.0.weight" in state_dict:
                entry["spatial_bridge_norm"] = torch.norm(state_dict["spatial_bridge.0.weight"].float()).item()
            else:
                entry["spatial_bridge_norm"] = 0.0

            # -------------------------------------------------------------
            # 3. GATv2 & GRAPH MESSAGE-PASSING PHYSICS
            # -------------------------------------------------------------
            if "att_temp" in state_dict:
                tau = F.softplus(state_dict["att_temp"].float()).clamp(min=0.05).item()
                entry["gat_att_temp"] = tau

            if "gamma" in state_dict:
                gamma_decay = F.softplus(state_dict["gamma"].float()).item()
                entry["gamma_decay"] = gamma_decay

            if "alpha_proj.weight" in state_dict:
                entry["alpha_proj_norm"] = torch.norm(state_dict["alpha_proj.weight"].float()).item()
                if "alpha_proj.bias" in state_dict:
                    base_alpha = torch.sigmoid(state_dict["alpha_proj.bias"].float()).item() * 0.85 + 0.10
                    entry["appnp_base_alpha"] = base_alpha

            # GATv2 Projections Norms
            for layer_k in ["gat_w_src.weight", "gat_w_dst.weight", "gat_w_edge.weight", "gat_a.weight", "mp_update.weight"]:
                if layer_k in state_dict:
                    entry[layer_k.replace(".weight", "_norm")] = torch.norm(state_dict[layer_k].float()).item()

            # -------------------------------------------------------------
            # 4. TRANSFORMER CROSS-ATTENTION & GATING
            # -------------------------------------------------------------
            for proj_k in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]:
                if proj_k in state_dict:
                    mat = state_dict[proj_k].float()
                    entry[proj_k.replace(".weight", "_norm")] = torch.norm(mat).item()
                    entry[proj_k.replace(".weight", "_cond")] = self._matrix_condition_number(mat)
                    entry[proj_k.replace(".weight", "_eff_rank")] = self._matrix_effective_rank(mat)

            if "context_gate.0.weight" in state_dict and "id_enc.0.weight" in state_dict:
                gate_norm = torch.norm(state_dict["context_gate.0.weight"].float()).item()
                id_norm = torch.norm(state_dict["id_enc.0.weight"].float()).item()
                entry["gate_to_id_ratio"] = gate_norm / max(1e-9, id_norm)

            if "dynamic_w_ema" in state_dict:
                entry["dynamic_w_ema"] = state_dict["dynamic_w_ema"].float().item()

            self.history.append(entry)

        df = pd.DataFrame(self.history).sort_values("epoch").reset_index(drop=True)
        return df

    def print_health_report(self, df: pd.DataFrame) -> None:
        """Evaluate failure modes and print formatted diagnostic scorecard."""
        latest = df.iloc[-1]
        earliest = df.iloc[0]

        print("\n" + "=" * 90)
        print(f"               LIBELLA TRAINING TRAJECTORY AUTOPSY REPORT")
        print(f"      Evaluated {len(df)} Checkpoints from Epoch {int(earliest['epoch'])} to {int(latest['epoch'])}")
        print("=" * 90)

        # -------------------------------------------------------------
        # METRIC PROGRESSION TABLE
        # -------------------------------------------------------------
        print("\n[1] METRIC & PARAMETER TRAJECTORY SUMMARY:")
        summary_cols = [
            ("Reconstruction Loss", "l_rec", ".4f"),
            ("Val Loss", "val_loss", ".4f"),
            ("Cell Sharpness (P_W %)", "p_w", ".1f"),
            ("Gene Peak (G_W %)", "g_w", ".1f"),
            ("Topic Max Overlap", "topic_max_overlap", ".3f"),
            ("Topic Effective Rank", "topic_effective_rank", ".1f"),
            ("Anchor Drift from T0", "anchor_drift", ".4f"),
            ("GAT Att Temp (Tau)", "gat_att_temp", ".3f"),
            ("Edge Decay (Gamma)", "gamma_decay", ".3f"),
            ("Gate-to-ID Ratio", "gate_to_id_ratio", ".3f"),
            ("Top Topic Utilization", "top_topic_pct", ".1f"),
        ]

        print(f"{'Metric':<28} | {'Epoch ' + str(int(earliest['epoch'])):<12} | {'Epoch ' + str(int(latest['epoch'])):<12} | {'Delta':<12} | {'Status'}")
        print("-" * 90)

        for name, col, fmt in summary_cols:
            if col in df.columns and not df[col].isna().all():
                v0 = earliest[col]
                v1 = latest[col]
                delta = v1 - v0
                
                # Health thresholds
                status = "✓ OK"
                if col == "topic_max_overlap" and v1 > 0.40:
                    status = "⚠️ REDUNDANT"
                elif col == "topic_effective_rank" and "K_topics" in latest and (v1 / latest["K_topics"]) < 0.40:
                    status = "🚨 COLLAPSED"
                elif col == "gat_att_temp" and (v1 < 0.06 or v1 > 5.0):
                    status = "⚠️ EXTREME TAU"
                elif col == "gamma_decay" and v1 < 0.05:
                    status = "⚠️ UNIFORM GRAPH"
                elif col == "gate_to_id_ratio" and (v1 < 0.05 or v1 > 10.0):
                    status = "⚠️ UNBALANCED GATE"
                elif col == "top_topic_pct" and v1 > 35.0:
                    status = "🚨 DOMINATED"
                elif col == "p_w" and v1 < 45.0:
                    status = "⚠️ BLURRY CELLS"

                print(f"{name:<28} | {format(v0, fmt):<12} | {format(v1, fmt):<12} | {format(delta, fmt):<12} | {status}")

        print("-" * 90)

        # -------------------------------------------------------------
        # STRUCTURAL INTEGRITY DIAGNOSTIC AUDIT
        # -------------------------------------------------------------
        print("\n[2] COMPONENT HEALTH & ANOMALY AUDIT:")

        issues = []
        strengths = []

        # Audit 1: Val Loss Divergence
        if "val_loss" in df.columns and not df["val_loss"].isna().all():
            val_delta = latest["val_loss"] - df["val_loss"].min()
            if val_delta > 0.05:
                issues.append(f"[OVERFITTING] Val loss increased by +{val_delta:.4f} from best checkpoint.")
            else:
                strengths.append("[GENERALIZATION] Validation loss tracked stably with training loss.")

        # Audit 2: Dictionary Orthogonality & Rank
        if "topic_max_overlap" in latest and latest["topic_max_overlap"] > 0.35:
            issues.append(f"[TOPIC COLLAPSE] High maximum topic collinearity detected ({latest['topic_max_overlap']:.3f} > 0.35). Some metaprograms are redundant.")
        else:
            strengths.append(f"[ORTHOGONALITY] Topic dictionary is well-separated (Max overlap: {latest.get('topic_max_overlap', 0):.3f}).")

        # Audit 3: Effective Rank Ratio
        if "topic_effective_rank_ratio" in latest:
            ratio = latest["topic_effective_rank_ratio"]
            if ratio < 0.50:
                issues.append(f"[DIMENSIONALITY] Dictionary effective rank ratio is {ratio:.1%} (< 50%). Model is only utilizing a fraction of its K topic capacity.")
            else:
                strengths.append(f"[RANK] Effective topic utilization is healthy ({ratio:.1%} of full K capacity).")

        # Audit 4: Attention Physics
        if "gat_att_temp" in latest:
            tau = latest["gat_att_temp"]
            if tau < 0.08:
                issues.append(f"[GAT ATTENTION FREEZE] Attention temperature tau={tau:.3f} is near minimum clamp. GATv2 is acting almost deterministically.")
            elif tau > 3.0:
                issues.append(f"[GAT ATTENTION DILUTION] Attention temperature tau={tau:.3f} is very soft. Graph weights approach uniform average.")
            else:
                strengths.append(f"[GAT DYNAMICS] Attention temperature is in optimal regime (tau={tau:.3f}).")

        # Audit 5: Edge Decay
        if "gamma_decay" in latest:
            gamma = latest["gamma_decay"]
            if gamma < 0.10:
                issues.append(f"[SPATIAL BLUR] Softplus(gamma)={gamma:.3f} is low; bilateral distance decay is barely pruning transcriptomic edges.")
            elif gamma > 10.0:
                issues.append(f"[GRAPH SEVERING] Softplus(gamma)={gamma:.3f} is aggressive; spatial neighbors are excessively downweighted.")
            else:
                strengths.append(f"[BILATERAL GRAPH] Spatial decay parameter gamma={gamma:.3f} maintains structured graph propagation.")

        # Audit 6: Transformer Gate Balance
        if "gate_to_id_ratio" in latest:
            ratio = latest["gate_to_id_ratio"]
            if ratio < 0.10:
                issues.append(f"[CONTEXT GATE SUPPRESSED] Gate-to-ID ratio is {ratio:.3f}. Spatial context is being largely ignored in favor of cell intrinsic identity.")
            elif ratio > 5.0:
                issues.append(f"[IDENTITY DROWNED] Gate-to-ID ratio is {ratio:.3f}. Spatial neighborhood is dominating intrinsic single-cell state.")
            else:
                strengths.append(f"[CROSS-ATTENTION] Balanced Context Gate vs Intrinsic Identity (Ratio: {ratio:.3f}).")

        # Audit 7: QKV Conditioning
        if "q_proj_cond" in latest and latest["q_proj_cond"] > 100.0:
            issues.append(f"[ILL-CONDITIONED Q] Query projection condition number is high ({latest['q_proj_cond']:.1f}). Potential gradient instability in cross-attention.")

        # Print findings
        for s in strengths:
            print(f"  [✓ PASS] {s}")
        if not issues:
            print("  [✓ EXCELLENT] No critical training dysfunctions or collapses detected.")
        else:
            for iss in issues:
                print(f"  [!] {iss}")

        print("=" * 90 + "\n")

    def generate_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot a publication-quality 12-panel autopsy diagnostic matrix."""
        print(f"[➤] Generating diagnostic visual dashboard: {output_path.name}...")
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.28)

        epochs = df["epoch"].to_numpy()

        # 1. Loss Dynamics
        ax1 = fig.add_subplot(gs[0, 0])
        if "train_loss" in df and not df["train_loss"].isna().all():
            ax1.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        if "val_loss" in df and not df["val_loss"].isna().all():
            ax1.plot(epochs, df["val_loss"], label="Val Loss", color="#d62728", lw=2, ls="--")
        if "l_rec" in df and not df["l_rec"].isna().all():
            ax1.plot(epochs, df["l_rec"], label="Pure Recon (L_rec)", color="#2ca02c", lw=1.5, ls=":")
        ax1.set_title("1. Loss & Reconstruction Trajectory", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")

        # 2. Regularization Balances
        ax2 = fig.add_subplot(gs[0, 1])
        if "l_anc" in df and not df["l_anc"].isna().all():
            ax2.plot(epochs, df["l_anc"], label="Anchor Loss (L_anc)", color="#9467bd", lw=1.8)
        if "l_ort" in df and not df["l_ort"].isna().all():
            ax2.plot(epochs, df["l_ort"], label="Ortho Penalty (L_ort)", color="#8c564b", lw=1.8)
        if "kl_weight" in df and not df["kl_weight"].isna().all():
            ax2_twin = ax2.twinx()
            ax2_twin.plot(epochs, df["kl_weight"], label="Dynamic KL Weight", color="#e377c2", lw=1.5, ls="--")
            ax2_twin.set_ylabel("KL Weight", color="#e377c2")
            ax2_twin.grid(False)
        ax2.set_title("2. Regularization & Loss Anchoring", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Penalty Scale")
        ax2.legend(loc="upper left")

        # 3. Cell & Gene Confidence Trajectories (P_W & G_W)
        ax3 = fig.add_subplot(gs[0, 2])
        if "p_w" in df and not df["p_w"].isna().all():
            ax3.plot(epochs, df["p_w"], label="Cell Topic Sharpness (P_W)", color="#ff7f0e", lw=2.2)
        if "g_w" in df and not df["g_w"].isna().all():
            ax3.plot(epochs, df["g_w"], label="Gene Anchor Peak (G_W)", color="#17becf", lw=2.2)
        ax3.axhline(70.0, color="#ff7f0e", ls=":", alpha=0.6, label="P_W Target (70%)")
        ax3.set_title("3. Cell & Gene Sharpness Squeeze", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Max Peak Probability (%)")
        ax3.legend(loc="lower right")

        # 4. Dictionary Orthogonality & Collinearity
        ax4 = fig.add_subplot(gs[1, 0])
        if "topic_max_overlap" in df:
            ax4.plot(epochs, df["topic_max_overlap"], label="Max Pairwise Overlap", color="#d62728", lw=2)
        if "topic_mean_overlap" in df:
            ax4.plot(epochs, df["topic_mean_overlap"], label="Mean Topic Overlap", color="#7f7f7f", lw=1.5)
        ax4.axhline(0.25, color="red", ls="--", alpha=0.5, label="Target Ortho Bound (0.25)")
        ax4.set_title("4. Topic Separation & Redundancy", fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Cosine Overlap")
        ax4.legend(loc="upper left")

        # 5. Effective Rank of Metaprograms
        ax5 = fig.add_subplot(gs[1, 1])
        if "topic_effective_rank" in df:
            ax5.plot(epochs, df["topic_effective_rank"], label="Effective Topic Rank", color="#2ca02c", lw=2.2)
            if "K_topics" in df:
                k_val = df["K_topics"].iloc[0]
                ax5.axhline(k_val, color="black", ls=":", alpha=0.5, label=f"Max K Capacity ({k_val})")
        ax5.set_title("5. Dictionary Dimensionality Health", fontweight="bold")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Continuous Effective Rank")
        ax5.legend(loc="lower right")

        # 6. Prior Anchor Drift & Spatial Bridge
        ax6 = fig.add_subplot(gs[1, 2])
        if "anchor_drift" in df:
            ax6.plot(epochs, df["anchor_drift"], label="Anchor Drift from T0", color="#9467bd", lw=2)
        if "spatial_bridge_norm" in df:
            ax6_twin = ax6.twinx()
            ax6_twin.plot(epochs, df["spatial_bridge_norm"], label="Spatial Bridge ||W||", color="#bcbd22", lw=1.5, ls="--")
            ax6_twin.set_ylabel("Bridge L2 Norm", color="#bcbd22")
            ax6_twin.grid(False)
        ax6.set_title("6. Dictionary Plasticity & Spatial Shift", fontweight="bold")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("1 - Cosine(T, T0)")
        ax6.legend(loc="upper left")

        # 7. GATv2 Attention Temperature & Graph Edge Decay
        ax7 = fig.add_subplot(gs[2, 0])
        if "gat_att_temp" in df:
            ax7.plot(epochs, df["gat_att_temp"], label="GAT Temp (Tau)", color="#e377c2", lw=2)
        if "gamma_decay" in df:
            ax7.plot(epochs, df["gamma_decay"], label="Edge Decay (Gamma)", color="#8c564b", lw=2, ls="-.")
        ax7.set_title("7. GATv2 Attention & Edge Physics", fontweight="bold")
        ax7.set_xlabel("Epoch")
        ax7.set_ylabel("Parameter Value")
        ax7.legend(loc="upper right")

        # 8. Transformer Attention Q, K, V Norms
        ax8 = fig.add_subplot(gs[2, 1])
        if "q_proj_norm" in df:
            ax8.plot(epochs, df["q_proj_norm"], label="||Q_proj||", lw=1.8)
        if "k_proj_norm" in df:
            ax8.plot(epochs, df["k_proj_norm"], label="||K_proj||", lw=1.8)
        if "v_proj_norm" in df:
            ax8.plot(epochs, df["v_proj_norm"], label="||V_proj||", lw=1.8)
        if "gat_w_src_norm" in df:
            ax8.plot(epochs, df["gat_w_src_norm"], label="||GAT_Src||", lw=1.2, ls=":")
        ax8.set_title("8. GNN & Attention Weight Norms", fontweight="bold")
        ax8.set_xlabel("Epoch")
        ax8.set_ylabel("Frobenius Norm")
        ax8.legend(loc="upper left")

        # 9. Context Gate vs Single-Cell Identity Ratio
        ax9 = fig.add_subplot(gs[2, 2])
        if "gate_to_id_ratio" in df:
            ax9.plot(epochs, df["gate_to_id_ratio"], label="Gate-to-ID Ratio", color="#17becf", lw=2)
            ax9.axhline(1.0, color="gray", ls="--", alpha=0.6, label="Parity (1.0)")
            ax9.set_yscale("log")
        ax9.set_title("9. Spatial Context vs Cell-Identity Gate", fontweight="bold")
        ax9.set_xlabel("Epoch")
        ax9.set_ylabel("Ratio (Gate / ID) [Log Scale]")
        ax9.legend(loc="upper right")

        # 10. Topic Dominance / Monopolization
        ax10 = fig.add_subplot(gs[3, 0])
        if "top_topic_pct" in df:
            ax10.plot(epochs, df["top_topic_pct"], label="Dominant Topic %", color="#d62728", lw=2)
            ax10.axhline(25.0, color="orange", ls=":", label="Warning Threshold (25%)")
            ax10.axhline(40.0, color="red", ls="--", label="Collapse Threshold (40%)")
        ax10.set_title("10. Topic Monopolization Risk", fontweight="bold")
        ax10.set_xlabel("Epoch")
        ax10.set_ylabel("Global Proportion of Cells (%)")
        ax10.legend(loc="upper left")

        # 11. EMA Entropy & Collapse Ratio
        ax11 = fig.add_subplot(gs[3, 1])
        if "entropy" in df and not df["entropy"].isna().all():
            ax11.plot(epochs, df["entropy"], label="Global Topic Entropy", color="#3399e6", lw=2)
        if "collapse_ratio" in df and not df["collapse_ratio"].isna().all():
            ax11_twin = ax11.twinx()
            ax11_twin.plot(epochs, df["collapse_ratio"], label="Collapse Ratio", color="#e63333", lw=1.5, ls="--")
            ax11_twin.set_ylabel("Collapse Ratio", color="#e63333")
            ax11_twin.grid(False)
        ax11.set_title("11. Information Entropy Dynamics", fontweight="bold")
        ax11.set_xlabel("Epoch")
        ax11.set_ylabel("Shannon Entropy (Nats)")
        ax11.legend(loc="lower left")

        # 12. Cross-Attention Matrix Condition Numbers
        ax12 = fig.add_subplot(gs[3, 2])
        if "q_proj_cond" in df:
            ax12.plot(epochs, df["q_proj_cond"], label="Cond(Q)", color="#1f77b4", lw=1.5)
        if "k_proj_cond" in df:
            ax12.plot(epochs, df["k_proj_cond"], label="Cond(K)", color="#ff7f0e", lw=1.5)
        if "v_proj_cond" in df:
            ax12.plot(epochs, df["v_proj_cond"], label="Cond(V)", color="#2ca02c", lw=1.5)
        ax12.set_yscale("log")
        ax12.set_title("12. Linear Head Conditioning (Ill-Conditioning Check)", fontweight="bold")
        ax12.set_xlabel("Epoch")
        ax12.set_ylabel("Condition Number (Log)")
        ax12.legend(loc="upper left")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[✓] Multi-panel visual report saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Libella Trajectory Health Diagnostic Autopsy")
    parser.add_argument(
        "--ckpt-dir", 
        type=str, 
        default="/Users/Hemato/project_3/benchmark/libella_output/run/autopsy_checkpoints",
        help="Path to the autopsy checkpoints directory"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save the autopsy reports (defaults to ckpt-dir parent)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ckpt_dir.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = out_dir / "trajectory_autopsy_metrics.csv"
    plot_png = out_dir / "trajectory_autopsy_dashboard.png"

    autopsy = TrajectoryAutopsy(ckpt_dir)
    df = autopsy.run_autopsy()

    # Save metrics CSV
    df.to_csv(report_csv, index=False)
    print(f"[✓] Full trajectory metrics saved to: {report_csv}")

    # Generate terminal report and diagnostic chart
    autopsy.print_health_report(df)
    autopsy.generate_dashboard(df, plot_png)


if __name__ == "__main__":
    main()