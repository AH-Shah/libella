import torch
import torch.nn.functional as F
from pathlib import Path
from libella.model import LibellaGNN # Adjust to your import
import numpy as np

def audit_graph_attention(checkpoint_path: str):
    print("\n[*] Loading Checkpoint for Graph Audit...")
    device = torch.device('cpu')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = LibellaGNN(in_channels=ckpt['model_state_dict']['decoder_weight'].shape[1], 
                       n_metaprograms=ckpt['model_state_dict']['decoder_weight'].shape[0])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"\n[1] Spatial Gain: {model.spatial_gain.item():.6f}")
    if model.spatial_gain.item() < 0.01:
        print("    -> VERDICT: The model turned the spatial highway OFF.")

    # Calculate actual temperatures
    gat_tau = F.softplus(model.att_temp).item()
    cross_tau = F.softplus(model.cross_temp).item() + 0.1
    print(f"\n[2] Attention Temperatures:")
    print(f"    -> GATv2 Tau: {gat_tau:.4f} (Ideal for sharpness: < 0.2)")
    print(f"    -> Cross-Attention Tau: {cross_tau:.4f} (Ideal for sharpness: < 0.2)")

    if gat_tau > 0.5:
        print("    -> VERDICT: Softmax is creating a uniform, blurry mush.")

if __name__ == "__main__":
    audit_graph_attention("/Users/Hemato/project_3/benchmark/libella_output/run_discovery/gnn_checkpoint.pt")