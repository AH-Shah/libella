import argparse
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Stress Test the Two-Phase Dynamic Tracker")
    parser.add_argument("manifest", type=str, help="Path to your manifest CSV (e.g., 500k cells or 3.8M cells)")
    parser.add_argument("--script", type=str, default="main.py", help="Your main entry script (e.g., main.py or module name)")
    parser.add_argument("--out-dir", type=str, default="./libella_calibration_test", help="Isolated output directory")
    parser.add_argument("--max-epochs", type=int, default=300, help="Give the tracker plenty of room")
    return parser.parse_args()

def run_and_capture(args):
    """Runs the Libella CLI and parses the tracker logs in real-time."""
    cmd = [
        "python", args.script, args.manifest,
        "--out-dir", args.out_dir,
        "--force-retrain",
        "--epochs", str(args.max_epochs),
        "--phase", "TRAIN"  # Only run up to training for the stress test
    ]
    
    print(f"🚀 Launching Stress Test: {' '.join(cmd)}\n")
    
    # Regex parsers based on your specific log format
    epoch_regex = re.compile(
        r"\[Ep\s+(\d+)\]\s+Pure_Rec:([\d.]+)\s+V_Loss:([\d.]+).*?P_W:([\d.]+)%.*?L_Anc:([\d.]+)"
    )
    
    tracker_data = []
    phase_2_start = None
    stop_epoch = None
    
    # Run the process and read output line-by-line
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line, end="") # Echo to console
        
        # 1. Parse Epoch Data
        match = epoch_regex.search(line)
        if match:
            tracker_data.append({
                "Epoch": int(match.group(1)),
                "Pure_Rec": float(match.group(2)),
                "V_Loss": float(match.group(3)),
                "P_W": float(match.group(4)),
                "L_Anc": float(match.group(5))
            })
            
        # 2. Track Phase Transitions
        if "[🚀] Phase 1 Complete" in line:
            # The last recorded epoch is where Phase 1 ended
            phase_2_start = tracker_data[-1]["Epoch"] if tracker_data else 0
            
        if "[✅] Sparsification complete" in line or "Forced Phase 2 transition" in line:
            stop_epoch = tracker_data[-1]["Epoch"] if tracker_data else 0

    process.wait()
    
    if not tracker_data:
        print("\n[!] No training data parsed. Did the pipeline crash before Epoch 1?")
        return
        
    df = pd.DataFrame(tracker_data)
    plot_calibration(df, phase_2_start, stop_epoch, Path(args.out_dir))

def plot_calibration(df: pd.DataFrame, phase_2_start: int, stop_epoch: int, out_dir: Path):
    """Generates a visual report of the Phase Tracker's behavior."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_report.png"
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- Left Axis: Losses (Representation) ---
    ax1.set_xlabel("Epoch", fontweight='bold')
    ax1.set_ylabel("Loss", color='tab:blue', fontweight='bold')
    l1 = ax1.plot(df["Epoch"], df["Pure_Rec"], color='tab:blue', linewidth=2.5, label="Pure Rec")
    l2 = ax1.plot(df["Epoch"], df["V_Loss"], color='tab:cyan', linestyle='--', alpha=0.6, label="Val Loss")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # --- Right Axis: Sparsification (P_W & L_Anc) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Cell Purity (P_W %) / Anchor Drift", color='tab:red', fontweight='bold')
    l3 = ax2.plot(df["Epoch"], df["P_W"], color='tab:red', linewidth=2.5, label="Cell Purity (P_W %)")
    l4 = ax2.plot(df["Epoch"], df["L_Anc"] * 100, color='tab:orange', linestyle=':', linewidth=2, label="Anchor Drift (Scaled)")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # --- Tracker Visualizations ---
    if phase_2_start is not None:
        ax1.axvline(x=phase_2_start, color='green', linestyle='--', linewidth=2, zorder=0)
        ax1.text(phase_2_start + 1, df["Pure_Rec"].max(), "🚀 Phase 2\n(Sparsification)", color='green', fontweight='bold')
        ax1.axvspan(0, phase_2_start, color='gray', alpha=0.1) # Shaded Phase 1
        
    if stop_epoch is not None:
        ax1.axvline(x=stop_epoch, color='black', linestyle='-.', linewidth=2, zorder=0)
        ax1.text(stop_epoch + 1, df["Pure_Rec"].mean(), "✅ Terminated", color='black', fontweight='bold')

    # Legends
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title(f"Tracker Calibration Report | Final P_W: {df['P_W'].iloc[-1]}%", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    
    print(f"\n" + "="*50)
    print(f"📊 CALIBRATION REPORT SAVED TO: {out_path}")
    print("="*50)
    print(f"Final Pure_Rec: {df['Pure_Rec'].iloc[-1]:.3f}")
    print(f"Final Cell Purity (P_W): {df['P_W'].iloc[-1]:.2f}%")
    if phase_2_start:
        print(f"Phase 2 Triggered At: Epoch {phase_2_start}")
    print("="*50)

if __name__ == "__main__":
    run_and_capture(parse_args())