#!/usr/bin/env python3
"""
Week 5: Plot Tracking Metrics

Generates visualization plots from tracking summary files:
- RMSE vs frame index
- Translation components over time
- Rotation angle over time
- Expression norm over time

Usage:
    python scripts/plot_tracking_metrics.py --seq 01
    python scripts/plot_tracking_metrics.py --seq 01 17 --output-dir outputs/analysis/plots
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will generate text summary only")

import numpy as np


def load_tracking_summary(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load tracking summary from JSON file."""
    if not json_path.exists():
        print(f"Warning: Summary file not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_tracking_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load tracking metrics from CSV file."""
    if not csv_path.exists():
        return []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return []


def plot_rmse_vs_frame(frames: List[Dict], output_path: Path, seq_id: str):
    """Plot RMSE vs frame index."""
    if not HAS_MATPLOTLIB:
        return
    
    frame_idx = [int(f.get('frame_idx', i)) for i, f in enumerate(frames)]
    rmse = [float(f.get('face_nn_rmse_mm', 0)) for f in frames]
    reinit = [f.get('was_reinit', 'False') == 'True' for f in frames]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frame_idx, rmse, 'b-o', linewidth=2, markersize=6, label='Face NN-RMSE')
    
    # Mark re-initialization points
    reinit_idx = [i for i, r in zip(frame_idx, reinit) if r]
    reinit_rmse = [rmse[frame_idx.index(i)] for i in reinit_idx]
    if reinit_idx:
        ax.scatter(reinit_idx, reinit_rmse, c='red', s=100, marker='x', 
                   linewidths=3, label='Re-initialization', zorder=5)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('RMSE (mm)', fontsize=12)
    ax.set_title(f'Face NN-RMSE vs Frame - Sequence {seq_id}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add mean line
    mean_rmse = np.mean(rmse)
    ax.axhline(y=mean_rmse, color='gray', linestyle='--', alpha=0.7, 
               label=f'Mean: {mean_rmse:.2f} mm')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_translation_vs_frame(frames: List[Dict], output_path: Path, seq_id: str):
    """Plot translation components vs frame index."""
    if not HAS_MATPLOTLIB:
        return
    
    frame_idx = [int(f.get('frame_idx', i)) for i, f in enumerate(frames)]
    tx = [float(f.get('translation_x', 0)) for f in frames]
    ty = [float(f.get('translation_y', 0)) for f in frames]
    tz = [float(f.get('translation_z', 0)) for f in frames]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(frame_idx, tx, 'r-o', linewidth=2, markersize=4)
    axes[0].set_ylabel('Translation X (m)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(frame_idx, ty, 'g-o', linewidth=2, markersize=4)
    axes[1].set_ylabel('Translation Y (m)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(frame_idx, tz, 'b-o', linewidth=2, markersize=4)
    axes[2].set_ylabel('Translation Z (m)', fontsize=11)
    axes[2].set_xlabel('Frame Index', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Pose Translation vs Frame - Sequence {seq_id}', fontsize=14)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_rotation_vs_frame(frames: List[Dict], output_path: Path, seq_id: str):
    """Plot rotation angle vs frame index."""
    if not HAS_MATPLOTLIB:
        return
    
    frame_idx = [int(f.get('frame_idx', i)) for i, f in enumerate(frames)]
    rot_angle = [float(f.get('rotation_angle_deg', 0)) for f in frames]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frame_idx, rot_angle, 'purple', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Rotation Angle (degrees)', fontsize=12)
    ax.set_title(f'Rotation Angle from Identity vs Frame - Sequence {seq_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_expression_vs_frame(frames: List[Dict], output_path: Path, seq_id: str):
    """Plot expression coefficient norm vs frame index."""
    if not HAS_MATPLOTLIB:
        return
    
    frame_idx = [int(f.get('frame_idx', i)) for i, f in enumerate(frames)]
    expr_norm = [float(f.get('expression_norm', 0)) for f in frames]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frame_idx, expr_norm, 'orange', linewidth=2, marker='o', markersize=4)
    ax.fill_between(frame_idx, expr_norm, alpha=0.3, color='orange')
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Expression Coefficient Norm', fontsize=12)
    ax.set_title(f'Expression Magnitude vs Frame - Sequence {seq_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_metrics(frames: List[Dict], output_path: Path, seq_id: str):
    """Create a combined metrics overview plot."""
    if not HAS_MATPLOTLIB:
        return
    
    frame_idx = [int(f.get('frame_idx', i)) for i, f in enumerate(frames)]
    rmse = [float(f.get('face_nn_rmse_mm', 0)) for f in frames]
    tz = [float(f.get('translation_z', 0)) for f in frames]
    rot_angle = [float(f.get('rotation_angle_deg', 0)) for f in frames]
    expr_norm = [float(f.get('expression_norm', 0)) for f in frames]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE
    axes[0, 0].plot(frame_idx, rmse, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_ylabel('RMSE (mm)')
    axes[0, 0].set_title('Face NN-RMSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Translation Z (depth)
    axes[0, 1].plot(frame_idx, tz, 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_ylabel('Translation Z (m)')
    axes[0, 1].set_title('Depth (Z Translation)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rotation
    axes[1, 0].plot(frame_idx, rot_angle, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Rotation (deg)')
    axes[1, 0].set_title('Rotation Angle')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Expression
    axes[1, 1].plot(frame_idx, expr_norm, 'orange', linewidth=2, marker='o', markersize=4)
    axes[1, 1].fill_between(frame_idx, expr_norm, alpha=0.3, color='orange')
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('Expression Norm')
    axes[1, 1].set_title('Expression Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Tracking Metrics Overview - Sequence {seq_id}', fontsize=16)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_text_summary(frames: List[Dict], summary: Dict, seq_id: str) -> str:
    """Generate a text summary of tracking metrics."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"TRACKING SUMMARY - Sequence {seq_id}")
    lines.append("=" * 60)
    lines.append("")
    
    num_frames = len(frames)
    lines.append(f"Total frames: {num_frames}")
    
    if frames:
        rmse = [float(f.get('face_nn_rmse_mm', 0)) for f in frames]
        lines.append(f"RMSE (mm):")
        lines.append(f"  Mean:   {np.mean(rmse):.2f}")
        lines.append(f"  Std:    {np.std(rmse):.2f}")
        lines.append(f"  Min:    {np.min(rmse):.2f}")
        lines.append(f"  Max:    {np.max(rmse):.2f}")
        
        lines.append("")
        
        reinit_count = sum(1 for f in frames if f.get('was_reinit', 'False') == 'True')
        lines.append(f"Re-initializations: {reinit_count}")
        
        lines.append("")
        lines.append("Per-frame metrics:")
        lines.append("-" * 50)
        lines.append(f"{'Frame':<8} {'RMSE(mm)':<12} {'Trans Z(m)':<12} {'Rot(deg)':<10}")
        lines.append("-" * 50)
        
        for f in frames[:20]:  # Show first 20 frames
            idx = f.get('frame_idx', '?')
            rmse_val = float(f.get('face_nn_rmse_mm', 0))
            tz = float(f.get('translation_z', 0))
            rot = float(f.get('rotation_angle_deg', 0))
            lines.append(f"{idx:<8} {rmse_val:<12.2f} {tz:<12.4f} {rot:<10.2f}")
        
        if num_frames > 20:
            lines.append(f"... ({num_frames - 20} more frames)")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plot tracking metrics")
    parser.add_argument("--seq", type=str, nargs="+", default=["01"],
                        help="Sequence ID(s) to plot")
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/analysis"),
                        help="Analysis output root directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: analysis_root/plots)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.analysis_root / "plots"
    
    print("=" * 60)
    print("Week 5: Tracking Metrics Plotter")
    print("=" * 60)
    print()
    
    for seq_id in args.seq:
        print(f"Processing sequence {seq_id}...")
        
        # Load data
        json_path = args.analysis_root / f"tracking_summary_{seq_id}.json"
        csv_path = args.analysis_root / f"tracking_summary_{seq_id}.csv"
        
        summary = load_tracking_summary(json_path)
        frames = load_tracking_csv(csv_path)
        
        if not frames and summary:
            frames = summary.get('frames', [])
        
        if not frames:
            print(f"  No tracking data found for sequence {seq_id}")
            continue
        
        print(f"  Loaded {len(frames)} frames")
        
        # Create output directory
        plot_dir = args.output_dir / seq_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        if HAS_MATPLOTLIB:
            plot_rmse_vs_frame(frames, plot_dir / "rmse_vs_frame.png", seq_id)
            plot_translation_vs_frame(frames, plot_dir / "translation_vs_frame.png", seq_id)
            plot_rotation_vs_frame(frames, plot_dir / "rotation_vs_frame.png", seq_id)
            plot_expression_vs_frame(frames, plot_dir / "expression_vs_frame.png", seq_id)
            plot_combined_metrics(frames, plot_dir / "metrics_overview.png", seq_id)
        
        # Generate text summary
        text_summary = generate_text_summary(frames, summary or {}, seq_id)
        print(text_summary)
        
        # Save text summary
        summary_path = plot_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(text_summary)
        print(f"Saved: {summary_path}")
        
        print()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
