#!/usr/bin/env python3
"""
Week 5 Demo Script - Sequential Face Tracking with Temporal Smoothing
====================================================================

This script demonstrates the results of temporal face tracking over 5 consecutive frames.
To view results, open the generated PLY files in MeshLab or CloudCompare.

Usage:
    python demo_week5.py

Outputs:
    - Tracked meshes per frame
    - 3D mesh-scan overlays (red mesh + cyan point cloud)  
    - Tracking metrics plot
    - Summary report
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("Week 5: Sequential Face Tracking Demo")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    outputs_dir = base_dir / "outputs"
    
    # Check if tracking results exist
    tracking_summary = outputs_dir / "analysis" / "tracking_summary_01.json"
    if not tracking_summary.exists():
        print("❌ No tracking results found!")
        print("Run: python pipeline/main.py --track --optimize --frames 5 --target-sequences 01")
        return
    
    # Load tracking metrics
    with open(tracking_summary) as f:
        data = json.load(f)
    
    print(f"✓ Found tracking data for {data['num_frames']} frames")
    
    # Display key results
    print("\n📊 TRACKING METRICS:")
    print("-" * 40)
    frames = data['frames']
    
    print(f"{'Frame':<8} {'Transl-X':<10} {'Transl-Y':<10} {'Transl-Z':<10} {'Rot-Angle':<12} {'Scale':<10}")
    print("-" * 70)
    
    for frame in frames:
        print(f"{frame['frame_idx']:<8} "
              f"{frame['translation_x']:<10.4f} "
              f"{frame['translation_y']:<10.4f} "
              f"{frame['translation_z']:<10.4f} "
              f"{frame['rotation_angle_deg']:<12.1f} "
              f"{frame['scale']:<10.6f}")
    
    # Check available visualization files
    print("\n📁 AVAILABLE FILES:")
    print("-" * 40)
    
    meshes_dir = outputs_dir / "meshes" / "01"
    overlays_dir = outputs_dir / "overlays_3d" / "01"
    
    tracked_meshes = sorted(meshes_dir.glob("*_tracked.ply"))
    rigid_overlays = sorted(overlays_dir.glob("*_overlay_rigid.ply"))
    opt_overlays = sorted(overlays_dir.glob("*_overlay_opt.ply"))
    
    print(f"🔹 Tracked meshes: {len(tracked_meshes)} files")
    for mesh in tracked_meshes:
        print(f"   → {mesh}")
    
    print(f"\n🔹 3D Mesh-Scan Overlays (Rigid): {len(rigid_overlays)} files")
    for overlay in rigid_overlays[:3]:  # Show first 3
        print(f"   → {overlay}")
    if len(rigid_overlays) > 3:
        print(f"   ... and {len(rigid_overlays) - 3} more")
    
    print(f"\n🔹 3D Mesh-Scan Overlays (Optimized): {len(opt_overlays)} files")
    for overlay in opt_overlays[:3]:  # Show first 3
        print(f"   → {overlay}")
    if len(opt_overlays) > 3:
        print(f"   ... and {len(opt_overlays) - 3} more")
    
    # Generate tracking plot
    try:
        create_tracking_plot(frames, outputs_dir)
        print(f"\n📈 Tracking plot saved to: outputs/analysis/tracking_plot.png")
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("HOW TO VIEW RESULTS:")
    print("=" * 60)
    print("1. Open MeshLab or CloudCompare")
    print("2. Load any overlay PLY file (e.g., frame_00000_overlay_opt.ply)")
    print("3. You should see:")
    print("   🔴 Red mesh (reconstructed face)")
    print("   🔵 Cyan/blue points (RGB-D scan)")
    print("4. Compare different frames to see temporal tracking")
    print("\n✨ Key Achievement:")
    print("   Sequential frames maintain pose continuity with temporal smoothing!")
    print("=" * 60)

def create_tracking_plot(frames, outputs_dir):
    """Create tracking metrics visualization"""
    frame_indices = [f['frame_idx'] for f in frames]
    translations_x = [f['translation_x'] for f in frames]
    translations_z = [f['translation_z'] for f in frames]
    rotation_angles = [f['rotation_angle_deg'] for f in frames]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Translation plot
    ax1.plot(frame_indices, translations_x, 'r-o', label='Translation X')
    ax1.plot(frame_indices, translations_z, 'b-o', label='Translation Z')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Translation (meters)')
    ax1.set_title('Head Translation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation plot
    ax2.plot(frame_indices, rotation_angles, 'g-o', label='Rotation Angle')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Rotation Angle (degrees)')
    ax2.set_title('Head Rotation Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = outputs_dir / "analysis" / "tracking_plot.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()