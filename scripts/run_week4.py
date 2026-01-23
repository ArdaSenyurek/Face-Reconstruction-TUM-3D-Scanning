#!/usr/bin/env python3
"""
Week 4: Complete Pipeline Runner

Runs the full Week 4 pipeline:
1. Rigid alignment with sanity checks
2. Overlay generation
3. Single-frame optimization
4. Weight tuning
5. Metrics collection
"""

import subprocess
import json
from pathlib import Path
import sys

def main():
    print("=" * 60)
    print("Week 4: Complete Pipeline Execution")
    print("=" * 60)
    print()
    
    # Check if BFM model exists
    bfm_model = Path("data/bfm/model2019_fullHead.h5")
    if not bfm_model.exists():
        print("ERROR: BFM model not found!")
        print("Please download model2019_fullHead.h5 and place it in data/bfm/")
        print("See README.md for instructions.")
        return 1
    
    # Step 1: Run pipeline with overlays
    print("Step 1: Running pipeline with pose init and overlays...")
    print("  Sequences: 01, 17")
    print("  Frame: frame_00000")
    print()
    
    cmd = [
        "python3", "pipeline/main.py",
        "--skip-download",  # Assume data already downloaded
        "--skip-convert",   # Assume already converted
        "--skip-model-setup",  # Assume model already set up
        "--frames", "1",  # Only first frame
        "--make-overlays",
        "--target-sequences", "01", "17",
        "--optimize",
        "--max-iterations", "20",
        "--verbose-optimize",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("ERROR: Pipeline execution failed")
        return 1
    
    print()
    print("=" * 60)
    print("Step 2: Verifying deliverables...")
    print("=" * 60)
    print()
    
    # Check deliverables
    deliverables = {
        "Rigid alignment reports": [
            "outputs/pose_init/01/frame_00000_rigid_report.json",
            "outputs/pose_init/17/frame_00000_rigid_report.json",
        ],
        "3D PLY overlays": [
            "outputs/overlays3d/01_frame_00000_mesh_scan_overlay.ply",
            "outputs/overlays3d/17_frame_00000_mesh_scan_overlay.ply",
        ],
        "2D PNG overlays": [
            "outputs/overlays2d/01_frame_00000_overlay.png",
            "outputs/overlays2d/17_frame_00000_overlay.png",
        ],
        "Depth comparison PNGs": [
            "outputs/depth_overlay/01_frame_00000_depth_compare.png",
            "outputs/depth_overlay/17_frame_00000_depth_compare.png",
        ],
        "Optimized meshes": [
            "outputs/meshes/01/frame_00000.ply",
            "outputs/meshes/17/frame_00000.ply",
        ],
    }
    
    all_found = True
    for category, files in deliverables.items():
        print(f"{category}:")
        for file_path in files:
            path = Path(file_path)
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path}")
            if not exists:
                all_found = False
        print()
    
    if not all_found:
        print("WARNING: Some deliverables are missing!")
        print("Check pipeline logs for errors.")
    else:
        print("✓ All deliverables found!")
    
    print()
    print("=" * 60)
    print("Deliverable Locations:")
    print("=" * 60)
    print()
    print("Rigid alignment reports:")
    print("  outputs/pose_init/{seq}/frame_00000_rigid_report.json")
    print()
    print("3D PLY overlays:")
    print("  outputs/overlays3d/{seq}_frame_00000_mesh_scan_overlay.ply")
    print()
    print("2D PNG overlays:")
    print("  outputs/overlays2d/{seq}_frame_00000_overlay.png")
    print()
    print("Depth comparison PNGs:")
    print("  outputs/depth_overlay/{seq}_frame_00000_depth_compare.png")
    print()
    print("Optimized meshes:")
    print("  outputs/meshes/{seq}/frame_00000.ply")
    print()
    
    return 0 if all_found else 1

if __name__ == "__main__":
    sys.exit(main())
