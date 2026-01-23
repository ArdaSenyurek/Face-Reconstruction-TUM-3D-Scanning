#!/usr/bin/env python3
"""
Week 4: Weight Tuning Script

Performs grid search over regularization weights and evaluates stability.
"""

import subprocess
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def run_reconstruction(rgb_path: Path, depth_path: Path, intrinsics_path: Path,
                      landmarks_path: Path, mapping_path: Path, model_dir: Path,
                      output_mesh: Path, lambda_landmark: float, lambda_depth: float,
                      lambda_reg: float, max_iter: int = 20) -> Dict:
    """Run reconstruction with given weights and return metrics."""
    binary = Path("build/bin/face_reconstruction")
    if not binary.exists():
        return {"error": "Binary not found"}
    
    cmd = [
        str(binary),
        "--rgb", str(rgb_path),
        "--depth", str(depth_path),
        "--intrinsics", str(intrinsics_path),
        "--landmarks", str(landmarks_path),
        "--mapping", str(mapping_path),
        "--model-dir", str(model_dir),
        "--output-mesh", str(output_mesh),
        "--optimize",
        "--max-iter", str(max_iter),
        "--lambda-landmark", str(lambda_landmark),
        "--lambda-depth", str(lambda_depth),
        "--lambda-reg", str(lambda_reg),
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        
        # Parse output for metrics (simplified - would need better parsing)
        output = result.stdout + result.stderr
        metrics = {
            "converged": "Converged" in output or "converged" in output.lower(),
            "iterations": max_iter,
            "success": result.returncode == 0 and output_mesh.exists(),
        }
        
        # Try to extract energy values from output
        for line in output.split('\n'):
            if "Final energy" in line:
                try:
                    metrics["final_energy"] = float(line.split(":")[-1].strip())
                except:
                    pass
            if "Landmark energy" in line:
                try:
                    metrics["landmark_energy"] = float(line.split(":")[-1].strip())
                except:
                    pass
            if "Depth energy" in line:
                try:
                    metrics["depth_energy"] = float(line.split(":")[-1].strip())
                except:
                    pass
        
        return metrics
    except subprocess.TimeoutExpired:
        return {"error": "Timeout", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

def main():
    if len(sys.argv) < 3:
        print("Usage: python week4_weight_tuning.py <seq01_dir> <seq17_dir>")
        print("  seq01_dir: Path to outputs/converted/01/")
        print("  seq17_dir: Path to outputs/converted/17/")
        sys.exit(1)
    
    seq01_dir = Path(sys.argv[1])
    seq17_dir = Path(sys.argv[2])
    
    model_dir = Path("data/model_bfm")
    mapping_path = Path("data/landmark_mapping.txt")
    landmarks_root = Path("outputs/landmarks")
    output_root = Path("outputs/weight_tuning")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Weight grid
    lambda_landmarks = [0.5, 1.0, 2.0]
    lambda_depths = [0.05, 0.1, 0.2]
    lambda_regs = [0.5, 1.0, 2.0]
    
    results = []
    
    for seq_dir, seq_id in [(seq01_dir, "01"), (seq17_dir, "17")]:
        rgb_path = seq_dir / "rgb" / "frame_00000.png"
        depth_path = seq_dir / "depth" / "frame_00000.png"
        intrinsics_path = seq_dir / "intrinsics.txt"
        landmarks_path = landmarks_root / seq_id / "frame_00000.txt"
        
        if not all(p.exists() for p in [rgb_path, depth_path, intrinsics_path, landmarks_path]):
            print(f"Warning: Missing files for {seq_id}, skipping")
            continue
        
        print(f"\n=== Tuning weights for sequence {seq_id} ===")
        
        for lambda_lm in lambda_landmarks:
            for lambda_d in lambda_depths:
                for lambda_r in lambda_regs:
                    print(f"Testing: λ_lm={lambda_lm}, λ_d={lambda_d}, λ_r={lambda_r}")
                    
                    output_mesh = output_root / f"{seq_id}_lm{lambda_lm}_d{lambda_d}_r{lambda_r}.ply"
                    
                    metrics = run_reconstruction(
                        rgb_path, depth_path, intrinsics_path,
                        landmarks_path, mapping_path, model_dir,
                        output_mesh, lambda_lm, lambda_d, lambda_r
                    )
                    
                    result = {
                        "sequence": seq_id,
                        "lambda_landmark": lambda_lm,
                        "lambda_depth": lambda_d,
                        "lambda_reg": lambda_r,
                        **metrics
                    }
                    results.append(result)
    
    # Save CSV
    csv_path = Path("outputs/analysis/week4_weight_sweep.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Saved results to {csv_path}")
    
    # Find best weights (lowest final energy, converged)
    best = None
    best_energy = float('inf')
    for r in results:
        if r.get("success") and r.get("converged") and "final_energy" in r:
            if r["final_energy"] < best_energy:
                best_energy = r["final_energy"]
                best = r
    
    if best:
        best_json = {
            "lambda_landmark": best["lambda_landmark"],
            "lambda_depth": best["lambda_depth"],
            "lambda_reg": best["lambda_reg"],
            "final_energy": best["final_energy"],
            "sequence": best["sequence"]
        }
        
        best_path = Path("outputs/analysis/best_weights.json")
        with open(best_path, 'w') as f:
            json.dump(best_json, f, indent=2)
        print(f"✓ Best weights saved to {best_path}")
        print(f"  λ_lm={best['lambda_landmark']}, λ_d={best['lambda_depth']}, λ_r={best['lambda_reg']}")
    else:
        print("Warning: No converged results found")

if __name__ == "__main__":
    main()
