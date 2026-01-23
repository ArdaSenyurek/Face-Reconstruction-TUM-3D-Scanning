#!/usr/bin/env python3
"""
Week 4: Metrics Collection Script

Collects metrics from rigid alignment reports and optimization results.
"""

import json
from pathlib import Path
from typing import Dict, List
import sys

def load_rigid_report(report_path: Path) -> Dict:
    """Load rigid alignment JSON report."""
    if not report_path.exists():
        return {}
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def collect_metrics(sequences: List[str] = ["01", "17"]) -> Dict:
    """Collect all Week 4 metrics."""
    metrics = {}
    
    for seq_id in sequences:
        seq_metrics = {
            "rigid_alignment": {},
            "optimization": {},
        }
        
        # Load rigid alignment report
        rigid_report = Path(f"outputs/pose_init/{seq_id}/frame_00000_rigid_report.json")
        if rigid_report.exists():
            report_data = load_rigid_report(rigid_report)
            seq_metrics["rigid_alignment"] = {
                "scale": report_data.get("transform", {}).get("scale"),
                "translation_norm": report_data.get("transform", {}).get("translation_norm"),
                "rotation_det": report_data.get("transform", {}).get("rotation_det"),
                "rmse_mm": report_data.get("alignment_errors", {}).get("rmse_mm"),
                "mean_error_mm": report_data.get("alignment_errors", {}).get("mean_error_mm"),
                "median_error_mm": report_data.get("alignment_errors", {}).get("median_error_mm"),
                "z_range_check": report_data.get("depth_z_range", {}).get("mesh_in_range"),
            }
        
        # Check for optimized mesh
        optimized_mesh = Path(f"outputs/meshes/{seq_id}/frame_00000_optimized.ply")
        seq_metrics["optimization"] = {
            "mesh_exists": optimized_mesh.exists(),
            "mesh_path": str(optimized_mesh) if optimized_mesh.exists() else None,
        }
        
        metrics[seq_id] = seq_metrics
    
    return metrics

def main():
    sequences = sys.argv[1:] if len(sys.argv) > 1 else ["01", "17"]
    
    print("Collecting Week 4 metrics...")
    print(f"Sequences: {', '.join(sequences)}")
    print()
    
    all_metrics = collect_metrics(sequences)
    
    # Save metrics
    output_path = Path("outputs/analysis/metrics_week4.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {output_path}")
    print()
    
    # Print summary
    print("Summary:")
    for seq_id, seq_metrics in all_metrics.items():
        print(f"\nSequence {seq_id}:")
        
        rigid = seq_metrics.get("rigid_alignment", {})
        if rigid:
            print(f"  Rigid Alignment:")
            print(f"    Scale: {rigid.get('scale', 'N/A')}")
            print(f"    RMSE: {rigid.get('rmse_mm', 'N/A')} mm")
            print(f"    Mean error: {rigid.get('mean_error_mm', 'N/A')} mm")
            print(f"    Z-range check: {rigid.get('z_range_check', 'N/A')}")
        
        opt = seq_metrics.get("optimization", {})
        print(f"  Optimization:")
        print(f"    Mesh exists: {opt.get('mesh_exists', False)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
