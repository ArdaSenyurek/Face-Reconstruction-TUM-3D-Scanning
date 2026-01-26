#!/usr/bin/env python3
"""
Sparse Alignment Analysis Script

Analyzes JSON reports from pose_init to evaluate mapping quality and alignment performance.
Generates summary statistics, comparison plots, and identifies problematic mappings or frames.

Usage:
    python scripts/analyze_sparse_alignment.py --reports-dir outputs/pose_init --output-dir outputs/analysis
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_json_report(filepath: Path) -> Optional[Dict]:
    """Load a JSON report file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def collect_reports(reports_dir: Path) -> List[Dict]:
    """Collect all JSON reports from a directory tree."""
    reports = []
    
    for json_file in reports_dir.rglob("*.json"):
        if json_file.name.endswith("_rigid_report.json"):
            report = load_json_report(json_file)
            if report:
                # Extract sequence and frame from path
                parts = json_file.parts
                seq_idx = -1
                frame_idx = -1
                for i, part in enumerate(parts):
                    if "pose_init" in part or "outputs" in part:
                        if i + 1 < len(parts):
                            seq_idx = i + 1
                        if i + 2 < len(parts):
                            frame_idx = i + 2
                        break
                
                if seq_idx >= 0 and frame_idx >= 0:
                    report["_sequence"] = parts[seq_idx] if seq_idx < len(parts) else "unknown"
                    report["_frame"] = json_file.stem.replace("_rigid_report", "")
                    report["_filepath"] = str(json_file)
                
                reports.append(report)
    
    return reports


def compute_summary_statistics(reports: List[Dict]) -> Dict:
    """Compute summary statistics across all reports."""
    stats = {
        "num_frames": len(reports),
        "mapping_quality": defaultdict(list),
        "procrustes_analysis": defaultdict(list),
        "alignment_errors": defaultdict(list),
    }
    
    for report in reports:
        # Mapping quality metrics
        if "mapping_quality" in report:
            mq = report["mapping_quality"]
            stats["mapping_quality"]["coverage_percent"].append(mq.get("coverage_percent", 0))
            stats["mapping_quality"]["pre_alignment_rmse_mm"].append(mq.get("pre_alignment_rmse_mm", 0))
            stats["mapping_quality"]["num_valid_mappings"].append(mq.get("num_valid_mappings", 0))
        
        # Procrustes analysis
        if "procrustes_analysis" in report:
            pa = report["procrustes_analysis"]
            stats["procrustes_analysis"]["improvement_percent"].append(pa.get("improvement_percent", 0))
            stats["procrustes_analysis"]["post_icp_rmse_mm"].append(pa.get("post_icp_rmse_mm", 0))
            stats["procrustes_analysis"]["post_procrustes_rmse_mm"].append(pa.get("post_procrustes_rmse_mm", 0))
        
        # Final alignment errors
        if "alignment_errors" in report:
            ae = report["alignment_errors"]
            stats["alignment_errors"]["rmse_mm"].append(ae.get("rmse_mm", 0))
            stats["alignment_errors"]["mean_error_mm"].append(ae.get("mean_error_mm", 0))
            stats["alignment_errors"]["median_error_mm"].append(ae.get("median_error_mm", 0))
    
    # Compute means and stds
    summary = {}
    for category, metrics in stats.items():
        if category == "num_frames":
            summary[category] = metrics
        else:
            summary[category] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary[category][f"{metric_name}_mean"] = np.mean(values)
                    summary[category][f"{metric_name}_std"] = np.std(values)
                    summary[category][f"{metric_name}_min"] = np.min(values)
                    summary[category][f"{metric_name}_max"] = np.max(values)
                    summary[category][f"{metric_name}_median"] = np.median(values)
    
    return summary


def identify_problematic_frames(reports: List[Dict], threshold_rmse_mm: float = 20.0) -> List[Dict]:
    """Identify frames with high alignment errors."""
    problematic = []
    
    for report in reports:
        rmse_mm = None
        if "alignment_errors" in report:
            rmse_mm = report["alignment_errors"].get("rmse_mm", None)
        elif "procrustes_analysis" in report:
            rmse_mm = report["procrustes_analysis"].get("post_icp_rmse_mm", None)
        
        if rmse_mm is not None and rmse_mm > threshold_rmse_mm:
            problematic.append({
                "sequence": report.get("_sequence", "unknown"),
                "frame": report.get("_frame", "unknown"),
                "rmse_mm": rmse_mm,
                "filepath": report.get("_filepath", "")
            })
    
    return sorted(problematic, key=lambda x: x["rmse_mm"], reverse=True)


def export_csv(reports: List[Dict], output_path: Path):
    """Export per-frame metrics to CSV."""
    fieldnames = [
        "sequence", "frame",
        "mapping_coverage_percent", "pre_alignment_rmse_mm", "num_valid_mappings",
        "post_procrustes_rmse_mm", "post_icp_rmse_mm", "improvement_percent",
        "final_rmse_mm", "final_mean_error_mm", "final_median_error_mm"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for report in reports:
            row = {
                "sequence": report.get("_sequence", "unknown"),
                "frame": report.get("_frame", "unknown"),
            }
            
            if "mapping_quality" in report:
                mq = report["mapping_quality"]
                row["mapping_coverage_percent"] = mq.get("coverage_percent", 0)
                row["pre_alignment_rmse_mm"] = mq.get("pre_alignment_rmse_mm", 0)
                row["num_valid_mappings"] = mq.get("num_valid_mappings", 0)
            
            if "procrustes_analysis" in report:
                pa = report["procrustes_analysis"]
                row["post_procrustes_rmse_mm"] = pa.get("post_procrustes_rmse_mm", 0)
                row["post_icp_rmse_mm"] = pa.get("post_icp_rmse_mm", 0)
                row["improvement_percent"] = pa.get("improvement_percent", 0)
            
            if "alignment_errors" in report:
                ae = report["alignment_errors"]
                row["final_rmse_mm"] = ae.get("rmse_mm", 0)
                row["final_mean_error_mm"] = ae.get("mean_error_mm", 0)
                row["final_median_error_mm"] = ae.get("median_error_mm", 0)
            
            writer.writerow(row)


def create_comparison_plots(reports: List[Dict], output_dir: Path):
    """Create comparison plots for before/after alignment errors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    pre_rmse = []
    post_procrustes_rmse = []
    post_icp_rmse = []
    improvement = []
    
    for report in reports:
        if "mapping_quality" in report and "procrustes_analysis" in report:
            mq = report["mapping_quality"]
            pa = report["procrustes_analysis"]
            
            pre_rmse.append(mq.get("pre_alignment_rmse_mm", 0))
            post_procrustes_rmse.append(pa.get("post_procrustes_rmse_mm", 0))
            post_icp_rmse.append(pa.get("post_icp_rmse_mm", 0))
            improvement.append(pa.get("improvement_percent", 0))
    
    if not pre_rmse:
        print("Warning: No data for plotting")
        return
    
    # Plot 1: Before/After comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pre_rmse))
    width = 0.25
    
    ax.bar(x - width, pre_rmse, width, label='Pre-alignment', color='red', alpha=0.7)
    ax.bar(x, post_procrustes_rmse, width, label='After Procrustes', color='orange', alpha=0.7)
    ax.bar(x + width, post_icp_rmse, width, label='After ICP', color='green', alpha=0.7)
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('RMSE (mm)')
    ax.set_title('Alignment Error: Before vs After')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "alignment_comparison.png", dpi=150)
    plt.close()
    
    # Plot 2: Improvement histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(improvement, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Improvement (%)')
    ax.set_ylabel('Number of Frames')
    ax.set_title('Alignment Improvement Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_histogram.png", dpi=150)
    plt.close()
    
    # Plot 3: Coverage vs Error scatter
    coverage = []
    final_rmse = []
    
    for report in reports:
        if "mapping_quality" in report and "alignment_errors" in report:
            mq = report["mapping_quality"]
            ae = report["alignment_errors"]
            coverage.append(mq.get("coverage_percent", 0))
            final_rmse.append(ae.get("rmse_mm", 0))
    
    if coverage and final_rmse:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coverage, final_rmse, alpha=0.6)
        ax.set_xlabel('Mapping Coverage (%)')
        ax.set_ylabel('Final RMSE (mm)')
        ax.set_title('Mapping Coverage vs Final Alignment Error')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "coverage_vs_error.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze sparse alignment results")
    parser.add_argument("--reports-dir", type=str, required=True,
                       help="Directory containing pose_init JSON reports")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--threshold-rmse", type=float, default=20.0,
                       help="RMSE threshold for identifying problematic frames (mm)")
    
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading reports from: {reports_dir}")
    reports = collect_reports(reports_dir)
    print(f"Found {len(reports)} reports")
    
    if not reports:
        print("No reports found!")
        return
    
    # Compute summary statistics
    print("Computing summary statistics...")
    summary = compute_summary_statistics(reports)
    
    # Save summary
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    # Identify problematic frames
    print(f"Identifying problematic frames (RMSE > {args.threshold_rmse} mm)...")
    problematic = identify_problematic_frames(reports, args.threshold_rmse)
    
    if problematic:
        problematic_path = output_dir / "problematic_frames.json"
        with open(problematic_path, 'w') as f:
            json.dump(problematic, f, indent=2)
        print(f"Found {len(problematic)} problematic frames")
        print(f"Saved to: {problematic_path}")
    else:
        print("No problematic frames found!")
    
    # Export CSV
    csv_path = output_dir / "per_frame_metrics.csv"
    export_csv(reports, csv_path)
    print(f"Exported CSV to: {csv_path}")
    
    # Create plots
    print("Creating comparison plots...")
    create_comparison_plots(reports, output_dir)
    print(f"Plots saved to: {output_dir}")
    
    # Print summary
    print("\n=== Summary Statistics ===")
    if "mapping_quality" in summary:
        mq = summary["mapping_quality"]
        print(f"Mapping Coverage: {mq.get('coverage_percent_mean', 0):.1f}% ± {mq.get('coverage_percent_std', 0):.1f}%")
        print(f"Pre-alignment RMSE: {mq.get('pre_alignment_rmse_mm_mean', 0):.2f} ± {mq.get('pre_alignment_rmse_mm_std', 0):.2f} mm")
    
    if "procrustes_analysis" in summary:
        pa = summary["procrustes_analysis"]
        print(f"Post-ICP RMSE: {pa.get('post_icp_rmse_mm_mean', 0):.2f} ± {pa.get('post_icp_rmse_mm_std', 0):.2f} mm")
        print(f"Improvement: {pa.get('improvement_percent_mean', 0):.1f}% ± {pa.get('improvement_percent_std', 0):.1f}%")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
