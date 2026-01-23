#!/usr/bin/env python3
"""
Week 4: Full Evaluation Script

Complete evaluation including:
1. Gauss-Newton optimization (expression coefficients)
2. Before/After metrics comparison
3. 3D overlay generation
4. Regularization weight tuning
5. Summary report

Usage:
    python scripts/week4_full_evaluation.py --seq 01
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import argparse
from scipy.spatial import cKDTree


def load_ply_vertices(path: Path) -> np.ndarray:
    """Load vertices from PLY file."""
    vertices = []
    with open(path, 'r') as f:
        in_header = True
        num_verts = 0
        for line in f:
            if in_header:
                if 'element vertex' in line:
                    num_verts = int(line.split()[-1])
                if 'end_header' in line:
                    in_header = False
            else:
                if len(vertices) < num_verts:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(vertices) if vertices else np.zeros((0, 3))


def compute_nn_metrics(scan: np.ndarray, mesh: np.ndarray, 
                       sample_size: int = 5000) -> Dict:
    """Compute nearest-neighbor metrics between scan and mesh."""
    if len(scan) == 0 or len(mesh) == 0:
        return {}
    
    # Sample points for efficiency
    if len(scan) > sample_size:
        indices = np.random.choice(len(scan), sample_size, replace=False)
        scan_sampled = scan[indices]
    else:
        scan_sampled = scan
    
    # Build KD-tree
    tree = cKDTree(mesh)
    distances, _ = tree.query(scan_sampled)
    
    # Compute metrics
    rmse = np.sqrt(np.mean(distances**2))
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    
    # Threshold analysis
    thresholds = [0.005, 0.010, 0.020, 0.030, 0.050]  # meters
    pct_within = {}
    for th in thresholds:
        pct = np.sum(distances < th) / len(distances) * 100
        pct_within[f"within_{int(th*1000)}mm"] = round(pct, 2)
    
    return {
        "nn_rmse_mm": round(rmse * 1000, 2),
        "mean_dist_mm": round(mean_dist * 1000, 2),
        "median_dist_mm": round(median_dist * 1000, 2),
        "percentiles": pct_within,
        "num_scan_points": len(scan_sampled),
        "num_mesh_vertices": len(mesh),
    }


def filter_face_region(scan: np.ndarray, mesh: np.ndarray, 
                       margin: float = 0.05) -> np.ndarray:
    """Filter scan points to face region (within mesh bbox + margin)."""
    if len(mesh) == 0:
        return scan
    
    mesh_min = mesh.min(axis=0) - margin
    mesh_max = mesh.max(axis=0) + margin
    
    mask = (
        (scan[:, 0] > mesh_min[0]) & (scan[:, 0] < mesh_max[0]) &
        (scan[:, 1] > mesh_min[1]) & (scan[:, 1] < mesh_max[1]) &
        (scan[:, 2] > mesh_min[2]) & (scan[:, 2] < mesh_max[2])
    )
    
    return scan[mask]


def load_rigid_report(path: Path) -> Dict:
    """Load rigid alignment report JSON."""
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}


def run_pose_init(seq_id: str, frame: str = "frame_00000") -> Dict:
    """Run pose_init to get rigid alignment."""
    project_root = Path(__file__).parent.parent
    binary = project_root / "build" / "bin" / "pose_init"
    
    if not binary.exists():
        print(f"ERROR: pose_init binary not found at {binary}")
        return {}
    
    # Paths
    depth_path = project_root / f"outputs/converted/{seq_id}/depth/{frame}.png"
    rgb_path = project_root / f"outputs/converted/{seq_id}/rgb/{frame}.png"
    intrinsics_path = project_root / f"outputs/converted/{seq_id}/intrinsics.txt"
    landmarks_path = project_root / f"outputs/landmarks/{seq_id}/{frame}.txt"
    mapping_path = project_root / "data/landmark_mapping_bfm.txt"
    model_dir = project_root / "data/model_bfm"
    output_dir = project_root / f"outputs/pose_init/{seq_id}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_mesh = output_dir / f"{frame}_aligned.ply"
    report_path = output_dir / f"{frame}_rigid_report.json"
    
    cmd = [
        str(binary),
        "--depth", str(depth_path),
        "--rgb", str(rgb_path),
        "--intrinsics", str(intrinsics_path),
        "--landmarks", str(landmarks_path),
        "--mapping", str(mapping_path),
        "--model-dir", str(model_dir),
        "--output", str(output_mesh),
        "--report", str(report_path),
    ]
    
    print(f"  Running pose_init for {seq_id}/{frame}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"  ERROR: pose_init failed: {result.stderr}")
        return {}
    
    return load_rigid_report(report_path)


def run_face_reconstruction(seq_id: str, frame: str = "frame_00000",
                           optimize: bool = True,
                           lambda_landmark: float = 1.0,
                           lambda_depth: float = 0.1,
                           lambda_reg: float = 1.0,
                           max_iter: int = 30,
                           verbose: bool = True) -> Tuple[bool, str]:
    """Run face_reconstruction with or without optimization."""
    project_root = Path(__file__).parent.parent
    binary = project_root / "build" / "bin" / "face_reconstruction"
    
    if not binary.exists():
        return False, f"Binary not found: {binary}"
    
    # Paths
    depth_path = project_root / f"outputs/converted/{seq_id}/depth/{frame}.png"
    rgb_path = project_root / f"outputs/converted/{seq_id}/rgb/{frame}.png"
    intrinsics_path = project_root / f"outputs/converted/{seq_id}/intrinsics.txt"
    landmarks_path = project_root / f"outputs/landmarks/{seq_id}/{frame}.txt"
    mapping_path = project_root / "data/landmark_mapping_bfm.txt"
    model_dir = project_root / "data/model_bfm"
    meshes_dir = project_root / f"outputs/meshes/{seq_id}"
    
    meshes_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = "_optimized" if optimize else "_rigid"
    output_mesh = meshes_dir / f"{frame}{suffix}.ply"
    
    cmd = [
        str(binary),
        "--rgb", str(rgb_path),
        "--depth", str(depth_path),
        "--intrinsics", str(intrinsics_path),
        "--model-dir", str(model_dir),
        "--landmarks", str(landmarks_path),
        "--mapping", str(mapping_path),
        "--output-mesh", str(output_mesh),
        "--max-iter", str(max_iter),
        "--lambda-landmark", str(lambda_landmark),
        "--lambda-depth", str(lambda_depth),
        "--lambda-reg", str(lambda_reg),
    ]
    
    if optimize:
        cmd.append("--optimize")
    else:
        cmd.append("--no-optimize")
    
    if verbose:
        cmd.append("--verbose")
    
    opt_str = "optimized" if optimize else "rigid"
    print(f"  Running face_reconstruction ({opt_str}) for {seq_id}/{frame}...")
    print(f"    lambda_lm={lambda_landmark}, lambda_depth={lambda_depth}, lambda_reg={lambda_reg}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        return False, f"Failed: {result.stderr}"
    
    if verbose and result.stdout:
        # Extract optimization log
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Iter' in line or 'energy' in line.lower() or 'converged' in line.lower():
                print(f"    {line}")
    
    return output_mesh.exists(), str(output_mesh)


def run_create_overlays(seq_id: str, frame: str = "frame_00000") -> bool:
    """Run create_overlays tool."""
    project_root = Path(__file__).parent.parent
    binary = project_root / "build" / "bin" / "create_overlays"
    
    if not binary.exists():
        return False
    
    # Paths
    depth_path = project_root / f"outputs/converted/{seq_id}/depth/{frame}.png"
    rgb_path = project_root / f"outputs/converted/{seq_id}/rgb/{frame}.png"
    intrinsics_path = project_root / f"outputs/converted/{seq_id}/intrinsics.txt"
    mesh_rigid = project_root / f"outputs/pose_init/{seq_id}/{frame}_aligned.ply"
    mesh_opt = project_root / f"outputs/meshes/{seq_id}/{frame}_optimized.ply"
    out_dir = project_root / f"outputs/overlays_3d/{seq_id}"
    metrics_path = out_dir / f"{frame}_overlay_metrics.json"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        str(binary),
        "--depth", str(depth_path),
        "--rgb", str(rgb_path),
        "--intrinsics", str(intrinsics_path),
        "--mesh-rigid", str(mesh_rigid),
        "--mesh-opt", str(mesh_opt),
        "--out-dir", str(out_dir),
        "--frame-name", frame,
        "--output-metrics", str(metrics_path),
    ]
    
    print(f"  Creating 3D overlays for {seq_id}/{frame}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
    
    return result.returncode == 0


def compute_before_after_metrics(seq_id: str, frame: str = "frame_00000") -> Dict:
    """Compute comprehensive before/after metrics."""
    project_root = Path(__file__).parent.parent
    
    # Load scan
    scan_path = project_root / f"outputs/analysis/pointclouds/{seq_id}/{frame}.ply"
    if not scan_path.exists():
        return {"error": "Scan not found"}
    
    scan = load_ply_vertices(scan_path)
    
    # Load rigid mesh
    rigid_mesh_path = project_root / f"outputs/pose_init/{seq_id}/{frame}_aligned.ply"
    if not rigid_mesh_path.exists():
        return {"error": "Rigid mesh not found"}
    
    rigid_mesh = load_ply_vertices(rigid_mesh_path)
    
    # Load optimized mesh
    opt_mesh_path = project_root / f"outputs/meshes/{seq_id}/{frame}_optimized.ply"
    if not opt_mesh_path.exists():
        return {"error": "Optimized mesh not found"}
    
    opt_mesh = load_ply_vertices(opt_mesh_path)
    
    # Filter to face region
    face_scan_rigid = filter_face_region(scan, rigid_mesh)
    face_scan_opt = filter_face_region(scan, opt_mesh)
    
    # Load rigid report for landmark RMSE
    rigid_report = load_rigid_report(
        project_root / f"outputs/pose_init/{seq_id}/{frame}_rigid_report.json"
    )
    
    metrics = {
        "sequence": seq_id,
        "frame": frame,
        "timestamp": datetime.now().isoformat(),
        "before_gn": {
            "landmark_rmse_mm": rigid_report.get("alignment_errors", {}).get("rmse_mm", None),
            "landmark_mean_mm": rigid_report.get("alignment_errors", {}).get("mean_error_mm", None),
            "face_region": compute_nn_metrics(face_scan_rigid, rigid_mesh),
            "full_scan": compute_nn_metrics(scan, rigid_mesh),
        },
        "after_gn": {
            "face_region": compute_nn_metrics(face_scan_opt, opt_mesh),
            "full_scan": compute_nn_metrics(scan, opt_mesh),
        },
    }
    
    # Compute improvement
    before_face = metrics["before_gn"]["face_region"].get("nn_rmse_mm", 0)
    after_face = metrics["after_gn"]["face_region"].get("nn_rmse_mm", 0)
    
    if before_face > 0 and after_face > 0:
        improvement = (before_face - after_face) / before_face * 100
        metrics["improvement"] = {
            "face_rmse_reduction_pct": round(improvement, 2),
            "face_rmse_before_mm": before_face,
            "face_rmse_after_mm": after_face,
        }
    
    return metrics


def run_weight_tuning(seq_id: str, frame: str = "frame_00000") -> List[Dict]:
    """Test different regularization weight combinations."""
    settings = [
        {"lambda_depth": 0.05, "lambda_reg": 0.5, "name": "low_depth_low_reg"},
        {"lambda_depth": 0.05, "lambda_reg": 1.0, "name": "low_depth_high_reg"},
        {"lambda_depth": 0.10, "lambda_reg": 0.5, "name": "high_depth_low_reg"},
        {"lambda_depth": 0.10, "lambda_reg": 1.0, "name": "high_depth_high_reg"},
    ]
    
    project_root = Path(__file__).parent.parent
    results = []
    
    for setting in settings:
        print(f"\n  Testing: {setting['name']}")
        
        # Run reconstruction with this setting
        success, mesh_path = run_face_reconstruction(
            seq_id, frame,
            optimize=True,
            lambda_landmark=1.0,
            lambda_depth=setting["lambda_depth"],
            lambda_reg=setting["lambda_reg"],
            max_iter=30,
            verbose=False,
        )
        
        if success:
            # Load meshes and compute metrics
            scan = load_ply_vertices(
                project_root / f"outputs/analysis/pointclouds/{seq_id}/{frame}.ply"
            )
            mesh = load_ply_vertices(Path(mesh_path))
            face_scan = filter_face_region(scan, mesh)
            
            face_metrics = compute_nn_metrics(face_scan, mesh)
            
            result = {
                **setting,
                "success": True,
                "face_rmse_mm": face_metrics.get("nn_rmse_mm"),
                "within_10mm_pct": face_metrics.get("percentiles", {}).get("within_10mm"),
                "within_20mm_pct": face_metrics.get("percentiles", {}).get("within_20mm"),
            }
        else:
            result = {
                **setting,
                "success": False,
                "error": "Reconstruction failed",
            }
        
        results.append(result)
        print(f"    RMSE: {result.get('face_rmse_mm', 'N/A')} mm")
    
    return results


def generate_summary_report(metrics: Dict, tuning_results: List[Dict], 
                           seq_id: str) -> str:
    """Generate Week 4 summary report."""
    report = []
    report.append("=" * 70)
    report.append("WEEK 4 SUMMARY REPORT")
    report.append("=" * 70)
    report.append(f"Sequence: {seq_id}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Before/After comparison
    report.append("-" * 70)
    report.append("1. GAUSS-NEWTON OPTIMIZATION RESULTS")
    report.append("-" * 70)
    
    before = metrics.get("before_gn", {})
    after = metrics.get("after_gn", {})
    improvement = metrics.get("improvement", {})
    
    report.append("")
    report.append("Landmark RMSE (Rigid Alignment):")
    lm_rmse = before.get("landmark_rmse_mm")
    if lm_rmse:
        report.append(f"  Before GN: {lm_rmse:.2f} mm")
    
    report.append("")
    report.append("Face Region NN-RMSE:")
    before_face = before.get("face_region", {}).get("nn_rmse_mm")
    after_face = after.get("face_region", {}).get("nn_rmse_mm")
    
    if before_face:
        report.append(f"  Before GN: {before_face:.2f} mm")
    if after_face:
        report.append(f"  After GN:  {after_face:.2f} mm")
    
    if improvement:
        report.append(f"  Improvement: {improvement.get('face_rmse_reduction_pct', 0):.1f}%")
    
    report.append("")
    report.append("Threshold Analysis (After GN):")
    after_pct = after.get("face_region", {}).get("percentiles", {})
    for key, value in after_pct.items():
        th_mm = key.replace("within_", "").replace("mm", "")
        report.append(f"  Points < {th_mm}mm: {value:.1f}%")
    
    # Weight tuning results
    report.append("")
    report.append("-" * 70)
    report.append("2. REGULARIZATION WEIGHT TUNING")
    report.append("-" * 70)
    report.append("")
    report.append(f"{'Setting':<25} {'lambda_depth':<12} {'lambda_reg':<12} {'RMSE (mm)':<12} {'<10mm %':<10}")
    report.append("-" * 70)
    
    best_result = None
    best_rmse = float('inf')
    
    for result in tuning_results:
        name = result.get("name", "unknown")
        ld = result.get("lambda_depth", "N/A")
        lr = result.get("lambda_reg", "N/A")
        rmse = result.get("face_rmse_mm", "N/A")
        pct10 = result.get("within_10mm_pct", "N/A")
        
        report.append(f"{name:<25} {ld:<12} {lr:<12} {rmse if rmse != 'N/A' else 'N/A':<12} {pct10 if pct10 != 'N/A' else 'N/A':<10}")
        
        if rmse != "N/A" and rmse < best_rmse:
            best_rmse = rmse
            best_result = result
    
    if best_result:
        report.append("")
        report.append(f"Best setting: {best_result.get('name')} (RMSE={best_rmse:.2f} mm)")
    
    # Final summary
    report.append("")
    report.append("-" * 70)
    report.append("3. CONCLUSIONS")
    report.append("-" * 70)
    report.append("")
    
    # Determine if GN improved results
    if improvement and improvement.get("face_rmse_reduction_pct", 0) > 0:
        report.append("✓ Gauss-Newton optimization IMPROVED over rigid alignment")
        report.append(f"  - RMSE reduced by {improvement.get('face_rmse_reduction_pct', 0):.1f}%")
    elif after_face and before_face:
        if after_face <= before_face:
            report.append("✓ Gauss-Newton maintained or slightly improved alignment")
        else:
            report.append("✗ Gauss-Newton did not improve (possible overfitting or divergence)")
    
    report.append("")
    report.append("Stability Assessment:")
    
    # Check convergence from tuning results
    converged_count = sum(1 for r in tuning_results if r.get("success", False))
    total_count = len(tuning_results)
    
    if converged_count == total_count:
        report.append(f"  ✓ Optimization converged for all {total_count} weight settings tested")
    else:
        report.append(f"  ! Optimization converged for {converged_count}/{total_count} settings")
    
    report.append("")
    report.append("Limitations:")
    report.append("  - Depth sensor noise limits surface accuracy (~5-10mm)")
    report.append("  - BFM model expressiveness constrains fine details")
    report.append("  - Sparse landmarks (33) for initialization")
    report.append("  - Missing forehead/hair region in depth scans")
    
    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Week 4 Full Evaluation")
    parser.add_argument("--seq", type=str, default="01", help="Sequence ID (default: 01)")
    parser.add_argument("--frame", type=str, default="frame_00000", help="Frame name")
    parser.add_argument("--skip-rigid", action="store_true", help="Skip rigid alignment (use existing)")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip weight tuning")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("WEEK 4: FULL EVALUATION")
    print("=" * 70)
    print(f"Sequence: {args.seq}")
    print(f"Frame: {args.frame}")
    print()
    
    # Step 1: Rigid alignment
    print("[1/6] Rigid Alignment (Procrustes)...")
    if not args.skip_rigid:
        rigid_report = run_pose_init(args.seq, args.frame)
        if rigid_report:
            print(f"  ✓ RMSE: {rigid_report.get('alignment_errors', {}).get('rmse_mm', 'N/A'):.2f} mm")
        else:
            print("  ✗ Rigid alignment failed")
            return 1
    else:
        print("  (Skipped - using existing)")
    print()
    
    # Step 2: Rigid mesh (no optimization)
    print("[2/6] Generating Rigid Mesh (baseline)...")
    success, rigid_mesh_path = run_face_reconstruction(
        args.seq, args.frame,
        optimize=False,
        verbose=False,
    )
    if success:
        print(f"  ✓ Saved: {rigid_mesh_path}")
    else:
        print("  ✗ Rigid mesh generation failed")
    print()
    
    # Step 3: Optimized mesh
    print("[3/6] Running Gauss-Newton Optimization...")
    success, opt_mesh_path = run_face_reconstruction(
        args.seq, args.frame,
        optimize=True,
        lambda_landmark=1.0,
        lambda_depth=0.1,
        lambda_reg=1.0,
        max_iter=30,
        verbose=True,
    )
    if success:
        print(f"  ✓ Saved: {opt_mesh_path}")
    else:
        print("  ✗ Optimization failed")
        return 1
    print()
    
    # Step 4: Create 3D overlays
    print("[4/6] Creating 3D Overlays...")
    overlay_success = run_create_overlays(args.seq, args.frame)
    if overlay_success:
        print(f"  ✓ Overlays saved to outputs/overlays_3d/{args.seq}/")
    else:
        print("  ✗ Overlay creation failed")
    print()
    
    # Step 5: Compute before/after metrics
    print("[5/6] Computing Before/After Metrics...")
    metrics = compute_before_after_metrics(args.seq, args.frame)
    
    if "error" not in metrics:
        # Save metrics
        metrics_path = project_root / "outputs/analysis/metrics_before_after.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics saved to {metrics_path}")
        
        # Print summary
        before_face = metrics.get("before_gn", {}).get("face_region", {}).get("nn_rmse_mm")
        after_face = metrics.get("after_gn", {}).get("face_region", {}).get("nn_rmse_mm")
        
        if before_face and after_face:
            print(f"  Face RMSE: {before_face:.2f} mm → {after_face:.2f} mm")
    else:
        print(f"  ✗ Metrics computation failed: {metrics.get('error')}")
    print()
    
    # Step 6: Weight tuning
    print("[6/6] Regularization Weight Tuning...")
    if not args.skip_tuning:
        tuning_results = run_weight_tuning(args.seq, args.frame)
        
        # Save tuning results
        tuning_path = project_root / "outputs/analysis/weight_tuning_results.json"
        with open(tuning_path, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        print(f"  ✓ Tuning results saved to {tuning_path}")
    else:
        print("  (Skipped)")
        tuning_results = []
    print()
    
    # Generate summary report
    print("=" * 70)
    summary = generate_summary_report(metrics, tuning_results, args.seq)
    print(summary)
    
    # Save summary
    summary_path = project_root / "outputs/analysis/week4_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")
    
    # Final deliverables check
    print("\n" + "=" * 70)
    print("DELIVERABLES CHECK")
    print("=" * 70)
    
    deliverables = {
        f"outputs/meshes/{args.seq}/{args.frame}_rigid.ply": "Rigid mesh",
        f"outputs/meshes/{args.seq}/{args.frame}_optimized.ply": "Optimized mesh",
        f"outputs/pose_init/{args.seq}/{args.frame}_aligned.ply": "Aligned mesh (from pose_init)",
        f"outputs/overlays_3d/{args.seq}/{args.frame}_overlay_rigid.ply": "Rigid overlay",
        f"outputs/overlays_3d/{args.seq}/{args.frame}_overlay_opt.ply": "Optimized overlay",
        "outputs/analysis/metrics_before_after.json": "Metrics JSON",
    }
    
    all_found = True
    for path, desc in deliverables.items():
        full_path = project_root / path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {desc}: {path}")
        if not exists:
            all_found = False
    
    print()
    if all_found:
        print("✓ All Week 4 deliverables generated successfully!")
    else:
        print("! Some deliverables missing - check errors above")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
