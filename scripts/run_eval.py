#!/usr/bin/env python3
"""
Week 6 Evaluation Script: 3-stage fitting protocol.

Stage 1 — Identity/Shape only (neutral frame):
  Optimize identity α only (δ=0, pose fixed from Procrustes). Strong λ_alpha.
  Outputs: outputs/week6/<seq>/<frame>/meshes/identity.ply, identity_state.json.

Stage 2 — Expression only (single frame):
  With identity fixed from Stage 1, optimize δ only. Pose fixed.
  Outputs: outputs/week6/<seq>/<frame>/meshes/expression.ply.

Stage 3 — Frame-by-frame expression (tracking):
  Identity fixed; estimate δ per frame for N frames. Optional EMA/SLERP smoothing.
  Outputs: outputs/week6/<seq>/<frame>/meshes/tracked.ply, metrics.json, convergence.json.

Usage:
  python scripts/run_week6_eval.py --sequences 01 14 17 19
  python scripts/run_week6_eval.py --sequences 01 --frames 5 --stage 1 2 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 6 evaluation: 3-stage protocol (identity → expression → tracking)")
    p.add_argument("--data-root", type=Path, default=Path("data"), help="Data root")
    p.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root (converted, landmarks, pose_init live here)")
    p.add_argument("--sequences", type=str, nargs="+", default=["01", "14", "17", "19"],
                   help="Sequence IDs to run (e.g. 01 14 17 19)")
    p.add_argument("--frames", type=int, default=10, help="Max frames for Stage 3 tracking")
    p.add_argument("--stage", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3],
                   help="Which stages to run (default: 1 2 3)")
    p.add_argument("--neutral-frame", type=str, default="frame_00000", help="Neutral frame for Stage 1/2")
    p.add_argument("--recon-binary", type=Path, default=REPO_ROOT / "build/bin/face_reconstruction")
    p.add_argument("--pose-init-binary", type=Path, default=REPO_ROOT / "build/bin/pose_init")
    p.add_argument("--overlay-binary", type=Path, default=REPO_ROOT / "build/bin/create_overlays")
    p.add_argument("--model-dir", type=Path, default=REPO_ROOT / "data/bfm/model_bfm")
    p.add_argument("--landmark-mapping", type=Path, default=REPO_ROOT / "data/bfm_landmark_68.txt")
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--lambda-alpha", type=float, default=2.0, help="Stage 1 identity regularization")
    p.add_argument("--lambda-delta", type=float, default=1.5, help="Expression regularization")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--temporal-smoothing", action="store_true", help="Stage 3: EMA/SLERP smoothing")
    p.add_argument("--week6-dir", type=Path, default=None,
                   help="Week6 output base (default: output_root/week6)")
    p.add_argument("--metrics", action="store_true", default=True,
                   help="Compute per-frame metrics (landmark, depth, surface error); default: on")
    p.add_argument("--no-metrics", dest="metrics", action="store_false",
                   help="Skip computing metrics")
    p.add_argument("--visuals", action="store_true", default=True,
                   help="Generate 2D overlay and depth comparison images; default: on")
    p.add_argument("--no-visuals", dest="visuals", action="store_false",
                   help="Skip generating visuals")
    p.add_argument("--overlays", action="store_true", default=True,
                   help="Export 3D overlay PLY (rigid + opt) per frame; default: on")
    p.add_argument("--no-overlays", dest="overlays", action="store_false",
                   help="Skip 3D overlay export")
    p.add_argument("--aggregate", action="store_true", default=True,
                   help="Run aggregate_summary to produce summary.csv; default: on")
    p.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                   help="Skip summary.csv aggregation")
    return p.parse_args()


def get_converted_seq_dirs(output_root: Path, sequences: List[str]) -> Dict[str, Path]:
    """Resolve converted/<seq> paths."""
    converted = output_root / "converted"
    out: Dict[str, Path] = {}
    for seq in sequences:
        d = converted / seq
        if d.exists():
            out[seq] = d
    return out


def run_metrics_for_frame(
    args: argparse.Namespace,
    week6_base: Path,
    seq: str,
    frame_name: str,
    mesh_ply: Path,
    seq_dir: Path,
    landmarks_file: Path,
) -> Optional[Dict[str, Any]]:
    """Run compute_metrics.py for one frame; return metrics dict or None."""
    if not mesh_ply.exists():
        return None
    depth = seq_dir / "depth" / f"{frame_name}.png"
    intrinsics = seq_dir / "intrinsics.txt"
    out_json = week6_base / seq / frame_name / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pointcloud = seq_dir / "pointclouds" / f"{frame_name}.ply"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compute_metrics.py"),
        "--mesh", str(mesh_ply),
        "--depth", str(depth),
        "--intrinsics", str(intrinsics),
        "--landmarks", str(landmarks_file),
        "--mapping", str(args.landmark_mapping),
        "--output", str(out_json),
    ]
    if pointcloud.exists():
        cmd += ["--pointcloud", str(pointcloud)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if out_json.exists():
        try:
            with open(out_json) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def run_visuals_for_frame(
    args: argparse.Namespace,
    week6_base: Path,
    seq: str,
    frame_name: str,
    mesh_ply: Path,
    seq_dir: Path,
) -> bool:
    """Run generate_visuals.py for one frame."""
    if not mesh_ply.exists():
        return False
    depth = seq_dir / "depth" / f"{frame_name}.png"
    intrinsics = seq_dir / "intrinsics.txt"
    rgb = seq_dir / "rgb" / f"{frame_name}.png"
    out_dir = week6_base / seq / frame_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_visuals.py"),
        "--seq", seq,
        "--frame", frame_name.replace("frame_", ""),
        "--mesh", str(mesh_ply),
        "--depth", str(depth),
        "--intrinsics", str(intrinsics),
        "--output-dir", str(out_dir),
    ]
    if rgb.exists():
        cmd += ["--rgb", str(rgb)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def run_overlays_for_frame(
    args: argparse.Namespace,
    week6_base: Path,
    seq: str,
    frame_name: str,
) -> None:
    """Run export_overlay_ply.py for rigid and opt for one frame."""
    frame_id = frame_name.replace("frame_", "")
    for stage in ("rigid", "opt"):
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_overlay_ply.py"),
            "--seq", seq,
            "--frame", frame_id,
            "--stage", stage,
            "--output-root", str(args.output_root),
            "--week6-dir", str(week6_base),
            "--overlay-binary", str(args.overlay_binary),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass


def run_stage1(
    args: argparse.Namespace,
    seq: str,
    seq_dir: Path,
    week6_base: Path,
    run_config: Dict[str, Any],
) -> bool:
    """Stage 1: Identity only on neutral frame. Pose from Procrustes."""
    frame_name = args.neutral_frame
    rgb = seq_dir / "rgb" / f"{frame_name}.png"
    depth = seq_dir / "depth" / f"{frame_name}.png"
    intrinsics = seq_dir / "intrinsics.txt"
    landmarks_root = args.output_root / "landmarks"
    landmarks_file = landmarks_root / seq / f"{frame_name}.txt"
    pose_init_root = args.output_root / "pose_init"
    aligned_mesh = pose_init_root / seq / f"{frame_name}_aligned.ply"
    report_json = pose_init_root / seq / f"{frame_name}_rigid_report.json"

    out_meshes = week6_base / seq / frame_name / "meshes"
    out_meshes.mkdir(parents=True, exist_ok=True)
    identity_ply = out_meshes / "identity.ply"
    identity_state = week6_base / seq / "identity_state.json"
    convergence_json = week6_base / seq / "convergence_stage1.json"

    if not all([rgb.exists(), depth.exists(), intrinsics.exists(), landmarks_file.exists()]):
        print(f"[Stage 1] Skip {seq}: missing rgb/depth/intrinsics/landmarks")
        return False

    # Run pose_init if rigid mesh not present
    if not aligned_mesh.exists() or not report_json.exists():
        cmd_pose = [
            str(args.pose_init_binary),
            "--rgb", str(rgb),
            "--depth", str(depth),
            "--intrinsics", str(intrinsics),
            "--landmarks", str(landmarks_file),
            "--model-dir", str(args.model_dir),
            "--mapping", str(args.landmark_mapping),
            "--output", str(aligned_mesh),
            "--report", str(report_json),
        ]
        subprocess.run(cmd_pose, capture_output=True, timeout=args.timeout)
        if not aligned_mesh.exists():
            print(f"[Stage 1] pose_init failed for {seq}")
            return False

    # face_reconstruction --stage id (identity only, delta=0, pose fixed)
    cmd = [
        str(args.recon_binary),
        "--rgb", str(rgb),
        "--depth", str(depth),
        "--intrinsics", str(intrinsics),
        "--model-dir", str(args.model_dir),
        "--landmarks", str(landmarks_file),
        "--mapping", str(args.landmark_mapping),
        "--output-mesh", str(identity_ply),
        "--optimize",
        "--stage", "id",
        "--max-iter", str(args.max_iter),
        "--lambda-alpha", str(args.lambda_alpha),
        "--lambda-reg", "1.0",
        "--output-state-json", str(identity_state),
        "--output-convergence-json", str(convergence_json),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=args.timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[Stage 1] face_reconstruction failed for {seq}: {e}")
        return False
    if not identity_ply.exists():
        return False
    print(f"[Stage 1] {seq} identity saved to {identity_ply}")
    if args.metrics:
        run_metrics_for_frame(args, week6_base, seq, frame_name, identity_ply, seq_dir, landmarks_file)
    if args.visuals:
        run_visuals_for_frame(args, week6_base, seq, frame_name, identity_ply, seq_dir)
    if args.overlays:
        run_overlays_for_frame(args, week6_base, seq, frame_name)
    return True


def run_stage2(
    args: argparse.Namespace,
    seq: str,
    seq_dir: Path,
    week6_base: Path,
) -> bool:
    """Stage 2: Expression only, identity fixed from Stage 1."""
    frame_name = args.neutral_frame
    identity_state = week6_base / seq / "identity_state.json"
    if not identity_state.exists():
        print(f"[Stage 2] Skip {seq}: no identity_state.json (run Stage 1 first)")
        return False

    rgb = seq_dir / "rgb" / f"{frame_name}.png"
    depth = seq_dir / "depth" / f"{frame_name}.png"
    intrinsics = seq_dir / "intrinsics.txt"
    landmarks_file = args.output_root / "landmarks" / seq / f"{frame_name}.txt"
    out_meshes = week6_base / seq / frame_name / "meshes"
    out_meshes.mkdir(parents=True, exist_ok=True)
    expression_ply = out_meshes / "expression.ply"
    convergence_json = week6_base / seq / "convergence_stage2.json"

    cmd = [
        str(args.recon_binary),
        "--rgb", str(rgb),
        "--depth", str(depth),
        "--intrinsics", str(intrinsics),
        "--model-dir", str(args.model_dir),
        "--landmarks", str(landmarks_file),
        "--mapping", str(args.landmark_mapping),
        "--output-mesh", str(expression_ply),
        "--optimize",
        "--stage", "expr",
        "--init-identity-json", str(identity_state),
        "--max-iter", str(args.max_iter),
        "--lambda-delta", str(args.lambda_delta),
        "--output-convergence-json", str(convergence_json),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=args.timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[Stage 2] face_reconstruction failed for {seq}: {e}")
        return False
    if not expression_ply.exists():
        return False
    print(f"[Stage 2] {seq} expression saved to {expression_ply}")
    if args.metrics:
        run_metrics_for_frame(args, week6_base, seq, frame_name, expression_ply, seq_dir, landmarks_file)
    if args.visuals:
        run_visuals_for_frame(args, week6_base, seq, frame_name, expression_ply, seq_dir)
    if args.overlays:
        run_overlays_for_frame(args, week6_base, seq, frame_name)
    return True


def run_stage3(
    args: argparse.Namespace,
    seq: str,
    seq_dir: Path,
    week6_base: Path,
) -> bool:
    """Stage 3: Frame-by-frame expression with fixed identity (from Stage 1)."""
    identity_state = week6_base / seq / "identity_state.json"
    if not identity_state.exists():
        print(f"[Stage 3] Skip {seq}: no identity_state.json")
        return False

    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    intrinsics = seq_dir / "intrinsics.txt"
    landmarks_root = args.output_root / "landmarks" / seq
    n_frames = args.frames
    frames = sorted(rgb_dir.glob("frame_*.png"))[:n_frames]
    all_metrics: List[Dict[str, Any]] = []
    all_convergence: List[Dict[str, Any]] = []

    for i, rgb_path in enumerate(frames):
        frame_name = rgb_path.stem
        depth_path = depth_dir / f"{frame_name}.png"
        landmarks_file = landmarks_root / f"{frame_name}.txt"
        if not depth_path.exists() or not landmarks_file.exists():
            continue
        out_meshes = week6_base / seq / frame_name / "meshes"
        out_meshes.mkdir(parents=True, exist_ok=True)
        tracked_ply = out_meshes / "tracked.ply"
        state_json = week6_base / seq / frame_name / "state_final.json"
        conv_json = week6_base / seq / frame_name / "convergence.json"
        state_json.parent.mkdir(parents=True, exist_ok=True)

        # Identity from Stage 1; pose from previous frame state (for i > 0) or from identity_state (frame 0)
        prev_state = week6_base / seq / frames[i - 1].stem / "state_final.json" if i > 0 else None

        cmd = [
            str(args.recon_binary),
            "--rgb", str(rgb_path),
            "--depth", str(depth_path),
            "--intrinsics", str(intrinsics),
            "--model-dir", str(args.model_dir),
            "--landmarks", str(landmarks_file),
            "--mapping", str(args.landmark_mapping),
            "--output-mesh", str(tracked_ply),
            "--optimize",
            "--stage", "expr",
            "--init-identity-json", str(identity_state),
            "--output-state-json", str(state_json),
            "--output-convergence-json", str(conv_json),
            "--max-iter", str(args.max_iter),
        ]
        if prev_state and prev_state.exists():
            idx = cmd.index("--output-state-json")
            cmd = cmd[:idx] + ["--init-pose-json", str(prev_state)] + cmd[idx:]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        except (subprocess.TimeoutExpired, Exception):
            continue
        if tracked_ply.exists():
            entry: Dict[str, Any] = {"sequence": seq, "frame": frame_name, "mesh": str(tracked_ply)}
            if args.metrics:
                metrics_dict = run_metrics_for_frame(
                    args, week6_base, seq, frame_name, tracked_ply, seq_dir, landmarks_file
                )
                if metrics_dict:
                    entry["metrics"] = metrics_dict
            all_metrics.append(entry)
            if args.visuals:
                run_visuals_for_frame(args, week6_base, seq, frame_name, tracked_ply, seq_dir)
            if args.overlays:
                run_overlays_for_frame(args, week6_base, seq, frame_name)
            if conv_json.exists():
                try:
                    with open(conv_json) as f:
                        all_convergence.append(json.load(f))
                except Exception:
                    pass
    if all_metrics:
        metrics_path = week6_base / seq / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"frames": all_metrics, "num_frames": len(all_metrics)}, f, indent=2)
        if all_convergence:
            conv_path = week6_base / seq / "convergence.json"
            with open(conv_path, "w") as f:
                json.dump({"per_frame": all_convergence}, f, indent=2)
        print(f"[Stage 3] {seq} tracked {len(all_metrics)} frames")
    return len(all_metrics) > 0


def save_run_config(week6_base: Path, args: argparse.Namespace) -> None:
    """Save run config for reproducibility."""
    cfg = {
        "sequences": args.sequences,
        "stage": args.stage,
        "frames": args.frames,
        "neutral_frame": args.neutral_frame,
        "max_iter": args.max_iter,
        "lambda_alpha": args.lambda_alpha,
        "lambda_delta": args.lambda_delta,
        "temporal_smoothing": args.temporal_smoothing,
        "model_dir": str(args.model_dir),
        "landmark_mapping": str(args.landmark_mapping),
    }
    path = week6_base / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Run config saved to {path}")


def main() -> int:
    args = parse_args()
    week6_base = args.week6_dir or (args.output_root / "week6")
    week6_base.mkdir(parents=True, exist_ok=True)
    seq_dirs = get_converted_seq_dirs(args.output_root, args.sequences)
    if not seq_dirs:
        print("No converted sequences found. Run pipeline with conversion first.")
        return 1

    run_config: Dict[str, Any] = {"args": vars(args)}
    save_run_config(week6_base, args)

    for seq in args.sequences:
        if seq not in seq_dirs:
            continue
        seq_dir = seq_dirs[seq]
        if 1 in args.stage:
            run_stage1(args, seq, seq_dir, week6_base, run_config)
        if 2 in args.stage:
            run_stage2(args, seq, seq_dir, week6_base)
        if 3 in args.stage:
            run_stage3(args, seq, seq_dir, week6_base)

    if args.aggregate:
        agg_script = REPO_ROOT / "scripts" / "aggregate_summary.py"
        if agg_script.exists():
            cmd = [sys.executable, str(agg_script), "--week6-dir", str(week6_base)]
            subprocess.run(cmd, capture_output=True, timeout=30)
            print("Summary written to", week6_base / "summary.csv")

    print("Week 6 evaluation done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
