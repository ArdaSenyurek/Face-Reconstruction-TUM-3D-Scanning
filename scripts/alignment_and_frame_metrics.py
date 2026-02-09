#!/usr/bin/env python3
"""
Report alignment quality and frame-to-frame differences.

1. Rigid alignment: use pose_init JSON reports (pre/post Procrustes/ICP RMSE).
   Run: python scripts/analyze_sparse_alignment.py --reports-dir outputs/pose_init --output-dir outputs/analysis

2. Rigid vs non-rigid (optimized): read overlay metrics JSONs (rigid.nn_rmse_m vs optimized.nn_rmse_m).
   This script finds *_overlay_metrics.json and prints improvement (rigid - opt) in mm.

3. Frame-to-frame differences: read tracking summary JSON/CSV and compute deltas
   (translation, rotation angle, expression_norm) between consecutive frames.

Usage:
  python scripts/alignment_and_frame_metrics.py --overlay-metrics-dir outputs/overlays_3d
  python scripts/alignment_and_frame_metrics.py --tracking-summary outputs/analysis/tracking_summary_01.json
  python scripts/alignment_and_frame_metrics.py --pose-init-dir outputs/pose_init --output-dir outputs/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_overlay_metrics(metrics_path: Path) -> Dict[str, Any] | None:
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def report_rigid_vs_opt(overlay_metrics_dir: Path) -> None:
    """Report rigid vs optimized NN-RMSE and improvement from overlay metrics."""
    if not overlay_metrics_dir.exists():
        print(f"Directory not found: {overlay_metrics_dir}")
        return
    files = list(overlay_metrics_dir.rglob("*_overlay_metrics.json"))
    if not files:
        print(f"No *_overlay_metrics.json under {overlay_metrics_dir}")
        return
    print("Rigid vs optimized (NN-RMSE mesh–scan, meters → mm)")
    print("-" * 60)
    for p in sorted(files):
        try:
            data = load_overlay_metrics(p)
        except Exception as e:
            print(f"  {p}: error {e}")
            continue
        rigid = data.get("rigid", {})
        opt = data.get("optimized")
        r_m = rigid.get("nn_rmse_m")
        if r_m is None:
            continue
        r_mm = r_m * 1000.0
        rel = p.relative_to(overlay_metrics_dir) if overlay_metrics_dir in p.parents else p.name
        print(f"  {rel}")
        print(f"    rigid NN-RMSE:    {r_mm:.2f} mm")
        if opt is not None:
            o_m = opt.get("nn_rmse_m")
            if o_m is not None:
                o_mm = o_m * 1000.0
                improvement_mm = r_mm - o_mm
                print(f"    optimized NN-RMSE: {o_mm:.2f} mm")
                print(f"    improvement:       {improvement_mm:.2f} mm (rigid - opt)")
        print()
    return


def report_frame_deltas(tracking_summary_path: Path) -> None:
    """Compute and print frame-to-frame differences from tracking summary JSON."""
    if not tracking_summary_path.exists():
        print(f"File not found: {tracking_summary_path}")
        return
    with open(tracking_summary_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if len(frames) < 2:
        print("Need at least 2 frames for deltas.")
        return
    print(f"Frame-to-frame differences: {tracking_summary_path.name}")
    print("-" * 60)
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]
        dx = curr.get("translation_x", 0) - prev.get("translation_x", 0)
        dy = curr.get("translation_y", 0) - prev.get("translation_y", 0)
        dz = curr.get("translation_z", 0) - prev.get("translation_z", 0)
        d_rot = curr.get("rotation_angle_deg", 0) - prev.get("rotation_angle_deg", 0)
        d_scale = curr.get("scale", 0) - prev.get("scale", 0)
        d_expr = curr.get("expression_norm", 0) - prev.get("expression_norm", 0)
        dist_t = np.sqrt(dx * dx + dy * dy + dz * dz)
        print(f"  {prev.get('frame_name')} -> {curr.get('frame_name')}:")
        print(f"    delta translation (m): ({dx:.5f}, {dy:.5f}, {dz:.5f})  norm={dist_t:.5f}")
        print(f"    delta rotation (deg):  {d_rot:.4f}")
        print(f"    delta scale:            {d_scale:.6f}")
        print(f"    delta expression_norm: {d_expr:.4f}")
        print()
    return


def main() -> int:
    ap = argparse.ArgumentParser(description="Report alignment and frame-to-frame metrics")
    ap.add_argument("--overlay-metrics-dir", type=Path, default=None,
                    help="Directory containing *_overlay_metrics.json (rigid vs opt)")
    ap.add_argument("--tracking-summary", type=Path, default=None,
                    help="Path to tracking_summary_<seq>.json for frame deltas")
    ap.add_argument("--pose-init-dir", type=Path, default=None,
                    help="Directory with pose_init reports; suggest running analyze_sparse_alignment.py")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Optional output dir for analyze_sparse_alignment")
    args = ap.parse_args()

    did_something = False
    if args.overlay_metrics_dir is not None:
        report_rigid_vs_opt(args.overlay_metrics_dir)
        did_something = True
    if args.tracking_summary is not None:
        report_frame_deltas(args.tracking_summary)
        did_something = True
    if args.pose_init_dir is not None:
        print("Rigid alignment (pose_init) metrics:")
        print("  Run: python scripts/analyze_sparse_alignment.py")
        print(f"       --reports-dir {args.pose_init_dir}")
        if args.output_dir is not None:
            print(f"       --output-dir {args.output_dir}")
        print("  This produces summary stats and before/after alignment plots.")
        did_something = True
    if not did_something:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
