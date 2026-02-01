#!/usr/bin/env python3
"""
Week 6: Export 3D overlay PLY for a given sequence and frame (rigid or optimized).

Calls create_overlays binary and writes to outputs/week6/<seq>/<frame>/overlays_3d/
or copies to reports/week6/figures/ when --report is set.

Usage:
  python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage rigid
  python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage opt
  python scripts/export_overlay_ply.py --seq 01 --frame 00000 --stage rigid --report

Open in MeshLab: point size 3–4, verify overlap (scan cyan, mesh red).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Export overlay PLY (rigid or optimized mesh + scan)")
    ap.add_argument("--seq", type=str, required=True, help="Sequence ID (e.g. 01)")
    ap.add_argument("--frame", type=str, default="00000", help="Frame index (e.g. 00000)")
    ap.add_argument("--stage", type=str, required=True, choices=["rigid", "opt"],
                    help="rigid = rigid-aligned mesh + scan; opt = optimized mesh + scan")
    ap.add_argument("--output-root", type=Path, default=Path("outputs"))
    ap.add_argument("--week6-dir", type=Path, default=None)
    ap.add_argument("--overlay-binary", type=Path, default=REPO_ROOT / "build/bin/create_overlays")
    ap.add_argument("--report", action="store_true", help="Copy to reports/week6/figures/")
    args = ap.parse_args()

    frame_name = f"frame_{args.frame}" if not args.frame.startswith("frame_") else args.frame
    week6 = args.week6_dir or (args.output_root / "week6")
    converted = args.output_root / "converted" / args.seq
    pose_init_root = args.output_root / "pose_init"
    meshes_root = args.output_root / "meshes"
    week6_meshes = week6 / args.seq / frame_name / "meshes"

    depth_path = converted / "depth" / f"{frame_name}.png"
    intrinsics_path = converted / "intrinsics.txt"
    rigid_mesh = pose_init_root / args.seq / f"{frame_name}_aligned.ply"
    opt_mesh = week6_meshes / "expression.ply"
    if not opt_mesh.exists():
        opt_mesh = week6_meshes / "tracked.ply"
    if not opt_mesh.exists():
        opt_mesh = meshes_root / args.seq / f"{frame_name}_optimized.ply"
    if not opt_mesh.exists():
        opt_mesh = meshes_root / args.seq / f"{frame_name}_tracked.ply"

    if not depth_path.exists() or not intrinsics_path.exists():
        print("Missing depth or intrinsics:", depth_path, intrinsics_path)
        return 1
    if args.stage == "rigid" and not rigid_mesh.exists():
        print("Missing rigid mesh:", rigid_mesh)
        return 1
    if args.stage == "opt" and not opt_mesh.exists():
        print("Missing optimized mesh:", opt_mesh)
        return 1
    if not args.overlay_binary.exists():
        print("Overlay binary not found:", args.overlay_binary)
        return 1

    out_dir = week6 / args.seq / frame_name / "overlays_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_rigid = rigid_mesh if rigid_mesh.exists() else opt_mesh
    mesh_opt = opt_mesh if opt_mesh.exists() else rigid_mesh
    cmd = [
        str(args.overlay_binary),
        "--mesh-rigid", str(mesh_rigid),
        "--mesh-opt", str(mesh_opt),
        "--depth", str(depth_path),
        "--intrinsics", str(intrinsics_path),
        "--out-dir", str(out_dir),
        "--frame-name", frame_name,
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    print("Overlay written to", out_dir)

    if args.report:
        report_fig = REPO_ROOT / "reports" / "week6" / "figures"
        report_fig.mkdir(parents=True, exist_ok=True)
        for ply in out_dir.glob("*.ply"):
            dest = report_fig / f"{args.seq}_{frame_name}_{args.stage}_{ply.name}"
            shutil.copy2(ply, dest)
            print("Copied to", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
