#!/usr/bin/env python3
"""
Compute depth stats, point-cloud exports, and mesh distance RMSE for selected frames.
Optionally measures reconstruction runtime by re-invoking the C++ binary.

Example:
  python scripts/eval_depth_and_runtime.py \
    --converted-root outputs/converted \
    --meshes-root outputs/meshes \
    --sequences 01 02 \
    --frames 0 1 2 3 4 \
    --binary build/bin/test_real_data \
    --model-dir data/model_biwi \
    --output outputs/analysis/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


def load_intrinsics(path: Path) -> Tuple[float, float, float, float]:
    with open(path, "r", encoding="utf-8") as f:
        parts = f.readline().strip().split()
    fx, fy, cx, cy = map(float, parts[:4])
    return fx, fy, cx, cy


def depth_to_points(depth_path: Path, intrinsics: Tuple[float, float, float, float], max_points: int = 50000) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth: {depth_path}")
    depth = depth.astype(np.float32)
    mask = depth > 0
    ys, xs = np.nonzero(mask)
    zs = depth[mask]
    if zs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    # convert mm -> meters if values look large
    if zs.max() > 20.0:
        zs = zs / 1000.0
    xs_f = (xs - cx) * zs / fx
    ys_f = (ys - cy) * zs / fy
    pts = np.stack([xs_f, ys_f, zs], axis=1)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts


def load_mesh_vertices(ply_path: Path) -> np.ndarray:
    # Minimal PLY ASCII reader (vertices only)
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines[0].strip().startswith("ply"):
        raise RuntimeError(f"Not a PLY file: {ply_path}")
    num_vertices = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[2])
        if line.strip() == "end_header":
            header_end = i + 1
            break
    verts = []
    for line in lines[header_end : header_end + num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(verts, dtype=np.float32)


def save_pointcloud_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_depth_vis(depth_path: Path, out_path: Path) -> Dict[str, float]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth: {depth_path}")
    depth_f = depth.astype(np.float32)
    mask = depth_f > 0
    stats = {
        "min": float(depth_f[mask].min()) if mask.any() else 0.0,
        "max": float(depth_f[mask].max()) if mask.any() else 0.0,
        "mean": float(depth_f[mask].mean()) if mask.any() else 0.0,
        "std": float(depth_f[mask].std()) if mask.any() else 0.0,
    }
    # visualize
    norm = np.zeros_like(depth_f, dtype=np.uint8)
    if mask.any():
        cv2.normalize(depth_f, norm, 0, 255, cv2.NORM_MINMAX)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return stats


def compute_cloud_to_mesh_rmse(cloud: np.ndarray, mesh: np.ndarray, sample: int = 20000) -> float:
    if cloud.shape[0] == 0 or mesh.shape[0] == 0:
        return float("nan")
    if cloud.shape[0] > sample:
        idx = np.random.choice(cloud.shape[0], sample, replace=False)
        cloud = cloud[idx]
    tree = cKDTree(mesh)
    dists, _ = tree.query(cloud, k=1)
    rmse = float(np.sqrt(np.mean(dists ** 2)))
    return rmse


def measure_runtime(binary: Path, rgb: Path, depth: Path, intrinsics: Path, model_dir: Path, out_mesh: Path, timeout: int) -> float:
    cmd = [
        str(binary),
        "--rgb",
        str(rgb),
        "--depth",
        str(depth),
        "--intrinsics",
        str(intrinsics),
        "--model-dir",
        str(model_dir),
        "--output-mesh",
        str(out_mesh),
    ]
    start = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    end = time.time()
    return end - start


def main() -> int:
    ap = argparse.ArgumentParser(description="Depth stats, point-cloud export, and mesh distance RMSE.")
    ap.add_argument("--converted-root", type=Path, default=Path("outputs/converted"))
    ap.add_argument("--meshes-root", type=Path, default=Path("outputs/meshes"))
    ap.add_argument("--output", type=Path, default=Path("outputs/analysis/eval_metrics.json"))
    ap.add_argument("--sequences", nargs="*", help="Sequence ids (e.g., 01 02). Default: auto-detect.")
    ap.add_argument("--frames", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--binary", type=Path, default=Path("build/bin/test_real_data"))
    ap.add_argument("--model-dir", type=Path, default=Path("data/model_biwi"))
    ap.add_argument("--measure-runtime", action="store_true")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    seq_ids: Sequence[str]
    if args.sequences:
        seq_ids = args.sequences
    else:
        seq_ids = [p.name for p in sorted(args.converted_root.iterdir()) if p.is_dir()]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sid in seq_ids:
        seq_dir = args.converted_root / sid
        mesh_dir = args.meshes_root / sid
        intr_path = seq_dir / "intrinsics.txt"
        if not intr_path.exists():
            print(f"[WARN] Missing intrinsics for {sid}, skipping")
            continue
        intr = load_intrinsics(intr_path)
        seq_res: Dict[str, Dict[str, float]] = {}

        for fi in args.frames:
            rgb = seq_dir / "rgb" / f"frame_{fi:05d}.png"
            depth = seq_dir / "depth" / f"frame_{fi:05d}.png"
            mesh = mesh_dir / f"frame_{fi:05d}.ply"
            if not (rgb.exists() and depth.exists() and mesh.exists()):
                continue

            # Depth stats + vis
            depth_vis_out = Path("outputs/analysis/depth_vis") / sid / f"frame_{fi:05d}.png"
            stats = save_depth_vis(depth, depth_vis_out)

            # Point cloud + PLY
            cloud = depth_to_points(depth, intr)
            pc_out = Path("outputs/analysis/pointclouds") / sid / f"frame_{fi:05d}.ply"
            save_pointcloud_ply(cloud, pc_out)

            # Mesh distance RMSE
            verts = load_mesh_vertices(mesh)
            rmse = compute_cloud_to_mesh_rmse(cloud, verts)

            seq_res[f"frame_{fi:05d}"] = {
                "depth_min": stats["min"],
                "depth_max": stats["max"],
                "depth_mean": stats["mean"],
                "depth_std": stats["std"],
                "cloud_points": int(cloud.shape[0]),
                "rmse_cloud_mesh_m": rmse,
            }

            # Optional runtime measurement (reruns reconstruction)
            if args.measure_runtime:
                tmp_mesh = Path("outputs/analysis/runtime_meshes") / sid / f"frame_{fi:05d}_tmp.ply"
                tmp_mesh.parent.mkdir(parents=True, exist_ok=True)
                try:
                    runtime = measure_runtime(args.binary, rgb, depth, intr_path, args.model_dir, tmp_mesh, args.timeout)
                    seq_res[f"frame_{fi:05d}"]["runtime_seconds"] = runtime
                except Exception as exc:  # pragma: no cover
                    seq_res[f"frame_{fi:05d}"]["runtime_seconds"] = float("nan")
                    print(f"[WARN] Runtime measurement failed for {sid} frame {fi:05d}: {exc}")

        if seq_res:
            results[sid] = seq_res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Wrote eval metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

