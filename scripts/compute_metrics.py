#!/usr/bin/env python3
"""
Week 6: Compute quantitative metrics for a single frame.

- 2D landmark reprojection error (mean/median/RMSE in pixels)
- Depth error: observed vs rendered depth (MAE/RMSE in mm) on valid overlap
- 3D surface error: pointcloud-to-mesh NN distances (mean/median/RMSE, % under 5/10/20 mm)

Usage:
  python scripts/compute_week6_metrics.py --mesh <ply> --depth <png> --intrinsics <txt> \\
    --landmarks <txt> --mapping <txt> --model-dir <dir> [--pointcloud <ply>] --output <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_ply_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path, "r") as f:
        in_header = True
        nv = 0
        for line in f:
            if in_header:
                if "element vertex" in line:
                    nv = int(line.split()[-1])
                if "end_header" in line:
                    in_header = False
            else:
                if len(verts) < nv:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))


def load_landmarks_txt(path: Path) -> np.ndarray:
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def load_intrinsics(path: Path) -> tuple[float, float, float, float]:
    with open(path) as f:
        parts = f.read().strip().split()
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def project_vertices(fx: float, fy: float, cx: float, cy: float, verts: np.ndarray) -> np.ndarray:
    z = verts[:, 2]
    mask = z > 1e-6
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    u[mask] = fx * verts[mask, 0] / z[mask] + cx
    v[mask] = fy * verts[mask, 1] / z[mask] + cy
    return np.stack([u, v], axis=1)


def landmark_reprojection_error(
    mesh_verts: np.ndarray,
    vertex_indices: list[int],
    landmarks_2d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> dict:
    if not vertex_indices or len(landmarks_2d) == 0:
        return {"mean_px": float("nan"), "median_px": float("nan"), "rmse_px": float("nan")}
    n = min(len(vertex_indices), len(landmarks_2d))
    errors = []
    for i in range(n):
        vi = vertex_indices[i]
        if vi < 0 or vi >= len(mesh_verts):
            continue
        p = mesh_verts[vi]
        if p[2] <= 0:
            continue
        u = fx * p[0] / p[2] + cx
        v = fy * p[1] / p[2] + cy
        d = np.sqrt((u - landmarks_2d[i, 0]) ** 2 + (v - landmarks_2d[i, 1]) ** 2)
        errors.append(d)
    if not errors:
        return {"mean_px": float("nan"), "median_px": float("nan"), "rmse_px": float("nan")}
    err = np.array(errors)
    return {
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
    }


def depth_error_mm(depth_obs: np.ndarray, depth_rend: np.ndarray, depth_scale: float = 1000.0) -> dict:
    valid = (depth_obs > 0) & (depth_rend > 0)
    if not np.any(valid):
        return {"mae_mm": float("nan"), "rmse_mm": float("nan"), "num_pixels": 0}
    do = depth_obs[valid].astype(np.float64) / depth_scale  # mm -> m if scale 1000
    dr = depth_rend[valid].astype(np.float64)
    diff_mm = np.abs(do - dr) * 1000.0  # m -> mm
    return {
        "mae_mm": float(np.mean(diff_mm)),
        "rmse_mm": float(np.sqrt(np.mean(diff_mm ** 2))),
        "num_pixels": int(np.sum(valid)),
    }


def surface_error_stats(cloud: np.ndarray, mesh_verts: np.ndarray, thresholds_mm: list[float]) -> dict:
    if cloud.shape[0] == 0 or mesh_verts.shape[0] == 0:
        return {
            "mean_mm": float("nan"), "median_mm": float("nan"), "rmse_mm": float("nan"),
            **{f"pct_under_{t}mm": float("nan") for t in thresholds_mm},
        }
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {
            "mean_mm": float("nan"), "median_mm": float("nan"), "rmse_mm": float("nan"),
            **{f"pct_under_{t}mm": float("nan") for t in thresholds_mm},
        }
    tree = cKDTree(mesh_verts)
    d, _ = tree.query(cloud, k=1)
    d_mm = d * 1000.0
    n = len(d_mm)
    out = {
        "mean_mm": float(np.mean(d_mm)),
        "median_mm": float(np.median(d_mm)),
        "rmse_mm": float(np.sqrt(np.mean(d_mm ** 2))),
    }
    for t in thresholds_mm:
        out[f"pct_under_{t}mm"] = float(100.0 * np.sum(d_mm <= t) / n)
    return out


def render_depth_simple(verts: np.ndarray, fx: float, fy: float, cx: float, cy: float, h: int, w: int) -> np.ndarray:
    depth = np.zeros((h, w), dtype=np.float32)
    for i in range(len(verts)):
        x, y, z = verts[i, 0], verts[i, 1], verts[i, 2]
        if z <= 0:
            continue
        u = int(round(fx * x / z + cx))
        v = int(round(fy * y / z + cy))
        if 0 <= u < w and 0 <= v < h:
            if depth[v, u] == 0 or z < depth[v, u]:
                depth[v, u] = z
    return depth


def load_mapping_vertex_indices(mapping_path: Path) -> list[int]:
    inds = []
    with open(mapping_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                inds.append(int(parts[1]))
    return inds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--depth", type=Path, required=True)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, default=REPO_ROOT / "data/bfm_landmark_68.txt")
    ap.add_argument("--pointcloud", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    args = ap.parse_args()

    import cv2
    mesh_verts = load_ply_vertices(args.mesh)
    landmarks_2d = load_landmarks_txt(args.landmarks)
    fx, fy, cx, cy = load_intrinsics(args.intrinsics)
    depth_img = cv2.imread(str(args.depth), cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        print("Failed to load depth", file=sys.stderr)
        return 1
    h, w = depth_img.shape
    depth_obs = depth_img.astype(np.float32) / args.depth_scale

    metrics: dict = {}

    # 2D landmark reprojection (need vertex indices from mapping; use first N landmarks as proxy if no mapping)
    mapping_inds = load_mapping_vertex_indices(args.mapping) if args.mapping.exists() else []
    if mapping_inds and len(mapping_inds) <= len(landmarks_2d):
        lm_err = landmark_reprojection_error(
            mesh_verts, mapping_inds[: len(landmarks_2d)], landmarks_2d, fx, fy, cx, cy
        )
        metrics["landmark_reprojection"] = lm_err
    else:
        metrics["landmark_reprojection"] = {"mean_px": float("nan"), "median_px": float("nan"), "rmse_px": float("nan")}

    # Depth error: render mesh depth (m) then compare to observed (m), report mm
    depth_rend = render_depth_simple(mesh_verts, fx, fy, cx, cy, h, w)
    valid = (depth_obs > 0) & (depth_rend > 0)
    if np.any(valid):
        diff_mm = np.abs(depth_obs[valid] - depth_rend[valid]) * 1000.0
        metrics["depth_error"] = {
            "mae_mm": float(np.mean(diff_mm)),
            "rmse_mm": float(np.sqrt(np.mean(diff_mm ** 2))),
            "num_pixels": int(np.sum(valid)),
        }
    else:
        metrics["depth_error"] = {"mae_mm": float("nan"), "rmse_mm": float("nan"), "num_pixels": 0}

    # 3D surface error: pointcloud vs mesh
    if args.pointcloud and args.pointcloud.exists():
        cloud = load_ply_vertices(args.pointcloud)
        metrics["surface_error"] = surface_error_stats(cloud, mesh_verts, [5, 10, 20])
    else:
        # Backproject depth to pointcloud
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        z = depth_obs.astype(np.float64)
        valid = z > 0
        x = (xx[valid] - cx) * z[valid] / fx
        y = (yy[valid] - cy) * z[valid] / fy
        cloud = np.stack([x, y, z[valid]], axis=1)
        if len(cloud) > 50000:
            rng = np.random.default_rng(42)
            cloud = cloud[rng.choice(len(cloud), 50000, replace=False)]
        metrics["surface_error"] = surface_error_stats(cloud, mesh_verts, [5, 10, 20])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
