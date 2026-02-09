#!/usr/bin/env python3
"""
Week 6: Generate 2D overlay (RGB + mesh silhouette) and depth comparison (obs / rend / residual).

Outputs:
  outputs/week6/<seq>/<frame>/overlay_rgb.png   — RGB with projected mesh edges/silhouette
  outputs/week6/<seq>/<frame>/depth_obs.png     — observed depth (colormap)
  outputs/week6/<seq>/<frame>/depth_rend.png    — rendered mesh depth (colormap)
  outputs/week6/<seq>/<frame>/depth_residual.png — residual (obs - rend) or side-by-side

Usage:
  python scripts/generate_week6_visuals.py --seq 01 --frame 00000 --mesh <ply> --depth <png> --intrinsics <txt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def project_verts(fx: float, fy: float, cx: float, cy: float, verts: np.ndarray) -> np.ndarray:
    z = verts[:, 2]
    mask = z > 1e-6
    u = np.zeros(len(verts), dtype=np.float32)
    v = np.zeros(len(verts), dtype=np.float32)
    u[mask] = fx * verts[mask, 0] / z[mask] + cx
    v[mask] = fy * verts[mask, 1] / z[mask] + cy
    return np.stack([u, v], axis=1)


def render_depth(verts: np.ndarray, fx: float, fy: float, cx: float, cy: float, h: int, w: int) -> np.ndarray:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=str, default="01")
    ap.add_argument("--frame", type=str, default="00000")
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--depth", type=Path, required=True)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--rgb", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    frame_name = f"frame_{args.frame}" if not args.frame.startswith("frame_") else args.frame
    out_dir = args.output_dir or (REPO_ROOT / "outputs" / "week6" / args.seq / frame_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts = load_ply_vertices(args.mesh)
    if len(verts) == 0:
        print("No vertices in mesh")
        return 1
    with open(args.intrinsics) as f:
        parts = f.read().strip().split()
    fx, fy, cx, cy = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    depth_img = cv2.imread(str(args.depth), cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        print("Failed to load depth")
        return 1
    h, w = depth_img.shape
    depth_obs = depth_img.astype(np.float32) / 1000.0

    # Rendered depth (mesh in same frame as observed)
    depth_rend = render_depth(verts, fx, fy, cx, cy, h, w)

    # Depth visualizations (colormap)
    def to_vis(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = d > 0 if mask is None else mask
        if not np.any(valid):
            return np.zeros((h, w, 3), dtype=np.uint8)
        v = d.copy()
        v[~valid] = np.nan
        mn, mx = np.nanmin(v), np.nanmax(v)
        if mx <= mn:
            mx = mn + 1
        vn = np.zeros_like(d)
        vn[valid] = (v[valid] - mn) / (mx - mn)
        vn = (np.clip(vn, 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(vn, cv2.COLORMAP_JET)

    valid_obs = depth_obs > 0
    valid_rend = depth_rend > 0
    depth_obs_vis = to_vis(depth_obs, valid_obs)
    depth_rend_vis = to_vis(depth_rend, valid_rend)
    valid_both = valid_obs & valid_rend
    residual = np.zeros_like(depth_obs)
    residual[valid_both] = (depth_obs[valid_both] - depth_rend[valid_both]) * 1000.0  # mm
    residual_vis = to_vis(np.clip(np.abs(residual), 0, 50), valid_both)

    cv2.imwrite(str(out_dir / "depth_obs.png"), depth_obs_vis)
    cv2.imwrite(str(out_dir / "depth_rend.png"), depth_rend_vis)
    cv2.imwrite(str(out_dir / "depth_residual.png"), residual_vis)
    print("Wrote depth_obs.png, depth_rend.png, depth_residual.png to", out_dir)

    # Overlay RGB: project mesh and draw edges/silhouette
    if args.rgb and args.rgb.exists():
        rgb = cv2.imread(str(args.rgb))
        if rgb is not None:
            uv = project_verts(fx, fy, cx, cy, verts)
            for i in range(len(uv)):
                u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
                if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                    cv2.circle(rgb, (u, v), 1, (0, 255, 0), -1)
            overlay_rgb_path = out_dir / "overlay_rgb.png"
            cv2.imwrite(str(overlay_rgb_path), rgb)
            print("Wrote overlay_rgb.png to", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
