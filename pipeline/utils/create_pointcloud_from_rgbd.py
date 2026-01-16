#!/usr/bin/env python3
"""
Create a point cloud PLY from an RGB-D frame and intrinsics.

This avoids calling the C++ recon binary (which requires a model). It simply
backprojects the depth map using intrinsics and writes an ASCII PLY of points.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def _load_intrinsics(intrinsics_path: Path) -> Tuple[float, float, float, float]:
    """
    Load intrinsics from a text file.
    Accepts either:
        fx fy cx cy   (single line)
    or two lines like Biwi depth.cal:
        fx 0 cx
        0 fy cy
    Falls back to Kinect defaults if parsing fails.
    """
    try:
        lines = intrinsics_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            raise ValueError("empty intrinsics file")
        first = lines[0].split()
        if len(first) >= 4:
            fx, fy, cx, cy = map(float, first[:4])
            return fx, fy, cx, cy
        if len(lines) >= 2:
            fx = float(lines[0].split()[0])
            cx = float(lines[0].split()[2])
            fy = float(lines[1].split()[1])
            cy = float(lines[1].split()[2])
            return fx, fy, cx, cy
    except Exception:
        pass
    # Fallback to Kinect v1 intrinsics
    return 525.0, 525.0, 319.5, 239.5


def _write_ply(points: np.ndarray, output_ply: Path) -> None:
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ply, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def create_pointcloud_from_rgbd(
    rgb_path: Path,
    depth_path: Path,
    intrinsics_path: Path,
    output_ply: Path,
    binary: Path | None = None,  # kept for API compatibility, unused
    depth_scale: float = 1000.0,
    timeout: int = 30,  # unused, kept for signature compatibility
) -> bool:
    """
    Backproject depth to a point cloud and save as PLY.
    Returns True on success.
    """
    if not depth_path.exists():
        return False

    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return False

    if depth.dtype != np.float32 and depth.dtype != np.float64:
        depth = depth.astype(np.float32) / float(depth_scale)

    h, w = depth.shape[:2]
    fx, fy, cx, cy = _load_intrinsics(intrinsics_path)

    u = np.repeat(np.arange(w)[np.newaxis, :], h, axis=0)
    v = np.repeat(np.arange(h)[:, np.newaxis], w, axis=1)

    z = depth
    mask = z > 0
    z = z[mask]
    if z.size == 0:
        return False

    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy

    points = np.stack((x, y, z), axis=1)
    _write_ply(points, output_ply)
    return True


__all__ = ["create_pointcloud_from_rgbd"]


