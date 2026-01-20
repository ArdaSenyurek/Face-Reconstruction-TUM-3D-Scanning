#!/usr/bin/env python3
"""
Create a simple mean-shape-only morphable model from a list of point clouds.

This mirrors the Week 3 commit logic:
- Load multiple PLY point clouds
- Downsample all to the smallest vertex count
- Compute vertex-wise mean
- Save mean_shape.bin plus empty PCA bases/stddevs

Note: No faces/connectivity are generated here. Use triangulation separately
      if you need faces (e.g., scripts/legacy/triangulate_pointcloud.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np


def _load_ply_vertices(filepath: Path) -> np.ndarray:
    """Load vertices from an ASCII PLY file."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            parts = line.split()
            num_vertices = int(parts[2])
        elif line.startswith("end_header"):
            header_end = i + 1
            break
    verts: List[List[float]] = []
    for line in lines[header_end : header_end + num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            verts.append([x, y, z])
    return np.asarray(verts, dtype=np.float64)


def _resample_to_min(vertices_list: List[np.ndarray]) -> List[np.ndarray]:
    """Downsample each vertex array to the minimum vertex count."""
    if not vertices_list:
        return []
    min_vertices = min(len(v) for v in vertices_list)
    resampled: List[np.ndarray] = []
    for verts in vertices_list:
        if len(verts) > min_vertices:
            idx = np.linspace(0, len(verts) - 1, min_vertices, dtype=int)
            resampled.append(verts[idx])
        else:
            resampled.append(verts)
    return resampled


def _save_binary_vector(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(np.float64).tofile(str(path))


def create_mean_shape_from_pointclouds(pointcloud_files: Iterable[str], output_dir: str, use_single: bool = False) -> bool:
    """
    Compute a mean-shape-only model from PLY point clouds.

    Args:
        pointcloud_files: iterable of PLY file paths
        output_dir: directory to write model files
        use_single: if True, only use the first valid point cloud
    Returns:
        True on success, False otherwise
    """
    files = [Path(p) for p in pointcloud_files if Path(p).exists()]
    if not files:
        print("No valid point cloud files provided.")
        return False

    if use_single:
        files = files[:1]

    vertices_list: List[np.ndarray] = []
    for pc in files:
        try:
            verts = _load_ply_vertices(pc)
            if len(verts) == 0:
                continue
            vertices_list.append(verts)
            print(f"Loaded {pc} ({len(verts)} vertices)")
        except Exception as e:
            print(f"Warning: failed to load {pc}: {e}")

    if not vertices_list:
        print("Failed to load any vertices from point clouds.")
        return False

    resampled = _resample_to_min(vertices_list)
    if not resampled:
        print("Resampling failed.")
        return False

    mean_shape = np.mean(resampled, axis=0)
    mean_flat = mean_shape.flatten()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_binary_vector(out_dir / "mean_shape.bin", mean_flat)

    # Empty PCA bases/stddevs (no variability)
    dim = len(mean_flat)
    _save_binary_vector(out_dir / "identity_basis.bin", np.zeros((dim, 0), dtype=np.float64))
    _save_binary_vector(out_dir / "expression_basis.bin", np.zeros((dim, 0), dtype=np.float64))
    _save_binary_vector(out_dir / "identity_stddev.bin", np.zeros((0,), dtype=np.float64))
    _save_binary_vector(out_dir / "expression_stddev.bin", np.zeros((0,), dtype=np.float64))

    print(f"✓ Saved mean shape to {out_dir}")
    print(f"  vertices: {len(mean_shape)}  dimension: {len(mean_flat)}")
    print("⚠️ Faces/connectivity not generated here; triangulate separately if needed.")
    return True


__all__ = ["create_mean_shape_from_pointclouds"]


