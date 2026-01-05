#!/usr/bin/env python3
"""
Create a simple face model (mean shape only) from Biwi point clouds.
Outputs mean_shape.bin and empty PCA bases/stddevs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np


def load_ply_vertices(filepath: str) -> np.ndarray:
    vertices: List[List[float]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        if "element vertex" in line:
            parts = line.split()
            num_vertices = int(parts[2])
        elif "end_header" in line:
            header_end = i + 1
            break
    for line in lines[header_end : header_end + num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            vertices.append([x, y, z])
    return np.array(vertices)


def create_mean_shape_from_pointclouds(pointcloud_files: List[str], output_dir: str, single: bool) -> bool:
    print(f"Loading {len(pointcloud_files)} point clouds...")
    all_vertices = []
    min_vertices = float("inf")
    if single and pointcloud_files:
        pointcloud_files = [pointcloud_files[0]]
    for pc_file in pointcloud_files:
        if not os.path.exists(pc_file):
            print(f"Warning: {pc_file} not found, skipping")
            continue
        vertices = load_ply_vertices(pc_file)
        all_vertices.append(vertices)
        min_vertices = min(min_vertices, len(vertices))
        print(f"  Loaded {pc_file}: {len(vertices)} vertices")
    if not all_vertices:
        print("Error: No valid point clouds found!")
        return False
    print(f"\nResampling to {min_vertices} vertices...")
    resampled = []
    for vertices in all_vertices:
        if len(vertices) > min_vertices:
            indices = np.linspace(0, len(vertices) - 1, min_vertices, dtype=int)
            resampled.append(vertices[indices])
        else:
            resampled.append(vertices)
    print("Computing mean shape...")
    mean_shape = np.mean(resampled, axis=0)
    mean_shape_flat = mean_shape.flatten()
    os.makedirs(output_dir, exist_ok=True)
    mean_shape_path = os.path.join(output_dir, "mean_shape.bin")
    mean_shape_flat.astype(np.float64).tofile(mean_shape_path)
    print(f"✓ Saved mean shape to: {mean_shape_path}")
    print(f"  Vertices: {len(mean_shape)}")
    print(f"  Dimension: {len(mean_shape_flat)}")
    num_vertices = len(mean_shape)
    dim = len(mean_shape_flat)
    identity_basis = np.zeros((dim, 0), dtype=np.float64)
    identity_basis_path = os.path.join(output_dir, "identity_basis.bin")
    identity_basis.tofile(identity_basis_path)
    print(f"✓ Saved identity_basis to: {identity_basis_path} (0 components)")
    expression_basis = np.zeros((dim, 0), dtype=np.float64)
    expression_basis_path = os.path.join(output_dir, "expression_basis.bin")
    expression_basis.tofile(expression_basis_path)
    print(f"✓ Saved expression_basis to: {expression_basis_path} (0 components)")
    identity_stddev = np.array([], dtype=np.float64)
    identity_stddev_path = os.path.join(output_dir, "identity_stddev.bin")
    identity_stddev.tofile(identity_stddev_path)
    print(f"✓ Saved identity_stddev to: {identity_stddev_path}")
    expression_stddev = np.array([], dtype=np.float64)
    expression_stddev_path = os.path.join(output_dir, "expression_stddev.bin")
    expression_stddev.tofile(expression_stddev_path)
    print(f"✓ Saved expression_stddev to: {expression_stddev_path}")
    print("\nNote: Face connectivity not generated automatically; triangulate separately if needed.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Create mean shape model from Biwi point clouds (no PCA).")
    ap.add_argument("--pointclouds", nargs="+", required=True, help="PLY point cloud files")
    ap.add_argument("--output", required=True, help="Output directory for model files")
    ap.add_argument("--single", action="store_true", help="Use only the first point cloud")
    args = ap.parse_args()
    ok = create_mean_shape_from_pointclouds(args.pointclouds, args.output, args.single)
    if ok:
        print("\n✓ Model created (mean shape only).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

