#!/usr/bin/env python3
"""
Triangulate point cloud to mesh using open3d (preferred) or scipy Delaunay fallback.
"""
from __future__ import annotations

import argparse
import os
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


def triangulate_with_open3d(vertices: np.ndarray, output_path: str) -> bool:
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        print("Estimating normals...")
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        print("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"✓ Saved mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
        return True
    except ImportError:
        print("Open3D not available.")
        return False


def triangulate_with_scipy(vertices: np.ndarray, output_path: str) -> bool:
    try:
        from scipy.spatial import Delaunay
        from sklearn.decomposition import PCA

        print("Computing PCA to find projection plane...")
        pca = PCA(n_components=3)
        pca.fit(vertices)
        projected = vertices @ pca.components_[:2].T
        print("Running Delaunay triangulation...")
        tri = Delaunay(projected)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(tri.simplices)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for simplex in tri.simplices:
                f.write(f"3 {simplex[0]} {simplex[1]} {simplex[2]}\n")
        print(f"✓ Saved mesh with {len(vertices)} vertices and {len(tri.simplices)} faces")
        return True
    except ImportError:
        print("scipy/sklearn not available")
        return False


def triangulate_simple(vertices: np.ndarray, output_path: str) -> None:
    print("Simple mode: writing point cloud only (no faces).")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    print(f"✓ Saved point cloud (no faces). Install open3d for full triangulation.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Triangulate point cloud to mesh.")
    ap.add_argument("input_ply", help="Input PLY point cloud file")
    ap.add_argument("output_ply", help="Output PLY mesh file")
    ap.add_argument("--method", choices=["auto", "open3d", "scipy", "simple"], default="auto")
    args = ap.parse_args()

    if not os.path.exists(args.input_ply):
        print(f"Error: {args.input_ply} not found")
        return 1

    print(f"Loading point cloud from: {args.input_ply}")
    vertices = load_ply_vertices(args.input_ply)
    print(f"Loaded {len(vertices)} vertices")

    success = False
    if args.method in ("auto", "open3d"):
        success = triangulate_with_open3d(vertices, args.output_ply)
    if not success and args.method in ("auto", "scipy"):
        success = triangulate_with_scipy(vertices, args.output_ply)
    if not success:
        triangulate_simple(vertices, args.output_ply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

