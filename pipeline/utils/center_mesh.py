#!/usr/bin/env python3
"""
Center a PLY mesh at the origin and optionally scale it.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np


def load_ply_vertices(filepath: str) -> Tuple[np.ndarray, List[List[int]], int, int, int, List[str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_end = 0
    num_vertices = 0
    num_faces = 0
    for i, line in enumerate(lines):
        if "element vertex" in line:
            parts = line.split()
            num_vertices = int(parts[2])
        elif "element face" in line:
            parts = line.split()
            num_faces = int(parts[2])
        elif "end_header" in line:
            header_end = i + 1
            break
    vertices = []
    for line in lines[header_end : header_end + num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            vertices.append([x, y, z])
    faces: List[List[int]] = []
    for line in lines[header_end + num_vertices : header_end + num_vertices + num_faces]:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "3":
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.array(vertices), faces, header_end, num_vertices, num_faces, lines


def save_ply(
    filepath: str, vertices: np.ndarray, faces: List[List[int]], header_end: int, original_lines: List[str]
) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(header_end):
            f.write(original_lines[i])
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python center_mesh.py <input.ply> <output.ply> [--scale SCALE]")
        return 1
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    scale = 1.0
    if "--scale" in sys.argv:
        idx = sys.argv.index("--scale")
        if idx + 1 < len(sys.argv):
            scale = float(sys.argv[idx + 1])

    print(f"Loading mesh from: {input_file}")
    vertices, faces, header_end, num_vertices, num_faces, lines = load_ply_vertices(input_file)
    print(f"Loaded {len(vertices)} vertices, {len(faces)} faces")

    centroid = vertices.mean(axis=0)
    print(f"Original centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    vertices_centered = vertices - centroid
    if scale != 1.0:
        vertices_centered *= scale
        print(f"Applied scale: {scale}")
    new_centroid = vertices_centered.mean(axis=0)
    min_vals = vertices_centered.min(axis=0)
    max_vals = vertices_centered.max(axis=0)
    print(f"New centroid: ({new_centroid[0]:.4f}, {new_centroid[1]:.4f}, {new_centroid[2]:.4f})")
    print("New bounds:")
    print(f"  X: [{min_vals[0]:.4f}, {max_vals[0]:.4f}]")
    print(f"  Y: [{min_vals[1]:.4f}, {max_vals[1]:.4f}]")
    print(f"  Z: [{min_vals[2]:.4f}, {max_vals[2]:.4f}]")

    save_ply(output_file, vertices_centered, faces, header_end, lines)
    print(f"✓ Saved centered mesh to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

