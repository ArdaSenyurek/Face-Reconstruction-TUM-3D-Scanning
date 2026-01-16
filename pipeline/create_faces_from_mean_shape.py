#!/usr/bin/env python3
"""
Create faces.bin file from mean_shape.bin using Open3D triangulation.
This generates face connectivity for the morphable model.
"""
from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np


def load_binary_vector(filepath: str) -> np.ndarray:
    """Load binary vector file (double precision floats)."""
    with open(filepath, "rb") as f:
        data = f.read()
    num_elements = len(data) // 8  # double = 8 bytes
    values = struct.unpack("d" * num_elements, data)
    return np.array(values, dtype=np.float64)


def load_mean_shape(model_dir: str) -> np.ndarray:
    """Load mean shape and reshape to vertices (N x 3)."""
    bin_path = os.path.join(model_dir, "mean_shape.bin")
    txt_path = os.path.join(model_dir, "mean_shape.txt")
    
    if os.path.exists(bin_path):
        mean_shape = load_binary_vector(bin_path)
    elif os.path.exists(txt_path):
        mean_shape = np.loadtxt(txt_path)
    else:
        raise FileNotFoundError(f"Mean shape not found in {model_dir}")
    
    num_vertices = len(mean_shape) // 3
    if len(mean_shape) % 3 != 0:
        raise ValueError(f"Mean shape dimension ({len(mean_shape)}) must be divisible by 3")
    
    return mean_shape.reshape(num_vertices, 3)


def triangulate_with_open3d(vertices: np.ndarray) -> np.ndarray:
    """Triangulate vertices using Open3D ball pivoting algorithm."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    print("Estimating normals...")
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    
    print("Running ball pivoting triangulation...")
    # Use ball pivoting to preserve exact vertex count
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    
    radii = [radius, radius * 2]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    # Verify vertex count matches
    mesh_vertices = np.asarray(mesh.vertices)
    if len(mesh_vertices) != len(vertices):
        print(f"Warning: Mesh has {len(mesh_vertices)} vertices, expected {len(vertices)}")
        print("Trying Poisson reconstruction instead...")
        # Fallback to Poisson if ball pivoting changes vertex count
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh_vertices = np.asarray(mesh.vertices)
        if len(mesh_vertices) != len(vertices):
            print(f"Warning: Poisson mesh has {len(mesh_vertices)} vertices, expected {len(vertices)}")
            print("Using mesh as-is, but vertex indices may not match exactly.")
    
    # Extract faces
    faces = np.asarray(mesh.triangles)
    
    # If vertex count doesn't match, we need to map faces to original vertices
    if len(mesh_vertices) != len(vertices):
        # Find closest original vertex for each mesh vertex
        from scipy.spatial.distance import cdist
        distances = cdist(mesh_vertices, vertices)
        closest_indices = np.argmin(distances, axis=1)
        # Remap face indices
        faces = closest_indices[faces]
    
    return faces


def triangulate_with_scipy(vertices: np.ndarray) -> np.ndarray:
    """Triangulate vertices using scipy Delaunay triangulation."""
    try:
        from scipy.spatial import Delaunay
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scipy and sklearn are required. Install with: pip install scipy scikit-learn")
    
    print("Computing PCA to find projection plane...")
    pca = PCA(n_components=3)
    pca.fit(vertices)
    projected = vertices @ pca.components_[:2].T
    
    print("Running Delaunay triangulation...")
    tri = Delaunay(projected)
    faces = tri.simplices
    
    return faces


def save_faces_binary(faces: np.ndarray, output_path: str) -> None:
    """Save faces in binary format compatible with C++ loader."""
    rows, cols = faces.shape
    if cols != 3:
        raise ValueError(f"Faces must have 3 columns, got {cols}")
    
    with open(output_path, "wb") as f:
        # Write header: rows, cols as int32
        f.write(struct.pack("i", rows))  # int32_t rows
        f.write(struct.pack("i", cols))  # int32_t cols
        
        # Write face data as int32
        faces_int32 = faces.astype(np.int32)
        f.write(faces_int32.tobytes())
    
    print(f"✓ Saved {rows} faces to binary file: {output_path}")


def save_faces_text(faces: np.ndarray, output_path: str) -> None:
    """Save faces in text format."""
    rows, cols = faces.shape
    with open(output_path, "w") as f:
        for i in range(rows):
            f.write(f"{faces[i, 0]} {faces[i, 1]} {faces[i, 2]}\n")
    
    print(f"✓ Saved {rows} faces to text file: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate faces.bin from mean_shape.bin using Open3D triangulation"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing mean_shape.bin"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: model_dir/faces.bin)"
    )
    parser.add_argument(
        "--format",
        choices=["bin", "txt", "both"],
        default="bin",
        help="Output format (default: bin)"
    )
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return 1
    
    print(f"Loading mean shape from: {model_dir}")
    try:
        vertices = load_mean_shape(str(model_dir))
        print(f"Loaded {len(vertices)} vertices")
    except Exception as e:
        print(f"Error loading mean shape: {e}")
        return 1
    
    print("Triangulating vertices...")
    faces = None
    # Try Open3D first (better quality)
    try:
        faces = triangulate_with_open3d(vertices)
        print(f"Generated {len(faces)} faces using Open3D")
    except ImportError:
        print("Open3D not available, trying scipy...")
        try:
            faces = triangulate_with_scipy(vertices)
            print(f"Generated {len(faces)} faces using scipy Delaunay")
        except Exception as e:
            print(f"Error triangulating: {e}")
            print("\nPlease install Open3D for better results:")
            print("  pip install open3d")
            print("\nOr install scipy and scikit-learn:")
            print("  pip install scipy scikit-learn")
            return 1
    except Exception as e:
        print(f"Error with Open3D triangulation: {e}")
        print("Trying scipy as fallback...")
        try:
            faces = triangulate_with_scipy(vertices)
            print(f"Generated {len(faces)} faces using scipy Delaunay")
        except Exception as e2:
            print(f"Error triangulating: {e2}")
            return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_dir / "faces.bin"
    
    # Save faces
    if args.format in ("bin", "both"):
        bin_path = output_path if args.format == "bin" else model_dir / "faces.bin"
        save_faces_binary(faces, str(bin_path))
    
    if args.format in ("txt", "both"):
        txt_path = output_path if args.format == "txt" else model_dir / "faces.txt"
        save_faces_text(faces, str(txt_path))
    
    print("\n✓ Face connectivity file(s) created successfully!")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

