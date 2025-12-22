#!/usr/bin/env python3
"""
Triangulate point cloud to create mesh with faces.
Uses Poisson reconstruction or Delaunay triangulation.
"""

import numpy as np
import sys
import os
import argparse

def load_ply_vertices(filepath):
    """Load vertices from PLY file"""
    vertices = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header_end = 0
    num_vertices = 0
    
    for i, line in enumerate(lines):
        if 'element vertex' in line:
            parts = line.split()
            num_vertices = int(parts[2])
        elif 'end_header' in line:
            header_end = i + 1
            break
    
    for line in lines[header_end:header_end+num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            vertices.append([x, y, z])
    
    return np.array(vertices)

def triangulate_with_open3d(vertices, output_path):
    """Triangulate using Open3D Poisson reconstruction"""
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        # Estimate normals
        print("Estimating normals...")
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Poisson surface reconstruction
        print("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low density vertices (outliers)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Save
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"✓ Saved mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
        return True
        
    except ImportError:
        print("Open3D not available, trying alternative method...")
        return False

def triangulate_with_scipy(vertices, output_path):
    """Triangulate using scipy Delaunay (2D projection)"""
    try:
        from scipy.spatial import Delaunay
        
        # Project to 2D (use X-Y plane, assuming face is roughly frontal)
        # Or use PCA to find best projection plane
        print("Computing PCA to find best projection plane...")
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        pca.fit(vertices)
        
        # Project to plane perpendicular to first principal component
        # (removes depth variation)
        projected = vertices @ pca.components_[:2].T
        
        # Delaunay triangulation
        print("Running Delaunay triangulation...")
        tri = Delaunay(projected)
        
        # Save PLY
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(tri.simplices)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces
            for simplex in tri.simplices:
                f.write(f"3 {simplex[0]} {simplex[1]} {simplex[2]}\n")
        
        print(f"✓ Saved mesh with {len(vertices)} vertices and {len(tri.simplices)} faces")
        return True
        
    except ImportError:
        print("scipy/sklearn not available")
        return False

def triangulate_simple(vertices, output_path):
    """Simple triangulation using nearest neighbors"""
    print("Using simple nearest-neighbor triangulation...")
    
    # Simple approach: create faces from nearest neighbors
    # This is a basic method, not ideal but works
    from scipy.spatial.distance import cdist
    
    # For each point, find 2 nearest neighbors and create triangle
    # (This is simplified, proper triangulation is better)
    print("Warning: Simple triangulation may not produce good results.")
    print("Consider installing open3d: pip install open3d")
    
    # For now, just save vertices (no faces)
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    
    print(f"✓ Saved point cloud (no faces - install open3d for triangulation)")
    return False

def main():
    parser = argparse.ArgumentParser(description='Triangulate point cloud to mesh')
    parser.add_argument('input_ply', help='Input PLY point cloud file')
    parser.add_argument('output_ply', help='Output PLY mesh file')
    parser.add_argument('--method', choices=['auto', 'open3d', 'scipy', 'simple'],
                       default='auto', help='Triangulation method')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_ply):
        print(f"Error: {args.input_ply} not found")
        return 1
    
    print(f"Loading point cloud from: {args.input_ply}")
    vertices = load_ply_vertices(args.input_ply)
    print(f"Loaded {len(vertices)} vertices")
    
    # Try different methods
    success = False
    
    if args.method == 'auto' or args.method == 'open3d':
        if triangulate_with_open3d(vertices, args.output_ply):
            success = True
    
    if not success and (args.method == 'auto' or args.method == 'scipy'):
        if triangulate_with_scipy(vertices, args.output_ply):
            success = True
    
    if not success:
        triangulate_simple(vertices, args.output_ply)
        print("\n⚠️  Could not create proper mesh. Install open3d for better results:")
        print("   pip install open3d")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

