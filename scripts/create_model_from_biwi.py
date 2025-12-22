#!/usr/bin/env python3
"""
Create a simple face model from Biwi dataset point clouds.
This creates a mean shape from Biwi depth data, not a full PCA model.
"""

import numpy as np
import struct
import os
import sys
import argparse
from pathlib import Path

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

def create_mean_shape_from_pointclouds(pointcloud_files, output_dir):
    """Create mean shape from multiple point clouds"""
    print(f"Loading {len(pointcloud_files)} point clouds...")
    
    all_vertices = []
    min_vertices = float('inf')
    
    # Load all point clouds
    for i, pc_file in enumerate(pointcloud_files):
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
    
    # Resample to same number of vertices (use minimum)
    print(f"\nResampling to {min_vertices} vertices...")
    resampled = []
    for vertices in all_vertices:
        if len(vertices) > min_vertices:
            # Simple downsampling: take every Nth vertex
            indices = np.linspace(0, len(vertices)-1, min_vertices, dtype=int)
            resampled.append(vertices[indices])
        else:
            resampled.append(vertices)
    
    # Compute mean
    print("Computing mean shape...")
    mean_shape = np.mean(resampled, axis=0)
    
    # Flatten to vector format [x1, y1, z1, x2, y2, z2, ...]
    mean_shape_flat = mean_shape.flatten()
    
    # Save as binary
    os.makedirs(output_dir, exist_ok=True)
    mean_shape_path = os.path.join(output_dir, 'mean_shape.bin')
    with open(mean_shape_path, 'wb') as f:
        mean_shape_flat.astype(np.float64).tofile(f)
    
    print(f"✓ Saved mean shape to: {mean_shape_path}")
    print(f"  Vertices: {len(mean_shape)}")
    print(f"  Dimension: {len(mean_shape_flat)}")
    
    # Create dummy basis (zero matrices - no PCA components)
    num_vertices = len(mean_shape)
    dim = len(mean_shape_flat)
    
    # Identity basis (empty - 0 components)
    identity_basis = np.zeros((dim, 0), dtype=np.float64)
    identity_basis_path = os.path.join(output_dir, 'identity_basis.bin')
    with open(identity_basis_path, 'wb') as f:
        identity_basis.tofile(f)
    print(f"✓ Saved identity_basis to: {identity_basis_path} (0 components)")
    
    # Expression basis (empty - 0 components)
    expression_basis = np.zeros((dim, 0), dtype=np.float64)
    expression_basis_path = os.path.join(output_dir, 'expression_basis.bin')
    with open(expression_basis_path, 'wb') as f:
        expression_basis.tofile(f)
    print(f"✓ Saved expression_basis to: {expression_basis_path} (0 components)")
    
    # Stddev (empty)
    identity_stddev = np.array([], dtype=np.float64)
    expression_stddev = np.array([], dtype=np.float64)
    
    identity_stddev_path = os.path.join(output_dir, 'identity_stddev.bin')
    with open(identity_stddev_path, 'wb') as f:
        identity_stddev.tofile(f)
    print(f"✓ Saved identity_stddev to: {identity_stddev_path}")
    
    expression_stddev_path = os.path.join(output_dir, 'expression_stddev.bin')
    with open(expression_stddev_path, 'wb') as f:
        expression_stddev.tofile(f)
    print(f"✓ Saved expression_stddev to: {expression_stddev_path}")
    
    # Load faces from first point cloud if available, or create simple faces
    # For now, we'll create a simple mesh connectivity
    # In practice, you'd want to use Delaunay triangulation or similar
    print(f"\n⚠️  Note: Face connectivity not generated automatically.")
    print(f"   You may need to triangulate the point cloud separately.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create face model from Biwi point clouds')
    parser.add_argument('--pointclouds', nargs='+', required=True,
                       help='PLY point cloud files from Biwi')
    parser.add_argument('--output', required=True,
                       help='Output directory for model files')
    parser.add_argument('--single', action='store_true',
                       help='Use single point cloud (no averaging)')
    
    args = parser.parse_args()
    
    if args.single and len(args.pointclouds) > 1:
        print("Using first point cloud only (--single mode)")
        args.pointclouds = [args.pointclouds[0]]
    
    if create_mean_shape_from_pointclouds(args.pointclouds, args.output):
        print(f"\n✓ Model created successfully in: {args.output}")
        print(f"  This is a simple mean shape model (no PCA components).")
        print(f"  It can be used for alignment but not for shape variation.")
    else:
        print("\n✗ Failed to create model")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

