#!/usr/bin/env python3
"""
Utility to center a PLY mesh at origin for better visualization.
Moves mesh centroid to (0,0,0) and optionally scales it.
"""

import sys
import numpy as np

def load_ply_vertices(filepath):
    """Load vertices from PLY file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header_end = 0
    num_vertices = 0
    num_faces = 0
    
    for i, line in enumerate(lines):
        if 'element vertex' in line:
            parts = line.split()
            if 'vertex' in parts:
                num_vertices = int(parts[2])
        elif 'element face' in line:
            parts = line.split()
            if 'face' in parts:
                num_faces = int(parts[2])
        elif 'end_header' in line:
            header_end = i + 1
            break
    
    vertices = []
    for line in lines[header_end:header_end+num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            vertices.append([x, y, z])
    
    faces = []
    for line in lines[header_end+num_vertices:header_end+num_vertices+num_faces]:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == '3':
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    
    return np.array(vertices), faces, header_end, num_vertices, num_faces

def save_ply(filepath, vertices, faces, header_end, num_vertices, num_faces, original_lines):
    """Save PLY file with updated vertices"""
    with open(filepath, 'w') as f:
        # Write header
        for i in range(header_end):
            f.write(original_lines[i])
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python center_mesh.py <input.ply> <output.ply> [--scale SCALE]")
        print("\nExample:")
        print("  python center_mesh.py build/aligned_mesh_step2.ply build/aligned_mesh_centered.ply")
        print("  python center_mesh.py build/aligned_mesh_step2.ply build/aligned_mesh_centered.ply --scale 2.0")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    scale = 1.0
    
    if '--scale' in sys.argv:
        idx = sys.argv.index('--scale')
        if idx + 1 < len(sys.argv):
            scale = float(sys.argv[idx + 1])
    
    print(f"Loading mesh from: {input_file}")
    vertices, faces, header_end, num_vertices, num_faces = load_ply_vertices(input_file)
    
    print(f"Loaded {len(vertices)} vertices, {len(faces)} faces")
    
    # Compute centroid
    centroid = vertices.mean(axis=0)
    print(f"\nOriginal centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    
    # Center at origin
    vertices_centered = vertices - centroid
    
    # Apply scale if requested
    if scale != 1.0:
        vertices_centered *= scale
        print(f"Applied scale: {scale}")
    
    new_centroid = vertices_centered.mean(axis=0)
    print(f"New centroid: ({new_centroid[0]:.4f}, {new_centroid[1]:.4f}, {new_centroid[2]:.4f})")
    
    # Compute bounds
    min_vals = vertices_centered.min(axis=0)
    max_vals = vertices_centered.max(axis=0)
    print(f"\nNew bounds:")
    print(f"  X: [{min_vals[0]:.4f}, {max_vals[0]:.4f}] ({max_vals[0]-min_vals[0]:.4f}m)")
    print(f"  Y: [{min_vals[1]:.4f}, {max_vals[1]:.4f}] ({max_vals[1]-min_vals[1]:.4f}m)")
    print(f"  Z: [{min_vals[2]:.4f}, {max_vals[2]:.4f}] ({max_vals[2]-min_vals[2]:.4f}m)")
    
    # Read original file for header
    with open(input_file, 'r') as f:
        original_lines = f.readlines()
    
    # Save centered mesh
    save_ply(output_file, vertices_centered, faces, header_end, num_vertices, num_faces, original_lines)
    print(f"\n✓ Saved centered mesh to: {output_file}")
    print("  You can now open it in MeshLab and use 'Fit to Screen'")

if __name__ == '__main__':
    main()

