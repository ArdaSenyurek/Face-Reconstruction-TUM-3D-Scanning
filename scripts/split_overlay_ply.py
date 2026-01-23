#!/usr/bin/env python3
"""
Split combined PLY overlay into separate scan and mesh files.
"""

import sys
from pathlib import Path

def split_ply_overlay(input_ply: Path, output_scan: Path, output_mesh: Path):
    """Split PLY overlay into scan points and mesh."""
    with open(input_ply, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header_end = 0
    num_vertices = 0
    num_faces = 0
    scan_vertex_count = 0
    
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[-1])
        elif line.startswith("element face"):
            num_faces = int(line.split()[-1])
        elif line.strip() == "end_header":
            header_end = i + 1
            break
    
    # Read vertices
    vertices = []
    for i in range(header_end, header_end + num_vertices):
        parts = lines[i].strip().split()
        if len(parts) >= 6:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            vertices.append((x, y, z, r, g, b))
    
    # Find where scan points end (blue points: 0, 0, 255)
    scan_count = 0
    for v in vertices:
        if v[3] == 0 and v[4] == 0 and v[5] == 255:
            scan_count += 1
        else:
            break
    
    print(f"Found {scan_count} scan points (blue) and {len(vertices) - scan_count} mesh vertices (red)")
    
    # Write scan points PLY
    output_scan.parent.mkdir(parents=True, exist_ok=True)
    with open(output_scan, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {scan_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(scan_count):
            v = vertices[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]} {v[4]} {v[5]}\n")
    
    # Read faces (they reference mesh vertices with offset)
    faces = []
    face_start = header_end + num_vertices
    for i in range(face_start, face_start + num_faces):
        parts = lines[i].strip().split()
        if len(parts) >= 4 and parts[0] == "3":
            v0, v1, v2 = int(parts[1]), int(parts[2]), int(parts[3])
            # Remove offset to get mesh vertex indices
            v0 -= scan_count
            v1 -= scan_count
            v2 -= scan_count
            if v0 >= 0 and v1 >= 0 and v2 >= 0:
                faces.append((v0, v1, v2))
    
    # Write mesh PLY
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    with open(output_mesh, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices) - scan_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(scan_count, len(vertices)):
            v = vertices[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]} {v[4]} {v[5]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"✓ Scan points saved to: {output_scan}")
    print(f"✓ Mesh saved to: {output_mesh}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 split_overlay_ply.py <overlay_ply> [output_scan_ply] [output_mesh_ply]")
        sys.exit(1)
    
    input_ply = Path(sys.argv[1])
    if len(sys.argv) >= 4:
        output_scan = Path(sys.argv[2])
        output_mesh = Path(sys.argv[3])
    else:
        # Auto-generate output names
        output_scan = input_ply.parent / f"{input_ply.stem}_scan.ply"
        output_mesh = input_ply.parent / f"{input_ply.stem}_mesh.ply"
    
    split_ply_overlay(input_ply, output_scan, output_mesh)
