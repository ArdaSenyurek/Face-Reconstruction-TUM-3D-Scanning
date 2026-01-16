#!/usr/bin/env python3
"""
Extract faces from a PLY mesh file and save as faces.bin for the morphable model.
This is a simpler alternative if you have a mesh file with faces already.
"""
from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import numpy as np


def load_ply_faces(ply_path: str) -> np.ndarray:
    """Load faces from a PLY file."""
    faces = []
    with open(ply_path, "r") as f:
        lines = f.readlines()
    
    header_end = 0
    num_vertices = 0
    num_faces = 0
    in_header = True
    
    for i, line in enumerate(lines):
        if "element vertex" in line:
            num_vertices = int(line.split()[2])
        elif "element face" in line:
            num_faces = int(line.split()[2])
        elif "end_header" in line:
            header_end = i + 1
            break
    
    if num_faces == 0:
        raise ValueError("PLY file has no faces (element face count is 0)")
    
    # Read faces
    for line in lines[header_end + num_vertices:header_end + num_vertices + num_faces]:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "3":
            # Format: "3 v1 v2 v3"
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    
    return np.array(faces, dtype=np.int32)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract faces from PLY mesh and save as faces.bin"
    )
    parser.add_argument(
        "ply_file",
        type=str,
        help="Input PLY mesh file with faces"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output faces.bin path (default: same directory as PLY)"
    )
    args = parser.parse_args()
    
    ply_path = Path(args.ply_file)
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        return 1
    
    print(f"Loading faces from: {ply_path}")
    try:
        faces = load_ply_faces(str(ply_path))
        print(f"Loaded {len(faces)} faces")
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: save in model_biwi directory
        output_path = Path("data/model_biwi/faces.bin")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save faces
    save_faces_binary(faces, str(output_path))
    
    print(f"\n✓ Face connectivity file created: {output_path}")
    print(f"  Faces: {len(faces)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


