#!/usr/bin/env python3
"""
Create PLY point cloud from RGB-D frame using C++ face_reconstruction binary.

This utility converts depth images to 3D point clouds using camera intrinsics.
Used when PLY files are missing from the dataset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def create_pointcloud_from_rgbd(
    rgb_path: Path,
    depth_path: Path,
    intrinsics_path: Path,
    output_ply: Path,
    binary_path: Path = Path("build/bin/face_reconstruction"),
    depth_scale: float = 1000.0,
    timeout: int = 60
) -> bool:
    """
    Create PLY point cloud from RGB-D frame using C++ binary.
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image
        intrinsics_path: Path to camera intrinsics file
        output_ply: Output PLY file path
        binary_path: Path to face_reconstruction binary
        depth_scale: Depth scale factor (default: 1000.0 for mm to meters)
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if not binary_path.exists():
        print(f"Error: Binary not found: {binary_path}", file=sys.stderr)
        return False
    
    if not rgb_path.exists():
        print(f"Error: RGB image not found: {rgb_path}", file=sys.stderr)
        return False
    
    if not depth_path.exists():
        print(f"Error: Depth image not found: {depth_path}", file=sys.stderr)
        return False
    
    if not intrinsics_path.exists():
        print(f"Error: Intrinsics file not found: {intrinsics_path}", file=sys.stderr)
        return False
    
    # Create output directory
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        str(binary_path),
        "--rgb", str(rgb_path),
        "--depth", str(depth_path),
        "--intrinsics", str(intrinsics_path),
        "--depth-scale", str(depth_scale),
        "--output-pointcloud", str(output_ply),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Verify output file was created
        if output_ply.exists() and output_ply.stat().st_size > 0:
            return True
        else:
            print(f"Error: Point cloud file not created or empty: {output_ply}", file=sys.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed: {e}", file=sys.stderr)
        if e.stderr:
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out after {timeout}s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


def create_pointclouds_from_converted_data(
    converted_root: Path,
    output_dir: Path,
    binary_path: Path = Path("build/bin/face_reconstruction"),
    max_sequences: Optional[int] = None,
    frames_per_sequence: int = 1,
    depth_scale: float = 1000.0
) -> list[Path]:
    """
    Create PLY point clouds from converted RGB-D data.
    
    Args:
        converted_root: Root directory of converted data (outputs/converted/)
        output_dir: Directory to save PLY files
        binary_path: Path to face_reconstruction binary
        max_sequences: Maximum number of sequences to process (None = all)
        frames_per_sequence: Number of frames per sequence to convert
        depth_scale: Depth scale factor
        
    Returns:
        List of created PLY file paths
    """
    created_ply_files = []
    
    # Find all sequences
    sequence_dirs = sorted([d for d in converted_root.iterdir() if d.is_dir()])
    
    if max_sequences:
        sequence_dirs = sequence_dirs[:max_sequences]
    
    for seq_dir in sequence_dirs:
        rgb_dir = seq_dir / "rgb"
        depth_dir = seq_dir / "depth"
        intrinsics_path = seq_dir / "intrinsics.txt"
        
        if not (rgb_dir.exists() and depth_dir.exists() and intrinsics_path.exists()):
            continue
        
        # Get frames
        rgb_frames = sorted(rgb_dir.glob("frame_*.png"))[:frames_per_sequence]
        
        for rgb_frame in rgb_frames:
            depth_frame = depth_dir / rgb_frame.name
            if not depth_frame.exists():
                continue
            
            # Create output PLY path
            output_ply = output_dir / seq_dir.name / f"{rgb_frame.stem}.ply"
            
            # Create point cloud
            success = create_pointcloud_from_rgbd(
                rgb_frame,
                depth_frame,
                intrinsics_path,
                output_ply,
                binary_path,
                depth_scale
            )
            
            if success:
                created_ply_files.append(output_ply)
    
    return created_ply_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create PLY point cloud from RGB-D frame"
    )
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image path")
    parser.add_argument("--depth", type=Path, required=True, help="Depth image path")
    parser.add_argument("--intrinsics", type=Path, required=True, help="Intrinsics file path")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY path")
    parser.add_argument("--binary", type=Path, default=Path("build/bin/face_reconstruction"),
                       help="Path to face_reconstruction binary")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                       help="Depth scale factor (default: 1000.0)")
    
    args = parser.parse_args()
    
    success = create_pointcloud_from_rgbd(
        args.rgb,
        args.depth,
        args.intrinsics,
        args.output,
        args.binary,
        args.depth_scale
    )
    
    sys.exit(0 if success else 1)

