#!/usr/bin/env python3
"""
Convert Biwi Kinect Head Pose Dataset to format compatible with face reconstruction pipeline.

Biwi dataset structure:
- RGB images: typically in a directory
- Depth images: typically in a directory
- Camera parameters: may be in a config file or hardcoded

This script:
1. Finds RGB and depth image pairs
2. Extracts or sets camera intrinsics
3. Prepares data in the expected format
"""

import os
import sys
import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path


def find_biwi_files(base_dir):
    """
    Find RGB and depth files in Biwi dataset directory.
    Biwi dataset structure can vary, so we try common patterns.
    """
    base_path = Path(base_dir)
    
    rgb_files = []
    depth_files = []
    
    # Common Biwi dataset patterns
    patterns = [
        # Pattern 1: RGB and depth in separate directories
        (base_path / "rgb", "*.png"),
        (base_path / "depth", "*.png"),
        # Pattern 2: Mixed in same directory with prefixes
        (base_path, "*rgb*.png"),
        (base_path, "*depth*.png"),
        # Pattern 3: Subdirectories for each person/sequence
        (base_path, "**/rgb*.png"),
        (base_path, "**/depth*.png"),
        # Pattern 4: Indexed files
        (base_path, "*_r*.png"),
        (base_path, "*_d*.png"),
    ]
    
    # Try to find RGB and depth files
    for pattern_base, pattern in patterns[::2]:  # RGB patterns
        if pattern_base.exists():
            rgb_files.extend(pattern_base.glob(pattern))
    
    for pattern_base, pattern in patterns[1::2]:  # Depth patterns
        if pattern_base.exists():
            depth_files.extend(pattern_base.glob(pattern))
    
    # Sort and match RGB-depth pairs
    rgb_files = sorted(set(rgb_files))
    depth_files = sorted(set(depth_files))
    
    return rgb_files, depth_files


def match_rgb_depth_pairs(rgb_files, depth_files):
    """Match RGB and depth files that likely correspond to the same frame."""
    pairs = []
    
    # Create a map of depth files by their base name
    depth_map = {}
    for depth_file in depth_files:
        stem = depth_file.stem
        # Extract frame number (e.g., "frame_00003" from "frame_00003_depth")
        base_name = stem.replace("_depth", "").replace("depth", "").replace("_D", "").replace("D", "")
        depth_map[(depth_file.parent, base_name)] = depth_file
    
    for rgb_file in rgb_files:
        rgb_stem = rgb_file.stem
        rgb_parent = rgb_file.parent
        
        # Extract frame number from RGB file
        base_name = rgb_stem.replace("_rgb", "").replace("rgb", "").replace("_RGB", "").replace("RGB", "")
        
        # Try to find matching depth file
        key = (rgb_parent, base_name)
        if key in depth_map:
            pairs.append((rgb_file, depth_map[key]))
        else:
            # Fallback: try same directory with depth suffix
            depth_candidate = rgb_parent / f"{base_name}_depth.bin"
            if depth_candidate.exists():
                pairs.append((rgb_file, depth_candidate))
    
    return pairs


def read_biwi_calibration(cal_file):
    """
    Read Biwi depth.cal file to get camera intrinsics.
    Format:
    fx 0 cx
    0 fy cy
    0 0 1
    ...
    640 480 (resolution)
    """
    try:
        with open(cal_file, 'r') as f:
            lines = f.readlines()
            # First line: fx 0 cx
            parts = lines[0].strip().split()
            fx = float(parts[0])
            cx = float(parts[2])
            # Second line: 0 fy cy
            parts = lines[1].strip().split()
            fy = float(parts[1])
            cy = float(parts[2])
            return fx, fy, cx, cy
    except:
        return None

def get_kinect_v1_intrinsics():
    """
    Get Kinect v1 camera intrinsics (typical for Biwi dataset).
    These are approximate values - you may need to calibrate for your specific setup.
    """
    # Kinect v1 typical intrinsics (640x480)
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    return fx, fy, cx, cy


def get_kinect_v2_intrinsics():
    """
    Get Kinect v2 camera intrinsics.
    """
    # Kinect v2 typical intrinsics (512x424)
    fx = 365.0
    fy = 365.0
    cx = 255.5
    cy = 211.5
    return fx, fy, cx, cy


def read_biwi_depth_binary(depth_path):
    """
    Read Biwi compressed binary depth format.
    Biwi dataset uses Run-Length Encoding (RLE) format:
    Format: [width:4][height:4][num_empty:4][num_full:4][depth_values:2*num_full][...]
    """
    try:
        with open(str(depth_path), 'rb') as f:
            # Check if it's a standard image format first
            first_bytes = f.read(8)
            if first_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                # It's a PNG, use OpenCV
                return None
            
            # Reset to beginning
            f.seek(0)
            
            # Read width and height (int32, little endian)
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            
            # Verify reasonable dimensions
            if width < 100 or width > 2000 or height < 100 or height > 2000:
                return None
            
            # Initialize depth image (zeros = invalid/empty pixels)
            depth_img = np.zeros((height, width), dtype=np.int16)
            
            # Decode RLE format
            p = 0  # Current pixel position in flattened array
            total_pixels = width * height
            
            while p < total_pixels:
                # Read number of empty pixels (zero depth)
                try:
                    num_empty = np.fromfile(f, dtype=np.int32, count=1)[0]
                except (EOFError, ValueError):
                    break
                
                # Skip empty pixels
                p += num_empty
                
                if p >= total_pixels:
                    break
                
                # Read number of pixels with depth values
                try:
                    num_full = np.fromfile(f, dtype=np.int32, count=1)[0]
                except (EOFError, ValueError):
                    break
                
                if num_full > 0:
                    # Read depth values (int16)
                    try:
                        depth_values = np.fromfile(f, dtype=np.int16, count=num_full)
                        # Write depth values to image
                        end_idx = min(p + num_full, total_pixels)
                        depth_img.flat[p:end_idx] = depth_values[:end_idx - p]
                        p += num_full
                    except (EOFError, ValueError):
                        break
                else:
                    break
            
            # Convert to uint16 (with zero as invalid)
            depth_img = np.clip(depth_img, 0, 65535).astype(np.uint16)
            
            return depth_img
            
    except Exception as e:
        print(f"Error reading binary depth: {e}")
        return None


def convert_depth_image(depth_path, output_path):
    """
    Convert Biwi depth image to 16-bit PNG format.
    Biwi depth may be in different formats:
    - Compressed binary format (special)
    - PNG format (standard)
    - Raw binary format
    """
    # First, try reading as Biwi binary format
    depth = read_biwi_depth_binary(depth_path)
    
    # If that failed, try OpenCV
    if depth is None:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    
    if depth is None:
        print(f"Warning: Could not load depth image {depth_path}")
        print(f"  File exists: {depth_path.exists()}")
        print(f"  File size: {depth_path.stat().st_size if depth_path.exists() else 'N/A'} bytes")
        return False
    
    # Convert to single channel if needed
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    
    # Check depth range and convert to 16-bit
    if depth.dtype == np.uint8:
        # Scale to 16-bit (assuming depth is in mm, but scale may vary)
        depth = depth.astype(np.uint16) * 256
    elif depth.dtype == np.uint16:
        # Already 16-bit, use as is
        pass
    elif depth.dtype == np.float32 or depth.dtype == np.float64:
        # Convert from float (likely in meters) to mm
        depth = (depth * 1000).astype(np.uint16)
        depth = np.clip(depth, 0, 65535)
    elif depth.dtype == np.int32 or depth.dtype == np.int64:
        # Convert from integer
        depth = np.clip(depth, 0, 65535).astype(np.uint16)
    else:
        # Unknown format, try to convert
        depth = depth.astype(np.uint16)
        depth = np.clip(depth, 0, 65535)
    
    # Ensure valid depth range (typical for Kinect: 0-8000mm)
    depth = np.clip(depth, 0, 8000)
    
    cv2.imwrite(str(output_path), depth)
    return True


def copy_or_convert_rgb(rgb_path, output_path):
    """Copy RGB image, converting if necessary."""
    img = cv2.imread(str(rgb_path))
    if img is None:
        print(f"Warning: Could not load RGB image {rgb_path}")
        return False
    
    cv2.imwrite(str(output_path), img)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Biwi Kinect Head Pose Dataset to face reconstruction format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert entire dataset
  python convert_biwi_dataset.py --input /path/to/biwi/dataset --output data/biwi_test
  
  # Convert specific sequence
  python convert_biwi_dataset.py --input /path/to/biwi/person01 --output data/biwi_test \\
      --max-frames 10 --kinect-version v1
  
  # Use custom intrinsics
  python convert_biwi_dataset.py --input /path/to/biwi --output data/biwi_test \\
      --intrinsics 525.0 525.0 320.0 240.0
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input Biwi dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for converted data')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='Maximum number of frames to convert (0 = all)')
    parser.add_argument('--kinect-version', choices=['v1', 'v2'], default='v1',
                       help='Kinect version (for default intrinsics)')
    parser.add_argument('--intrinsics', type=float, nargs=4, metavar=('FX', 'FY', 'CX', 'CY'),
                       help='Camera intrinsics: fx fy cx cy')
    parser.add_argument('--depth-scale', type=float, default=1000.0,
                       help='Depth scale factor (default: 1000.0 for mm)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning Biwi dataset: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find RGB and depth files
    rgb_files, depth_files = find_biwi_files(input_dir)
    print(f"Found {len(rgb_files)} RGB files and {len(depth_files)} depth files")
    
    # Match pairs
    pairs = match_rgb_depth_pairs(rgb_files, depth_files)
    print(f"Matched {len(pairs)} RGB-depth pairs")
    
    if len(pairs) == 0:
        print("Error: No RGB-depth pairs found!")
        print("\nTroubleshooting:")
        print("1. Check that RGB and depth files exist in the dataset")
        print("2. Verify file naming patterns")
        print("3. Try specifying a subdirectory with --input")
        sys.exit(1)
    
    # Limit number of frames if specified
    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]
        print(f"Processing first {len(pairs)} pairs")
    
    # Get camera intrinsics
    if args.intrinsics:
        fx, fy, cx, cy = args.intrinsics
        print(f"Using provided intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    else:
        # Try to read from depth.cal file in dataset
        cal_file = input_dir / "depth.cal"
        if cal_file.exists():
            cal_result = read_biwi_calibration(cal_file)
            if cal_result:
                fx, fy, cx, cy = cal_result
                print(f"Using intrinsics from depth.cal: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            else:
                if args.kinect_version == 'v1':
                    fx, fy, cx, cy = get_kinect_v1_intrinsics()
                else:
                    fx, fy, cx, cy = get_kinect_v2_intrinsics()
                print(f"Could not read depth.cal, using {args.kinect_version} default intrinsics")
        else:
            if args.kinect_version == 'v1':
                fx, fy, cx, cy = get_kinect_v1_intrinsics()
            else:
                fx, fy, cx, cy = get_kinect_v2_intrinsics()
            print(f"Using {args.kinect_version} default intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Save intrinsics
    intrinsics_file = output_dir / "intrinsics.txt"
    with open(intrinsics_file, 'w') as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")
    print(f"Saved intrinsics to: {intrinsics_file}")
    
    # Process each pair
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    successful_rgb = 0
    successful_depth = 0
    for i, (rgb_path, depth_path) in enumerate(pairs):
        # Create output filenames
        rgb_out = rgb_dir / f"frame_{i:05d}.png"
        depth_out = depth_dir / f"frame_{i:05d}.png"
        
        # Convert/copy files
        rgb_ok = copy_or_convert_rgb(rgb_path, rgb_out)
        if rgb_ok:
            successful_rgb += 1
        
        depth_ok = convert_depth_image(depth_path, depth_out)
        if depth_ok:
            successful_depth += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")
    
    print(f"\n✅ Successfully converted:")
    print(f"   RGB: {successful_rgb}/{len(pairs)} frames")
    print(f"   Depth: {successful_depth}/{len(pairs)} frames")
    
    if successful_depth == 0 and successful_rgb > 0:
        print(f"\n⚠️  Note: Depth files could not be converted (binary format issue).")
        print(f"   You can still test with RGB-only + landmarks:")
        print(f"   python scripts/detect_landmarks.py --image {rgb_dir}/frame_00000.png ...")
    print(f"\nConverted data is in: {output_dir}")
    print(f"\nTo test with a single frame:")
    print(f"  cd build")
    print(f"  ./bin/test_real_data \\")
    print(f"    --rgb ../{output_dir}/rgb/frame_00000.png \\")
    print(f"    --depth ../{output_dir}/depth/frame_00000.png \\")
    print(f"    --intrinsics ../{output_dir}/intrinsics.txt \\")
    print(f"    --model-dir ../data/model \\")
    print(f"    --output-mesh ../{output_dir}/reconstructed_00000.ply")


if __name__ == '__main__':
    main()
