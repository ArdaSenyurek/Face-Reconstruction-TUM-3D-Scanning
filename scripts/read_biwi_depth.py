#!/usr/bin/env python3
"""
Experimental script to read Biwi depth.bin files.
This attempts various decoding strategies based on the binary format.
"""

import struct
import numpy as np
import sys
from pathlib import Path

def read_biwi_depth_attempt1(depth_path):
    """
    Attempt 1: Header + zlib compressed
    Format: [width:4][height:4][compressed_size:4][zlib_data...]
    """
    try:
        import zlib
        with open(depth_path, 'rb') as f:
            data = f.read()
        
        if len(data) < 12:
            return None
        
        width = struct.unpack('<I', data[0:4])[0]
        height = struct.unpack('<I', data[4:8])[0]
        compressed_size = struct.unpack('<I', data[8:12])[0]
        
        compressed_data = data[12:]
        try:
            decompressed = zlib.decompress(compressed_data)
            if len(decompressed) == width * height * 2:
                depth = np.frombuffer(decompressed, dtype=np.uint16).reshape(height, width)
                return depth
        except:
            pass
    except:
        pass
    return None

def read_biwi_depth_attempt2(depth_path):
    """
    Attempt 2: Simple RLE decoding (if format is run-length encoded)
    This is a simplified attempt - real format may be more complex
    """
    try:
        with open(depth_path, 'rb') as f:
            data = f.read()
        
        width = struct.unpack('<I', data[0:4])[0]
        height = struct.unpack('<I', data[4:8])[0]
        
        # Skip header and potential size field
        idx = 8
        depth = np.zeros((height, width), dtype=np.uint16)
        
        # This is a placeholder - actual RLE format needs reverse engineering
        # For now, try reading as uint16 values directly if size matches
        remaining = len(data) - idx
        if remaining >= width * height * 2:
            depth_flat = np.frombuffer(data[idx:idx+width*height*2], dtype=np.uint16)
            depth = depth_flat.reshape(height, width)
            return depth
    except:
        pass
    return None

def read_biwi_depth_simple(depth_path, width=640, height=480):
    """
    Simple approach: If we know dimensions, try reading raw uint16
    """
    try:
        with open(depth_path, 'rb') as f:
            # Skip header (8 bytes)
            f.seek(8)
            # Try reading as uint16 array
            data = f.read()
            if len(data) >= width * height * 2:
                depth = np.frombuffer(data[:width*height*2], dtype=np.uint16)
                depth = depth.reshape(height, width)
                # Filter out obviously invalid values
                depth = np.clip(depth, 0, 10000)  # Reasonable depth range in mm
                return depth
    except Exception as e:
        print(f"Error: {e}")
    return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python read_biwi_depth.py <depth.bin> [output.png]")
        sys.exit(1)
    
    depth_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Try different methods
    depth = None
    method = None
    
    # Try attempt 1 (zlib)
    depth = read_biwi_depth_attempt1(depth_path)
    if depth is not None:
        method = "zlib decompression"
    
    # Try attempt 2
    if depth is None:
        depth = read_biwi_depth_attempt2(depth_path)
        if depth is not None:
            method = "RLE/raw decode"
    
    # Try simple approach
    if depth is None:
        depth = read_biwi_depth_simple(depth_path)
        if depth is not None:
            method = "simple raw read"
    
    if depth is not None:
        print(f"Successfully read depth using: {method}")
        print(f"Shape: {depth.shape}, Range: {depth.min()}-{depth.max()}")
        
        if output_path:
            import cv2
            # Save as 16-bit PNG
            cv2.imwrite(output_path, depth.astype(np.uint16))
            print(f"Saved to: {output_path}")
    else:
        print("Failed to read depth file")
        sys.exit(1)
