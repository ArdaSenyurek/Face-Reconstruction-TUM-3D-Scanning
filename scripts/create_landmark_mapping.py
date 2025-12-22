#!/usr/bin/env python3
"""
Helper script to create landmark-to-model vertex mapping.

This script helps identify which model vertices correspond to dlib 68-point landmarks.
The mapping is created by analyzing the mean shape geometry and identifying
key facial features.

dlib 68-point landmark order:
  0-16:   Jaw line
  17-21:  Right eyebrow
  22-26:  Left eyebrow
  27-35:  Nose
  36-41:  Right eye
  42-47:  Left eye
  48-67:  Mouth
"""

import numpy as np
import struct
import sys
import os

def load_binary_vector(filepath):
    """Load binary vector file (double precision)"""
    with open(filepath, 'rb') as f:
        data = f.read()
        num_elements = len(data) // 8  # double = 8 bytes
        values = struct.unpack('d' * num_elements, data)
        return np.array(values, dtype=np.float64)

def load_mean_shape(model_dir):
    """Load mean shape from model directory"""
    mean_shape_path = os.path.join(model_dir, 'mean_shape.bin')
    if not os.path.exists(mean_shape_path):
        mean_shape_path = os.path.join(model_dir, 'mean_shape.txt')
        if not os.path.exists(mean_shape_path):
            raise FileNotFoundError(f"Mean shape not found in {model_dir}")
        # Load text format
        mean_shape = np.loadtxt(mean_shape_path)
    else:
        # Load binary format
        mean_shape = load_binary_vector(mean_shape_path)
    
    # Reshape to N x 3
    num_vertices = len(mean_shape) // 3
    vertices = mean_shape.reshape(num_vertices, 3)
    return vertices

def find_vertex_by_position(vertices, target_pos, tolerance=0.01):
    """Find vertex closest to target position"""
    distances = np.linalg.norm(vertices - target_pos, axis=1)
    min_idx = np.argmin(distances)
    if distances[min_idx] < tolerance:
        return min_idx
    return None

def identify_landmark_vertices(vertices):
    """
    Identify model vertices corresponding to key facial landmarks.
    
    This is a heuristic approach based on geometry:
    - Nose tip: highest Z (most forward)
    - Eye corners: lateral extremes at eye level
    - Mouth corners: lateral extremes at mouth level
    - Jaw: lowest Y (bottom of face)
    """
    num_vertices = len(vertices)
    
    # Find bounding box
    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    min_z, max_z = vertices[:, 2].min(), vertices[:, 2].max()
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    print(f"Model bounds: X:[{min_x:.3f}, {max_x:.3f}], Y:[{min_y:.3f}, {max_y:.3f}], Z:[{min_z:.3f}, {max_z:.3f}]")
    print(f"Center: ({center_x:.3f}, {center_y:.3f})")
    
    mapping = {}
    
    # Key landmarks to identify (dlib 68-point indices)
    # We'll identify a subset of stable landmarks
    
    # 1. Nose tip (landmark 30) - highest Z (most forward)
    nose_tip_idx = np.argmax(vertices[:, 2])
    mapping[30] = int(nose_tip_idx)
    print(f"Landmark 30 (nose tip) -> vertex {nose_tip_idx}: {vertices[nose_tip_idx]}")
    
    # 2. Left eye corner (landmark 36) - leftmost at eye level
    eye_level = vertices[nose_tip_idx, 1] + 0.02  # Slightly above nose
    eye_candidates = vertices[(vertices[:, 1] > eye_level - 0.02) & 
                              (vertices[:, 1] < eye_level + 0.02) &
                              (vertices[:, 0] < center_x)]
    if len(eye_candidates) > 0:
        left_eye_idx = np.argmin(eye_candidates[:, 0])
        # Find original index
        for i, v in enumerate(vertices):
            if np.allclose(v, eye_candidates[left_eye_idx]):
                mapping[36] = i
                print(f"Landmark 36 (left eye corner) -> vertex {i}: {vertices[i]}")
                break
    
    # 3. Right eye corner (landmark 45) - rightmost at eye level
    eye_candidates = vertices[(vertices[:, 1] > eye_level - 0.02) & 
                              (vertices[:, 1] < eye_level + 0.02) &
                              (vertices[:, 0] > center_x)]
    if len(eye_candidates) > 0:
        right_eye_idx = np.argmax(eye_candidates[:, 0])
        for i, v in enumerate(vertices):
            if np.allclose(v, eye_candidates[right_eye_idx]):
                mapping[45] = i
                print(f"Landmark 45 (right eye corner) -> vertex {i}: {vertices[i]}")
                break
    
    # 4. Left mouth corner (landmark 48) - leftmost at mouth level
    mouth_level = vertices[nose_tip_idx, 1] - 0.03  # Below nose
    mouth_candidates = vertices[(vertices[:, 1] > mouth_level - 0.02) & 
                                (vertices[:, 1] < mouth_level + 0.02) &
                                (vertices[:, 0] < center_x)]
    if len(mouth_candidates) > 0:
        left_mouth_idx = np.argmin(mouth_candidates[:, 0])
        for i, v in enumerate(vertices):
            if np.allclose(v, mouth_candidates[left_mouth_idx]):
                mapping[48] = i
                print(f"Landmark 48 (left mouth corner) -> vertex {i}: {vertices[i]}")
                break
    
    # 5. Right mouth corner (landmark 54) - rightmost at mouth level
    mouth_candidates = vertices[(vertices[:, 1] > mouth_level - 0.02) & 
                                (vertices[:, 1] < mouth_level + 0.02) &
                                (vertices[:, 0] > center_x)]
    if len(mouth_candidates) > 0:
        right_mouth_idx = np.argmax(mouth_candidates[:, 0])
        for i, v in enumerate(vertices):
            if np.allclose(v, mouth_candidates[right_mouth_idx]):
                mapping[54] = i
                print(f"Landmark 54 (right mouth corner) -> vertex {i}: {vertices[i]}")
                break
    
    # 6. Chin (landmark 8) - lowest Y (bottom of jaw)
    chin_idx = np.argmin(vertices[:, 1])
    mapping[8] = int(chin_idx)
    print(f"Landmark 8 (chin) -> vertex {chin_idx}: {vertices[chin_idx]}")
    
    # 7. Left jaw (landmark 4) - leftmost at jaw level
    jaw_level = vertices[chin_idx, 1] + 0.01
    jaw_candidates = vertices[(vertices[:, 1] > jaw_level - 0.02) & 
                             (vertices[:, 1] < jaw_level + 0.02) &
                             (vertices[:, 0] < center_x)]
    if len(jaw_candidates) > 0:
        left_jaw_idx = np.argmin(jaw_candidates[:, 0])
        for i, v in enumerate(vertices):
            if np.allclose(v, jaw_candidates[left_jaw_idx]):
                mapping[4] = i
                print(f"Landmark 4 (left jaw) -> vertex {i}: {vertices[i]}")
                break
    
    # 8. Right jaw (landmark 12) - rightmost at jaw level
    jaw_candidates = vertices[(vertices[:, 1] > jaw_level - 0.02) & 
                             (vertices[:, 1] < jaw_level + 0.02) &
                             (vertices[:, 0] > center_x)]
    if len(jaw_candidates) > 0:
        right_jaw_idx = np.argmax(jaw_candidates[:, 0])
        for i, v in enumerate(vertices):
            if np.allclose(v, jaw_candidates[right_jaw_idx]):
                mapping[12] = i
                print(f"Landmark 12 (right jaw) -> vertex {i}: {vertices[i]}")
                break
    
    return mapping

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_landmark_mapping.py <model_dir> <output_mapping.txt>")
        print("\nExample:")
        print("  python create_landmark_mapping.py data/model data/landmark_mapping.txt")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    print("=== Landmark-to-Model Vertex Mapping Creator ===")
    print(f"Model directory: {model_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Load mean shape
    print("Loading mean shape...")
    vertices = load_mean_shape(model_dir)
    print(f"Loaded {len(vertices)} vertices")
    print()
    
    # Identify landmark vertices
    print("Identifying landmark vertices...")
    mapping = identify_landmark_vertices(vertices)
    print()
    
    if len(mapping) < 6:
        print(f"WARNING: Only {len(mapping)} landmarks identified. Need at least 6 for stable alignment.")
        print("You may need to manually adjust the mapping.")
    
    # Save mapping
    print(f"Saving mapping to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("# Landmark to Model Vertex Mapping\n")
        f.write("# Format: landmark_index model_vertex_index\n")
        f.write("# dlib 68-point landmarks -> model vertex indices\n")
        f.write("# Created automatically - verify and adjust if needed\n")
        f.write("\n")
        
        for landmark_idx in sorted(mapping.keys()):
            vertex_idx = mapping[landmark_idx]
            f.write(f"{landmark_idx} {vertex_idx}\n")
    
    print(f"✓ Saved {len(mapping)} mappings to {output_file}")
    print()
    print("Next steps:")
    print("  1. Review the mapping file")
    print("  2. Run test_landmark_mapping to visualize mapped points")
    print("  3. Adjust manually if needed")

if __name__ == '__main__':
    main()

