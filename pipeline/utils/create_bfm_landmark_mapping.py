#!/usr/bin/env python3
"""
Create BFM Landmark Mapping

Generates accurate landmark-to-vertex mapping by:
1. Loading BFM semantic landmarks from the h5 file
2. Finding closest vertices to each landmark coordinate
3. Mapping to dlib's 68-landmark scheme

Usage:
    python pipeline/utils/create_bfm_landmark_mapping.py

Landmark mapping uses the BFM mean shape: for each dlib landmark we have a BFM
semantic 3D position; we map to a mesh vertex. By default we use "anti-center-bias":
among the k nearest vertices we pick the one most outward from the face center
toward the semantic point, so peripheral landmarks (eyebrows, lip corners) are not
pulled inward and overlays align better.

NOTE: The BFM model only provides ~30 semantic landmarks (eyes, nose, mouth, etc.)
      but NOT the jaw contour (dlib landmarks 0-16). The current mapping covers:
      - Chin tip (dlib 8)
      - Eyebrows (dlib 17, 19, 22, 24)
      - Nose (dlib 30-35)
      - Eyes (dlib 36-47)
      - Mouth (dlib 48-67)

      To add jaw contour landmarks (dlib 0-7 and 9-16), you would need to:
      1. Load the BFM mean shape in MeshLab or similar 3D viewer
      2. Visually identify vertices along the jaw contour
      3. Record vertex indices and add to DLIB_TO_BFM_MAP below
      
      The current 31 landmarks provide good alignment for most use cases.
"""

import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping from dlib 68 landmarks to BFM semantic landmarks
# dlib landmark indices -> BFM landmark name
DLIB_TO_BFM_MAP = {
    # Jaw contour (0-16) - BFM doesn't have full contour, only chin tip
    8: 'center.chin.tip',  # Chin tip (center of jaw)
    
    # Right eyebrow (17-21)
    17: 'right.eyebrow.inner_lower',   # Inner end
    19: 'right.eyebrow.bend.lower',    # Arch/bend (correct name with dots)
    
    # Left eyebrow (22-26)
    22: 'left.eyebrow.inner_lower',    # Inner end
    24: 'left.eyebrow.bend.lower',     # Arch/bend (correct name with dots)
    
    # Nose (27-35)
    30: 'center.nose.tip',              # Nose tip
    31: 'right.nose.wing.outer',        # Right nostril outer
    32: 'right.nose.wing.tip',          # Right nostril tip  
    33: 'center.nose.attachement_to_philtrum',  # Under nose center
    34: 'left.nose.wing.tip',           # Left nostril tip
    35: 'left.nose.wing.outer',         # Left nostril outer
    
    # Right eye (36-41)
    36: 'right.eye.corner_outer',       # Outer corner
    37: 'right.eye.top',                # Top
    38: 'right.eye.top',                # Top (duplicate for stability)
    39: 'right.eye.corner_inner',       # Inner corner
    40: 'right.eye.bottom',             # Bottom
    41: 'right.eye.bottom',             # Bottom (duplicate for stability)
    
    # Left eye (42-47)
    42: 'left.eye.corner_inner',        # Inner corner
    43: 'left.eye.top',                 # Top
    44: 'left.eye.top',                 # Top (duplicate for stability)
    45: 'left.eye.corner_outer',        # Outer corner
    46: 'left.eye.bottom',              # Bottom
    47: 'left.eye.bottom',              # Bottom (duplicate for stability)
    
    # Outer lips (48-59)
    48: 'right.lips.corner',            # Right corner
    49: 'right.lips.philtrum_ridge',    # Right upper
    50: 'right.lips.philtrum_ridge',    # (approximate)
    51: 'center.lips.upper.outer',      # Upper lip center
    52: 'left.lips.philtrum_ridge',     # Left upper
    53: 'left.lips.philtrum_ridge',     # (approximate)
    54: 'left.lips.corner',             # Left corner
    57: 'center.lips.lower.outer',      # Lower lip center
    
    # Inner lips (60-67)
    62: 'center.lips.upper.inner',      # Upper lip inner center
    66: 'center.lips.lower.inner',      # Lower lip inner center
}


def load_bfm_landmarks(bfm_path: Path) -> Dict[str, np.ndarray]:
    """Load semantic landmarks from BFM h5 file."""
    landmarks = {}
    
    with h5py.File(bfm_path, 'r') as f:
        landmarks_data = f['metadata/landmarks/json'][()]
        s = landmarks_data.tobytes().decode('utf-8')
        landmark_list = json.loads(s)
        
        for lm in landmark_list:
            name = lm['id']
            coords = np.array(lm['coordinates'])
            landmarks[name] = coords
    
    print(f"Loaded {len(landmarks)} BFM semantic landmarks")
    return landmarks


def load_bfm_mean_shape(bfm_path: Path) -> np.ndarray:
    """Load mean shape vertices from BFM h5 file."""
    with h5py.File(bfm_path, 'r') as f:
        mean_shape = np.array(f['shape/model/mean'])
    
    # Reshape to (N, 3)
    num_vertices = len(mean_shape) // 3
    vertices = mean_shape.reshape(num_vertices, 3)
    
    print(f"Loaded mean shape: {num_vertices} vertices")
    return vertices


def find_closest_vertex(vertices: np.ndarray, target: np.ndarray) -> int:
    """Find the vertex index closest to the target 3D point."""
    distances = np.linalg.norm(vertices - target, axis=1)
    return int(np.argmin(distances))


def find_k_nearest_vertex_indices(vertices: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k vertices closest to target (ascending distance)."""
    distances = np.linalg.norm(vertices - target, axis=1)
    return np.argsort(distances)[:k]


def choose_vertex_outward(
    vertices: np.ndarray,
    target: np.ndarray,
    face_center: np.ndarray,
    k: int = 15,
) -> int:
    """
    Among the k-nearest vertices to target, choose the one that is most 'outward'
    from the face center toward the target. This reduces center bias: the raw
    nearest vertex on a curved mesh can lie between the semantic point and the
    center; picking the most outward vertex in that direction improves overlay
    alignment at eyebrows, jaw, and lip corners.
    """
    if k <= 0:
        return find_closest_vertex(vertices, target)
    idxs = find_k_nearest_vertex_indices(vertices, target, min(k, len(vertices)))
    direction = target - face_center
    n = np.linalg.norm(direction)
    if n < 1e-9:
        return int(idxs[0])
    direction = direction / n
    best_i = 0
    best_proj = -np.inf
    for i in idxs:
        proj = np.dot(vertices[i] - face_center, direction)
        if proj > best_proj:
            best_proj = proj
            best_i = i
    return int(best_i)


def create_mapping(
    bfm_path: Path,
    output_path: Path,
    visualize: bool = True,
    anti_center_bias: bool = True,
    k_outward: int = 15,
) -> Dict[int, int]:
    """Create landmark mapping from dlib indices to BFM vertex indices.
    
    If anti_center_bias is True (default), for each BFM semantic landmark we choose
    among the k_outward nearest vertices the one that is most 'outward' from the
    face center toward the landmark. This reduces the bias where nearest-vertex
    snapping pulls peripheral landmarks (eyebrows, lip corners, etc.) toward the
    center and worsens overlay alignment.
    """
    # Load BFM data
    bfm_landmarks = load_bfm_landmarks(bfm_path)
    vertices = load_bfm_mean_shape(bfm_path)
    
    # Face center: mean of all vertices (used to define "outward" direction)
    face_center = np.mean(vertices, axis=0)
    print(f"\nFace center (mean of vertices): {face_center}")
    
    # Print available BFM landmarks for reference
    print("\nAvailable BFM landmarks:")
    for name in sorted(bfm_landmarks.keys()):
        print(f"  {name}")
    
    # Create mapping
    mapping = {}
    print("\n=== Dlib to BFM Vertex Mapping ===")
    if anti_center_bias:
        print(f"  Using anti-center-bias: among k={k_outward} nearest, pick most outward from center")
    
    for dlib_idx, bfm_name in sorted(DLIB_TO_BFM_MAP.items()):
        if bfm_name not in bfm_landmarks:
            print(f"  Warning: '{bfm_name}' not found in BFM landmarks")
            continue
        
        # Get BFM landmark coordinate
        target_coord = bfm_landmarks[bfm_name]
        
        # Choose vertex: either classic nearest or outward among k-nearest
        if anti_center_bias:
            vertex_idx = choose_vertex_outward(vertices, target_coord, face_center, k=k_outward)
        else:
            vertex_idx = find_closest_vertex(vertices, target_coord)
        
        # Get actual vertex position for verification
        vertex_pos = vertices[vertex_idx]
        dist = np.linalg.norm(target_coord - vertex_pos)
        
        mapping[dlib_idx] = vertex_idx
        print(f"  dlib[{dlib_idx:2d}] -> vertex[{vertex_idx:5d}] ({bfm_name}, dist={dist:.2f}mm)")
    
    # Save mapping file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# BFM Landmark Mapping (dlib 68 -> BFM vertex index)\n")
        f.write("# Generated from BFM semantic landmarks; vertex = k-nearest then most outward from center\n")
        f.write("# Format: dlib_landmark_idx BFM_vertex_idx\n")
        for dlib_idx in sorted(mapping.keys()):
            vertex_idx = mapping[dlib_idx]
            bfm_name = DLIB_TO_BFM_MAP.get(dlib_idx, "unknown")
            f.write(f"{dlib_idx} {vertex_idx}  # {bfm_name}\n")
    
    print(f"\nSaved mapping to: {output_path}")
    print(f"Total mappings: {len(mapping)}")
    
    # Visualize if requested
    if visualize:
        visualize_mapping(vertices, mapping, bfm_landmarks)
    
    return mapping


def visualize_mapping(
    vertices: np.ndarray,
    mapping: Dict[int, int],
    bfm_landmarks: Dict[str, np.ndarray]
):
    """Create a simple visualization of the landmark mapping."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # View 1: Front view (X vs Y)
    ax1 = fig.add_subplot(131)
    ax1.scatter(vertices[::100, 0], vertices[::100, 1], s=0.5, c='lightgray', alpha=0.5)
    
    for dlib_idx, vertex_idx in mapping.items():
        v = vertices[vertex_idx]
        ax1.scatter(v[0], v[1], s=50, c='red', marker='o')
        ax1.annotate(str(dlib_idx), (v[0], v[1]), fontsize=6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Front View (XY)')
    ax1.set_aspect('equal')
    
    # View 2: Side view (Z vs Y)
    ax2 = fig.add_subplot(132)
    ax2.scatter(vertices[::100, 2], vertices[::100, 1], s=0.5, c='lightgray', alpha=0.5)
    
    for dlib_idx, vertex_idx in mapping.items():
        v = vertices[vertex_idx]
        ax2.scatter(v[2], v[1], s=50, c='red', marker='o')
        ax2.annotate(str(dlib_idx), (v[2], v[1]), fontsize=6)
    
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    ax2.set_title('Side View (ZY)')
    ax2.set_aspect('equal')
    
    # View 3: 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(vertices[::100, 0], vertices[::100, 2], vertices[::100, 1], 
                s=0.5, c='lightgray', alpha=0.3)
    
    for dlib_idx, vertex_idx in mapping.items():
        v = vertices[vertex_idx]
        ax3.scatter(v[0], v[2], v[1], s=50, c='red', marker='o')
        ax3.text(v[0], v[2], v[1], str(dlib_idx), fontsize=6)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('Y')
    ax3.set_title('3D View')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs/debug/bfm_landmark_mapping.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create BFM landmark mapping")
    parser.add_argument("--bfm", type=str, default="data/bfm/model2019_fullHead.h5",
                       help="Path to BFM h5 file")
    parser.add_argument("--output", type=str, default="data/bfm_landmark_68.txt",
                       help="Output mapping file")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Skip visualization")
    parser.add_argument("--no-anti-center-bias", action="store_true",
                       help="Use classic nearest vertex only (can bias peripheral landmarks inward)")
    parser.add_argument("--k-outward", type=int, default=15,
                       help="Among k nearest vertices, pick most outward (default 15)")
    
    args = parser.parse_args()
    
    bfm_path = Path(args.bfm)
    output_path = Path(args.output)
    
    if not bfm_path.exists():
        print(f"Error: BFM file not found: {bfm_path}")
        return 1
    
    create_mapping(
        bfm_path,
        output_path,
        visualize=not args.no_visualize,
        anti_center_bias=not args.no_anti_center_bias,
        k_outward=max(1, args.k_outward),
    )
    return 0


if __name__ == "__main__":
    exit(main())
