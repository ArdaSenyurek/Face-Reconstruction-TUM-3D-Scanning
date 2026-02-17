#!/usr/bin/env python3
"""
Convert Basel Face Model (BFM) to project format.

Supports:
- BFM 2009 (.mat format)
- BFM 2017 (.h5 format)

Output format:
- mean_shape.bin: (N*3,) float64, flattened [x1,y1,z1,x2,y2,z2,...]
- identity_basis.bin: (N*3, num_id) float64
- expression_basis.bin: (N*3, num_exp) float64
- identity_stddev.bin: (num_id,) float64
- expression_stddev.bin: (num_exp,) float64
- faces.bin: (F, 3) int32 with header [rows, cols]

Usage:
    python convert_bfm_to_project.py data/bfm/01_MorphableModel.mat data/model_bfm/
    python convert_bfm_to_project.py data/bfm/model2017-1_face12_nomouth.h5 data/model_bfm/
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Try to import scipy for .mat files
try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import h5py for .h5 files
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _save_binary_vector(path: Path, array: np.ndarray) -> None:
    """Save numpy array as raw binary float64."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(np.float64).tofile(str(path))


def _save_binary_matrix(path: Path, array: np.ndarray) -> None:
    """Save numpy matrix as raw binary float64 in column-major order (Eigen default)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # C++ MorphableModel loads into Eigen::MatrixXd (column-major); file must be column-major
    arr = np.asarray(array, dtype=np.float64)
    arr.flatten(order="F").tofile(str(path))


def _save_faces_binary(path: Path, faces: np.ndarray) -> None:
    """Save faces as binary with int32 header [rows, cols]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faces = faces.astype(np.int32)
    rows, cols = faces.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", rows, cols))
        f.write(faces.tobytes())


def _center_vertices(vertices: np.ndarray) -> np.ndarray:
    """Center vertices at origin (important for alignment)."""
    centroid = vertices.mean(axis=0)
    return vertices - centroid


def _flip_yz_axes(vertices: np.ndarray) -> np.ndarray:
    """
    Flip Y and Z axes to convert from BFM to camera coordinates.
    BFM: Y-up, face looks along +Z
    Camera: Y-down, face should look along -Z (towards camera)
    vertices: (N, 3) array
    """
    vertices = vertices.copy()
    vertices[:, 1] *= -1  # Flip Y (up -> down)
    vertices[:, 2] *= -1  # Flip Z (face towards camera)
    return vertices


def _flip_yz_axes_flat(flat_coords: np.ndarray) -> np.ndarray:
    """
    Flip Y and Z axes for flattened coordinates [x1,y1,z1,x2,y2,z2,...].
    """
    flat_coords = flat_coords.copy()
    # Y values are at indices 1, 4, 7, 10, ...
    flat_coords[1::3] *= -1
    # Z values are at indices 2, 5, 8, 11, ...
    flat_coords[2::3] *= -1
    return flat_coords


def load_bfm_2009(mat_path: Path) -> Optional[dict]:
    """
    Load BFM 2009 from .mat file.
    
    Expected keys:
    - shapeMU: mean shape (3N, 1)
    - shapePC: shape PCA basis (3N, num_components)
    - shapeEV: shape eigenvalues (num_components, 1)
    - tl: triangle list (F, 3), 1-indexed
    
    Expression model (if available):
    - expMU, expPC, expEV
    """
    if not HAS_SCIPY:
        print("scipy not installed. Install with: pip install scipy")
        return None
    
    try:
        data = loadmat(str(mat_path))
    except Exception as e:
        print(f"Failed to load .mat file: {e}")
        return None
    
    result = {}
    
    # Shape model
    if "shapeMU" in data:
        result["mean_shape"] = data["shapeMU"].flatten()
    elif "meanshape" in data:
        result["mean_shape"] = data["meanshape"].flatten()
    else:
        print("No mean shape found in .mat file")
        return None
    
    if "shapePC" in data:
        result["identity_basis"] = data["shapePC"]
    elif "idBase" in data:
        result["identity_basis"] = data["idBase"]
    else:
        print("Warning: No identity basis found, using empty")
        result["identity_basis"] = np.zeros((len(result["mean_shape"]), 0))
    
    if "shapeEV" in data:
        result["identity_stddev"] = np.sqrt(data["shapeEV"].flatten())
    elif "idEV" in data:
        result["identity_stddev"] = np.sqrt(data["idEV"].flatten())
    else:
        result["identity_stddev"] = np.ones(result["identity_basis"].shape[1])
    
    # Expression model
    if "expPC" in data:
        result["expression_basis"] = data["expPC"]
        result["expression_stddev"] = np.sqrt(data.get("expEV", np.ones((data["expPC"].shape[1], 1))).flatten())
    elif "exBase" in data:
        result["expression_basis"] = data["exBase"]
        result["expression_stddev"] = np.sqrt(data.get("exEV", np.ones((data["exBase"].shape[1], 1))).flatten())
    else:
        print("Warning: No expression basis found, using empty")
        result["expression_basis"] = np.zeros((len(result["mean_shape"]), 0))
        result["expression_stddev"] = np.zeros(0)
    
    # Faces (triangles)
    if "tl" in data:
        result["faces"] = data["tl"].astype(np.int32) - 1  # Convert to 0-indexed
    elif "tri" in data:
        result["faces"] = data["tri"].astype(np.int32) - 1
    else:
        print("Warning: No faces found")
        result["faces"] = np.zeros((0, 3), dtype=np.int32)
    
    return result


def load_bfm_2017(h5_path: Path) -> Optional[dict]:
    """
    Load BFM 2017 from .h5 file.
    
    Structure:
    - shape/model/mean
    - shape/model/pcaBasis
    - shape/model/pcaVariance
    - expression/model/mean (if available)
    - expression/model/pcaBasis
    - expression/model/pcaVariance
    - shape/representer/cells (faces)
    """
    if not HAS_H5PY:
        print("h5py not installed. Install with: pip install h5py")
        return None
    
    try:
        f = h5py.File(str(h5_path), "r")
    except Exception as e:
        print(f"Failed to load .h5 file: {e}")
        return None
    
    result = {}
    
    try:
        # Shape model
        if "shape/model/mean" in f:
            result["mean_shape"] = f["shape/model/mean"][:].flatten()
        else:
            print("No mean shape found in .h5 file")
            f.close()
            return None
        
        if "shape/model/pcaBasis" in f:
            result["identity_basis"] = f["shape/model/pcaBasis"][:]
        else:
            result["identity_basis"] = np.zeros((len(result["mean_shape"]), 0))
        
        if "shape/model/pcaVariance" in f:
            result["identity_stddev"] = np.sqrt(f["shape/model/pcaVariance"][:].flatten())
        else:
            result["identity_stddev"] = np.ones(result["identity_basis"].shape[1])
        
        # Expression model
        if "expression/model/pcaBasis" in f:
            result["expression_basis"] = f["expression/model/pcaBasis"][:]
            if "expression/model/pcaVariance" in f:
                result["expression_stddev"] = np.sqrt(f["expression/model/pcaVariance"][:].flatten())
            else:
                result["expression_stddev"] = np.ones(result["expression_basis"].shape[1])
        else:
            result["expression_basis"] = np.zeros((len(result["mean_shape"]), 0))
            result["expression_stddev"] = np.zeros(0)
        
        # Faces (BFM/Statismo H5 may use 1-based indexing; convert to 0-based like .mat)
        if "shape/representer/cells" in f:
            result["faces"] = f["shape/representer/cells"][:].T.astype(np.int32)
            # If min index is 1, treat as 1-based and convert to 0-based (avoids "exploded" mesh)
            if result["faces"].size > 0 and result["faces"].min() == 1:
                result["faces"] = result["faces"] - 1
                print("  Converted face indices from 1-based to 0-based")
        else:
            result["faces"] = np.zeros((0, 3), dtype=np.int32)
        
    finally:
        f.close()
    
    return result


def load_face3d_bfm(bfm_path: Path) -> Optional[dict]:
    """
    Load BFM from face3d library format (.npy files).
    
    Expected files:
    - BFM.npy or similar containing model data
    """
    try:
        # Try loading as a single .npy file with a dict
        data = np.load(str(bfm_path), allow_pickle=True).item()
        
        result = {}
        
        if "shapeMU" in data:
            result["mean_shape"] = data["shapeMU"].flatten()
        
        if "shapePC" in data:
            result["identity_basis"] = data["shapePC"]
        
        if "shapeEV" in data:
            result["identity_stddev"] = np.sqrt(data["shapeEV"].flatten())
        
        if "expPC" in data:
            result["expression_basis"] = data["expPC"]
            result["expression_stddev"] = np.sqrt(data.get("expEV", np.ones(data["expPC"].shape[1])).flatten())
        else:
            result["expression_basis"] = np.zeros((len(result["mean_shape"]), 0))
            result["expression_stddev"] = np.zeros(0)
        
        if "tri" in data:
            result["faces"] = data["tri"].astype(np.int32)
        
        return result
        
    except Exception as e:
        print(f"Failed to load face3d format: {e}")
        return None


def convert_bfm_to_project(bfm_path: Path, output_dir: Path, 
                           num_identity: int = 80, 
                           num_expression: int = 64,
                           center: bool = True) -> bool:
    """
    Convert BFM to project format.
    
    Args:
        bfm_path: Path to BFM file (.mat, .h5, or .npy)
        output_dir: Output directory for converted model
        num_identity: Number of identity components to keep (default: 80)
        num_expression: Number of expression components to keep (default: 64)
        center: Whether to center the mean shape at origin
    
    Returns:
        True on success, False on failure
    """
    print(f"Converting BFM from {bfm_path} to {output_dir}")
    
    # Determine file format and load
    suffix = bfm_path.suffix.lower()
    
    if suffix == ".mat":
        data = load_bfm_2009(bfm_path)
    elif suffix in [".h5", ".hdf5"]:
        data = load_bfm_2017(bfm_path)
    elif suffix == ".npy":
        data = load_face3d_bfm(bfm_path)
    else:
        print(f"Unsupported file format: {suffix}")
        return False
    
    if data is None:
        return False
    
    # Extract and validate data
    mean_shape = data["mean_shape"]
    identity_basis = data["identity_basis"]
    identity_stddev = data["identity_stddev"]
    expression_basis = data["expression_basis"]
    expression_stddev = data["expression_stddev"]
    faces = data["faces"]
    
    num_vertices = len(mean_shape) // 3
    print(f"  Vertices: {num_vertices}")
    print(f"  Identity components: {identity_basis.shape[1]}")
    print(f"  Expression components: {expression_basis.shape[1]}")
    print(f"  Faces: {faces.shape[0]}")
    
    # Truncate to requested number of components
    if identity_basis.shape[1] > num_identity:
        print(f"  Truncating identity basis to {num_identity} components")
        identity_basis = identity_basis[:, :num_identity]
        identity_stddev = identity_stddev[:num_identity]
    
    if expression_basis.shape[1] > num_expression:
        print(f"  Truncating expression basis to {num_expression} components")
        expression_basis = expression_basis[:, :num_expression]
        expression_stddev = expression_stddev[:num_expression]
    
    # Flip Y and Z: BFM uses Y-up, Z out of face; camera uses Y-down, Z into scene
    print("  Flipping Y and Z (BFM -> camera frame)")
    mean_shape = _flip_yz_axes_flat(mean_shape)
    identity_basis = _flip_yz_axes_flat(identity_basis)  # Same structure: [x,y,z,...] per row
    expression_basis = _flip_yz_axes_flat(expression_basis)
    
    # Center mean shape at origin if requested
    if center:
        vertices = mean_shape.reshape(-1, 3)
        vertices = _center_vertices(vertices)
        mean_shape = vertices.flatten()
        print("  Centered mean shape at origin")
    
    # Print bounds
    vertices = mean_shape.reshape(-1, 3)
    print(f"  X range: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"  Y range: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"  Z range: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
    
    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    _save_binary_vector(output_dir / "mean_shape.bin", mean_shape)
    _save_binary_matrix(output_dir / "identity_basis.bin", identity_basis)
    _save_binary_matrix(output_dir / "expression_basis.bin", expression_basis)
    _save_binary_vector(output_dir / "identity_stddev.bin", identity_stddev)
    _save_binary_vector(output_dir / "expression_stddev.bin", expression_stddev)
    _save_faces_binary(output_dir / "faces.bin", faces)
    
    # Also save faces as text for debugging
    np.savetxt(output_dir / "faces.txt", faces, fmt="%d")
    
    print(f"✓ Saved model to {output_dir}")
    print(f"  mean_shape.bin: {num_vertices * 3} values")
    print(f"  identity_basis.bin: {identity_basis.shape}")
    print(f"  expression_basis.bin: {expression_basis.shape}")
    print(f"  faces.bin: {faces.shape[0]} triangles")
    
    return True


def find_bfm_file(bfm_dir: Path) -> Optional[Path]:
    """Find BFM file in directory."""
    # Check for various BFM file patterns
    patterns = [
        "*.mat",
        "*.h5",
        "*.hdf5",
        "*.npy",
        "01_MorphableModel.mat",
        "model2017-1_face12_nomouth.h5",
        "BFM.mat",
        "BFM.npy",
    ]
    
    for pattern in patterns:
        files = list(bfm_dir.glob(pattern))
        if files:
            return files[0]
    
    return None


def main():
    """Command-line entry point."""
    if len(sys.argv) < 3:
        print("Usage: python convert_bfm_to_project.py <bfm_file> <output_dir>")
        print("       python convert_bfm_to_project.py <bfm_dir> <output_dir>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # If input is a directory, find BFM file
    if input_path.is_dir():
        bfm_file = find_bfm_file(input_path)
        if bfm_file is None:
            print(f"No BFM file found in {input_path}")
            sys.exit(1)
        input_path = bfm_file
    
    success = convert_bfm_to_project(input_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
