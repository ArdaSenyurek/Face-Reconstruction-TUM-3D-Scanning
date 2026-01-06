# Generating faces.bin for the Morphable Model

## Problem
The meshes appear as point clouds because the `faces.bin` file is missing from `data/model_biwi/`. This file contains the triangle connectivity information needed to render proper meshes.

## Solution

You have two options:

### Option 1: Generate faces using Python script (Recommended)

1. **Install Open3D** (if not already installed):
   ```bash
   pip install open3d
   ```

2. **Run the face generation script**:
   ```bash
   python3 scripts/create_faces_from_mean_shape.py data/model_biwi
   ```

   This will create `data/model_biwi/faces.bin` using ball pivoting triangulation.

3. **If Open3D is not available**, the script will try scipy as a fallback:
   ```bash
   pip install scipy scikit-learn
   python3 scripts/create_faces_from_mean_shape.py data/model_biwi
   ```

### Option 2: Extract faces from existing mesh

If you have a PLY mesh file with faces (e.g., from a previous run), you can extract them:

```bash
python3 scripts/extract_faces_from_ply.py path/to/mesh_with_faces.ply --output data/model_biwi/faces.bin
```

## Verification

After generating `faces.bin`, verify it works:

1. Rebuild your project:
   ```bash
   cd build && make
   ```

2. Run a reconstruction:
   ```bash
   build/bin/face_reconstruction --rgb <path> --depth <path> \
                                  --intrinsics <path> --model-dir data/model_biwi \
                                  --output-mesh test.ply
   ```

3. Open `test.ply` in MeshLab or another 3D viewer. You should see a proper mesh with faces, not just points.

## Expected Result

- **Vertices**: 1000 (from mean_shape.bin)
- **Faces**: ~1570 triangles (generated via triangulation)

The faces.bin file format:
- Header: 2 int32 values (rows, cols = 3)
- Data: rows × 3 int32 values (face vertex indices)

