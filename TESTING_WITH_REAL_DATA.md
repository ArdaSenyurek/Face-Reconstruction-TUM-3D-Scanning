# Testing with Real Data

This guide explains how to test the face reconstruction pipeline with real RGB-D data.

## Prerequisites

1. **RGB-D Camera Data** (or RGB image + depth map)
   - RGB image (PNG, JPEG)
   - Depth image (16-bit PNG recommended)
   - Camera intrinsics

2. **PCA Morphable Model** (optional if you have a real model)
   - Or use the dummy model for testing

3. **Landmark Detection** (optional but recommended)
   - Python script provided
   - Requires dlib or MediaPipe

## Step-by-Step Guide

### Step 1: Prepare Your Data Directory

Create a directory structure for your test data:

```bash
mkdir -p data/test
cd data/test

# Place your files here:
# - rgb.png (or rgb.jpg)
# - depth.png (16-bit depth map)
# - intrinsics.txt
```

### Step 2: Prepare Camera Intrinsics

Create `intrinsics.txt` with your camera parameters:

```
fx fy cx cy
```

Example for a typical RGB-D camera:
```
525.0 525.0 320.0 240.0
```

To get intrinsics from your camera:
- **Kinect/RealSense**: Check camera documentation or calibrate
- **iPhone/iPad**: Use ARKit intrinsics
- **Custom camera**: Use camera calibration tools (OpenCV, ROS)

### Step 3: Prepare Depth Image

Your depth image should be:
- **Format**: 16-bit PNG (recommended) or other OpenCV-supported format
- **Units**: Typically millimeters (will be scaled by default scale factor 1000.0)

**Converting depth formats:**

If you have depth in a different format, you can convert it:

```python
import cv2
import numpy as np

# Example: Load depth from numpy array
depth = np.load('depth.npy')  # Your depth data

# Convert to 16-bit PNG (assuming depth is in meters)
depth_mm = (depth * 1000).astype(np.uint16)
cv2.imwrite('depth.png', depth_mm)
```

### Step 4: Detect Facial Landmarks (Optional)

Use the provided Python script to detect landmarks:

#### Using MediaPipe (Recommended - No downloads needed):

```bash
# Install MediaPipe
pip install mediapipe opencv-python

# Detect landmarks
python scripts/detect_landmarks.py \
    --image data/test/rgb.png \
    --method mediapipe \
    --output data/test/landmarks.txt \
    --visualize
```

#### Using dlib (More accurate but requires model file):

```bash
# Install dlib
pip install dlib opencv-python

# Download shape predictor (68-point model)
# From: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract to your project directory

# Detect landmarks
python scripts/detect_landmarks.py \
    --image data/test/rgb.png \
    --method dlib \
    --predictor shape_predictor_68_face_landmarks.dat \
    --output data/test/landmarks.txt \
    --visualize
```

### Step 5: Prepare PCA Model

You have two options:

#### Option A: Use Dummy Model (Quick Test)

```bash
python scripts/prepare_model.py --generate-dummy --output data/model
```

#### Option B: Use Real Model

If you have a real morphable face model (e.g., Basel Face Model):

1. Convert to the required format (see `DATA_FORMAT.md`)
2. Place files in `data/model/` directory:
   - `mean_shape.bin` or `mean_shape.txt`
   - `identity_basis.bin` or `identity_basis.txt`
   - `expression_basis.bin` or `expression_basis.txt`
   - `faces.bin` or `faces.txt` (face connectivity)
   - `identity_stddev.bin` or `identity_stddev.txt`
   - `expression_stddev.bin` or `expression_stddev.txt`

### Step 6: Run Reconstruction

Run the test program with your data:

```bash
cd build
./bin/test_real_data \
    --rgb ../data/test/rgb.png \
    --depth ../data/test/depth.png \
    --depth-scale 1000.0 \
    --intrinsics ../data/test/intrinsics.txt \
    --model-dir ../data/model \
    --landmarks ../data/test/landmarks.txt \
    --output-mesh ../data/test/reconstructed.ply
```

**Arguments explained:**
- `--rgb`: Path to RGB image
- `--depth`: Path to depth image
- `--depth-scale`: Scale factor (1000.0 = depth in mm, 1.0 = depth in meters)
- `--intrinsics`: Camera intrinsics file
- `--model-dir`: Directory containing PCA model files
- `--landmarks`: Detected landmarks file (optional)
- `--output-mesh`: Output PLY/OBJ file path

### Step 7: View Results

Open the reconstructed mesh:

```bash
# Using MeshLab
meshlab data/test/reconstructed.ply

# Or using Blender
# File → Import → Stanford (.ply)
```

## Common Issues and Solutions

### Issue: "Failed to load depth image"

**Solutions:**
- Check that depth image is 16-bit PNG format
- Verify file path is correct
- Try adjusting `--depth-scale` (try 1.0 if depth is in meters)

### Issue: "No face detected in image"

**Solutions:**
- Ensure face is clearly visible in RGB image
- Try different landmark detection method (MediaPipe vs dlib)
- Check image quality and lighting

### Issue: "Model loading failed"

**Solutions:**
- Verify all required model files exist in the model directory
- Check file formats (binary vs text)
- Ensure file permissions are correct

### Issue: "Depth values seem wrong"

**Solutions:**
- Adjust `--depth-scale` parameter
  - If depth is in millimeters: use 1000.0
  - If depth is in meters: use 1.0
  - If depth is in centimeters: use 100.0
- Check depth image encoding (may need conversion)

### Issue: "Landmarks don't match model"

**Note:** Currently, landmark-to-model correspondence is not automatically mapped. The `model_index` field in landmarks is set to -1 (unknown). In Week 2+, we'll implement landmark correspondence mapping.

## Example Workflow Script

Create `scripts/test_real_data.sh`:

```bash
#!/bin/bash

# Configuration
RGB_IMAGE="data/test/rgb.png"
DEPTH_IMAGE="data/test/depth.png"
INTRINSICS="data/test/intrinsics.txt"
MODEL_DIR="data/model"
OUTPUT_MESH="data/test/reconstructed.ply"

# Step 1: Detect landmarks (optional)
echo "Detecting landmarks..."
python scripts/detect_landmarks.py \
    --image "$RGB_IMAGE" \
    --method mediapipe \
    --output data/test/landmarks.txt

# Step 2: Run reconstruction
echo "Running reconstruction..."
cd build
./bin/test_real_data \
    --rgb "../$RGB_IMAGE" \
    --depth "../$DEPTH_IMAGE" \
    --intrinsics "../$INTRINSICS" \
    --model-dir "../$MODEL_DIR" \
    --landmarks "../data/test/landmarks.txt" \
    --output-mesh "../$OUTPUT_MESH"

echo "Done! Mesh saved to $OUTPUT_MESH"
```

Make it executable:
```bash
chmod +x scripts/test_real_data.sh
./scripts/test_real_data.sh
```

## Next Steps (Week 2+)

1. **Landmark Correspondence Mapping**: Map detected landmarks to model vertex indices
2. **Procrustes Alignment**: Align model to landmarks
3. **Optimization**: Use Gauss-Newton to optimize model coefficients
4. **Dense Alignment**: Align model to dense depth data

## Getting Real RGB-D Data

### Option 1: Use Your Own Camera

- **Intel RealSense**: Use RealSense SDK to capture RGB-D
- **Azure Kinect**: Use Azure Kinect SDK
- **iPhone/iPad**: Use ARKit to capture depth

### Option 2: Use Public Datasets

- **Biwi Kinect Head Pose Database**: RGB-D face data
  - See [BLWI_DATASET_GUIDE.md](BLWI_DATASET_GUIDE.md) for detailed instructions
  - Quick start:
    ```bash
    python scripts/convert_biwi_dataset.py \
        --input /path/to/biwi/dataset \
        --output data/biwi_test \
        --kinect-version v1
    ```
- **3DFAW Dataset**: 3D face alignment with depth
- **RGB-D Face Dataset**: Various RGB-D face datasets

### Option 3: Synthetic Data

Generate synthetic RGB-D data for testing:
- Render 3D face models
- Project to RGB + depth using virtual camera
