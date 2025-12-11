# Using Biwi Kinect Head Pose Dataset

This guide explains how to use the Biwi Kinect Head Pose Dataset with the face reconstruction pipeline.

## Dataset Overview

The Biwi Kinect Head Pose Dataset contains:
- RGB images
- Depth images  
- Head pose annotations (yaw, pitch, roll)
- Multiple subjects/sequences

## Quick Start

### Step 1: Convert Dataset

Use the provided conversion script:

```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi/dataset \
    --output data/biwi_test \
    --kinect-version v1 \
    --max-frames 50
```

**Arguments:**
- `--input`: Path to Biwi dataset directory
- `--output`: Output directory for converted data
- `--kinect-version`: Use `v1` or `v2` (affects default intrinsics)
- `--max-frames`: Limit number of frames to convert (for quick testing)

### Step 2: Test with Single Frame

```bash
cd build
./bin/test_real_data \
    --rgb ../data/biwi_test/rgb/frame_00000.png \
    --depth ../data/biwi_test/depth/frame_00000.png \
    --intrinsics ../data/biwi_test/intrinsics.txt \
    --model-dir ../data/model \
    --output-mesh ../data/biwi_test/reconstructed_00000.ply
```

### Step 3: Detect Landmarks (Optional)

```bash
python scripts/detect_landmarks.py \
    --image data/biwi_test/rgb/frame_00000.png \
    --method mediapipe \
    --output data/biwi_test/landmarks_00000.txt \
    --visualize
```

### Step 4: View Results

```bash
meshlab data/biwi_test/reconstructed_00000.ply
```

## Quick Start (Turkish)

Biwi dataset'i kullanmak için:

```bash
# 1. Dataset'i dönüştür
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi/dataset \
    --output data/biwi_test \
    --kinect-version v1

# 2. Test et
cd build
./bin/test_real_data \
    --rgb ../data/biwi_test/rgb/frame_00000.png \
    --depth ../data/biwi_test/depth/frame_00000.png \
    --intrinsics ../data/biwi_test/intrinsics.txt \
    --model-dir ../data/model \
    --output-mesh ../data/biwi_test/reconstructed.ply
```

Daha kısa rehber için: [QUICK_BIWI_GUIDE.md](QUICK_BIWI_GUIDE.md)

## Common Biwi Dataset Structures

The Biwi dataset can be organized in different ways. The conversion script tries to handle common patterns:

### Pattern 1: Separate RGB/Depth Directories
```
biwi_dataset/
├── rgb/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── depth/
    ├── 0000.png
    ├── 0001.png
    └── ...
```

### Pattern 2: Person/Sequence Subdirectories
```
biwi_dataset/
├── person01/
│   ├── rgb_0000.png
│   ├── depth_0000.png
│   └── ...
├── person02/
│   └── ...
```

### Pattern 3: Mixed Files
```
biwi_dataset/
├── frame_0000_rgb.png
├── frame_0000_depth.png
├── frame_0001_rgb.png
└── ...
```

## Camera Intrinsics

### Default Kinect v1 Intrinsics
The script uses these default values for Kinect v1 (640x480):
- fx = 525.0
- fy = 525.0
- cx = 319.5
- cy = 239.5

### Default Kinect v2 Intrinsics
For Kinect v2 (512x424):
- fx = 365.0
- fy = 365.0
- cx = 255.5
- cy = 211.5

### Custom Intrinsics

If you know the exact intrinsics for your dataset:

```bash
python scripts/convert_biwi_dataset.py \
    --input /path/to/biwi \
    --output data/biwi_test \
    --intrinsics 525.0 525.0 320.0 240.0
```

## Processing Multiple Frames

### Batch Processing Script

Create `scripts/process_biwi_frames.sh`:

```bash
#!/bin/bash

BIWI_DIR="data/biwi_test"
MODEL_DIR="data/model"
OUTPUT_DIR="data/biwi_test/output"

mkdir -p "$OUTPUT_DIR"

# Process first 10 frames
for i in {0..9}; do
    frame_num=$(printf "%05d" $i)
    echo "Processing frame $frame_num..."
    
    cd build
    ./bin/test_real_data \
        --rgb "../$BIWI_DIR/rgb/frame_${frame_num}.png" \
        --depth "../$BIWI_DIR/depth/frame_${frame_num}.png" \
        --intrinsics "../$BIWI_DIR/intrinsics.txt" \
        --model-dir "../$MODEL_DIR" \
        --output-mesh "../$OUTPUT_DIR/reconstructed_${frame_num}.ply" \
        2>&1 | grep -E "(Error|Success)" || echo "Frame $frame_num processed"
    cd ..
done

echo "Done! Results in $OUTPUT_DIR"
```

Make it executable:
```bash
chmod +x scripts/process_biwi_frames.sh
./scripts/process_biwi_frames.sh
```

## Troubleshooting

### Issue: "No RGB-depth pairs found"

**Solutions:**
1. Check the dataset structure - files may be in subdirectories
2. Try specifying a subdirectory:
   ```bash
   python scripts/convert_biwi_dataset.py \
       --input /path/to/biwi/person01 \
       --output data/biwi_test
   ```
3. Check file naming patterns manually:
   ```bash
   ls -la /path/to/biwi/*rgb*
   ls -la /path/to/biwi/*depth*
   ```

### Issue: Depth values seem wrong

**Solutions:**
1. Adjust depth scale:
   ```bash
   python scripts/convert_biwi_dataset.py \
       --input /path/to/biwi \
       --output data/biwi_test \
       --depth-scale 5000.0  # Try different values
   ```
2. Check depth image format:
   ```python
   import cv2
   depth = cv2.imread('depth.png', cv2.IMREAD_ANYDEPTH)
   print(f"Depth range: {depth.min()} - {depth.max()}")
   ```

### Issue: Face not detected in landmarks

**Solutions:**
1. Biwi dataset may have faces at angles - try different frames
2. Use dlib instead of MediaPipe (more robust):
   ```bash
   python scripts/detect_landmarks.py \
       --image data/biwi_test/rgb/frame_00000.png \
       --method dlib \
       --predictor shape_predictor_68_face_landmarks.dat \
       --output data/biwi_test/landmarks.txt
   ```

### Issue: Intrinsics don't match

**Solutions:**
1. Check image dimensions:
   ```bash
   file data/biwi_test/rgb/frame_00000.png
   ```
2. Adjust intrinsics based on resolution:
   - 640x480: Use Kinect v1 defaults
   - 512x424: Use Kinect v2 defaults
   - Other: Calibrate or estimate

## Using Head Pose Annotations

Biwi dataset includes head pose annotations (yaw, pitch, roll). These can be useful for:
- Validating alignment
- Filtering frames with extreme poses
- Future pose estimation features

To extract pose information, check the dataset documentation or annotation files.

## Next Steps

1. **Landmark Correspondence**: Map detected landmarks to model vertices
2. **Pose Alignment**: Use Procrustes alignment with landmarks
3. **Optimization**: Optimize model coefficients for each frame
4. **Sequence Processing**: Process entire sequences for temporal consistency

## References

- Biwi Kinect Head Pose Dataset: [Original paper/publication]
- Dataset download: [URL if available]
