#!/bin/bash
# Example script to test face reconstruction with real data

# Configuration - modify these paths for your data
RGB_IMAGE="${1:-data/test/rgb.png}"
DEPTH_IMAGE="${2:-data/test/depth.png}"
INTRINSICS="${3:-data/test/intrinsics.txt}"
MODEL_DIR="${4:-data/model}"
OUTPUT_DIR="${5:-data/test}"

mkdir -p "$OUTPUT_DIR"

# Step 1: Detect landmarks (if RGB image exists)
if [ -f "$RGB_IMAGE" ]; then
    echo "[1] Detecting landmarks from $RGB_IMAGE..."
    python3 scripts/detect_landmarks.py \
        --image "$RGB_IMAGE" \
        --method mediapipe \
        --output "$OUTPUT_DIR/landmarks.txt" 2>/dev/null || \
    echo "  (Landmark detection skipped - install mediapipe: pip install mediapipe)"
else
    echo "[1] RGB image not found: $RGB_IMAGE"
    echo "    Skipping landmark detection"
fi

# Step 2: Run reconstruction
echo ""
echo "[2] Running face reconstruction..."
cd build

./bin/test_real_data \
    --rgb "../$RGB_IMAGE" \
    --depth "../$DEPTH_IMAGE" \
    --intrinsics "../$INTRINSICS" \
    --model-dir "../$MODEL_DIR" \
    --landmarks "../$OUTPUT_DIR/landmarks.txt" \
    --output-mesh "../$OUTPUT_DIR/reconstructed.ply" 2>&1

cd ..

if [ -f "$OUTPUT_DIR/reconstructed.ply" ]; then
    echo ""
    echo "✅ Success! Mesh saved to: $OUTPUT_DIR/reconstructed.ply"
    echo ""
    echo "To view the mesh:"
    echo "  meshlab $OUTPUT_DIR/reconstructed.ply"
    echo "  or"
    echo "  Open in Blender: File → Import → Stanford (.ply)"
else
    echo ""
    echo "❌ Failed to create mesh. Check error messages above."
fi
