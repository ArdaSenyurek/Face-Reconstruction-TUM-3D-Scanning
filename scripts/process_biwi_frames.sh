#!/bin/bash
# Batch process multiple frames from Biwi dataset

BIWI_DIR="${1:-data/biwi_test}"
MODEL_DIR="${2:-data/model}"
OUTPUT_DIR="${3:-data/biwi_test/output}"
MAX_FRAMES="${4:-10}"

mkdir -p "$OUTPUT_DIR"

echo "Processing Biwi frames..."
echo "  Input: $BIWI_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Max frames: $MAX_FRAMES"

# Process frames
for i in $(seq 0 $((MAX_FRAMES - 1))); do
    frame_num=$(printf "%05d" $i)
    rgb_file="$BIWI_DIR/rgb/frame_${frame_num}.png"
    depth_file="$BIWI_DIR/depth/frame_${frame_num}.png"
    intrinsics_file="$BIWI_DIR/intrinsics.txt"
    output_file="$OUTPUT_DIR/reconstructed_${frame_num}.ply"
    
    if [ ! -f "$rgb_file" ] || [ ! -f "$depth_file" ]; then
        echo "Skipping frame $frame_num (files not found)"
        continue
    fi
    
    echo "Processing frame $frame_num..."
    
    cd build
    ./bin/test_real_data \
        --rgb "../$rgb_file" \
        --depth "../$depth_file" \
        --intrinsics "../$intrinsics_file" \
        --model-dir "../$MODEL_DIR" \
        --output-mesh "../$output_file" \
        > /dev/null 2>&1
    
    if [ -f "../$output_file" ]; then
        echo "  ✓ Frame $frame_num saved"
    else
        echo "  ✗ Frame $frame_num failed"
    fi
    
    cd ..
done

echo ""
echo "Done! Results saved in: $OUTPUT_DIR"
echo "Processed frames: $(ls -1 "$OUTPUT_DIR"/*.ply 2>/dev/null | wc -l)"
