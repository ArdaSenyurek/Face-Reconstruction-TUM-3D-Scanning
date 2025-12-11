# Data Format Documentation

This document describes the expected file formats for RGB-D data, PCA models, landmarks, and camera intrinsics.

## RGB-D Data

### RGB Images
- **Supported formats**: PNG, JPEG, BMP (any format supported by OpenCV)
- **Color space**: BGR (OpenCV standard)

### Depth Images
- **Supported formats**: 16-bit PNG (most common), or other formats supported by OpenCV
- **Units**: Depth values are expected in millimeters by default
- **Scale factor**: Can be specified when loading (default: 1000.0 for mm→meters conversion)
- **Invalid depths**: Zero values are treated as invalid and converted to NaN

Example:
```cpp
RGBDFrame frame;
frame.loadRGB("data/rgb.png");
frame.loadDepth("data/depth.png", 1000.0);  // 1000.0 = scale factor (mm to meters)
```

## Camera Intrinsics

**File format**: Simple text file with 4 space-separated values on one line

```
fx fy cx cy
```

Example:
```
525.0 525.0 320.0 240.0
```

Where:
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point coordinates in pixels

## PCA Morphable Model Files

The model loader supports both **binary** and **text** formats. It will automatically try binary first, then fall back to text.

### Required Files

All files should be in the same directory (specified by `model_dir`):

1. **Mean Shape**: `mean_shape.bin` or `mean_shape.txt`
2. **Identity Basis**: `identity_basis.bin` or `identity_basis.txt` (optional, can be empty)
3. **Expression Basis**: `expression_basis.bin` or `expression_basis.txt` (optional, can be empty)
4. **Identity Std Dev**: `identity_stddev.bin` or `identity_stddev.txt` (optional)
5. **Expression Std Dev**: `expression_stddev.bin` or `expression_stddev.txt` (optional)

### Binary Format

**Vector files** (mean_shape, stddev):
- Raw binary array of `double` values
- No header, just raw data
- Size: `num_elements * sizeof(double)` bytes

**Matrix files** (basis vectors):
- Option 1: With header (recommended)
  - First 4 bytes: `int32_t rows`
  - Next 4 bytes: `int32_t cols`
  - Then: `rows * cols * sizeof(double)` bytes of data
- Option 2: Without header
  - Raw binary array (dimensions inferred from file size and expected rows)
  - Requires `expected_rows` to be specified

### Text Format

**Vector files**:
- One value per line, or space-separated values
- Comments starting with `#` are ignored
- Example:
```
1.23
4.56
7.89
```

**Matrix files**:
- One row per line
- Space-separated values within each row
- Comments starting with `#` are ignored
- Example:
```
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
```

### Data Dimensions

- **Mean shape**: Vector of size `3N` where `N` = number of vertices
  - Format: `[x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]`
  
- **Identity basis**: Matrix of size `3N × M_id` where `M_id` = number of identity components
  - Each column is a basis vector
  
- **Expression basis**: Matrix of size `3N × M_exp` where `M_exp` = number of expression components
  - Each column is a basis vector

- **Std deviations**: Vectors of size matching the number of components

## Landmarks

### TXT Format

One landmark per line:
```
x y model_index
```

Example:
```
100.5 200.3 0
250.1 205.7 1
175.2 300.8 2
```

Where:
- `x`, `y`: Pixel coordinates
- `model_index`: Index of corresponding vertex in the morphable model (-1 if unknown)

### JSON Format

```json
{
  "landmarks": [
    {"x": 100.5, "y": 200.3, "model_index": 0},
    {"x": 250.1, "y": 205.7, "model_index": 1},
    {"x": 175.2, "y": 300.8, "model_index": 2}
  ]
}
```

## Example Directory Structure

```
data/
├── rgb.png                  # RGB image
├── depth.png                # Depth image (16-bit PNG)
├── intrinsics.txt           # Camera intrinsics (fx fy cx cy)
├── landmarks.txt            # 2D landmarks
└── model/                   # PCA model directory
    ├── mean_shape.bin
    ├── identity_basis.bin
    ├── expression_basis.bin
    ├── identity_stddev.bin
    └── expression_stddev.bin
```

## Converting Models from Other Formats

If you have a model in a different format (e.g., Basel Face Model), you'll need to convert it. Here's a Python example for converting from NumPy arrays:

```python
import numpy as np

# Load your model (example)
mean_shape = np.load('mean_shape.npy')  # Shape: (3*N,)
identity_basis = np.load('identity_basis.npy')  # Shape: (3*N, M_id)
identity_stddev = np.load('identity_stddev.npy')  # Shape: (M_id,)

# Save as binary
mean_shape.astype(np.float64).tofile('model/mean_shape.bin')
identity_basis.astype(np.float64).tofile('model/identity_basis.bin')
identity_stddev.astype(np.float64).tofile('model/identity_stddev.bin')

# Or save as text
np.savetxt('model/mean_shape.txt', mean_shape)
np.savetxt('model/identity_basis.txt', identity_basis)
np.savetxt('model/identity_stddev.txt', identity_stddev)
```

For binary matrices with header (recommended):
```python
def save_matrix_with_header(filename, matrix):
    rows, cols = matrix.shape
    with open(filename, 'wb') as f:
        f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        f.write(matrix.astype(np.float64).tobytes())

save_matrix_with_header('model/identity_basis.bin', identity_basis)
```
