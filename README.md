# 3D Face Reconstruction - Week 1

This project implements a 3D face reconstruction pipeline using RGB-D data and a PCA-based Morphable Face Model.

## Project Structure

```
face_reconstruction/
├── CMakeLists.txt          # Main CMake build file
├── README.md
├── include/                # Header files
│   ├── camera/
│   │   └── CameraIntrinsics.h
│   ├── data/
│   │   └── RGBDFrame.h
│   ├── utils/
│   │   └── DepthUtils.h
│   ├── model/
│   │   └── MorphableModel.h
│   ├── landmarks/
│   │   └── LandmarkData.h
│   └── alignment/
│       └── Procrustes.h
├── src/                    # Implementation files
│   ├── main.cpp
│   ├── camera/
│   ├── data/
│   ├── utils/
│   ├── model/
│   ├── landmarks/
│   └── alignment/
└── build/                  # Build directory (in gitignore)
```

## Dependencies

- **CMake** (version 3.15 or higher)
- **C++17** compatible compiler (GCC, Clang, MSVC)
- **OpenCV** (version 3.x or 4.x)
- **Eigen3** (linear algebra library)

### Installing Dependencies

#### macOS (Homebrew)
```bash
brew install opencv eigen cmake
```

#### Ubuntu/Debian
```bash
sudo apt-get install libopencv-dev libeigen3-dev cmake build-essential
```

#### Windows (vcpkg)
```bash
vcpkg install opencv eigen3
```

## Building

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# Or using make (Linux/macOS)
make -j4
```

## Running

### Basic test program
```bash
./bin/FaceReconstruction
```

### Test with real RGB-D data

**Quick test script:**
```bash
./scripts/test_real_data.sh data/test/rgb.png data/test/depth.png \
    data/test/intrinsics.txt data/model data/test
```

**Manual test:**
```bash
./bin/test_real_data --rgb data/rgb.png --depth data/depth.png \
    --intrinsics data/intrinsics.txt --model-dir data/model \
    --landmarks data/landmarks.txt --output-mesh output/face.ply
```

**Detect landmarks first:**
```bash
# Using MediaPipe (recommended)
python3 scripts/detect_landmarks.py \
    --image data/rgb.png \
    --method mediapipe \
    --output data/landmarks.txt \
    --visualize
```

For more options:
```bash
./bin/test_real_data --help
python3 scripts/detect_landmarks.py --help
```

## Week 1 Scope

Week 1 focuses on preparation and foundation:

### 1. Project Structure ✅
- CMake-based C++ project
- Modular folder structure
- Header/source separation

### 2. Data Loading (RGB-D) ✅
- `RGBDFrame`: Load RGB and depth images
- `CameraIntrinsics`: Camera intrinsic parameters
- Invalid depth value masking

### 3. Depth to 3D Backprojection ✅
- `DepthUtils`: Convert depth image to 3D point cloud
- Backprojection for individual pixels and entire images

### 4. PCA Morphable Model ✅
- `MorphableModel`: Model structure and loading interface
- Basic face reconstruction function
- Currently uses dummy data; implement `loadFromFiles()` for actual model files

### 5. Landmark Detection Interface ✅
- `LandmarkData`: Store 2D landmark data
- Support for TXT and JSON formats (JSON uses simple parser)
- Mapping to model vertex indices

### 6. Procrustes Alignment ✅
- `SimilarityTransform`: Scale, rotation, translation
- `estimateSimilarityTransform()`: Procrustes analysis implementation

### 7. Example Main Program ✅
- Example code that tests all modules
- Works even without actual data files

## Testing with Real Data

See **[TESTING_WITH_REAL_DATA.md](TESTING_WITH_REAL_DATA.md)** for a complete guide on testing with real RGB-D data.

### Quick Start: Generate Test Model

First, generate a dummy model for testing:

```bash
python3 scripts/prepare_model.py --generate-dummy --output data/model
```

This creates test model files in `data/model/` that you can use immediately.

### Preparing Your Own Model

See [scripts/README.md](scripts/README.md) for detailed instructions on:
- Converting from NumPy files
- Converting from Basel Face Model
- Converting from PyTorch/TensorFlow models
- Converting between binary and text formats

### Data Format Documentation

See [DATA_FORMAT.md](DATA_FORMAT.md) for detailed information about:
- RGB-D data formats
- PCA model file formats (binary and text)
- Landmark file formats
- Camera intrinsics format
- Example directory structure

### Quick Start with Real Data

1. **Prepare your data**:
   - RGB image (PNG/JPEG)
   - Depth image (16-bit PNG)
   - Camera intrinsics file
   - PCA model files (in a directory)
   - Landmarks file (TXT or JSON)

2. **Run the test program**:
   ```bash
   ./bin/test_real_data --rgb path/to/rgb.png \
       --depth path/to/depth.png \
       --intrinsics path/to/intrinsics.txt \
       --model-dir path/to/model/ \
       --landmarks path/to/landmarks.txt \
       --output-mesh output/face.ply
   ```

3. **View the mesh**: Open the generated PLY/OBJ file in MeshLab, Blender, or similar software.

## Next Steps (Week 2+)

- [x] Load actual PCA model files
- [x] Mesh export (PLY/OBJ)
- [ ] Landmark detection (Python wrapper + dlib/MediaPipe)
- [ ] Gauss-Newton / Levenberg-Marquardt optimization
- [ ] Dense depth alignment
- [ ] Rendering pipeline

## Notes

- `MorphableModel::loadFromFiles()` currently generates dummy data. Implement this function to load your actual PCA model files.
- A separate Python script is recommended for landmark detection (using dlib/MediaPipe), saving results as JSON/TXT to be loaded by C++.
- OpenCV and Eigen paths are automatically detected in CMakeLists.txt. If issues occur, configure them manually.

## License

This project is for educational purposes.
