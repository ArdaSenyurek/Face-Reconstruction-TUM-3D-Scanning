# 3D Face Reconstruction from RGB-D Data

Real-time parametric face reconstruction using the Basel Face Model (BFM) and RGB-D data from the Biwi Kinect Head Pose Database.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Basel Face Model (see below)
# Place model2019_fullHead.h5 in data/bfm/

# 3. Run the pipeline
python pipeline/main.py --download --frames 2 --optimize
```

## Prerequisites

### System Requirements
- **OS**: macOS, Linux (Windows untested)
- **Python**: 3.8+
- **CMake**: 3.14+
- **C++ Compiler**: C++17 support required

### Required Libraries
```bash
# macOS
brew install cmake eigen opencv

# Ubuntu/Debian
sudo apt-get install cmake libeigen3-dev libopencv-dev
```

## Setup

### Step 1: Clone and Install Python Dependencies

```bash
git clone <repository-url>
cd Face-Reconstruction-TUM-3D-Scanning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download the Basel Face Model (Required)

The Basel Face Model (BFM) 2019 is required for reconstruction.

1. **Register** at: https://faces.dmi.unibas.ch/bfm/bfm2019.html
2. **Download**: `model2019_fullHead.h5`
3. **Place** the file in: `data/bfm/model2019_fullHead.h5`

```bash
mkdir -p data/bfm
# Copy your downloaded model here
cp ~/Downloads/model2019_fullHead.h5 data/bfm/
```

### Step 3: Set Up Kaggle API (For Dataset Download)

The Biwi dataset is downloaded automatically from Kaggle.

1. **Create** a Kaggle account at https://www.kaggle.com
2. **Get API credentials**: Kaggle → Settings → API → Create New Token
3. **Save** `kaggle.json` to `~/.kaggle/kaggle.json`

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Alternatively, set environment variables:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

## Usage

### Basic Usage

```bash
# Download dataset + run pipeline (2 frames per sequence)
python pipeline/main.py --download --frames 2

# With Gauss-Newton optimization (recommended)
python pipeline/main.py --download --frames 2 --optimize
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--download` | Download Biwi dataset from Kaggle | Off |
| `--frames N` | Process N frames per sequence | 5 |
| `--optimize` | Enable Gauss-Newton optimization | Off |
| `--pose-only` | Optimize pose only (faster) | Off |
| `--sequence SEQ` | Process specific sequence(s) only | All |

### Skip Steps (for re-runs)

```bash
# Skip steps that were already completed
python pipeline/main.py --skip-convert --skip-model-setup --frames 5 --optimize
```

| Skip Option | What it skips |
|-------------|---------------|
| `--skip-download` | Dataset download |
| `--skip-convert` | Data conversion |
| `--skip-model-setup` | BFM model conversion |
| `--skip-landmarks` | Landmark detection |
| `--skip-pose-init` | Initial pose estimation |
| `--skip-reconstruct` | 3D reconstruction |
| `--no-analysis` | Analysis step |

### Advanced Optimization Settings

```bash
python pipeline/main.py --optimize \
    --max-iterations 20 \
    --lambda-landmark 1.0 \
    --lambda-depth 0.1 \
    --lambda-reg 1.0 \
    --verbose-optimize
```

## Output Structure

```
outputs/
├── converted/           # Processed RGB-D frames
│   └── 01/
│       ├── rgb/         # RGB images (PNG)
│       ├── depth/       # Depth maps (PNG, 16-bit)
│       ├── depth_vis/   # Depth visualizations
│       └── intrinsics.txt
├── landmarks/           # Detected 2D landmarks (TXT)
├── overlays/            # Landmark visualization on RGB
├── pose_init/           # Initial pose estimation (PLY)
├── meshes/              # Final reconstructed faces (PLY)
│   └── 01/
│       ├── frame_00000.ply
│       └── frame_00001.ply
├── analysis/
│   ├── metrics.json     # RMSE and statistics
│   └── pointclouds/     # Depth point clouds
└── logs/
    ├── pipeline_summary.json
    └── pipeline_*.log
```

## Viewing Results

### PLY Files
Open `.ply` files with:
- **MeshLab** (free): https://www.meshlab.net
- **CloudCompare** (free): https://cloudcompare.org
- **Blender** (free): https://www.blender.org

### Metrics
Check reconstruction quality in `outputs/analysis/metrics.json`:
```json
{
  "01": {
    "frame_00000": {
      "rmse_cloud_mesh_m": 0.167,  // Lower is better
      "cloud_points": 56336
    }
  }
}
```

## Pipeline Steps

1. **Preflight**: Check dependencies, build C++ binaries
2. **Download**: Fetch Biwi dataset from Kaggle
3. **Conversion**: Convert raw data to standard format
4. **Model Setup**: Convert BFM to project format
5. **Landmark Model**: Download dlib face landmark model
6. **Landmarks**: Detect 68 facial landmarks
7. **Mapping**: Validate landmark-to-vertex mapping
8. **Pose Init**: Initial pose via Procrustes alignment
9. **Reconstruction**: Gauss-Newton optimization
10. **Analysis**: Compute RMSE metrics

## Troubleshooting

### "Model not found"
```bash
# Ensure BFM model is in the correct location
ls data/bfm/model2019_fullHead.h5
```

### "Kaggle API not configured"
```bash
# Check kaggle.json exists
ls ~/.kaggle/kaggle.json
# Or set environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

### Build Errors
```bash
# Rebuild from scratch
rm -rf build
mkdir build && cd build
cmake ..
make -j4
```

### dlib Installation Issues
```bash
# On macOS, ensure Xcode tools are installed
xcode-select --install

# Use conda if pip fails
conda install -c conda-forge dlib
```

## Project Structure

```
├── pipeline/           # Python orchestration
│   ├── main.py         # Entry point
│   └── steps/          # Pipeline steps
├── src/                # C++ implementation
│   ├── tools/          # Executables
│   ├── model/          # Morphable model
│   ├── optimization/   # Gauss-Newton optimizer
│   └── alignment/      # Procrustes, ICP
├── include/            # C++ headers
├── data/               # Models and datasets
└── outputs/            # Results
```

## Citation

If you use this code, please cite:
- Basel Face Model: https://faces.dmi.unibas.ch
- Biwi Kinect Head Pose Database: Fanelli et al., IJCV 2013

## License

See LICENSE file.
