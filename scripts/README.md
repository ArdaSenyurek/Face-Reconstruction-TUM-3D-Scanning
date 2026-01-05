# Scripts Directory

## Structure

```
scripts/
├── pipeline.py              # Main entry point (CLI)
│
├── pipeline/                 # Pipeline package
│   ├── __init__.py          # Package exports
│   ├── base.py              # Base PipelineStep class
│   ├── logger.py            # Logging utilities
│   ├── orchestrator.py     # Pipeline orchestrator
│   ├── download_utils.py    # Download/network utilities (renamed from utils.py)
│   ├── io.py                # File I/O helpers (renamed from data_helpers.py)
│   │
│   └── steps/               # Pipeline steps
│       ├── download.py      # Dataset download
│       ├── conversion.py    # RGB-D format conversion
│       ├── landmarks.py     # Landmark detection
│       ├── mapping.py      # Mapping validation (calls C++ binary)
│       ├── pose_init.py     # Pose initialization (calls C++ binary)
│       ├── reconstruction.py # 3D reconstruction (calls C++ binary)
│       └── analysis.py     # Analysis/metrics (calls C++ binary)
│
└── utils/                   # Standalone utility scripts
    └── create_landmark_mapping.py  # Manual mapping creation tool
```

## File Organization

### Core Pipeline Files
- **pipeline.py** - Main CLI entry point
- **pipeline/base.py** - Base class for all pipeline steps
- **pipeline/orchestrator.py** - Orchestrates step execution
- **pipeline/logger.py** - Logging system

### Utility Modules
- **pipeline/download_utils.py** - Download and network utilities (used by download step)
- **pipeline/io.py** - File I/O helpers (used by conversion and analysis steps)

### Pipeline Steps
All steps inherit from `PipelineStep` and implement `execute()`:
- **download.py** - Downloads BIWI dataset
- **conversion.py** - Converts RGB-D to standard format
- **landmarks.py** - Detects 2D facial landmarks (Python libraries)
- **mapping.py** - Validates/creates mapping (calls C++ `validate_mapping`)
- **pose_init.py** - Initializes pose (calls C++ `pose_init`)
- **reconstruction.py** - Reconstructs 3D meshes (calls C++ `face_reconstruction`)
- **analysis.py** - Computes metrics (calls C++ `analysis`)

### Standalone Tools
- **utils/create_landmark_mapping.py** - Manual mapping creation/editing

## Design Principles

1. **Separation of Concerns**: Each step is independent and can be skipped
2. **C++ for 3D Computation**: All 3D computation is in C++ executables
3. **Python for Orchestration**: Python handles file management, calling binaries, and data loading
4. **Clear Naming**: Files named by purpose (io.py, download_utils.py, not generic "utils")

## Usage

```bash
# Run full pipeline
python scripts/pipeline.py --sequence 01 --frames 5

# Run specific steps
python scripts/pipeline.py --skip-convert --skip-pose-init

# Manual mapping creation
python scripts/utils/create_landmark_mapping.py --output data/landmark_mapping.txt --pairs "4:700 8:810"
```

