# Depth-Based Parametric Face Reconstruction from RGB-D Data

3D face reconstruction pipeline for the Biwi Kinect Head Pose dataset. Uses a parametric morphable model (identity + expression), rigid alignment (Procrustes + ICP), and Gauss-Newton optimization with sparse landmark and dense depth terms. Developed for TUM 3D Scanning & Motion Capture.

**Repository:** [https://github.com/ArdaSenyurek/Face-Reconstruction-TUM-3D-Scanning](https://github.com/ArdaSenyurek/Face-Reconstruction-TUM-3D-Scanning)

---

## Pipeline design and implementation

- **Design goals:** Single entrypoint (`pipeline/main.py`), config-driven run, optional steps (download, convert, model setup, pose init, reconstruct, analysis), and clear data flow aligned with the course project (depth-based parametric reconstruction, identity + expression, evaluation).
- **Why Python + C++:** Python handles orchestration, I/O, dataset discovery, and CLI; C++ handles numeric-heavy work (Procrustes, morphable model, Gauss-Newton, depth rendering, RMSE). Pipeline steps call C++ binaries via `subprocess` with paths and flags.
- **Modular steps:** Each step is a class (e.g. `DownloadStep`, `ConversionStep`, `PoseInitStep`, `ReconstructionStep`, `TrackingStep`) with `name`, `description`, and `execute()`. The orchestrator runs them in a fixed order with shared `config` and `state`; steps can be skipped via CLI (e.g. `--skip-convert`) or when prerequisites are missing.
- **Data flow:** Raw data → `converted/` (RGB, depth, intrinsics) → `landmarks/` (dlib) → **Pose Init (Procrustes + ICP)** → `pose_init/` → `meshes/` (C++ `face_reconstruction`) or tracking outputs; overlay and analysis steps consume these. Conversion reports drive all downstream steps.
- **Why this structure:** Enables partial runs, restart from a step, and reuse of the same C++ tools from scripts (e.g. evaluation) with consistent paths and options.

---

## How we use this codebase for the project

The project is described in the course proposal ([LaTeXAuthor_Guidelines_for_Proceedings.pdf](LaTeXAuthor_Guidelines_for_Proceedings.pdf)). Mapping from proposal to codebase:

| Proposal section | Implementation |
|------------------|----------------|
| **Technical approach (Section 2):** Parametric model M(α,δ), camera/depth rendering, energy E_sparse + E_depth + E_reg, optimization (Procrustes → Gauss-Newton) | C++ `include/` + `src/`: `MorphableModel`, `DepthRenderer`, `EnergyFunction`, `GaussNewton`, `Procrustes`, `LandmarkMapping`; exposed via `pose_init` and `face_reconstruction` binaries. |
| **Outputs (Section 2.5):** Identity mesh, per-frame expression/pose, depth renderings and overlays, quantitative error plots | Pipeline outputs under `meshes/`, `pose_init/`, tracking state JSON; `create_overlays`; `analysis` binary; `scripts/compute_metrics.py`, `aggregate_summary.py`. |
| **Evaluation (Section 4):** Depth error, landmark reprojection error, energy convergence, runtime | C++ `analysis` (RMSE, depth stats); `compute_metrics.py` (landmark/depth/surface metrics); pipeline runtime logging; qualitative overlays via `create_overlays` and `generate_visuals.py`. |
| **Milestones (Section 5):** Week 2–6 (model, landmarks, pose init, renderer, optimization, tracking, evaluation) | `ModelSetupStep`, `LandmarkDetectionStep`, `MappingSetupStep`, `PoseInitStep` + C++ `pose_init`; `DepthRenderer`, `EnergyFunction`; `face_reconstruction` with `--optimize`, `ReconstructionStep`, `Week4OverlayStep` (in `pipeline/steps/overlays.py`); `TrackingStep`; `run_eval.py`, `compute_metrics.py`, `aggregate_summary.py`, `generate_visuals.py`. |

---

## Architecture and flow diagrams

### General architecture

```mermaid
flowchart LR
  subgraph repo [Repository]
    subgraph py [Python]
      main[pipeline/main.py]
      steps[pipeline/steps]
      utils[pipeline/utils]
    end
    subgraph cpp [C++ build/bin]
      pose_init[pose_init]
      face_recon[face_reconstruction]
      create_overlays[create_overlays]
      validate_mapping[validate_mapping]
      analysis_bin[analysis]
    end
    scripts[scripts/]
  end
  main --> steps
  steps --> pose_init
  steps --> face_recon
  steps --> create_overlays
  steps --> validate_mapping
  steps --> analysis_bin
  scripts --> pose_init
  scripts --> face_recon
  scripts --> create_overlays
```

**Where C++ lives:** `src/tools/` (face_reconstruction, pose_init, create_overlays, validate_mapping, analysis), `src/alignment/` (Procrustes, ICP, LandmarkMapping), `src/model/`, `src/optimization/` (GaussNewton, EnergyFunction), `src/rendering/` (DepthRenderer), `src/camera/`, `src/landmarks/`, `src/utils/`. These are compiled into the binaries in `build/bin/` called by Python.

### Pipeline flow (step order; algorithm names)

```mermaid
flowchart TD
  Start([Start]) --> Preflight[Preflight]
  Preflight --> Download{skip_download?}
  Download -->|No| DownloadStep[Download]
  Download -->|Yes| FindData[Use existing data]
  DownloadStep --> Convert[Conversion]
  FindData --> Convert
  Convert --> ModelSetup[Model Setup]
  ModelSetup --> LandmarkModel[Landmark Model Download]
  LandmarkModel --> LandmarkDetect[Landmark Detection]
  LandmarkDetect --> MappingSetup[Mapping Setup]
  MappingSetup --> PoseInit["Rigid alignment (Pose Init): Procrustes + ICP"]
  PoseInit --> Overlays{make_overlays?}
  Overlays -->|Yes| Week4Overlay[Overlays C++]
  Overlays -->|No| ReconstructBranch{track?}
  Week4Overlay --> ReconstructBranch
  ReconstructBranch -->|Yes| Tracking["Tracking: frame0 Procrustes+ICP, then warm-start + Gauss-Newton per frame"]
  ReconstructBranch -->|No| Reconstruction["Reconstruction C++: Procrustes init + Gauss-Newton (E_sparse + E_depth + E_reg)"]
  Tracking --> Analysis[Analysis C++]
  Reconstruction --> Analysis
  Analysis --> End([End])
```

- **Rigid alignment (Pose Init):** Explicit step; C++ `pose_init` runs **Procrustes** (similarity transform from landmark correspondences) then **ICP** refinement; outputs rigid-aligned mesh (PLY) and report JSON.
- **Reconstruction:** Single-frame; C++ `face_reconstruction` with optional **Gauss-Newton** (see optimization cycle below).
- **Tracking:** Frame 0 uses rigid alignment; frames 1..N use warm-start from previous frame’s state + Gauss-Newton; optional temporal smoothing (EMA/SLERP).

### Tracking flow (Frame 0 vs Frame 1..N)

```mermaid
flowchart LR
  subgraph frame0 [Frame 0]
    P0[Pose Init: Procrustes + ICP]
    S0[State R,t,scale]
    R0[face_reconstruction optional GN]
    F0[Final state JSON]
    P0 --> S0 --> R0 --> F0
  end
  subgraph frameN [Frame 1..N]
    L[Load previous state]
    R[face_reconstruction with init-pose-json + Gauss-Newton]
    F[Final state JSON]
    T[Optional EMA/SLERP smoothing]
    L --> R --> F --> T
  end
  F0 -.-> L
```

### Gauss-Newton optimization (short)

When `--optimize` is used, Reconstruction (and Tracking) run a **Gauss-Newton** loop: compute residuals r (landmark + depth + reg) → compute Jacobian J → solve (J^T J + damping)·δ = −J^T r → line search → update parameters → convergence check. Energy: **E = E_sparse + E_depth + E_reg**.

### Full Gauss-Newton optimization cycle (appendix)

```mermaid
flowchart TD
  E["Energy E = E_sparse + E_depth + E_reg"]
  R[Compute residuals r]
  J[Compute Jacobian J]
  Solve["Solve (J'T J + damping) delta = -J'T r"]
  LS[Line search]
  Upd[Update params]
  Conv{Converged?}
  Out[Output mesh]
  E --> R
  R --> J
  J --> Solve
  Solve --> LS
  LS --> Upd
  Upd --> Conv
  Conv -->|No| R
  Conv -->|Yes| Out
```

### Step → C++ binary → src/

| Pipeline step   | C++ binary         | Main src components |
|-----------------|--------------------|----------------------|
| Pose Init       | `pose_init`        | `src/tools/pose_init.cpp`, `src/alignment/Procrustes.cpp`, `LandmarkMapping.cpp`, `MorphableModel.cpp`, `ICP.cpp` |
| Reconstruction  | `face_reconstruction` | `src/tools/face_reconstruction.cpp`, `src/optimization/`, `src/rendering/`, `src/model/`, `src/alignment/` |
| Overlays        | `create_overlays`  | `src/tools/create_overlays.cpp`, `DepthRenderer`, `DepthUtils` |
| Mapping check   | `validate_mapping` | `src/tools/validate_mapping.cpp`, `MorphableModel`, `LandmarkMapping` |
| Analysis        | `analysis`         | `src/tools/analysis.cpp` |

---

## Entrypoints

### Main pipeline

| Entrypoint | Purpose | Main options |
|------------|---------|--------------|
| **pipeline/main.py** | End-to-end Biwi pipeline: download (optional), convert RGB-D, landmarks, BFM model setup, **Pose Init (Procrustes + ICP)**, reconstruction or tracking, optional overlays and Week 6 eval. | `--data-root`, `--output-root`, `--frames`, `--download`, `--skip-convert`, `--skip-pose-init`, `--skip-reconstruct`, `--no-analysis`, `--optimize`, `--track`, `--make-overlays`, `--week6-eval`, `--recon-binary`, `--pose-init-binary`, `--model-dir`, `--landmark-mapping` |

**Launch (from repo root):**
```bash
python pipeline/main.py [options]
```

**Example:**
```bash
python pipeline/main.py --download --frames 5
python pipeline/main.py --no-analysis --optimize --track
```

**Outputs:** Under `output_root`: `converted/`, `landmarks/`, `pose_init/`, `meshes/`, `analysis/`, `logs/`. With `--week6-eval`: `outputs/week6/`.

---

### C++ tools (build/bin/)

Built with CMake; called by the pipeline or scripts.

| Binary | Purpose | Main inputs | Outputs |
|--------|---------|-------------|---------|
| **face_reconstruction** | 3D mesh from RGB-D using PCA model + optional **Gauss-Newton** (E_sparse + E_depth + E_reg); supports tracking (init/state JSON). | `--rgb`, `--depth`, `--intrinsics`, `--model-dir`, `--landmarks`, `--mapping`, `--output-mesh`; `--optimize`, `--init-pose-json`, `--output-state-json` | PLY mesh; optional state JSON |
| **pose_init** | **Rigid alignment:** **Procrustes** + **ICP** from landmarks and depth. | `--rgb`, `--depth`, `--intrinsics`, `--landmarks`, `--model-dir`, `--mapping`, `--output`, `--report` | Aligned mesh (PLY); optional report JSON |
| **create_overlays** | Mesh-vs-scan overlays (rigid and optional optimized). | `--mesh-rigid`, `--depth`, `--intrinsics`, `--out-dir`; optional `--mesh-opt`, `--rgb`, `--output-metrics` | 3D overlay PLY, 2D PNG, depth comparison, metrics JSON |
| **validate_mapping** | Check landmark-to-model mapping against PCA model. | `--mapping`, `--model-dir`, optional `--min-count` | Exit code + console |
| **analysis** | 3D metrics (cloud-to-mesh RMSE, depth stats). | `--pointcloud`, `--mesh`; optional `--depth`, `--output-vis`, `--output-json` | JSON; optional PNG |

---

### Scripts (scripts/)

| Script | Purpose | Main options | Outputs |
|--------|---------|--------------|---------|
| **run_eval.py** | Three-stage evaluation: identity → expression → tracking. | `--data-root`, `--output-root`, `--sequences`, `--frames`, `--stage`, `--recon-binary`, `--pose-init-binary`, etc. | `outputs/week6/<seq>/<frame>/meshes/` (identity.ply, expression.ply, tracked.ply), metrics, visuals |
| **export_overlay_ply.py** | Export 3D overlay PLY for one seq/frame (rigid or opt). | `--seq`, `--stage` (rigid \| opt), `--frame`, `--output-root`, `--report` | Overlay PLY under week6 or reports/week6/figures |
| **generate_visuals.py** | 2D overlay (RGB + mesh) and depth comparison images. | `--seq`, `--frame`, `--mesh`, `--depth`, `--intrinsics`, `--output-dir` | overlay_rgb.png, depth_obs/rend/residual PNGs |
| **compute_metrics.py** | Per-frame metrics: 2D landmark, depth, 3D surface error. | `--mesh`, `--depth`, `--intrinsics`, `--landmarks`, `--mapping`, `--model-dir`, `--output`; optional `--pointcloud` | JSON |
| **aggregate_summary.py** | Aggregate week6 metrics into one CSV. | `--week6-dir` (default `outputs/week6`) | `outputs/week6/summary.csv` |
| **analyze_sparse_alignment.py** | Analyze pose_init JSON reports. | `--reports-dir`, `--output-dir` | Summary stats, plots, CSV |

---

### Pipeline utils (runnable as scripts)

| File | Purpose | Main usage |
|------|---------|------------|
| **pipeline/utils/convert_bfm_to_project.py** | Convert BFM (.mat/.h5) to project binary format. | BFM path + output model dir → mean_shape.bin, identity/expression basis, faces.bin |
| **pipeline/utils/create_bfm_landmark_mapping.py** | Build landmark mapping from BFM semantic indices. | `--bfm`, `--output` |
| **pipeline/utils/verify_bfm_semantic_mapping.py** | Verify BFM landmark mapping. | `--bfm`, mapping path |
| **pipeline/utils/triangulate_pointcloud.py** | Triangulate point cloud to mesh. | input PLY, output path |
| **pipeline/utils/center_mesh.py** | Center mesh at origin. | input/output mesh paths |
| **pipeline/utils/debug_alignment.py** | Debug alignment visualization. | `--seq`, paths |
| **pipeline/steps/overlays.py** | Generate overlays for sequences (e.g. 01, 17). | Run as module or via pipeline `--make-overlays` |

---

## Dependencies

### Python

- **File:** [requirements.txt](requirements.txt)  
  numpy, opencv-python, scipy, matplotlib, setuptools, wheel, dlib, kagglehub, h5py (and optional alternatives in comments).

**Install:**
```bash
pip install -r requirements.txt
```
Use Python 3.8+; a virtual environment is recommended (`python -m venv .venv` then activate).

### C++ (for pipeline tools)

- **CMake** 3.15+, **C++17** compiler, **Eigen3**, **OpenCV** (required for some tools).

**Build:**
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
Executables appear in `build/bin/`.

**System packages (examples):**
- macOS: `brew install eigen opencv`
- Ubuntu: `sudo apt install libeigen3-dev libopencv-dev`

---

## Installation

1. Clone the repo and go to the project root.
2. Install Python deps: `pip install -r requirements.txt`
3. Build C++ tools: `mkdir build && cd build && cmake .. && cmake --build .`
4. (Optional) Obtain Biwi data: run with `--download` (requires Kaggle credentials) or place data under `data/` as expected by the pipeline.
5. (Optional) BFM model: place BFM file (.mat or .h5) in `data/bfm/` (see Large assets below).

---

## Large assets

Files larger than ~15 MB (e.g. BFM model) are **not** included in the repository or zip. Obtain the BFM (e.g. BFM 2017 or 2019 full head) from the official source or course resources and place it in `data/bfm/`. If you host such files on cloud storage (Google Drive, institutional host, etc.), add the link here for others to download.

---

## License

See [LICENSE](LICENSE).
