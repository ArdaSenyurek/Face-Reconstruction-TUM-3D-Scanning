#!/usr/bin/env python3
"""
Unified pipeline for the Biwi Kinect Head Pose dataset.

This module contains all pipeline infrastructure:
- Base classes for pipeline steps
- Logging system
- Pipeline orchestrator
- Download and I/O utilities

Pipeline steps are in pipeline/steps/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

# Try to import NumPy with better error handling
# NumPy can crash with segfaults if OpenBLAS is corrupted
try:
    import numpy as np
except (ImportError, AttributeError, OSError) as e:
    # If NumPy import fails, provide a helpful error message
    import sys
    print(f"ERROR: Failed to import NumPy: {e}", file=sys.stderr)
    print("This is likely due to a corrupted NumPy/OpenBLAS installation.", file=sys.stderr)
    print("Solutions:", file=sys.stderr)
    print("  1. Reinstall NumPy: pip install --upgrade --force-reinstall numpy", file=sys.stderr)
    print("  2. Set environment variable: export OPENBLAS_NUM_THREADS=1", file=sys.stderr)
    print("  3. Use a different Python environment with a working NumPy", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Catch any other exceptions including segfaults that might be reported as exceptions
    import sys
    print(f"ERROR: NumPy import caused an error: {e}", file=sys.stderr)
    print("This might be a segfault in OpenBLAS. Check the crash report.", file=sys.stderr)
    print("Try: pip install --upgrade --force-reinstall numpy", file=sys.stderr)
    sys.exit(1)


# Lazy import scipy to avoid hanging on macOS
_HAS_CKDTREE = None

def _check_scipy():
    """Lazy check for scipy availability."""
    global _HAS_CKDTREE
    if _HAS_CKDTREE is not None:
        return _HAS_CKDTREE
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("scipy import timed out")
        
        # Set a timeout for scipy import (5 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        try:
            from scipy.spatial import cKDTree
            _HAS_CKDTREE = True
        finally:
            signal.alarm(0)  # Cancel alarm
    except (ImportError, TimeoutError, Exception):
        _HAS_CKDTREE = False
    return _HAS_CKDTREE

# Pipeline steps are imported lazily in the run() method to avoid circular imports
# Do not import here at module level

# ============================================================================
# Logging System
# ============================================================================

class LogLevel(Enum):
    """Logging levels for the pipeline."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PipelineLogger:
    """
    Enhanced logger with progress tracking and colored terminal output.
    """
    
    def __init__(self, name: str = "pipeline", log_file: Optional[Path] = None, level: LogLevel = LogLevel.INFO):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        self.logger.setLevel(getattr(logging, level.value))
        self.log_file: Optional[Path] = log_file
        self.start_time = time.time()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_console_formatter())
        self.logger.addHandler(console_handler)
        
        # File handler if log file specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._get_file_formatter())
            self.logger.addHandler(file_handler)
    
    @staticmethod
    def _get_console_formatter() -> logging.Formatter:
        """Get formatter for console output with colors."""
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',      # Cyan
                'INFO': '\033[32m',       # Green
                'WARNING': '\033[33m',    # Yellow
                'ERROR': '\033[31m',      # Red
                'CRITICAL': '\033[35m',   # Magenta
                'RESET': '\033[0m'
            }
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
                reset = self.COLORS['RESET']
                record.levelname = f"{log_color}{record.levelname}{reset}"
                return super().format(record)
        
        return ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    
    @staticmethod
    def _get_file_formatter() -> logging.Formatter:
        """Get formatter for file output (no colors)."""
        return logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def step_start(self, step_name: str, description: str = ""):
        """Log the start of a pipeline step."""
        msg = f"▶ {step_name}"
        if description:
            msg += f": {description}"
        self.info(msg)
    
    def step_success(self, step_name: str, message: str = ""):
        """Log successful completion of a step."""
        msg = f"✓ {step_name} completed"
        if message:
            msg += f": {message}"
        self.info(msg)
    
    def step_skip(self, step_name: str, reason: str = ""):
        """Log that a step was skipped."""
        msg = f"⊘ {step_name} skipped"
        if reason:
            msg += f": {reason}"
        self.info(msg)
    
    def step_error(self, step_name: str, error: str):
        """Log an error in a step."""
        self.error(f"✗ {step_name} failed: {error}")
    
    def progress(self, current: int, total: int, item: str = "items"):
        """Log progress information."""
        percentage = (current / total * 100) if total > 0 else 0
        self.info(f"Progress: {current}/{total} {item} ({percentage:.1f}%)")
    
    def elapsed_time(self) -> float:
        """Get elapsed time since logger creation."""
        return time.time() - self.start_time
    
    def summary(self, message: str):
        """Log a summary message with separator."""
        self.info("=" * 60)
        self.info(message)
        self.info("=" * 60)


def setup_logging(log_dir: Path, level: str = "INFO") -> tuple[PipelineLogger, Path]:
    """
    Set up logging for the pipeline.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Tuple of (logger instance, log file path)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    log_level = LogLevel[level.upper()] if level.upper() in [e.name for e in LogLevel] else LogLevel.INFO
    logger = PipelineLogger("pipeline", log_file, log_level)
    
    return logger, log_file


# ============================================================================
# Base Pipeline Step Classes
# ============================================================================

class StepStatus(Enum):
    """Status of a pipeline step execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StepResult:
    """Result of executing a pipeline step."""
    status: StepStatus
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if step succeeded."""
        return self.status == StepStatus.SUCCESS
    
    @property
    def skipped(self) -> bool:
        """Check if step was skipped."""
        return self.status == StepStatus.SKIPPED


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.
    
    Each step should inherit from this class and implement:
    - name: A descriptive name for the step
    - description: A brief description of what the step does
    - execute(): The main execution logic
    """
    
    def __init__(self, logger: PipelineLogger, config: Dict[str, Any]):
        """
        Initialize a pipeline step.
        
        Args:
            logger: Logger instance for this step
            config: Configuration dictionary for this step
        """
        self.logger = logger
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this step."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this step does."""
        pass
    
    @abstractmethod
    def execute(self) -> StepResult:
        """
        Execute this pipeline step.
        
        Returns:
            StepResult indicating success, failure, or skip status
        """
        pass
    
    def should_run(self) -> bool:
        """
        Check if this step should run.
        
        Override this method to add conditional logic.
        By default, checks for a 'skip' flag in config.
        
        Returns:
            True if step should run, False otherwise
        """
        return not self.config.get("skip", False)
    
    def run(self) -> StepResult:
        """
        Run this step with logging and error handling.
        
        Returns:
            StepResult with execution status
        """
        if not self.should_run():
            self.logger.step_skip(self.name, self.config.get("skip_reason", "configured to skip"))
            return StepResult(StepStatus.SKIPPED, "Step skipped by configuration")
        
        self.logger.step_start(self.name, self.description)
        
        try:
            result = self.execute()
            
            if result.success:
                self.logger.step_success(self.name, result.message)
            elif result.skipped:
                self.logger.step_skip(self.name, result.message)
            else:
                self.logger.step_error(self.name, result.error or result.message)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.logger.step_error(self.name, error_msg)
            return StepResult(StepStatus.FAILED, f"Step failed with exception: {error_msg}", error=error_msg)


# ============================================================================
# File I/O Utilities
# ============================================================================

def read_biwi_calibration(cal_file: Path) -> Optional[Tuple[float, float, float, float]]:
    """Parse Biwi calibration file (depth.cal) to intrinsics."""
    try:
        with open(cal_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            fx = float(lines[0].strip().split()[0])
            cx = float(lines[0].strip().split()[2])
            fy = float(lines[1].strip().split()[1])
            cy = float(lines[1].strip().split()[2])
            return fx, fy, cx, cy
    except Exception:
        return None


def default_intrinsics() -> Tuple[float, float, float, float]:
    """Kinect v1 defaults for 640x480."""
    return 525.0, 525.0, 319.5, 239.5


def read_biwi_depth_binary(depth_path: Path) -> Optional[np.ndarray]:
    """Decode Biwi RLE depth binary format to uint16 image."""
    try:
        with open(depth_path, "rb") as f:
            header = f.read(8)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return None
            f.seek(0)
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            if width < 100 or width > 2000 or height < 100 or height > 2000:
                return None

            depth_img = np.zeros((height, width), dtype=np.int16)
            total = width * height
            p = 0
            while p < total:
                num_empty = np.fromfile(f, dtype=np.int32, count=1)
                if len(num_empty) == 0:
                    break
                p += int(num_empty[0])
                if p >= total:
                    break
                num_full = np.fromfile(f, dtype=np.int32, count=1)
                if len(num_full) == 0:
                    break
                num_full_val = int(num_full[0])
                if num_full_val > 0:
                    vals = np.fromfile(f, dtype=np.int16, count=num_full_val)
                    end_idx = min(p + num_full_val, total)
                    depth_img.flat[p:end_idx] = vals[: end_idx - p]
                    p += num_full_val
                else:
                    break

            return np.clip(depth_img, 0, 65535).astype(np.uint16)
    except Exception:
        return None


def convert_depth_image(depth_path: Path, output_path: Path) -> bool:
    """Convert depth image to standardized format."""
    depth = read_biwi_depth_binary(depth_path)
    if depth is None:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if depth is None:
        return False
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    if depth.dtype == np.uint8:
        depth = depth.astype(np.uint16) * 256
    elif depth.dtype in (np.float32, np.float64):
        depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    elif depth.dtype in (np.int32, np.int64):
        depth = np.clip(depth, 0, 65535).astype(np.uint16)
    depth = np.clip(depth, 0, 8000).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), depth)
    return True


def copy_rgb(rgb_path: Path, output_path: Path) -> bool:
    """Copy RGB image to output location."""
    img = cv2.imread(str(rgb_path))
    if img is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return True


def create_depth_visualization(depth_path: Path, output_path: Path, 
                              colormap: int = None,
                              normalize_percentile: tuple = None) -> bool:
    """
    Create a visually meaningful depth image from raw depth data.
    
    Args:
        depth_path: Path to raw depth image (uint16, in mm)
        output_path: Path to save visualization (uint8 BGR, colormapped)
        colormap: OpenCV colormap (default: JET for depth visualization)
        normalize_percentile: Percentile range for normalization (min, max)
                             Use percentiles to ignore outliers
    
    Returns:
        True if successful, False otherwise
    """
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    if normalize_percentile is None:
        normalize_percentile = (1.0, 99.0)
    
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return False
    
    # Get valid depth pixels (non-zero)
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # No valid depth data
        return False
    
    # Get depth values for normalization
    valid_depths = depth[valid_mask]
    
    # Use percentile-based normalization to handle outliers
    min_percentile, max_percentile = normalize_percentile
    depth_min = np.percentile(valid_depths, min_percentile)
    depth_max = np.percentile(valid_depths, max_percentile)
    
    # Normalize to 0-255 range
    depth_normalized = np.zeros_like(depth, dtype=np.float32)
    depth_normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-6)
    depth_normalized = np.clip(depth_normalized * 255, 0, 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]
    
    # Save visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), depth_colored)
    return True


def find_rgb_depth_pairs(seq_dir: Path) -> List[Tuple[Path, Path]]:
    """Discover matching RGB/depth files inside a Biwi sequence directory."""
    rgb_candidates = list(seq_dir.glob("*rgb*.png")) + list((seq_dir / "rgb").glob("*.png"))
    depth_candidates = list(seq_dir.glob("*depth*.*")) + list((seq_dir / "depth").glob("*.*"))
    rgb_candidates = [p for p in rgb_candidates if p.is_file()]
    depth_candidates = [p for p in depth_candidates if p.is_file()]

    depth_map: dict[tuple[str, str], Path] = {}
    for depth in depth_candidates:
        stem = depth.stem.lower()
        for suffix in ("_depth", "-depth"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        depth_map[(str(depth.parent), stem)] = depth

    pairs: List[Tuple[Path, Path]] = []
    for rgb in sorted(rgb_candidates):
        stem = rgb.stem.lower()
        for suffix in ("_rgb", "-rgb"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        key = (str(rgb.parent), stem)
        if key in depth_map:
            pairs.append((rgb, depth_map[key]))
        else:
            fallback = rgb.parent / f"{stem}_depth.bin"
            if fallback.exists():
                pairs.append((rgb, fallback))

    return pairs


def load_mesh_vertices(ply_path: Path) -> np.ndarray:
    """Load vertices from PLY file."""
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines or not lines[0].strip().startswith("ply"):
        return np.empty((0, 3), dtype=np.float32)
    num_vertices = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[2])
        if line.strip() == "end_header":
            header_end = i + 1
            break
    verts = []
    for line in lines[header_end : header_end + num_vertices]:
        parts = line.strip().split()
        if len(parts) >= 3:
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(verts, dtype=np.float32)


def compute_cloud_to_mesh_rmse(cloud: np.ndarray, mesh: np.ndarray, sample: int = 20000) -> float:
    """Compute RMSE between point cloud and mesh vertices.
    
    Note: This function is kept for backward compatibility but is no longer used
    in the main pipeline. The C++ analysis binary handles RMSE computation.
    """
    if cloud.shape[0] == 0 or mesh.shape[0] == 0 or not _check_scipy():
        return float("nan")
    if cloud.shape[0] > sample:
        idx = np.random.choice(cloud.shape[0], sample, replace=False)
        cloud = cloud[idx]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh)
    except (ImportError, Exception):
        return float("nan")
    dists, _ = tree.query(cloud, k=1)
    rmse = float(np.sqrt(np.mean(dists ** 2)))
    return rmse


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class PipelineOrchestrator:
    """
    Orchestrates the execution of pipeline steps in order.
    
    Each step receives configuration and can pass data to subsequent steps
    through the shared state dictionary.
    """
    
    def __init__(self, logger: PipelineLogger, config: Dict[str, Any]):
        """
        Initialize the orchestrator.
        
        Args:
            logger: Logger instance
            config: Global configuration dictionary
        """
        self.logger = logger
        self.config = config
        self.state: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Summary dictionary with execution results
        """
        # Import steps here to avoid circular imports
        from pipeline.steps import (
            AnalysisStep,
            ConversionStep,
            DownloadStep,
            LandmarkDetectionStep,
            LandmarkModelDownloadStep,
            MappingSetupStep,
            ModelSetupStep,
            PoseInitStep,
            ReconstructionStep,
        )
        from pipeline.steps.week4_overlays import Week4OverlayStep
        from pipeline.steps.preflight import PreflightStep
        from pipeline.steps.tracking import TrackingStep
        
        start_time = time.time()
        self.logger.summary("Starting Pipeline Execution")
        
        # Step 0: Preflight checks (auto-build if needed)
        step = PreflightStep(self.logger, {
            "model_dir": self.config.get("model_dir", DEFAULT_MODEL_DIR),
            "bfm_dir": self.config.get("bfm_dir", DEFAULT_BFM_DIR),
            "auto_build": self.config.get("auto_build", True),
        })
        result = step.run()
        self._record_result("preflight", result)
        if not result.success and not result.skipped:
            return self._build_summary(start_time, success=False)
        
        # Step 1: Download & Extract
        if not self.config.get("skip_download", False):
            step = DownloadStep(self.logger, {
                "download_dir": self.config["data_root"] / "biwi_download",
                "kaggle_dataset": self.config.get("kaggle_dataset", "kmader/biwi-kinect-head-pose-database"),
                "skip": self.config.get("skip_download", False),
            })
            result = step.run()
            self._record_result("download", result)
            if result.success:
                self.state["extract_root"] = result.data.get("extract_root")
            elif not result.skipped:
                return self._build_summary(start_time, success=False)
        else:
            # Try to find existing data
            download_dir = self.config["data_root"] / "biwi_download"
            extract_root = download_dir / "hpdb"
            preexisting_faces = download_dir / "faces_0"
            if extract_root.exists():
                self.state["extract_root"] = str(extract_root)
            elif preexisting_faces.exists():
                self.state["extract_root"] = str(preexisting_faces)
        
        # Step 2: Data Conversion
        if not self.config.get("skip_convert", False) and "extract_root" in self.state:
            step = ConversionStep(self.logger, {
                "raw_root": Path(self.state["extract_root"]),
                "output_root": self.config["output_root"] / "converted",
                "max_frames": self.config.get("max_frames", 25),
                "sequence": self.config.get("sequence"),
                "intrinsics": self.config.get("intrinsics"),
            })
            result = step.run()
            self._record_result("conversion", result)
            if result.success:
                reports = result.data.get("reports", [])
                filtered_reports = [
                    r for r in reports
                    if not self._should_ignore_sequence(r.get("sequence", ""))
                ]
                if len(filtered_reports) != len(reports):
                    self.logger.info("Skipping ignored temporary sequences")
                self.state["conversion_reports"] = filtered_reports
                self._persist_conversion_reports(self.state["conversion_reports"])
            elif not result.skipped:
                return self._build_summary(start_time, success=False)
        else:
            self.logger.step_skip("Data Conversion", "configured to skip")
            # Try to load persisted conversion reports
            self._load_conversion_reports()
        
        # Step 3: Model Setup (BFM conversion if needed)
        # Uses ModelSetupStep which handles BFM conversion internally
        if not self.config.get("skip_model_setup", False):
            step = ModelSetupStep(self.logger, {
                "model_dir": self.config.get("model_dir", DEFAULT_MODEL_DIR),
                "bfm_dir": self.config.get("bfm_dir", DEFAULT_BFM_DIR),
            })
            result = step.run()
            self._record_result("model_setup", result)
            
            if result.success:
                self.state["model_dir"] = result.data.get("model_dir", self.config.get("model_dir", DEFAULT_MODEL_DIR))
            elif not result.skipped:
                if self.config.get("require_model", True):
                    self.logger.error("Model setup failed and model is required. Pipeline cannot continue.")
                    return self._build_summary(start_time, success=False)
                else:
                    self.logger.warning("Model setup failed, but continuing without model (require_model=False)")
        
        # Step 4: Model Download (dlib predictor)
        step = LandmarkModelDownloadStep(self.logger, {
            "models_dir": self.config.get("models_dir", Path("data/models")),
        })
        result = step.run()
        self._record_result("model_download", result)
        
        # Step 5: Landmark Detection (dlib only)
        if "conversion_reports" in self.state:
            step = LandmarkDetectionStep(self.logger, {
                "conversion_reports": self.state.get("conversion_reports", []),
                "landmarks_root": self.config["output_root"] / "landmarks",
                "overlays_root": self.config["output_root"] / "overlays",
                "run_frames": self.config.get("run_frames", 5),
            })
            result = step.run()
            self._record_result("landmarks", result)
        else:
            self.logger.step_skip("Landmark Detection", "no conversion data")
        
        # Step 6: Landmark Mapping Setup (C++ binary)
        if not self.config.get("skip_reconstruct", False):
            step = MappingSetupStep(self.logger, {
                "landmark_mapping": self.config.get("landmark_mapping", "data/landmark_mapping.txt"),
                "model_dir": self.config.get("model_dir", Path("data/model_biwi")),
                "auto_generate_mapping": self.config.get("auto_generate_mapping", True),  # Default: enabled
                "min_mapping_count": self.config.get("min_mapping_count", 30),  # Default: 30 mappings
                "validate_mapping_binary": self.config.get("validate_mapping_binary", Path("build/bin/validate_mapping")),
            })
            result = step.run()
            self._record_result("mapping", result)
            if result.success:
                self.state["mapping_file"] = result.data.get("mapping_file")
            elif not result.skipped:
                self.logger.warning("Landmark mapping setup failed, but continuing...")
        
        # Step 7: Pose Initialization (Procrustes alignment)
        if not self.config.get("skip_pose_init", False) and "conversion_reports" in self.state:
            step = PoseInitStep(self.logger, {
                "pose_init_binary": self.config.get("pose_init_binary", Path("build/bin/pose_init")),
                "model_dir": self.config.get("model_dir", Path("data/model_biwi")),
                "landmark_mapping": self.config.get("landmark_mapping", "data/landmark_mapping.txt"),
                "conversion_reports": self.state.get("conversion_reports", []),
                "landmarks_root": self.config["output_root"] / "landmarks",
                "pose_init_root": self.config["output_root"] / "pose_init",
                "run_frames": self.config.get("run_frames", 5),
                "timeout": self.config.get("timeout", 60),
            })
            result = step.run()
            self._record_result("pose_init", result)
            if result.success:
                self.state["pose_init_reports"] = result.data.get("reports", [])
        else:
            self.logger.step_skip("Pose Initialization", "configured to skip or no data")
        
        # Step 7.5: Week 4 Overlays (after rigid alignment)
        if (self.config.get("make_overlays", False) and 
            "pose_init_reports" in self.state and
            "conversion_reports" in self.state):
            step = Week4OverlayStep(self.logger, {
                "overlay_binary": self.config.get("overlay_binary", Path("build/bin/create_overlays")),
                "pose_init_reports": self.state.get("pose_init_reports", []),
                "conversion_reports": self.state.get("conversion_reports", []),
                "overlays_3d_root": self.config["output_root"] / "overlays3d",
                "overlays_2d_root": self.config["output_root"] / "overlays2d",
                "depth_overlay_root": self.config["output_root"] / "depth_overlay",
                "target_sequences": self.config.get("target_sequences", ["01", "17"]),
            })
            result = step.run()
            self._record_result("week4_overlays", result)
        else:
            self.logger.step_skip("Week 4 Overlays", "configured to skip or no data")
        
        # Step 8: Reconstruction or Tracking (Week 5: tracking mode)
        if not self.config.get("skip_reconstruct", False) and "conversion_reports" in self.state:
            if self.config.get("track", False):
                # Week 5: Use TrackingStep for sequential processing with warm-start
                step = TrackingStep(self.logger, {
                    "binary": self.config.get("recon_binary", Path("build/bin/face_reconstruction")),
                    "pose_init_binary": self.config.get("pose_init_binary", Path("build/bin/pose_init")),
                    "overlay_binary": self.config.get("overlay_binary", Path("build/bin/create_overlays")),
                    "model_dir": self.config.get("model_dir", DEFAULT_MODEL_DIR),
                    "conversion_reports": self.state.get("conversion_reports", []),
                    "meshes_root": self.config["output_root"] / "meshes",
                    "analysis_root": self.config.get("analysis_root", self.config["output_root"] / "analysis"),
                    "landmarks_root": self.config["output_root"] / "landmarks",
                    "pose_init_root": self.config["output_root"] / "pose_init",
                    "tracking_root": self.config["output_root"] / "tracking",
                    "overlays_3d_root": self.config["output_root"] / "overlays_3d",
                    "run_frames": self.config.get("run_frames", 5),
                    "timeout": self.config.get("timeout", 60),
                    "target_sequences": self.config.get("target_sequences", ["01", "17"]),
                    # Optimization settings
                    "optimize": self.config.get("optimize", False),
                    "landmark_mapping": self.config.get("landmark_mapping", "data/landmark_mapping.txt"),
                    "verbose": self.config.get("verbose_optimize", False),
                    "max_iterations": self.config.get("max_iterations", 50),
                    "lambda_landmark": self.config.get("lambda_landmark", 1.0),
                    "lambda_depth": self.config.get("lambda_depth", 0.1),
                    "lambda_reg": self.config.get("lambda_reg", 1.0),
                    # Week 5: Tracking settings
                    "temporal_smoothing": self.config.get("temporal_smoothing", False),
                    "smooth_pose_alpha": self.config.get("smooth_pose_alpha", 0.7),
                    "smooth_expr_alpha": self.config.get("smooth_expr_alpha", 0.7),
                    "reinit_every": self.config.get("reinit_every", 0),
                    "drift_rmse_thresh": self.config.get("drift_rmse_thresh", 80.0),
                    "save_overlays_3d": self.config.get("save_overlays_3d", True),
                    "save_depth_residual_vis": self.config.get("save_depth_residual_vis", True),
                })
                result = step.run()
                self._record_result("tracking", result)
                if result.success:
                    self.state["tracking_reports"] = result.data.get("reports", [])
                    self.state["recon_reports"] = result.data.get("reports", [])
            else:
                # Standard single-frame reconstruction (Week 4)
                step = ReconstructionStep(self.logger, {
                    "binary": self.config.get("recon_binary", Path("build/bin/face_reconstruction")),
                    "model_dir": self.config.get("model_dir", DEFAULT_MODEL_DIR),
                    "conversion_reports": self.state.get("conversion_reports", []),
                    "meshes_root": self.config["output_root"] / "meshes",
                    "analysis_root": self.config.get("analysis_root", self.config["output_root"] / "analysis"),
                    "landmarks_root": self.config["output_root"] / "landmarks",
                    "run_frames": self.config.get("run_frames", 5),
                    "timeout": self.config.get("timeout", 60),
                    "save_pointclouds": self.config.get("save_pointclouds", False),
                    # Week 4: Optimization settings
                    "optimize": self.config.get("optimize", False),
                    "landmark_mapping": self.config.get("landmark_mapping", "data/landmark_mapping.txt"),
                    "verbose": self.config.get("verbose_optimize", False),
                    "max_iterations": self.config.get("max_iterations", 50),
                    "lambda_landmark": self.config.get("lambda_landmark", 1.0),
                    "lambda_depth": self.config.get("lambda_depth", 0.1),
                    "lambda_reg": self.config.get("lambda_reg", 1.0),
                })
                result = step.run()
                self._record_result("reconstruction", result)
                if result.success:
                    self.state["recon_reports"] = result.data.get("reports", [])
        else:
            self.logger.step_skip("3D Reconstruction", "configured to skip")
        
        # Step 9: Analysis (C++ binary)
        if "recon_reports" in self.state:
            step = AnalysisStep(self.logger, {
                "recon_reports": self.state.get("recon_reports", []),
                "conversion_reports": self.state.get("conversion_reports", []),
                "analysis_root": self.config.get("analysis_root", self.config["output_root"] / "analysis"),
                "save_pointclouds": self.config.get("save_pointclouds", False),
                "save_depth_vis": self.config.get("save_depth_vis", False),
                "save_metrics": self.config.get("save_metrics", False),
                "measure_runtime": self.config.get("measure_runtime", False),
                "recon_binary": self.config.get("recon_binary", Path("build/bin/face_reconstruction")),
                "analysis_binary": self.config.get("analysis_binary", Path("build/bin/analysis")),
                "model_dir": self.config.get("model_dir", Path("data/model_biwi")),
                "timeout": self.config.get("timeout", 60),
            })
            result = step.run()
            self._record_result("analysis", result)
        
        # Save summary
        summary = self._build_summary(start_time, success=True)
        self._save_summary(summary)
        
        return summary
    
    def _record_result(self, step_name: str, result: StepResult):
        """Record a step result."""
        self.results.append({
            "step": step_name,
            "status": result.status.value,
            "message": result.message,
            "data": result.data,
        })
    
    def _build_summary(self, start_time: float, success: bool) -> Dict[str, Any]:
        """Build execution summary."""
        elapsed = time.time() - start_time
        return {
            "success": success,
            "started_at": start_time,
            "finished_at": time.time(),
            "elapsed_seconds": elapsed,
            "steps": self.results,
            "log_file": str(self.logger.log_file) if self.logger.log_file else None,
        }
    
    @staticmethod
    def _should_ignore_sequence(seq_name: str) -> bool:
        """Filter out temporary helper folders that should never be processed."""
        return seq_name.startswith("temp_")
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save summary to file."""
        summary_path = self.config["output_root"] / "logs" / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Pipeline summary saved to {summary_path}")

    def _persist_conversion_reports(self, reports: List[Dict[str, Any]]):
        """Persist conversion reports so downstream steps can be restarted without rerun."""
        if not reports:
            return
        path = self.config["output_root"] / "logs" / "conversion_reports.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(reports, f, indent=2, default=str)
            self.logger.info(f"Saved conversion reports to {path}")
        except Exception as exc:
            self.logger.warning(f"Failed to save conversion reports: {exc}")

    def _load_conversion_reports(self):
        """Load conversion reports from persisted file when skipping conversion step."""
        path = self.config["output_root"] / "logs" / "conversion_reports.json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    reports = json.load(f)
                self.state["conversion_reports"] = reports
                self.logger.info(f"Loaded {len(reports)} conversion reports from {path}")
            except Exception as exc:
                self.logger.warning(f"Failed to load conversion reports: {exc}")
        else:
            self.logger.warning(f"No conversion reports found at {path} - downstream steps may be skipped")


# ============================================================================
# CLI Entry Point
# ============================================================================

# Default configuration
DEFAULT_MODEL_DIR = Path("data/model_bfm")  # Week 4: Use BFM model
DEFAULT_BFM_DIR = Path("data/bfm")
DEFAULT_RECON_BIN = Path("build/bin/face_reconstruction")
DEFAULT_KAGGLE_DATASET = "kmader/biwi-kinect-head-pose-database"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="End-to-end Biwi pipeline with modular architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all steps (analysis enabled by default)
  python scripts/pipeline.py --sequence 01 --frames 5

  # Download and run full pipeline
  python scripts/pipeline.py --download --sequence 01 --frames 5

  # Disable analysis (skip point clouds, metrics, visualizations)
  python scripts/pipeline.py --no-analysis
        """
    )
    
    # Essential paths
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                      help="Root data directory (default: data)")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"),
                      help="Root output directory (default: outputs)")
    
    # Essential processing options
    parser.add_argument("--sequence", type=str, nargs="*", default=None,
                      help="(Ignored) All sequences are processed")
    parser.add_argument("--frames", type=int, default=5,
                      help="Number of frames to process per sequence (default: 5, also limits conversion)")
    parser.add_argument("--max-frames", type=int, default=None,
                      help="Max frames to convert per sequence (0 = all, default: uses --frames value)")
    
    # Step toggles
    parser.add_argument("--download", action="store_true",
                      help="Download dataset using kagglehub (default: skip, use existing data)")
    parser.add_argument("--skip-convert", action="store_true",
                      help="Skip conversion step")
    parser.add_argument("--skip-model-setup", action="store_true",
                      help="Skip automatic model setup (create model manually)")
    parser.add_argument("--skip-pose-init", action="store_true",
                      help="Skip pose initialization step")
    parser.add_argument("--skip-reconstruct", action="store_true",
                      help="Skip reconstruction step")
    parser.add_argument("--no-require-model", dest="require_model", action="store_false",
                      default=True, help="Continue pipeline even if model setup fails (default: fail if model missing)")
    parser.add_argument("--use-single-pointcloud", action="store_true",
                      help="Use only first point cloud for model creation (faster, less accurate)")
    
    # Reconstruction options (with defaults)
    parser.add_argument("--recon-binary", type=Path, default=DEFAULT_RECON_BIN,
                      help=f"Reconstruction binary path (default: {DEFAULT_RECON_BIN})")
    parser.add_argument("--pose-init-binary", type=Path, default=Path("build/bin/pose_init"),
                      help="Pose initialization binary path (default: build/bin/pose_init)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                      help=f"Model directory (default: {DEFAULT_MODEL_DIR})")
    
    # Landmark mapping - use BFM semantic landmarks (correct correspondences)
    parser.add_argument("--landmark-mapping", type=Path, default=Path("data/bfm_landmark_68.txt"),
                      help="Path to landmark-to-model mapping file (default: data/bfm_landmark_68.txt)")
    parser.add_argument("--no-auto-generate-mapping", dest="auto_generate_mapping", action="store_false",
                      default=True, help="Disable auto-generation of landmark mapping (default: auto-generation is enabled)")
    parser.add_argument("--min-mapping-count", type=int, default=30,
                      help="Minimum number of landmark mappings required (default: 30)")
    
    # Analysis (enabled by default)
    parser.add_argument("--no-analysis", action="store_true",
                      help="Disable analysis (point clouds, depth visualization, metrics, and runtime)")
    
    # Week 4: Optimization settings
    parser.add_argument("--optimize", action="store_true",
                      help="Enable Gauss-Newton optimization (default: off for mean shape)")
    parser.add_argument("--max-iterations", type=int, default=10,
                      help="Maximum optimization iterations (default: 10)")
    parser.add_argument("--timeout", type=int, default=120,
                      help="Timeout in seconds per frame for reconstruction (default: 120)")
    parser.add_argument("--lambda-landmark", type=float, default=1.0,
                      help="Landmark term weight (default: 1.0)")
    parser.add_argument("--lambda-depth", type=float, default=0.1,
                      help="Depth term weight (default: 0.1)")
    parser.add_argument("--lambda-reg", type=float, default=1.0,
                      help="Regularization weight (default: 1.0)")
    parser.add_argument("--verbose-optimize", action="store_true",
                      help="Print detailed optimization output")
    
    # BFM settings
    parser.add_argument("--bfm-dir", type=Path, default=DEFAULT_BFM_DIR,
                      help=f"BFM source directory (default: {DEFAULT_BFM_DIR})")
    
    # Week 4: Overlay generation
    parser.add_argument("--make-overlays", action="store_true",
                      help="Generate mesh-scan overlay visualizations (Week 4)")
    parser.add_argument("--target-sequences", type=str, nargs="+", default=["01", "17"],
                      help="Sequences to generate overlays for (default: 01 17)")
    
    # Week 5: Tracking mode
    parser.add_argument("--track", action="store_true",
                      help="Enable sequential tracking mode (warm-start from previous frame)")
    parser.add_argument("--temporal-smoothing", action="store_true",
                      help="Enable temporal smoothing across frames (EMA + SLERP)")
    parser.add_argument("--smooth-pose-alpha", type=float, default=0.7,
                      help="EMA alpha for pose smoothing (0=no smoothing, 1=full smoothing, default: 0.7)")
    parser.add_argument("--smooth-expr-alpha", type=float, default=0.7,
                      help="EMA alpha for expression smoothing (default: 0.7)")
    parser.add_argument("--reinit-every", type=int, default=0,
                      help="Re-run Procrustes every K frames (0=never, default: 0)")
    parser.add_argument("--drift-rmse-thresh", type=float, default=80.0,
                      help="RMSE threshold (mm) for drift detection and auto re-init (default: 80.0)")
    parser.add_argument("--save-overlays-3d", action="store_true", default=True,
                      help="Save per-frame 3D overlay PLY files (default: True)")
    parser.add_argument("--save-depth-residual-vis", action="store_true", default=True,
                      help="Save per-frame depth residual visualizations (default: True)")
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Set up logging (fixed level, rarely changed)
    log_dir = args.output_root / "logs"
    logger, log_file = setup_logging(log_dir, "INFO")
    
    logger.info("=" * 60)
    logger.info("3D Face Reconstruction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output root: {args.output_root}")
    
    # Build configuration with sensible defaults
    # If --max-frames not explicitly set, use --frames value
    max_frames = args.max_frames if args.max_frames is not None else args.frames
    
    config = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "analysis_root": args.output_root / "analysis",  # Always default
        "max_frames": max_frames,
        "sequence": None,  # Ignored; all sequences are processed
        "kaggle_dataset": DEFAULT_KAGGLE_DATASET,
        "intrinsics": None,  # Use calibration file or defaults
        "skip_download": not args.download,  # Skip by default, enable with --download
        "skip_convert": args.skip_convert,
        "skip_model_setup": args.skip_model_setup,
        "skip_pose_init": args.skip_pose_init,
        "skip_reconstruct": args.skip_reconstruct,
        "require_model": args.require_model,
        "auto_setup_model": True,  # Auto-create model if missing (default)
        "use_single_pointcloud": args.use_single_pointcloud,
        "run_frames": args.frames,  # Renamed from run_frames
        "recon_binary": args.recon_binary,
        "pose_init_binary": args.pose_init_binary,
        "model_dir": args.model_dir,
        "bfm_dir": args.bfm_dir,  # Week 4: BFM source directory
        "timeout": args.timeout,  # Configurable via --timeout
        "landmark_mapping": args.landmark_mapping,
        "auto_generate_mapping": args.auto_generate_mapping,  # Default: True (set via argparse)
        "min_mapping_count": args.min_mapping_count,
        # Analysis enabled by default (disable with --no-analysis)
        "save_pointclouds": not args.no_analysis,
        "save_depth_vis": not args.no_analysis,
        "save_metrics": not args.no_analysis,
        "measure_runtime": not args.no_analysis,
        "force_download": False,  # Rarely used, removed from CLI
        # Week 4: Optimization settings
        "optimize": args.optimize,
        "max_iterations": args.max_iterations,
        "lambda_landmark": args.lambda_landmark,
        "lambda_depth": args.lambda_depth,
        "lambda_reg": args.lambda_reg,
        "verbose_optimize": args.verbose_optimize,
        # Week 4: Overlay generation
        "make_overlays": args.make_overlays,
        "target_sequences": args.target_sequences,
        # Week 5: Tracking settings
        "track": args.track,
        "temporal_smoothing": args.temporal_smoothing,
        "smooth_pose_alpha": args.smooth_pose_alpha,
        "smooth_expr_alpha": args.smooth_expr_alpha,
        "reinit_every": args.reinit_every,
        "drift_rmse_thresh": args.drift_rmse_thresh,
        "save_overlays_3d": args.save_overlays_3d,
        "save_depth_residual_vis": args.save_depth_residual_vis,
    }
    
    # Run pipeline
    try:
        orchestrator = PipelineOrchestrator(logger, config)
        summary = orchestrator.run()
        
        if summary["success"]:
            logger.summary(f"Pipeline completed successfully in {summary['elapsed_seconds']:.1f}s")
            return 0
        else:
            logger.error("Pipeline failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as exc:
        logger.error(f"Pipeline failed with exception: {exc}")
        import traceback
        # Always print traceback to stderr for debugging
        traceback.print_exc()
        # Also log it at error level so it appears in log file
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    # Ensure repo root and pipeline/ are on sys.path
    pipeline_dir = Path(__file__).parent
    repo_root = pipeline_dir.parent
    for p in (repo_root, pipeline_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.exit(main())
