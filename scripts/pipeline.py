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
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import certifi  # type: ignore
    _CERT_PATH = certifi.where()
except Exception:
    _CERT_PATH = None

try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False

# Import pipeline steps (from pipeline/steps/)
# Note: This import happens after all infrastructure is defined above
try:
    from pipeline.steps import (
        AnalysisStep,
        ConversionStep,
        DownloadStep,
        LandmarkDetectionStep,
        MappingSetupStep,
        PoseInitStep,
        ReconstructionStep,
    )
except ImportError:
    # If running as script directly, steps may not be importable yet
    # This will be resolved when package structure is complete
    pass

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
# Download Utilities
# ============================================================================

def http_get(url: str, target: Path, ssl_ctx: ssl.SSLContext) -> bool:
    """Download with a user-agent header to reduce 403s from some mirrors."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pip-like downloader)"})
    with urllib.request.urlopen(req, context=ssl_ctx) as resp, open(target, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        chunk_size = 1024 * 512
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100.0 / total
                print(f"Download progress: {pct:.1f}% ({downloaded}/{total} bytes)")
    return True


def extract_zip_find_tgz(zip_path: Path, target_dir: Path) -> Path:
    """Extract a Kaggle zip; if it contains a .tgz, return that path."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=target_dir)
        members = zf.namelist()
    tgz_candidates = [target_dir / m for m in members if m.lower().endswith((".tgz", ".tar.gz"))]
    if tgz_candidates:
        return tgz_candidates[0]
    hpdb_candidates = [target_dir / m for m in members if m.strip("/").endswith("hpdb")]
    if hpdb_candidates:
        return hpdb_candidates[0]
    faces_candidates = [target_dir / m for m in members if m.strip("/").endswith("faces_0")]
    if faces_candidates:
        return faces_candidates[0]
    raise RuntimeError("Zip extracted but no .tgz or hpdb directory found.")


def get_ssl_context():
    """Build SSL context that works on macOS when system certs are not available."""
    return ssl.create_default_context(cafile=_CERT_PATH) if _CERT_PATH else ssl.create_default_context()


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
    if cloud.shape[0] == 0 or mesh.shape[0] == 0 or not _HAS_CKDTREE:
        return float("nan")
    if cloud.shape[0] > sample:
        idx = np.random.choice(cloud.shape[0], sample, replace=False)
        cloud = cloud[idx]
    tree = cKDTree(mesh)
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
            MappingSetupStep,
            PoseInitStep,
            ReconstructionStep,
        )
        
        start_time = time.time()
        self.logger.summary("Starting Pipeline Execution")
        
        # Step 1: Download & Extract
        if not self.config.get("skip_download", False):
            step = DownloadStep(self.logger, {
                "download_dir": self.config["data_root"] / "biwi_download",
                "dataset_urls": self.config.get("dataset_urls", []),
                "kaggle_dataset": self.config.get("kaggle_dataset"),
                "force": self.config.get("force_download", False),
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
        
        # Step 2: Conversion
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
                self.state["conversion_reports"] = result.data.get("reports", [])
            elif not result.skipped:
                return self._build_summary(start_time, success=False)
        else:
            self.logger.step_skip("Data Conversion", "configured to skip")
        
        # Step 3: Landmark Detection
        if self.config.get("landmarks") != "none" and "conversion_reports" in self.state:
            step = LandmarkDetectionStep(self.logger, {
                "method": self.config.get("landmarks", "mediapipe"),
                "conversion_reports": self.state.get("conversion_reports", []),
                "landmarks_root": self.config["output_root"] / "landmarks",
                "overlays_root": self.config["output_root"] / "overlays",
                "run_frames": self.config.get("run_frames", 5),
            })
            result = step.run()
            self._record_result("landmarks", result)
        else:
            self.logger.step_skip("Landmark Detection", "disabled or no conversion data")
        
        # Step 4: Landmark Mapping Setup (C++ binary)
        if not self.config.get("skip_reconstruct", False):
            step = MappingSetupStep(self.logger, {
                "landmark_mapping": self.config.get("landmark_mapping", "data/landmark_mapping.txt"),
                "model_dir": self.config.get("model_dir", Path("data/model_biwi")),
                "auto_generate_mapping": self.config.get("auto_generate_mapping", True),  # Default: enabled
                "min_mapping_count": self.config.get("min_mapping_count", 15),
                "validate_mapping_binary": self.config.get("validate_mapping_binary", Path("build/bin/validate_mapping")),
            })
            result = step.run()
            self._record_result("mapping", result)
            if result.success:
                self.state["mapping_file"] = result.data.get("mapping_file")
            elif not result.skipped:
                self.logger.warning("Landmark mapping setup failed, but continuing...")
        
        # Step 5: Pose Initialization (Procrustes alignment)
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
        
        # Step 6: Reconstruction
        if not self.config.get("skip_reconstruct", False) and "conversion_reports" in self.state:
            step = ReconstructionStep(self.logger, {
                "binary": self.config.get("recon_binary", Path("build/bin/face_reconstruction")),
                "model_dir": self.config.get("model_dir", Path("data/model_biwi")),
                "conversion_reports": self.state.get("conversion_reports", []),
                "meshes_root": self.config["output_root"] / "meshes",
                "analysis_root": self.config.get("analysis_root", self.config["output_root"] / "analysis"),
                "run_frames": self.config.get("run_frames", 5),
                "timeout": self.config.get("timeout", 60),
                "save_pointclouds": self.config.get("save_pointclouds", False),
            })
            result = step.run()
            self._record_result("reconstruction", result)
            if result.success:
                self.state["recon_reports"] = result.data.get("reports", [])
        else:
            self.logger.step_skip("3D Reconstruction", "configured to skip")
        
        # Step 7: Analysis (C++ binary)
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
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save summary to file."""
        summary_path = self.config["output_root"] / "logs" / "pipeline_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Pipeline summary saved to {summary_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

# Default configuration
DATASET_URL = "https://data.vision.ee.ethz.ch/cvl/gfanelli/kinect_head_pose_db.tgz"
ALT_DATASET_URLS = [
    DATASET_URL,
    "https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose/resolve/main/kinect_head_pose_db.tgz",
]
DEFAULT_MODEL_DIR = Path("data/model_biwi")
DEFAULT_RECON_BIN = Path("build/bin/face_reconstruction")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="End-to-end Biwi pipeline with modular architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all steps (analysis enabled by default)
  python scripts/pipeline.py --sequence 01 --frames 5

  # Download dataset first
  python scripts/pipeline.py --download --sequence 01

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
                      help="Process specific sequence(s) (e.g. 01 or 01 02 03). Default: all sequences")
    parser.add_argument("--frames", type=int, default=5,
                      help="Number of frames to process per sequence (default: 5)")
    parser.add_argument("--max-frames", type=int, default=25,
                      help="Max frames to convert per sequence (0 = all, default: 25)")
    
    # Step toggles
    parser.add_argument("--download", action="store_true",
                      help="Download dataset (default: skip, use existing data)")
    parser.add_argument("--skip-convert", action="store_true",
                      help="Skip conversion step")
    parser.add_argument("--skip-pose-init", action="store_true",
                      help="Skip pose initialization step")
    parser.add_argument("--skip-reconstruct", action="store_true",
                      help="Skip reconstruction step")
    
    # Reconstruction options (with defaults)
    parser.add_argument("--recon-binary", type=Path, default=DEFAULT_RECON_BIN,
                      help=f"Reconstruction binary path (default: {DEFAULT_RECON_BIN})")
    parser.add_argument("--pose-init-binary", type=Path, default=Path("build/bin/pose_init"),
                      help="Pose initialization binary path (default: build/bin/pose_init)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                      help=f"Model directory (default: {DEFAULT_MODEL_DIR})")
    
    # Landmark detection
    parser.add_argument("--landmarks", choices=["none", "mediapipe", "dlib", "face_alignment"],
                      default="mediapipe", help="Landmark detection method (default: mediapipe)")
    
    # Landmark mapping (auto-generation enabled by default)
    parser.add_argument("--landmark-mapping", type=Path, default=Path("data/landmark_mapping.txt"),
                      help="Path to landmark-to-model mapping file (default: data/landmark_mapping.txt)")
    parser.add_argument("--no-auto-generate-mapping", dest="auto_generate_mapping", action="store_false",
                      default=True, help="Disable auto-generation of landmark mapping (default: auto-generation is enabled)")
    parser.add_argument("--min-mapping-count", type=int, default=15,
                      help="Minimum number of landmark mappings required (default: 15)")
    
    # Analysis (enabled by default)
    parser.add_argument("--no-analysis", action="store_true",
                      help="Disable analysis (point clouds, depth visualization, metrics, and runtime)")
    
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
    config = {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "analysis_root": args.output_root / "analysis",  # Always default
        "max_frames": args.max_frames,
        "sequence": args.sequence if args.sequence else None,  # List of sequences or None for all
        "dataset_urls": ALT_DATASET_URLS,  # Use hardcoded defaults
        "kaggle_dataset": None,  # Rarely used, removed from CLI
        "intrinsics": None,  # Use calibration file or defaults
        "skip_download": not args.download,  # Skip by default, enable with --download
        "skip_convert": args.skip_convert,
        "skip_pose_init": args.skip_pose_init,
        "skip_reconstruct": args.skip_reconstruct,
        "run_frames": args.frames,  # Renamed from run_frames
        "recon_binary": args.recon_binary,
        "pose_init_binary": args.pose_init_binary,
        "model_dir": args.model_dir,
        "timeout": 60,  # Fixed default
        "landmarks": args.landmarks,
        "landmark_mapping": args.landmark_mapping,
        "auto_generate_mapping": args.auto_generate_mapping,  # Default: True (set via argparse)
        "min_mapping_count": args.min_mapping_count,
        # Analysis enabled by default (disable with --no-analysis)
        "save_pointclouds": not args.no_analysis,
        "save_depth_vis": not args.no_analysis,
        "save_metrics": not args.no_analysis,
        "measure_runtime": not args.no_analysis,
        "force_download": False,  # Rarely used, removed from CLI
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
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
