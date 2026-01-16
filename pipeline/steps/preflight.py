"""
Preflight checks for required Python deps, C++ binaries, and model assets.
Supports fresh start: auto-compiles binaries if build directory is missing.
Fail fast with clear messaging; warn (not fallback) on non-fatal gaps.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

from main import PipelineStep, StepResult, StepStatus


def _check_python_packages(packages: List[str]) -> List[str]:
    """Check which Python packages are missing."""
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def _check_binaries(binaries: List[Path]) -> List[Path]:
    """Check which binaries are missing."""
    return [b for b in binaries if not b.exists()]


def _get_cpu_count() -> int:
    """Get number of CPUs for parallel build."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _run_cmake_build(project_root: Path, logger) -> Tuple[bool, str]:
    """
    Run cmake and make to build the project.
    
    Returns:
        Tuple of (success, message)
    """
    build_dir = project_root / "build"
    
    # Check for cmake
    if not shutil.which("cmake"):
        return False, "cmake not found. Please install cmake."
    
    # Check for make (or ninja)
    make_cmd = "make"
    if not shutil.which("make"):
        if shutil.which("ninja"):
            make_cmd = "ninja"
        else:
            return False, "make/ninja not found. Please install build tools."
    
    try:
        # Create build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Run cmake
        logger.info("Running cmake...")
        cmake_result = subprocess.run(
            ["cmake", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if cmake_result.returncode != 0:
            return False, f"cmake failed: {cmake_result.stderr}"
        
        # Run make with parallel jobs
        cpu_count = _get_cpu_count()
        logger.info(f"Building with {make_cmd} -j{cpu_count}...")
        
        make_args = [make_cmd, f"-j{cpu_count}"]
        make_result = subprocess.run(
            make_args,
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for build
        )
        
        if make_result.returncode != 0:
            return False, f"{make_cmd} failed: {make_result.stderr}"
        
        return True, "Build completed successfully"
        
    except subprocess.TimeoutExpired:
        return False, "Build timed out"
    except Exception as e:
        return False, f"Build error: {e}"


class PreflightStep(PipelineStep):
    """
    Validate environment before running the pipeline.
    
    Supports fresh start by auto-compiling if binaries are missing.
    """

    @property
    def name(self) -> str:
        return "Preflight"

    @property
    def description(self) -> str:
        return "Verify Python deps, build binaries if needed, check model assets"

    def execute(self) -> StepResult:
        # Get project root (parent of scripts/)
        project_root = Path(__file__).parent.parent.parent
        
        # Required Python packages
        required_pkgs = self.config.get(
            "required_python_packages",
            ["numpy", "cv2", "scipy"],
        )
        
        # Required binaries (core tools only, not test binaries)
        required_bins = [
            Path(p)
            for p in self.config.get(
                "required_binaries",
                [
                    "build/bin/face_reconstruction",
                    "build/bin/pose_init",
                    "build/bin/validate_mapping",
                    "build/bin/analysis",
                ],
            )
        ]
        
        model_dir = Path(self.config.get("model_dir", "data/model_bfm"))
        auto_build = self.config.get("auto_build", True)

        # Step 1: Check Python packages
        missing_pkgs = _check_python_packages(required_pkgs)
        if missing_pkgs:
            msg = (
                "Missing required Python packages: "
                + ", ".join(missing_pkgs)
                + ". Install them with: pip install " + " ".join(missing_pkgs)
            )
            return StepResult(StepStatus.FAILED, msg, {"missing_packages": missing_pkgs})

        # Step 2: Check binaries, auto-build if missing
        missing_bins = _check_binaries(required_bins)
        
        if missing_bins and auto_build:
            self.logger.info("Some binaries missing, attempting to build...")
            success, message = _run_cmake_build(project_root, self.logger)
            
            if not success:
                return StepResult(
                    StepStatus.FAILED,
                    f"Auto-build failed: {message}",
                    {"build_error": message}
                )
            
            # Re-check binaries after build
            missing_bins = _check_binaries(required_bins)
        
        if missing_bins:
            msg = (
                "Missing required binaries: "
                + ", ".join(str(b) for b in missing_bins)
                + ". Build manually with: cd build && cmake .. && make"
            )
            return StepResult(StepStatus.FAILED, msg, {"missing_binaries": [str(b) for b in missing_bins]})

        # Step 3: Check for BFM model (warn only, BFM setup step will handle it)
        bfm_dir = Path(self.config.get("bfm_dir", "data/bfm"))
        if not model_dir.exists():
            # Check if BFM source exists
            bfm_files = list(bfm_dir.glob("*.mat")) + list(bfm_dir.glob("*.h5"))
            if not bfm_files:
                self.logger.warning(
                    f"No BFM model found. Download from https://faces.dmi.unibas.ch/bfm/ "
                    f"and place .mat or .h5 file in {bfm_dir}"
                )

        # Step 4: Warn if model exists but faces connectivity is missing
        if model_dir.exists():
            faces_bin = model_dir / "faces.bin"
            faces_txt = model_dir / "faces.txt"
            if not faces_bin.exists() and not faces_txt.exists():
                self.logger.warning(f"No faces connectivity found in {model_dir}")

        return StepResult(StepStatus.SUCCESS, "Preflight checks passed")
