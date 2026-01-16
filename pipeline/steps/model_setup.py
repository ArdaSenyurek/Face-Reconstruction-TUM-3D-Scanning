"""
Model setup step: Setup morphable model from BFM or validate existing model.

This step has been simplified to rely on BFM (Basel Face Model) instead of
creating a model from averaged point clouds, which produced broken results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from main import PipelineStep, StepResult, StepStatus

# Import BFM converter
_UTILS_PATH = Path(__file__).parent.parent / "utils"
if _UTILS_PATH.exists():
    sys.path.insert(0, str(_UTILS_PATH))

try:
    from convert_bfm_to_project import convert_bfm_to_project, find_bfm_file
    _HAS_CONVERTER = True
except ImportError:
    _HAS_CONVERTER = False


def is_valid_model(model_dir: Path) -> bool:
    """Check if model directory contains required files with non-zero size."""
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    
    required_files = [
        "mean_shape.bin",
        "identity_basis.bin",
        "expression_basis.bin",
        "identity_stddev.bin",
        "expression_stddev.bin",
    ]
    
    for filename in required_files:
        filepath = model_dir / filename
        if not filepath.exists():
            return False
        # mean_shape must have content
        if filename == "mean_shape.bin" and filepath.stat().st_size == 0:
            return False
    
    return True


def find_bfm_source(bfm_dir: Path) -> Optional[Path]:
    """Find BFM source file in directory."""
    if not bfm_dir.exists():
        return None
    
    patterns = ["*.mat", "*.h5", "*.hdf5", "*.npy"]
    for pattern in patterns:
        files = list(bfm_dir.glob(pattern))
        if files:
            return files[0]
    
    return None


class ModelSetupStep(PipelineStep):
    """
    Setup morphable model for face reconstruction.
    
    This step checks if a valid BFM-based model exists, and if not,
    attempts to convert from BFM source files.
    
    Note: The old approach of creating models from averaged point clouds
    has been removed as it produced broken results (non-corresponding
    vertices were averaged, creating a "smashed" mean shape).
    """
    
    @property
    def name(self) -> str:
        return "Model Setup"
    
    @property
    def description(self) -> str:
        return "Validate or create morphable model from BFM"
    
    def execute(self) -> StepResult:
        model_dir = Path(self.config.get("model_dir", "data/model_bfm"))
        bfm_dir = Path(self.config.get("bfm_dir", "data/bfm"))
        
        # Step 1: Check if valid model already exists
        if is_valid_model(model_dir):
            self.logger.info(f"Model already exists at {model_dir}")
            
            # Ensure faces file exists
            faces_bin = model_dir / "faces.bin"
            faces_txt = model_dir / "faces.txt"
            if not faces_bin.exists() and not faces_txt.exists():
                self.logger.warning(f"No faces connectivity in {model_dir}")
            
            return StepResult(
                StepStatus.SUCCESS,
                f"Model ready at {model_dir}",
                {"model_dir": str(model_dir), "existing": True}
            )
        
        # Step 2: Try to create from BFM
        bfm_source = find_bfm_source(bfm_dir)
        
        if bfm_source is None:
            # Provide clear instructions
            self.logger.error("No valid model found and no BFM source available!")
            self.logger.info("=" * 60)
            self.logger.info("This project requires the Basel Face Model (BFM).")
            self.logger.info("")
            self.logger.info("To set up the model:")
            self.logger.info("  1. Register at: https://faces.dmi.unibas.ch/bfm/")
            self.logger.info("  2. Download BFM 2009 (01_MorphableModel.mat)")
            self.logger.info(f"  3. Place the .mat file in: {bfm_dir}/")
            self.logger.info("  4. Re-run the pipeline")
            self.logger.info("=" * 60)
            
            return StepResult(
                StepStatus.FAILED,
                f"BFM source not found in {bfm_dir}",
                {"model_dir": str(model_dir), "bfm_dir": str(bfm_dir)}
            )
        
        # Step 3: Convert BFM
        if not _HAS_CONVERTER:
            return StepResult(
                StepStatus.FAILED,
                "BFM converter not available",
                {}
            )
        
        self.logger.info(f"Converting BFM from {bfm_source}...")
        
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            
            success = convert_bfm_to_project(
                bfm_source,
                model_dir,
                num_identity=80,
                num_expression=64,
                center=True
            )
            
            if not success:
                return StepResult(
                    StepStatus.FAILED,
                    "BFM conversion failed",
                    {"bfm_source": str(bfm_source)}
                )
            
            if not is_valid_model(model_dir):
                return StepResult(
                    StepStatus.FAILED,
                    "Conversion produced invalid model",
                    {"model_dir": str(model_dir)}
                )
            
        except Exception as e:
            return StepResult(
                StepStatus.FAILED,
                f"Model creation error: {e}",
                {"exception": str(e)}
            )
        
        self.logger.info(f"Model created at {model_dir}")
        
        return StepResult(
            StepStatus.SUCCESS,
            f"Model created from BFM at {model_dir}",
            {"model_dir": str(model_dir), "created": True}
        )
