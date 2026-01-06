"""
Model setup step: Create morphable model from BIWI point clouds if missing.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional

from pipeline import PipelineStep, StepResult, StepStatus

# Import model creation function from utils
_UTILS_PATH = Path(__file__).parent.parent / "utils"
if _UTILS_PATH.exists():
    sys.path.insert(0, str(_UTILS_PATH))

try:
    from create_mean_shape_from_pointclouds import create_mean_shape_from_pointclouds
    _HAS_MODEL_CREATION = True
except ImportError:
    # Try legacy path as fallback
    _LEGACY_TOOLS_PATH = Path(__file__).parent.parent.parent.parent / "tools" / "legacy"
    if _LEGACY_TOOLS_PATH.exists():
        sys.path.insert(0, str(_LEGACY_TOOLS_PATH))
    try:
        from create_mean_shape_from_pointclouds import create_mean_shape_from_pointclouds
        _HAS_MODEL_CREATION = True
    except ImportError:
        _HAS_MODEL_CREATION = False


def is_valid_model(model_dir: Path) -> bool:
    """Check if model directory contains required files."""
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
        if not (model_dir / filename).exists():
            return False
    
    return True


class ModelSetupStep(PipelineStep):
    """Setup morphable model from BIWI point clouds if missing."""
    
    @property
    def name(self) -> str:
        return "Model Setup"
    
    @property
    def description(self) -> str:
        return "Create morphable model from BIWI point clouds if missing"
    
    def execute(self) -> StepResult:
        """Check if model exists, create if missing."""
        model_dir = Path(self.config.get("model_dir", "data/model_biwi"))
        biwi_root = Path(self.config.get("biwi_root"))
        auto_setup = self.config.get("auto_setup_model", True)
        use_single = self.config.get("use_single_pointcloud", False)
        
        if not biwi_root or not biwi_root.exists():
            return StepResult(
                StepStatus.FAILED,
                f"BIWI root directory not found: {biwi_root}",
                {"model_dir": str(model_dir)}
            )
        
        # Check if model already exists and is valid
        if is_valid_model(model_dir):
            self.logger.info(f"✓ Model already exists at {model_dir}")
            return StepResult(
                StepStatus.SUCCESS,
                f"Model already exists at {model_dir}",
                {"model_dir": str(model_dir), "existing": True}
            )
        
        # Model doesn't exist - check if we should create it
        if not auto_setup:
            return StepResult(
                StepStatus.SKIPPED,
                "Model setup skipped (auto_setup_model=False)",
                {"model_dir": str(model_dir)}
            )
        
        # Get PLY files from converted RGB-D data
        conversion_reports = self.config.get("conversion_reports", [])
        if not conversion_reports:
            return StepResult(
                StepStatus.FAILED,
                "No converted data available. Run conversion step first.",
                {"model_dir": str(model_dir)}
            )
        
        # Create PLY files from converted RGB-D data
        self.logger.info("Creating PLY point clouds from converted RGB-D data...")
        ply_files = self._create_ply_from_converted_data(conversion_reports)
        
        if not ply_files:
            return StepResult(
                StepStatus.FAILED,
                "Failed to create PLY point clouds from converted RGB-D data",
                {"model_dir": str(model_dir)}
            )
        
        self.logger.info(f"Created {len(ply_files)} PLY point cloud file(s)")
        
        if not ply_files:
            error_msg = (
                f"No PLY point cloud files found in {biwi_root}\n"
                "Attempted to create from RGB-D data but failed.\n"
                "The morphable model cannot be created automatically.\n"
                "Options:\n"
                "  1. Provide PLY files in the BIWI dataset\n"
                "  2. Use an existing model: --model-dir <path>\n"
                "  3. Create model manually: python scripts/utils/setup_model.py\n"
                "  4. Ensure converted RGB-D data exists (run conversion step first)"
            )
            return StepResult(
                StepStatus.FAILED,
                error_msg,
                {
                    "model_dir": str(model_dir),
                    "biwi_root": str(biwi_root),
                    "ply_files_found": 0
                }
            )
        
        self.logger.info(f"Found {len(ply_files)} PLY file(s)")
        
        # Check if model creation function is available
        if not _HAS_MODEL_CREATION:
            legacy_path_str = str(_LEGACY_TOOLS_PATH) if '_LEGACY_TOOLS_PATH' in globals() and _LEGACY_TOOLS_PATH.exists() else "tools/legacy"
            error_msg = (
                f"Model creation function not available.\n"
                f"Could not import create_mean_shape_from_pointclouds from {_UTILS_PATH} or {legacy_path_str}\n"
                f"Make sure scripts/pipeline/utils/create_mean_shape_from_pointclouds.py exists."
            )
            return StepResult(
                StepStatus.FAILED,
                error_msg,
                {
                    "model_dir": str(model_dir),
                    "legacy_path": str(_LEGACY_TOOLS_PATH)
                }
            )
        
        # Create model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to string paths (same as setup_model.py)
        pointcloud_files = [str(p) for p in ply_files]
        
        # Note: create_mean_shape_from_pointclouds handles single filtering internally
        # Pass the full list and let the function handle it
        if use_single:
            self.logger.info("Using only first point cloud (--use-single-pointcloud)")
        
        self.logger.info(f"Creating model from {len(pointcloud_files)} point cloud(s)...")
        
        try:
            # Call the function (exact same signature as setup_model.py)
            # The function handles single filtering internally when use_single=True
            # It also handles resampling and mean computation
            self.logger.debug(f"Calling create_mean_shape_from_pointclouds with {len(pointcloud_files)} files")
            success = create_mean_shape_from_pointclouds(
                pointcloud_files,
                str(model_dir),
                use_single
            )
            
            # Verify model was created successfully
            if not success:
                return StepResult(
                    StepStatus.FAILED,
                    "Model creation function returned False",
                    {"model_dir": str(model_dir)}
                )
            
            # Double-check that all required files exist
            if not is_valid_model(model_dir):
                missing_files = []
                required_files = [
                    "mean_shape.bin",
                    "identity_basis.bin",
                    "expression_basis.bin",
                    "identity_stddev.bin",
                    "expression_stddev.bin",
                ]
                for filename in required_files:
                    if not (model_dir / filename).exists():
                        missing_files.append(filename)
                
                return StepResult(
                    StepStatus.FAILED,
                    f"Model creation incomplete - missing files: {', '.join(missing_files)}",
                    {"model_dir": str(model_dir), "missing_files": missing_files}
                )
            
            # Success!
            actual_count = len(pointcloud_files) if not use_single else 1
            self.logger.info(f"✓ Model created successfully at {model_dir}")
            return StepResult(
                StepStatus.SUCCESS,
                f"Model created from {actual_count} point cloud(s)",
                {
                    "model_dir": str(model_dir),
                    "pointclouds_used": actual_count,
                    "pointclouds_available": len(pointcloud_files),
                    "created": True
                }
            )
                
        except ImportError as e:
            error_msg = (
                f"Failed to import model creation function: {e}\n"
                f"Make sure tools/legacy/create_mean_shape_from_pointclouds.py exists and is importable."
            )
            self.logger.error(error_msg)
            return StepResult(
                StepStatus.FAILED,
                error_msg,
                {"model_dir": str(model_dir), "exception": str(e), "legacy_path": str(_LEGACY_TOOLS_PATH)}
            )
        except Exception as e:
            error_msg = f"Error creating model: {e}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return StepResult(
                StepStatus.FAILED,
                error_msg,
                {"model_dir": str(model_dir), "exception": str(e)}
            )
    
    def _create_ply_from_converted_data(self, conversion_reports: List[dict]) -> List[Path]:
        """Create PLY files from converted RGB-D data."""
        created_files = []
        
        # Import utility function
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
            from create_pointcloud_from_rgbd import create_pointcloud_from_rgbd
        except ImportError:
            self.logger.warning("Could not import pointcloud creation utility")
            return created_files
        
        binary = Path(self.config.get("recon_binary", "build/bin/face_reconstruction"))
        if not binary.exists():
            self.logger.warning(f"Reconstruction binary not found: {binary}")
            return created_files
        
        # Create temporary directory for PLY files
        temp_ply_dir = Path(self.config.get("biwi_root")).parent / "temp_pointclouds"
        temp_ply_dir.mkdir(parents=True, exist_ok=True)
        
        # Process a few frames from each sequence
        max_frames_per_sequence = 3  # Use a few frames to create model
        
        for report in conversion_reports[:5]:  # Limit to first 5 sequences
            seq_dir = Path(report["output_dir"])
            seq_name = seq_dir.name
            
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            intrinsics_path = seq_dir / "intrinsics.txt"
            
            if not (rgb_dir.exists() and depth_dir.exists() and intrinsics_path.exists()):
                continue
            
            # Get a few frames
            rgb_frames = sorted(rgb_dir.glob("frame_*.png"))[:max_frames_per_sequence]
            
            for rgb_frame in rgb_frames:
                depth_frame = depth_dir / rgb_frame.name
                if not depth_frame.exists():
                    continue
                
                # Create output PLY
                output_ply = temp_ply_dir / seq_name / f"{rgb_frame.stem}.ply"
                
                try:
                    success = create_pointcloud_from_rgbd(
                        rgb_frame,
                        depth_frame,
                        intrinsics_path,
                        output_ply,
                        binary,
                        depth_scale=1000.0,
                        timeout=30
                    )
                    
                    if success and output_ply.exists():
                        created_files.append(output_ply)
                        self.logger.debug(f"Created PLY: {output_ply}")
                except Exception as e:
                    self.logger.debug(f"Failed to create PLY from {rgb_frame}: {e}")
        
        return created_files

