"""
Modular pipeline architecture for 3D face reconstruction.

All infrastructure is consolidated in the parent pipeline.py module.
Steps are in pipeline/steps/.
"""

# Import from parent directory's pipeline.py
# Use importlib to avoid circular imports
import sys
import importlib.util
from pathlib import Path

# Get the parent directory (scripts/)
_parent_dir = Path(__file__).parent.parent

# Add parent directory to path if not already there
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Load pipeline.py as a module file (not as a package)
# This avoids circular import issues
_pipeline_file = _parent_dir / "pipeline.py"
if not _pipeline_file.exists():
    raise ImportError(f"pipeline.py not found at {_pipeline_file}")

# Always load pipeline.py as a separate module to avoid circular imports
# This ensures we get a fully initialized module even when pipeline.py is being executed
spec = importlib.util.spec_from_file_location("_pipeline_core", _pipeline_file)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to create spec for {_pipeline_file}")

# Use a stable module name to avoid conflicts
_module_name = "_pipeline_core"

# Check if already loaded
if _module_name in sys.modules:
    pipeline_module = sys.modules[_module_name]
else:
    pipeline_module = importlib.util.module_from_spec(spec)
    # Set module attributes before execution so classes have correct __module__
    pipeline_module.__name__ = _module_name
    pipeline_module.__file__ = str(_pipeline_file)
    # Register in sys.modules BEFORE execution so dataclass can find it
    sys.modules[_module_name] = pipeline_module
    # Now execute the module
    spec.loader.exec_module(pipeline_module)

# Re-export everything after module is fully loaded
PipelineStep = pipeline_module.PipelineStep
StepResult = pipeline_module.StepResult
StepStatus = pipeline_module.StepStatus
PipelineLogger = pipeline_module.PipelineLogger
setup_logging = pipeline_module.setup_logging
PipelineOrchestrator = pipeline_module.PipelineOrchestrator
read_biwi_calibration = pipeline_module.read_biwi_calibration
default_intrinsics = pipeline_module.default_intrinsics
read_biwi_depth_binary = pipeline_module.read_biwi_depth_binary
convert_depth_image = pipeline_module.convert_depth_image
copy_rgb = pipeline_module.copy_rgb
create_depth_visualization = pipeline_module.create_depth_visualization
find_rgb_depth_pairs = pipeline_module.find_rgb_depth_pairs
load_mesh_vertices = pipeline_module.load_mesh_vertices
compute_cloud_to_mesh_rmse = pipeline_module.compute_cloud_to_mesh_rmse

__all__ = [
    "PipelineStep",
    "StepResult",
    "StepStatus",
    "PipelineLogger",
    "setup_logging",
    "PipelineOrchestrator",
    "read_biwi_calibration",
    "default_intrinsics",
    "read_biwi_depth_binary",
    "convert_depth_image",
    "copy_rgb",
    "create_depth_visualization",
    "find_rgb_depth_pairs",
    "load_mesh_vertices",
    "compute_cloud_to_mesh_rmse",
]

