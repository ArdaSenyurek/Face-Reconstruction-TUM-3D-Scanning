"""
Modular pipeline architecture for 3D face reconstruction.

All infrastructure is consolidated in the parent pipeline.py module.
Steps are in pipeline/steps/.
"""

# Import from parent directory's pipeline.py
# We need to import from the parent script file
import sys
import importlib.util
from pathlib import Path

# Load pipeline.py as a module
_pipeline_file = Path(__file__).parent.parent / "pipeline.py"
if _pipeline_file.exists():
    spec = importlib.util.spec_from_file_location("pipeline_module", _pipeline_file)
    pipeline_module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_module"] = pipeline_module
    spec.loader.exec_module(pipeline_module)
    
    # Re-export everything
    PipelineStep = pipeline_module.PipelineStep
    StepResult = pipeline_module.StepResult
    StepStatus = pipeline_module.StepStatus
    PipelineLogger = pipeline_module.PipelineLogger
    setup_logging = pipeline_module.setup_logging
    PipelineOrchestrator = pipeline_module.PipelineOrchestrator
    http_get = pipeline_module.http_get
    extract_zip_find_tgz = pipeline_module.extract_zip_find_tgz
    get_ssl_context = pipeline_module.get_ssl_context
    read_biwi_calibration = pipeline_module.read_biwi_calibration
    default_intrinsics = pipeline_module.default_intrinsics
    read_biwi_depth_binary = pipeline_module.read_biwi_depth_binary
    convert_depth_image = pipeline_module.convert_depth_image
    copy_rgb = pipeline_module.copy_rgb
    find_rgb_depth_pairs = pipeline_module.find_rgb_depth_pairs
    load_mesh_vertices = pipeline_module.load_mesh_vertices
    compute_cloud_to_mesh_rmse = pipeline_module.compute_cloud_to_mesh_rmse
else:
    # Fallback: define minimal stubs if file not found
    raise ImportError("pipeline.py not found in parent directory")

__all__ = [
    "PipelineStep",
    "StepResult",
    "StepStatus",
    "PipelineLogger",
    "setup_logging",
    "PipelineOrchestrator",
    "http_get",
    "extract_zip_find_tgz",
    "get_ssl_context",
    "read_biwi_calibration",
    "default_intrinsics",
    "read_biwi_depth_binary",
    "convert_depth_image",
    "copy_rgb",
    "find_rgb_depth_pairs",
    "load_mesh_vertices",
    "compute_cloud_to_mesh_rmse",
]
    PipelineStep,
    StepResult,
    StepStatus,
    PipelineLogger,
    setup_logging,
    PipelineOrchestrator,
    # Utilities
    http_get,
    extract_zip_find_tgz,
    get_ssl_context,
    read_biwi_calibration,
    default_intrinsics,
    read_biwi_depth_binary,
    convert_depth_image,
    copy_rgb,
    find_rgb_depth_pairs,
    load_mesh_vertices,
    compute_cloud_to_mesh_rmse,
)

__all__ = [
    "PipelineStep",
    "StepResult",
    "StepStatus",
    "PipelineLogger",
    "setup_logging",
    "PipelineOrchestrator",
    # Utilities
    "http_get",
    "extract_zip_find_tgz",
    "get_ssl_context",
    "read_biwi_calibration",
    "default_intrinsics",
    "read_biwi_depth_binary",
    "convert_depth_image",
    "copy_rgb",
    "find_rgb_depth_pairs",
    "load_mesh_vertices",
    "compute_cloud_to_mesh_rmse",
]

