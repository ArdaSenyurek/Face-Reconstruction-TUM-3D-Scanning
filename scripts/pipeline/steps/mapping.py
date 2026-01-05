"""
Landmark mapping setup step: Create or validate landmark-to-model vertex mapping.
Calls C++ validate_mapping binary for all computation.
"""
import subprocess
from pathlib import Path

from pipeline import PipelineStep, StepResult, StepStatus


class MappingSetupStep(PipelineStep):
    """Setup landmark-to-model vertex mapping file."""
    
    @property
    def name(self) -> str:
        return "Landmark Mapping Setup"
    
    @property
    def description(self) -> str:
        return "Create or validate landmark-to-model vertex mapping"
    
    def execute(self) -> StepResult:
        """Check if mapping exists, optionally create it. Uses C++ validate_mapping binary."""
        mapping_path = Path(self.config.get("landmark_mapping", "data/landmark_mapping.txt"))
        model_dir = Path(self.config.get("model_dir", "data/model_biwi"))
        auto_generate = self.config.get("auto_generate_mapping", True)  # Default: enabled
        min_mappings = self.config.get("min_mapping_count", 15)
        validate_binary = Path(self.config.get("validate_mapping_binary", "build/bin/validate_mapping"))
        
        if not validate_binary.exists():
            return StepResult(StepStatus.FAILED, f"validate_mapping binary not found: {validate_binary}")
        
        # Try to validate existing mapping first
        if mapping_path.exists():
            result = self._validate_mapping(validate_binary, mapping_path, model_dir, min_mappings)
            if result.success:
                return result
        
        # Try to auto-generate if enabled
        if auto_generate:
            return self._create_default_mapping(validate_binary, mapping_path, model_dir)
        else:
            self.logger.error(f"✗ Mapping file not found: {mapping_path}")
            return StepResult(StepStatus.FAILED, "Mapping file not found and auto-generation disabled",
                           {"mapping_file": str(mapping_path)})
    
    def _validate_mapping(self, binary: Path, mapping_path: Path, model_dir: Path, min_count: int) -> StepResult:
        """Validate mapping using C++ binary."""
        cmd = [
            str(binary),
            "--mapping", str(mapping_path),
            "--model-dir", str(model_dir),
            "--min-count", str(min_count),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode == 0:
                # Parse output: "OK <count>"
                output = result.stdout.strip()
                if output.startswith("OK"):
                    count = int(output.split()[1])
                    self.logger.info(f"✓ Mapping file validated with {count} entries: {mapping_path}")
                    return StepResult(StepStatus.SUCCESS, f"Mapping validated ({count} entries)",
                                   {"mapping_file": str(mapping_path), "count": count})
            else:
                self.logger.warning(f"Mapping validation failed: {result.stderr}")
        except Exception as e:
            self.logger.debug(f"Error validating mapping: {e}")
        
        return StepResult(StepStatus.FAILED, "Mapping validation failed", {})
    
    def _create_default_mapping(self, binary: Path, mapping_path: Path, model_dir: Path) -> StepResult:
        """Create default mapping using C++ binary."""
        cmd = [
            str(binary),
            "--mapping", str(mapping_path),
            "--model-dir", str(model_dir),
            "--create-default",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode == 0:
                # Parse output: "CREATED <count>"
                output = result.stdout.strip()
                if output.startswith("CREATED"):
                    count = int(output.split()[1])
                    self.logger.info(f"✓ Created default mapping with {count} entries")
                    self.logger.warning("⚠ Using default landmark pairs - consider manual refinement for better accuracy")
                    return StepResult(StepStatus.SUCCESS, f"Created default mapping ({count} entries)",
                                   {"mapping_file": str(mapping_path), "count": count, "auto_generated": True})
        except Exception as e:
            self.logger.warning(f"Failed to create default mapping: {e}")
        
        return StepResult(StepStatus.FAILED, "Failed to create default mapping", {})

