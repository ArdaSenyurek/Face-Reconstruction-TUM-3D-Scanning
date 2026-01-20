"""
Landmark mapping setup step: Create or validate landmark-to-model vertex mapping.
Calls C++ validate_mapping binary for all computation.
"""
import subprocess
from pathlib import Path

from main import PipelineStep, StepResult, StepStatus


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
        mapping_path = Path(self.config.get("landmark_mapping", "data/bfm_landmark_68.txt"))
        model_dir = Path(self.config.get("model_dir", "data/model_biwi"))
        auto_generate = self.config.get("auto_generate_mapping", True)  # Default: enabled
        min_mappings = self.config.get("min_mapping_count", 30)  # Default: 30 mappings
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
                # Parse output: "OK <count>" (may have warnings before it)
                output_lines = result.stdout.strip().split('\n')
                # Get the last line which should contain "OK <count>"
                last_line = output_lines[-1] if output_lines else ""
                
                if last_line.startswith("OK"):
                    count = int(last_line.split()[1])
                    self.logger.info(f"✓ Mapping file validated with {count} entries: {mapping_path}")
                    return StepResult(StepStatus.SUCCESS, f"Mapping validated ({count} entries)",
                                   {"mapping_file": str(mapping_path), "count": count})
            else:
                stderr_output = result.stderr.strip() if result.stderr else ""
                self.logger.warning(f"Mapping validation failed (return code {result.returncode})")
                if stderr_output:
                    self.logger.warning(f"  stderr: {stderr_output}")
        except Exception as e:
            self.logger.warning(f"Error validating mapping: {e}")
        
        return StepResult(StepStatus.FAILED, "Mapping validation failed", {})
    
    def _create_default_mapping(self, binary: Path, mapping_path: Path, model_dir: Path) -> StepResult:
        """Create default mapping using C++ binary."""
        # If mapping file exists, remove it first so the binary will create a new one
        # Otherwise, the binary will validate the existing file instead of creating
        file_existed = mapping_path.exists()
        if file_existed:
            self.logger.info(f"Removing existing mapping file to create new default mapping")
            try:
                mapping_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove existing mapping file: {e}")
                # Continue anyway - the binary might still work
        
        cmd = [
            str(binary),
            "--mapping", str(mapping_path),
            "--model-dir", str(model_dir),
            "--create-default",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode == 0:
                # Parse output: "CREATED <count>" or "OK <count>" (if file was re-validated)
                output_lines = result.stdout.strip().split('\n')
                # Get the last line which should contain CREATED or OK
                last_line = output_lines[-1] if output_lines else ""
                
                if last_line.startswith("CREATED"):
                    count = int(last_line.split()[1])
                    self.logger.info(f"✓ Created default mapping with {count} entries")
                    return StepResult(StepStatus.SUCCESS, f"Created default mapping ({count} entries)",
                                   {"mapping_file": str(mapping_path), "count": count, "auto_generated": True})
                elif last_line.startswith("OK"):
                    # File was validated instead of created (shouldn't happen if we removed it)
                    count = int(last_line.split()[1])
                    self.logger.info(f"✓ Mapping file exists and is valid with {count} entries")
                    return StepResult(StepStatus.SUCCESS, f"Mapping validated ({count} entries)",
                                   {"mapping_file": str(mapping_path), "count": count})
            else:
                # Log stderr for debugging
                stderr_output = result.stderr.strip() if result.stderr else ""
                if stderr_output:
                    self.logger.warning(f"validate_mapping stderr: {stderr_output}")
        except Exception as e:
            self.logger.warning(f"Failed to create default mapping: {e}")
        
        return StepResult(StepStatus.FAILED, "Failed to create default mapping", {})

