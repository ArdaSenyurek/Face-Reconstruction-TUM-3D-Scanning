"""
Post-run sanity checks using available C++ test utilities.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from main import PipelineStep, StepResult, StepStatus


class TestsStep(PipelineStep):
    """Run lightweight C++ sanity checks and report results."""

    @property
    def name(self) -> str:
        return "Post-run Tests"

    @property
    def description(self) -> str:
        return "Run C++ test utilities (mapping validation, mean-shape export)"

    def execute(self) -> StepResult:
        model_dir = Path(self.config.get("model_dir", "data/model_biwi"))
        mapping_file = Path(self.config.get("landmark_mapping", "data/landmark_mapping.txt"))
        validate_bin = Path(self.config.get("validate_mapping_binary", "build/bin/validate_mapping"))
        mean_shape_bin = Path(self.config.get("test_mean_shape_binary", "build/bin/test_export_mean_shape"))
        analysis_root = Path(self.config.get("analysis_root", "outputs/analysis"))

        checks: List[str] = []

        # 1) Validate mapping if binary and mapping exist
        if validate_bin.exists() and mapping_file.exists():
            try:
                cmd = [
                    str(validate_bin),
                    "--mapping",
                    str(mapping_file),
                    "--model-dir",
                    str(model_dir),
                    "--min-count",
                    str(self.config.get("min_mapping_count", 15)),
                ]
                self.logger.info(f"Running mapping validation: {' '.join(cmd)}")
                res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
                if res.stdout.strip():
                    self.logger.info(res.stdout.strip())
                checks.append("validate_mapping: ok")
            except subprocess.CalledProcessError as exc:
                self.logger.warning(f"validate_mapping failed: {exc.stderr.strip()}")
                return StepResult(StepStatus.FAILED, "validate_mapping failed", {"stderr": exc.stderr})
            except Exception as exc:
                return StepResult(StepStatus.FAILED, f"validate_mapping error: {exc}")
        else:
            self.logger.info("Skipping mapping validation (binary or mapping missing)")

        # 2) Export mean shape if binary exists
        if mean_shape_bin.exists():
            try:
                out_dir = analysis_root / "tests"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_ply = out_dir / "mean_shape_test.ply"
                cmd = [str(mean_shape_bin), str(model_dir), str(out_ply)]
                self.logger.info(f"Running mean-shape export: {' '.join(cmd)}")
                res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
                if res.stdout.strip():
                    self.logger.info(res.stdout.strip())
                if out_ply.exists():
                    checks.append(f"mean_shape_export: ok ({out_ply})")
                else:
                    return StepResult(StepStatus.FAILED, "mean_shape_export did not produce output")
            except subprocess.CalledProcessError as exc:
                self.logger.warning(f"mean_shape_export failed: {exc.stderr.strip()}")
                return StepResult(StepStatus.FAILED, "mean_shape_export failed", {"stderr": exc.stderr})
            except Exception as exc:
                return StepResult(StepStatus.FAILED, f"mean_shape_export error: {exc}")
        else:
            self.logger.info("Skipping mean-shape export (binary missing)")

        if not checks:
            return StepResult(StepStatus.SKIPPED, "No tests executed (binaries/mapping missing)")

        return StepResult(StepStatus.SUCCESS, "; ".join(checks), {"checks": checks})



