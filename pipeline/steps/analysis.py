"""
Analysis step: Compute metrics, visualizations, and statistics.
Calls C++ analysis binary for all 3D computation.

Metrics units:
- depth_min, depth_max, depth_mean, depth_std: meters (from C++ analysis binary).
- rmse_cloud_mesh_m: meters.
- runtime_seconds: seconds.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from main import PipelineStep, StepResult, StepStatus


class AnalysisStep(PipelineStep):
    """Analyze reconstruction results and generate metrics."""
    
    @property
    def name(self) -> str:
        return "Analysis"
    
    @property
    def description(self) -> str:
        return "Compute metrics and generate visualizations"
    
    def execute(self) -> StepResult:
        """Run analysis on reconstructed frames."""
        recon_reports = self.config.get("recon_reports", [])
        conversion_reports = self.config.get("conversion_reports", [])
        analysis_root = Path(self.config["analysis_root"])
        save_pointclouds = self.config.get("save_pointclouds", False)
        save_depth_vis = self.config.get("save_depth_vis", False)
        save_metrics = self.config.get("save_metrics", False)
        measure_runtime = self.config.get("measure_runtime", False)
        
        if not (save_pointclouds or save_depth_vis or save_metrics or measure_runtime):
            return StepResult(StepStatus.SKIPPED, "No analysis options enabled")
        
        # Build conversion report lookup
        conv_lookup = {Path(r["output_dir"]).name: r for r in conversion_reports}
        
        metrics = {}
        for recon_report in recon_reports:
            if not recon_report.get("success", False):
                continue
            
            seq_name = recon_report["sequence"]
            frame_name = recon_report["frame"]
            frame_stem = Path(frame_name).stem
            
            try:
                seq_dir = Path(conv_lookup[seq_name]["output_dir"])
                intrinsics_path = seq_dir / "intrinsics.txt"
                depth_frame = seq_dir / "depth" / f"{frame_stem}.png"
                mesh_path = Path(recon_report["mesh"])
                
                entry = {}
                
                pc_out = analysis_root / "pointclouds" / seq_name / f"{frame_stem}.ply"
                if save_pointclouds and pc_out.exists():
                    entry["cloud_points"] = float(self._count_ply_points(pc_out))
                
                # Build inputs for a single C++ analysis call per frame (avoids 2 subprocesses per frame)
                need_depth = save_depth_vis or save_metrics or measure_runtime
                need_rmse = save_metrics and mesh_path.exists()
                if need_rmse and not pc_out.exists() and depth_frame.exists() and intrinsics_path.exists():
                    try:
                        from pipeline.utils.create_pointcloud_from_rgbd import create_pointcloud_from_rgbd
                        pc_out.parent.mkdir(parents=True, exist_ok=True)
                        create_pointcloud_from_rgbd(
                            depth_frame, depth_frame, intrinsics_path, pc_out,
                            depth_scale=1000.0
                        )
                    except Exception as e:
                        self.logger.debug(f"Could not create pointcloud for {seq_name}/{frame_stem}: {e}")
                
                analysis_binary = Path(self.config.get("analysis_binary", "build/bin/analysis"))
                if analysis_binary.exists() and (need_depth or (need_rmse and pc_out.exists())):
                    vis_out = (analysis_root / "depth_vis" / seq_name / f"{frame_stem}.png") if save_depth_vis else None
                    pc_arg = pc_out if (need_rmse and pc_out.exists()) else None
                    mesh_arg = mesh_path if need_rmse else None
                    depth_arg = depth_frame if (need_depth and depth_frame.exists()) else None
                    metrics_result = self._run_analysis(
                        analysis_binary, pc_arg, mesh_arg, depth_arg, vis_out, None
                    )
                    if depth_arg and metrics_result:
                        entry.update({
                            "depth_min": metrics_result.get("depth_min", 0.0),
                            "depth_max": metrics_result.get("depth_max", 0.0),
                            "depth_mean": metrics_result.get("depth_mean", 0.0),
                            "depth_std": metrics_result.get("depth_std", 0.0),
                        })
                    if pc_arg and mesh_arg and metrics_result:
                        if "rmse_cloud_mesh_m" in metrics_result:
                            entry["rmse_cloud_mesh_m"] = metrics_result["rmse_cloud_mesh_m"]
                        if "cloud_points" in metrics_result:
                            entry["cloud_points"] = metrics_result["cloud_points"]
                
                # Runtime: use value recorded during reconstruction (no re-run, no runtime_meshes)
                if measure_runtime:
                    runtime = recon_report.get("runtime_seconds")
                    if runtime is not None and not (isinstance(runtime, float) and (runtime != runtime)):  # not NaN
                        entry["runtime_seconds"] = float(runtime)
                    else:
                        entry["runtime_seconds"] = float("nan")
                
                if entry:
                    metrics.setdefault(seq_name, {})[frame_stem] = entry
                    
            except Exception as e:
                self.logger.warning(f"Analysis failed for {seq_name} {frame_name}: {e}")
        
        # Save metrics (depth_* and rmse_cloud_mesh_m are in meters; see module docstring)
        if metrics:
            metrics_path = analysis_root / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            self.logger.info(f"Saved metrics to {metrics_path}")
        
        return StepResult(StepStatus.SUCCESS, f"Analyzed {len(metrics)} sequences",
                         {"metrics": metrics})
    
    def _count_ply_points(self, ply_path: Path) -> int:
        """Count points in PLY file (simple helper, not 3D computation)."""
        try:
            with open(ply_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("element vertex"):
                        return int(line.split()[2])
        except Exception:
            pass
        return 0
    
    def _run_analysis(self, binary: Path, pointcloud: Optional[Path], mesh: Optional[Path],
                     depth: Optional[Path], output_vis: Optional[Path], output_json: Optional[Path]) -> Dict[str, float]:
        """Run C++ analysis binary and parse metrics."""
        cmd = [str(binary)]
        
        if pointcloud:
            cmd.extend(["--pointcloud", str(pointcloud)])
        if mesh:
            cmd.extend(["--mesh", str(mesh)])
        if depth:
            cmd.extend(["--depth", str(depth)])
        if output_vis:
            cmd.extend(["--output-vis", str(output_vis)])
            output_vis.parent.mkdir(parents=True, exist_ok=True)
        if output_json:
            cmd.extend(["--output-json", str(output_json)])
            output_json.parent.mkdir(parents=True, exist_ok=True)
        
        metrics = {}
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                # Parse output: "key=value" lines
                for line in result.stdout.strip().split("\n"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            pass
        except Exception as e:
            self.logger.debug(f"Analysis binary failed: {e}")
        
        return metrics
