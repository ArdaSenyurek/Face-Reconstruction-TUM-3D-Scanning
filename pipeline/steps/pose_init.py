"""
Pose initialization step using C++ Procrustes alignment.
"""
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

from main import PipelineStep, StepResult, StepStatus, select_frames


class PoseInitStep(PipelineStep):
    """Initialize model pose using Procrustes alignment with landmarks and depth."""
    
    @property
    def name(self) -> str:
        return "Pose Initialization"
    
    @property
    def description(self) -> str:
        return "Initialize model pose using Procrustes alignment"
    
    def execute(self) -> StepResult:
        """Run pose initialization for all frames."""
        binary = Path(self.config.get("pose_init_binary", "build/bin/pose_init")).resolve()
        if not binary.exists():
            return StepResult(StepStatus.FAILED, f"Pose init binary not found: {binary}")
        self.logger.info(f"Using pose_init binary: {binary}")
        
        model_dir = Path(self.config.get("model_dir", "data/model_biwi"))
        mapping_path = Path(self.config.get("landmark_mapping", "data/bfm_landmark_68.txt"))
        conversion_reports = self.config.get("conversion_reports", [])
        landmarks_root = Path(self.config.get("landmarks_root", "outputs/landmarks"))
        pose_init_root = Path(self.config.get("pose_init_root", "outputs/pose_init"))
        run_frames = self.config.get("run_frames", 4)
        frame_step = self.config.get("frame_step", 20)
        frame_indices = self.config.get("frame_indices")
        timeout = self.config.get("timeout", 60)
        
        if not mapping_path.exists():
            return StepResult(StepStatus.FAILED, f"Landmark mapping file not found: {mapping_path}")
        
        pose_reports = []
        for seq_report in conversion_reports:
            output_dir = seq_report.get("output_dir")
            if not output_dir:
                continue  # skip failed conversions with no output dir
            seq_dir = Path(output_dir)
            seq_id = seq_dir.name
            intrinsics_path = seq_dir / "intrinsics.txt"
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            landmarks_dir = landmarks_root / seq_id
            
            if not intrinsics_path.exists():
                self.logger.warning(f"Missing intrinsics for {seq_id}, skipping")
                continue
            
            frames = select_frames(rgb_dir, frame_step, run_frames, frame_indices)
            
            for frame in frames:
                depth_frame = depth_dir / frame.name
                landmark_file = landmarks_dir / f"{frame.stem}.txt"
                
                if not (depth_frame.exists() and landmark_file.exists()):
                    continue
                
                output_mesh = pose_init_root / seq_id / f"{frame.stem}_aligned.ply"
                report_json = pose_init_root / seq_id / f"{frame.stem}_rigid_report.json"
                
                success = self._run_pose_init(
                    binary, model_dir, mapping_path, intrinsics_path,
                    frame, depth_frame, landmark_file, output_mesh, report_json, timeout
                )
                
                # Collect visualization file paths (overlay is overwritten each run)
                viz_files = {}
                if success and report_json.exists():
                    # Check for visualization files
                    base_path = output_mesh.parent / output_mesh.stem
                    pre_alignment_ply = base_path.parent / f"{base_path.name}_pre_alignment.ply"
                    post_procrustes_ply = base_path.parent / f"{base_path.name}_post_procrustes.ply"
                    landmarks_overlay_png = base_path.parent / f"{base_path.name}_landmarks_overlay.png"
                    
                    if pre_alignment_ply.exists():
                        viz_files["pre_alignment_ply"] = str(pre_alignment_ply)
                    if post_procrustes_ply.exists():
                        viz_files["post_procrustes_ply"] = str(post_procrustes_ply)
                    if landmarks_overlay_png.exists():
                        viz_files["landmarks_overlay_png"] = str(landmarks_overlay_png)
                
                pose_reports.append({
                    "sequence": seq_id,
                    "frame": frame.name,
                    "aligned_mesh": str(output_mesh),
                    "rigid_report": str(report_json) if report_json.exists() else None,
                    "visualization_files": viz_files,
                    "success": success
                })
                
                if success:
                    self.logger.info(f"✓ {seq_id}/{frame.name}")
                else:
                    self.logger.warning(f"✗ {seq_id}/{frame.name} failed")
        
        # Collect and log summary statistics
        successful = sum(1 for r in pose_reports if r["success"])
        summary_stats = self._collect_summary_statistics(pose_reports)
        
        if summary_stats:
            self.logger.info("\n=== Pose Initialization Summary ===")
            if "mapping_quality" in summary_stats:
                mq = summary_stats["mapping_quality"]
                self.logger.info(f"Average mapping coverage: {mq.get('avg_coverage', 0):.1f}%")
                self.logger.info(f"Average pre-alignment RMSE: {mq.get('avg_pre_rmse', 0):.2f} mm")
            
            if "procrustes_analysis" in summary_stats:
                pa = summary_stats["procrustes_analysis"]
                self.logger.info(f"Average post-Procrustes RMSE: {pa.get('avg_post_procrustes_rmse', 0):.2f} mm")
                self.logger.info(f"Average improvement: {pa.get('avg_improvement', 0):.1f}%")
            self.logger.info("=" * 40)
        
        return StepResult(StepStatus.SUCCESS, f"Initialized pose for {successful}/{len(pose_reports)} frames",
                         {"reports": pose_reports, "summary_statistics": summary_stats})
    
    def _run_pose_init(self, binary: Path, model_dir: Path, mapping_path: Path,
                      intrinsics: Path, rgb: Path, depth: Path, landmarks: Path,
                      output_mesh: Path, report_json: Path, timeout: Optional[int]) -> bool:
        """Run the pose initialization binary."""
        cmd = [
            str(binary),
            "--rgb", str(rgb),
            "--depth", str(depth),
            "--intrinsics", str(intrinsics),
            "--landmarks", str(landmarks),
            "--model-dir", str(model_dir),
            "--mapping", str(mapping_path),
            "--output", str(output_mesh),
            "--report", str(report_json),
        ]
        
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
        
        # No time limit when timeout is 0 or None (same as reconstruction step)
        run_timeout = None if (timeout is None or timeout <= 0) else timeout
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=run_timeout
            )
            return output_mesh.exists()
        except subprocess.CalledProcessError as exc:
            self.logger.debug(f"Pose init failed: {exc.stderr}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Pose init timed out for {rgb.name}")
            return False
    
    def _collect_summary_statistics(self, pose_reports: List[Dict]) -> Dict:
        """Collect summary statistics from JSON reports."""
        stats = {
            "mapping_quality": defaultdict(list),
            "procrustes_analysis": defaultdict(list),
        }
        
        for report in pose_reports:
            if not report.get("success", False):
                continue
            
            report_json_path = report.get("rigid_report")
            if not report_json_path or not Path(report_json_path).exists():
                continue
            
            try:
                with open(report_json_path, 'r') as f:
                    data = json.load(f)
                
                # Collect mapping quality metrics
                if "mapping_quality" in data:
                    mq = data["mapping_quality"]
                    stats["mapping_quality"]["coverage"].append(mq.get("coverage_percent", 0))
                    stats["mapping_quality"]["pre_rmse"].append(mq.get("pre_alignment_rmse_mm", 0))
                
                # Collect Procrustes analysis metrics
                if "procrustes_analysis" in data:
                    pa = data["procrustes_analysis"]
                    stats["procrustes_analysis"]["post_procrustes_rmse"].append(pa.get("post_procrustes_rmse_mm", 0))
                    stats["procrustes_analysis"]["improvement"].append(pa.get("improvement_percent", 0))
            
            except Exception as e:
                self.logger.debug(f"Failed to load report {report_json_path}: {e}")
                continue
        
        # Compute averages
        summary = {}
        for category, metrics in stats.items():
            summary[category] = {}
            for metric_name, values in metrics.items():
                if values:
                    if "coverage" in metric_name:
                        summary[category][f"avg_{metric_name}"] = sum(values) / len(values)
                    elif "rmse" in metric_name:
                        summary[category][f"avg_{metric_name}"] = sum(values) / len(values)
                    elif "improvement" in metric_name:
                        summary[category][f"avg_{metric_name}"] = sum(values) / len(values)
        
        return summary

