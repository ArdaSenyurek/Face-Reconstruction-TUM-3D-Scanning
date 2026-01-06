"""
Pose initialization step using C++ Procrustes alignment.
"""
import subprocess
from pathlib import Path
from typing import Optional

from pipeline import PipelineStep, StepResult, StepStatus


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
        
        model_dir = Path(self.config.get("model_dir", "data/model_biwi"))
        mapping_path = Path(self.config.get("landmark_mapping", "data/landmark_mapping.txt"))
        conversion_reports = self.config.get("conversion_reports", [])
        landmarks_root = Path(self.config.get("landmarks_root", "outputs/landmarks"))
        pose_init_root = Path(self.config.get("pose_init_root", "outputs/pose_init"))
        run_frames = self.config.get("run_frames", 5)
        timeout = self.config.get("timeout", 60)
        
        if not mapping_path.exists():
            return StepResult(StepStatus.FAILED, f"Landmark mapping file not found: {mapping_path}")
        
        pose_reports = []
        for seq_report in conversion_reports:
            seq_dir = Path(seq_report["output_dir"])
            seq_id = seq_dir.name
            intrinsics_path = seq_dir / "intrinsics.txt"
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            landmarks_dir = landmarks_root / seq_id
            
            if not intrinsics_path.exists():
                self.logger.warning(f"Missing intrinsics for {seq_id}, skipping")
                continue
            
            frames = sorted(rgb_dir.glob("frame_*.png"))[:run_frames]
            
            for frame in frames:
                depth_frame = depth_dir / frame.name
                landmark_file = landmarks_dir / f"{frame.stem}.txt"
                
                if not (depth_frame.exists() and landmark_file.exists()):
                    continue
                
                output_mesh = pose_init_root / seq_id / f"{frame.stem}_aligned.ply"
                
                success = self._run_pose_init(
                    binary, model_dir, mapping_path, intrinsics_path,
                    frame, depth_frame, landmark_file, output_mesh, timeout
                )
                
                pose_reports.append({
                    "sequence": seq_id,
                    "frame": frame.name,
                    "aligned_mesh": str(output_mesh),
                    "success": success
                })
                
                if success:
                    self.logger.info(f"✓ {seq_id}/{frame.name}")
                else:
                    self.logger.warning(f"✗ {seq_id}/{frame.name} failed")
        
        successful = sum(1 for r in pose_reports if r["success"])
        return StepResult(StepStatus.SUCCESS, f"Initialized pose for {successful}/{len(pose_reports)} frames",
                         {"reports": pose_reports})
    
    def _run_pose_init(self, binary: Path, model_dir: Path, mapping_path: Path,
                      intrinsics: Path, rgb: Path, depth: Path, landmarks: Path,
                      output_mesh: Path, timeout: int) -> bool:
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
        ]
        
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=timeout
            )
            return output_mesh.exists()
        except subprocess.CalledProcessError as exc:
            self.logger.debug(f"Pose init failed: {exc.stderr}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Pose init timed out for {rgb.name}")
            return False

