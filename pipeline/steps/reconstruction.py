"""
3D face reconstruction step using C++ binary.

Week 4: Updated to support optimization with landmarks and mapping.
"""

import subprocess
from pathlib import Path
from typing import Optional

from main import PipelineStep, StepResult, StepStatus


class ReconstructionStep(PipelineStep):
    """Run 3D face reconstruction using C++ binary."""
    
    @property
    def name(self) -> str:
        return "3D Reconstruction"
    
    @property
    def description(self) -> str:
        return "Reconstruct 3D face meshes from RGB-D data"
    
    def execute(self) -> StepResult:
        """Run reconstruction for all frames."""
        binary = Path(self.config["binary"]).resolve()
        if not binary.exists():
            return StepResult(StepStatus.FAILED, f"Binary not found: {binary}")
        
        model_dir = Path(self.config["model_dir"])
        conversion_reports = self.config.get("conversion_reports", [])
        meshes_root = Path(self.config["meshes_root"])
        analysis_root = Path(self.config.get("analysis_root", meshes_root.parent / "analysis"))
        landmarks_root = Path(self.config.get("landmarks_root", meshes_root.parent / "landmarks"))
        run_frames = self.config.get("run_frames", 5)
        timeout = self.config.get("timeout", 60)
        save_pointclouds = self.config.get("save_pointclouds", False)
        
        # Week 4: Optimization settings
        optimize = self.config.get("optimize", False)
        pose_only = self.config.get("pose_only", False)
        landmark_mapping = Path(self.config.get("landmark_mapping", "data/landmark_mapping.txt"))
        verbose = self.config.get("verbose", False)
        max_iterations = self.config.get("max_iterations", 50)
        lambda_landmark = self.config.get("lambda_landmark", 1.0)
        lambda_depth = self.config.get("lambda_depth", 0.1)
        lambda_reg = self.config.get("lambda_reg", 1.0)
        
        recon_reports = []
        for seq_report in conversion_reports:
            seq_dir = Path(seq_report["output_dir"])
            intrinsics_path = seq_dir / "intrinsics.txt"
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            
            frames = sorted(rgb_dir.glob("frame_*.png"))[:run_frames]
            
            for frame in frames:
                depth_frame = depth_dir / frame.name
                mesh_out = meshes_root / seq_dir.name / f"{frame.stem}.ply"
                
                # Landmark file for this frame (saved as frame_00000.txt, not frame_00000_landmarks.txt)
                landmarks_file = landmarks_root / seq_dir.name / f"{frame.stem}.txt"
                
                pc_out = None
                if save_pointclouds:
                    pc_out = analysis_root / "pointclouds" / seq_dir.name / f"{frame.stem}.ply"
                
                success = self._run_reconstruction(
                    binary=binary,
                    model_dir=model_dir,
                    intrinsics=intrinsics_path,
                    rgb=frame,
                    depth=depth_frame,
                    output_mesh=mesh_out,
                    timeout=timeout,
                    output_pointcloud=pc_out,
                    landmarks=landmarks_file if landmarks_file.exists() else None,
                    mapping=landmark_mapping if landmark_mapping.exists() else None,
                    optimize=optimize,
                    pose_only=pose_only,
                    verbose=verbose,
                    max_iterations=max_iterations,
                    lambda_landmark=lambda_landmark,
                    lambda_depth=lambda_depth,
                    lambda_reg=lambda_reg,
                )
                
                recon_reports.append({
                    "sequence": seq_dir.name,
                    "frame": frame.name,
                    "mesh": str(mesh_out),
                    "success": success,
                    "optimized": optimize and landmarks_file.exists(),
                })
                
                if success:
                    opt_str = " (optimized)" if optimize and landmarks_file.exists() else ""
                    self.logger.info(f"✓ {seq_dir.name}/{frame.name}{opt_str}")
                else:
                    self.logger.warning(f"✗ {seq_dir.name}/{frame.name} failed")
        
        successful = sum(1 for r in recon_reports if r["success"])
        optimized = sum(1 for r in recon_reports if r.get("optimized", False))
        msg = f"Reconstructed {successful}/{len(recon_reports)} frames"
        if optimize:
            msg += f" ({optimized} optimized)"
        
        return StepResult(
            StepStatus.SUCCESS,
            msg,
            {"reports": recon_reports, "optimized_count": optimized}
        )
    
    def _run_reconstruction(
        self,
        binary: Path,
        model_dir: Path,
        intrinsics: Path,
        rgb: Path,
        depth: Path,
        output_mesh: Path,
        timeout: int,
        output_pointcloud: Optional[Path] = None,
        landmarks: Optional[Path] = None,
        mapping: Optional[Path] = None,
        optimize: bool = False,
        pose_only: bool = False,
        verbose: bool = False,
        max_iterations: int = 50,
        lambda_landmark: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_reg: float = 1.0,
    ) -> bool:
        """Run the C++ reconstruction binary."""
        cmd = [
            str(binary),
            "--rgb", str(rgb),
            "--depth", str(depth),
            "--intrinsics", str(intrinsics),
            "--model-dir", str(model_dir),
            "--output-mesh", str(output_mesh),
        ]
        
        if output_pointcloud is not None:
            cmd.extend(["--output-pointcloud", str(output_pointcloud)])
            output_pointcloud.parent.mkdir(parents=True, exist_ok=True)
        
        # Week 4: Add landmarks and optimization params
        if landmarks is not None and landmarks.exists():
            cmd.extend(["--landmarks", str(landmarks)])
        
        if mapping is not None and mapping.exists():
            cmd.extend(["--mapping", str(mapping)])
        
        if optimize:
            cmd.append("--optimize")
            if pose_only:
                cmd.append("--pose-only")
            cmd.extend(["--max-iter", str(max_iterations)])
            cmd.extend(["--lambda-landmark", str(lambda_landmark)])
            cmd.extend(["--lambda-depth", str(lambda_depth)])
            cmd.extend(["--lambda-reg", str(lambda_reg)])
        else:
            cmd.append("--no-optimize")
        
        if verbose:
            cmd.append("--verbose")
        
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
            mesh_ok = output_mesh.exists()
            cloud_ok = output_pointcloud is None or output_pointcloud.exists()
            return mesh_ok and cloud_ok
        except subprocess.CalledProcessError as exc:
            self.logger.debug(f"Reconstruction failed: {exc.stderr}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Reconstruction timed out for {rgb.name}")
            return False
