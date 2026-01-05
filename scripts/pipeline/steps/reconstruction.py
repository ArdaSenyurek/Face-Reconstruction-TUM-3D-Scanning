"""
3D face reconstruction step using C++ binary.
"""

import subprocess
from pathlib import Path
from typing import Optional

from pipeline import PipelineStep, StepResult, StepStatus


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
        run_frames = self.config.get("run_frames", 5)
        timeout = self.config.get("timeout", 60)
        save_pointclouds = self.config.get("save_pointclouds", False)
        
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
                
                pc_out = None
                if save_pointclouds:
                    pc_out = analysis_root / "pointclouds" / seq_dir.name / f"{frame.stem}.ply"
                
                success = self._run_reconstruction(binary, model_dir, intrinsics_path, 
                                                  frame, depth_frame, mesh_out, timeout, pc_out)
                
                recon_reports.append({
                    "sequence": seq_dir.name,
                    "frame": frame.name,
                    "mesh": str(mesh_out),
                    "success": success
                })
                
                if success:
                    self.logger.info(f"✓ {seq_dir.name}/{frame.name}")
                else:
                    self.logger.warning(f"✗ {seq_dir.name}/{frame.name} failed")
        
        successful = sum(1 for r in recon_reports if r["success"])
        return StepResult(StepStatus.SUCCESS, f"Reconstructed {successful}/{len(recon_reports)} frames",
                         {"reports": recon_reports})
    
    def _run_reconstruction(self, binary: Path, model_dir: Path, intrinsics: Path,
                           rgb: Path, depth: Path, output_mesh: Path, timeout: int,
                           output_pointcloud: Optional[Path] = None) -> bool:
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

