"""
Week 4: Mesh-Scan Overlay Generation Step

Creates visualization overlays after rigid alignment:
1. 3D PLY overlay (cyan scan + red mesh)
2. Standalone scan and mesh PLY files
3. Quantitative sanity check metrics
4. 2D PNG overlay (RGB with projected points)

Output structure:
  outputs/overlays_3d/<seq>/
    frame_00000_scan.ply           # Cyan point cloud
    frame_00000_mesh_rigid.ply     # Red rigid-aligned mesh
    frame_00000_overlay_rigid.ply  # Combined overlay
    frame_00000_mesh_opt.ply       # Red optimized mesh (if available)
    frame_00000_overlay_opt.ply    # Combined with optimized mesh
    frame_00000_overlay_metrics.json  # Quantitative metrics
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from main import PipelineStep, StepResult, StepStatus


class Week4OverlayStep(PipelineStep):
    """Generate mesh-scan overlays after rigid alignment."""
    
    @property
    def name(self) -> str:
        return "Week 4 Overlays"
    
    @property
    def description(self) -> str:
        return "Generate 3D mesh-scan overlay visualizations for MeshLab/CloudCompare"
    
    def execute(self) -> StepResult:
        """Generate overlays for pose-initialized frames."""
        binary = Path(self.config.get("overlay_binary", "build/bin/create_overlays")).resolve()
        if not binary.exists():
            return StepResult(StepStatus.FAILED, f"Overlay binary not found: {binary}")
        
        pose_init_reports = self.config.get("pose_init_reports", [])
        conversion_reports = self.config.get("conversion_reports", [])
        
        # New folder structure: outputs/overlays_3d/<seq>/
        overlays_root = Path(self.config.get("overlays_3d_root", "outputs/overlays_3d"))
        meshes_root = Path(self.config.get("meshes_root", "outputs/meshes"))
        
        if not pose_init_reports:
            return StepResult(StepStatus.SKIPPED, "No pose init data available")
        
        # Build conversion map
        conversion_map = {}
        for conv_report in conversion_reports:
            seq_id = Path(conv_report.get("output_dir", "")).name
            if seq_id:
                conversion_map[seq_id] = conv_report
        
        overlay_reports = []
        successful = 0
        all_metrics = {}
        
        # Process only frame_00000 for selected sequences
        target_sequences = self.config.get("target_sequences", ["01", "17"])
        target_frame = "frame_00000"
        
        for pose_report in pose_init_reports:
            if not pose_report.get("success", False):
                continue
            
            seq_id = pose_report.get("sequence", "")
            frame_name = pose_report.get("frame", "")
            
            if seq_id not in target_sequences or frame_name != target_frame + ".png":
                continue
            
            # Rigid-aligned mesh from pose_init
            aligned_mesh = Path(pose_report.get("aligned_mesh", ""))
            if not aligned_mesh.exists():
                self.logger.warning(f"Rigid mesh not found: {aligned_mesh}")
                continue
            
            # Optimized mesh (if available)
            frame_stem = Path(frame_name).stem
            optimized_mesh = meshes_root / seq_id / f"{frame_stem}_optimized.ply"
            
            conv_data = conversion_map.get(seq_id)
            if not conv_data:
                self.logger.warning(f"No conversion data for {seq_id}")
                continue
            
            output_dir = Path(conv_data.get("output_dir", ""))
            rgb_path = output_dir / "rgb" / frame_name
            depth_path = output_dir / "depth" / frame_name
            intrinsics_path = output_dir / "intrinsics.txt"
            
            if not all(p.exists() for p in [depth_path, intrinsics_path]):
                self.logger.warning(f"Missing depth or intrinsics for {seq_id}")
                continue
            
            # Output directory: outputs/overlays_3d/<seq>/
            seq_out_dir = overlays_root / seq_id
            seq_out_dir.mkdir(parents=True, exist_ok=True)
            
            success, metrics = self._create_overlays(
                binary=binary,
                mesh_rigid=aligned_mesh,
                mesh_opt=optimized_mesh if optimized_mesh.exists() else None,
                depth=depth_path,
                rgb=rgb_path if rgb_path.exists() else None,
                intrinsics=intrinsics_path,
                out_dir=seq_out_dir,
                frame_name=frame_stem
            )
            
            if success:
                successful += 1
                self.logger.info(f"✓ {seq_id}/{frame_stem} overlays created")
                
                # Collect metrics
                if metrics:
                    all_metrics[seq_id] = {frame_stem: metrics}
            else:
                self.logger.warning(f"✗ {seq_id}/{frame_stem} overlay generation failed")
            
            overlay_reports.append({
                "sequence": seq_id,
                "frame": frame_stem,
                "out_dir": str(seq_out_dir),
                "scan_ply": str(seq_out_dir / f"{frame_stem}_scan.ply"),
                "mesh_rigid_ply": str(seq_out_dir / f"{frame_stem}_mesh_rigid.ply"),
                "overlay_rigid_ply": str(seq_out_dir / f"{frame_stem}_overlay_rigid.ply"),
                "mesh_opt_ply": str(seq_out_dir / f"{frame_stem}_mesh_opt.ply") if optimized_mesh.exists() else None,
                "overlay_opt_ply": str(seq_out_dir / f"{frame_stem}_overlay_opt.ply") if optimized_mesh.exists() else None,
                "metrics_json": str(seq_out_dir / f"{frame_stem}_overlay_metrics.json"),
                "success": success
            })
        
        # Write combined overlay_checks.json to analysis folder
        analysis_dir = Path("outputs/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        overlay_checks_path = analysis_dir / "overlay_checks.json"
        
        if all_metrics:
            with open(overlay_checks_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            self.logger.info(f"Wrote combined metrics to {overlay_checks_path}")
        
        return StepResult(
            StepStatus.SUCCESS,
            f"Created {successful}/{len(overlay_reports)} overlay sets",
            {"reports": overlay_reports, "overlay_checks": str(overlay_checks_path)}
        )
    
    def _create_overlays(
        self,
        binary: Path,
        mesh_rigid: Path,
        mesh_opt: Optional[Path],
        depth: Path,
        rgb: Optional[Path],
        intrinsics: Path,
        out_dir: Path,
        frame_name: str
    ) -> tuple:
        """
        Run overlay generation binary with new CLI interface.
        
        Returns:
            tuple: (success: bool, metrics: dict or None)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(binary),
            "--mesh-rigid", str(mesh_rigid),
            "--depth", str(depth),
            "--intrinsics", str(intrinsics),
            "--out-dir", str(out_dir),
            "--frame-name", frame_name,
        ]
        
        if mesh_opt and mesh_opt.exists():
            cmd.extend(["--mesh-opt", str(mesh_opt)])
        
        if rgb and rgb.exists():
            cmd.extend(["--rgb", str(rgb)])
        
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=120
            )
            
            # Check if key output files exist
            overlay_rigid = out_dir / f"{frame_name}_overlay_rigid.ply"
            scan_ply = out_dir / f"{frame_name}_scan.ply"
            metrics_json = out_dir / f"{frame_name}_overlay_metrics.json"
            
            success = overlay_rigid.exists() and scan_ply.exists()
            
            # Load metrics if available
            metrics = None
            if metrics_json.exists():
                try:
                    with open(metrics_json) as f:
                        metrics = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            return success, metrics
            
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"Overlay generation failed: {exc.stderr}")
            return False, None
        except subprocess.TimeoutExpired:
            self.logger.error("Overlay generation timed out")
            return False, None


def run_standalone(sequences: List[str] = None):
    """
    Standalone function to generate overlays without full pipeline.
    
    Usage:
        from pipeline.steps.week4_overlays import run_standalone
        run_standalone(["01", "17"])
    """
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("overlay_standalone")
    
    if sequences is None:
        sequences = ["01", "17"]
    
    binary = Path("build/bin/create_overlays").resolve()
    if not binary.exists():
        logger.error(f"Binary not found: {binary}")
        return
    
    pose_init_root = Path("outputs/pose_init")
    converted_root = Path("outputs/converted")
    meshes_root = Path("outputs/meshes")
    overlays_root = Path("outputs/overlays_3d")
    
    for seq_id in sequences:
        logger.info(f"Processing sequence {seq_id}...")
        
        frame_name = "frame_00000"
        
        # Input paths
        mesh_rigid = pose_init_root / seq_id / f"{frame_name}_aligned.ply"
        mesh_opt = meshes_root / seq_id / f"{frame_name}_optimized.ply"
        depth = converted_root / seq_id / "depth" / f"{frame_name}.png"
        rgb = converted_root / seq_id / "rgb" / f"{frame_name}.png"
        intrinsics = converted_root / seq_id / "intrinsics.txt"
        
        if not mesh_rigid.exists():
            logger.warning(f"Rigid mesh not found: {mesh_rigid}")
            continue
        if not depth.exists():
            logger.warning(f"Depth not found: {depth}")
            continue
        
        # Output directory
        out_dir = overlays_root / seq_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(binary),
            "--mesh-rigid", str(mesh_rigid),
            "--depth", str(depth),
            "--intrinsics", str(intrinsics),
            "--out-dir", str(out_dir),
            "--frame-name", frame_name,
        ]
        
        if mesh_opt.exists():
            cmd.extend(["--mesh-opt", str(mesh_opt)])
        if rgb.exists():
            cmd.extend(["--rgb", str(rgb)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            print(result.stdout)
            logger.info(f"✓ {seq_id} completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("Timeout")


if __name__ == "__main__":
    import sys
    seqs = sys.argv[1:] if len(sys.argv) > 1 else ["01", "17"]
    run_standalone(seqs)
