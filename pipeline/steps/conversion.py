"""
Data conversion step: Convert raw Biwi data to standardized format.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from main import (
    PipelineStep,
    StepResult,
    StepStatus,
    convert_depth_image,
    copy_rgb,
    create_depth_visualization,
    default_intrinsics,
    find_rgb_depth_pairs,
    read_biwi_calibration,
)

# Lazy import for point cloud creation (optional - for visualization only)
try:
    from pipeline.utils.create_pointcloud_from_rgbd import create_pointcloud_from_rgbd
except Exception:
    create_pointcloud_from_rgbd = None


class ConversionStep(PipelineStep):
    """Convert raw Biwi sequences to standardized format."""
    
    @staticmethod
    def _should_skip_sequence(name: str) -> bool:
        """Ignore temporary or helper folders that are not real sequences."""
        return name.startswith("temp_")
    
    @property
    def name(self) -> str:
        return "Data Conversion"
    
    @property
    def description(self) -> str:
        return "Convert RGB/depth frames to standardized layout"
    
    def execute(self) -> StepResult:
        """Convert sequences to standardized format."""
        raw_root = Path(self.config["raw_root"])
        output_root = Path(self.config["output_root"])
        max_frames = self.config.get("max_frames", 0)
        intrinsics_override = self.config.get("intrinsics")
        
        # Optionally restrict to selected sequence IDs (e.g. faces 1, 14, 18, 20)
        sequences_filter = self.config.get("sequences")
        sequence_dirs = sorted([
            p for p in raw_root.iterdir()
            if p.is_dir() and not self._should_skip_sequence(p.name)
        ])
        if sequences_filter:
            sequence_dirs = [p for p in sequence_dirs if p.name in sequences_filter]
        
        if not sequence_dirs:
            return StepResult(StepStatus.FAILED, "No sequences found to process")
        
        frame_step = self.config.get("frame_step", 1)
        frame_indices = self.config.get("frame_indices")
        conversion_reports = []
        for seq_dir in sequence_dirs:
            try:
                report = self._convert_sequence(seq_dir, output_root / seq_dir.name,
                                               max_frames, frame_step, frame_indices, intrinsics_override)
                conversion_reports.append(report)
                self.logger.info(f"Converted {seq_dir.name}: {report['rgb_ok']}/{report['frames_total']} RGB, "
                               f"{report['depth_ok']}/{report['frames_total']} depth")
            except Exception as e:
                self.logger.error(f"Failed to convert {seq_dir.name}: {e}")
                conversion_reports.append({
                    "sequence": seq_dir.name,
                    "frames_total": 0,
                    "rgb_ok": 0,
                    "depth_ok": 0,
                    "error": str(e)
                })
        
        return StepResult(StepStatus.SUCCESS, f"Converted {len(conversion_reports)} sequences",
                         {"reports": conversion_reports})
    
    def _convert_sequence(self, seq_dir: Path, output_dir: Path, max_frames: int,
                          frame_step: int,
                          frame_indices: Optional[List[int]],
                          intrinsics_override: Optional[Tuple[float, float, float, float]]) -> dict:
        """Convert a single sequence. Use frame_indices (e.g. [1,5,10]) if set; else step/max_frames."""
        all_pairs = find_rgb_depth_pairs(seq_dir)
        if not all_pairs:
            raise RuntimeError(f"No RGB/depth pairs found in {seq_dir}")
        if frame_indices is not None:
            indices = [i for i in sorted(frame_indices) if i < len(all_pairs)]
        elif max_frames > 0 and frame_step > 0:
            indices = list(range(0, len(all_pairs), frame_step))[:max_frames]
        else:
            indices = list(range(len(all_pairs))) if max_frames <= 0 else list(range(min(max_frames, len(all_pairs))))
        
        depth_cal = seq_dir / "depth.cal"
        intrinsics = intrinsics_override or read_biwi_calibration(depth_cal) or default_intrinsics()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "intrinsics.txt", "w", encoding="utf-8") as f:
            f.write(f"{intrinsics[0]} {intrinsics[1]} {intrinsics[2]} {intrinsics[3]}\n")
        
        rgb_out_dir = output_dir / "rgb"
        depth_out_dir = output_dir / "depth"
        depth_vis_dir = output_dir / "depth_vis"
        pc_out_dir = output_dir / "pointclouds"
        success_rgb = success_depth = 0
        
        for pos, idx in enumerate(indices):
            if idx >= len(all_pairs):
                break
            rgb_path, depth_path = all_pairs[idx]
            rgb_out = rgb_out_dir / f"frame_{idx:05d}.png"
            depth_out = depth_out_dir / f"frame_{idx:05d}.png"
            depth_vis_out = depth_vis_dir / f"frame_{idx:05d}.png"
            pc_out = pc_out_dir / f"frame_{idx:05d}.ply"
            
            if copy_rgb(rgb_path, rgb_out):
                success_rgb += 1
            if convert_depth_image(depth_path, depth_out):
                success_depth += 1
                create_depth_visualization(depth_out, depth_vis_out)
                if create_pointcloud_from_rgbd:
                    try:
                        create_pointcloud_from_rgbd(
                            rgb_out,
                            depth_out,
                            output_dir / "intrinsics.txt",
                            pc_out,
                            depth_scale=1000.0,
                        )
                    except Exception:
                        pass
            if (pos + 1) % 10 == 0 or pos == len(indices) - 1:
                self.logger.progress(pos + 1, len(indices), f"frames in {seq_dir.name}")
        
        return {
            "sequence": seq_dir.name,
            "frames_total": len(indices),
            "rgb_ok": success_rgb,
            "depth_ok": success_depth,
            "intrinsics": intrinsics,
            "output_dir": str(output_dir),
        }

