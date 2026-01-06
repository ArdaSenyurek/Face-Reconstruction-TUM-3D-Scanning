"""
Data conversion step: Convert raw Biwi data to standardized format.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from pipeline import (
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


class ConversionStep(PipelineStep):
    """Convert raw Biwi sequences to standardized format."""
    
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
        sequence_filter = self.config.get("sequence")  # Can be None, single string, or list
        intrinsics_override = self.config.get("intrinsics")
        
        # Handle multiple sequence filters
        if sequence_filter is None:
            # Process all sequences
            sequence_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
        elif isinstance(sequence_filter, list):
            # Process specific sequences
            filter_set = set(sequence_filter)
            sequence_dirs = sorted([p for p in raw_root.iterdir() 
                                   if p.is_dir() and p.name in filter_set])
        else:
            # Single sequence (backward compatibility)
            sequence_dirs = sorted([p for p in raw_root.iterdir() 
                                   if p.is_dir() and p.name == sequence_filter])
        
        if not sequence_dirs:
            return StepResult(StepStatus.FAILED, "No sequences found to process")
        
        conversion_reports = []
        for seq_dir in sequence_dirs:
            try:
                report = self._convert_sequence(seq_dir, output_root / seq_dir.name, 
                                               max_frames, intrinsics_override)
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
                          intrinsics_override: Optional[Tuple[float, float, float, float]]) -> dict:
        """Convert a single sequence."""
        pairs = find_rgb_depth_pairs(seq_dir)
        if max_frames > 0:
            pairs = pairs[:max_frames]
        
        if not pairs:
            raise RuntimeError(f"No RGB/depth pairs found in {seq_dir}")
        
        depth_cal = seq_dir / "depth.cal"
        intrinsics = intrinsics_override or read_biwi_calibration(depth_cal) or default_intrinsics()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "intrinsics.txt", "w", encoding="utf-8") as f:
            f.write(f"{intrinsics[0]} {intrinsics[1]} {intrinsics[2]} {intrinsics[3]}\n")
        
        rgb_out_dir = output_dir / "rgb"
        depth_out_dir = output_dir / "depth"
        depth_vis_dir = output_dir / "depth_vis"
        success_rgb = success_depth = 0
        
        for idx, (rgb_path, depth_path) in enumerate(pairs):
            rgb_out = rgb_out_dir / f"frame_{idx:05d}.png"
            depth_out = depth_out_dir / f"frame_{idx:05d}.png"
            depth_vis_out = depth_vis_dir / f"frame_{idx:05d}.png"
            
            if copy_rgb(rgb_path, rgb_out):
                success_rgb += 1
            if convert_depth_image(depth_path, depth_out):
                success_depth += 1
                # Create depth visualization for better viewing
                create_depth_visualization(depth_out, depth_vis_out)
            if (idx + 1) % 10 == 0 or idx == len(pairs) - 1:
                self.logger.progress(idx + 1, len(pairs), f"frames in {seq_dir.name}")
        
        return {
            "sequence": seq_dir.name,
            "frames_total": len(pairs),
            "rgb_ok": success_rgb,
            "depth_ok": success_depth,
            "intrinsics": intrinsics,
            "output_dir": str(output_dir),
        }

