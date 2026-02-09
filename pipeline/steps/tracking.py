"""
Week 5: Tracking Step for Sequential Face Reconstruction

Processes multiple frames with temporal continuity:
- Warm-start optimization from previous frame
- Optional temporal smoothing (EMA for translation/expression, SLERP for rotation)
- Drift detection and automatic re-initialization
- Per-frame metrics and visualizations
"""

import json
import subprocess
import csv
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pipeline.main import PipelineStep, StepResult, StepStatus


@dataclass
class TrackingState:
    """State carried between frames for tracking."""
    # Pose parameters
    rotation: List[List[float]] = field(default_factory=lambda: [[1,0,0],[0,1,0],[0,0,1]])
    translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    
    # Shape coefficients
    expression: List[float] = field(default_factory=list)
    identity: List[float] = field(default_factory=list)
    
    # Tracking metadata
    frame_idx: int = 0
    reinit_count: int = 0
    last_rmse_mm: float = 0.0  # Actually optimization final_energy when from C++; not geometric RMSE in mm
    final_energy: float = 0.0   # Optimization total energy (same as last_rmse_mm from C++)
    depth_rmse_mm: float = -1.0  # Per-pixel depth RMSE in mm (from C++; -1 if not set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackingState':
        """Create from dictionary (only known fields; missing keys use dataclass defaults)."""
        allowed = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)
    
    def save(self, path: Path):
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrackingState':
        """Load state from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_idx: int
    frame_name: str
    sequence: str
    
    # Alignment quality
    landmark_rmse_mm: float = 0.0   # 3D landmark error in mm (optional; use pose_init for rigid)
    landmark_reprojection_rmse_px: float = -1.0  # 2D reprojection RMSE in pixels (-1 if not computed)
    optimization_energy: float = 0.0  # Optimization total energy (from C++ final_energy); not geometric RMSE
    final_energy: float = 0.0       # Same as optimization_energy; use overlay/analysis for geometric mesh-scan RMSE
    depth_rmse_mm: float = -1.0     # Per-pixel depth RMSE in mm (from C++; -1 if not available)
    
    # Pose parameters
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    rotation_angle_deg: float = 0.0
    scale: float = 1.0
    
    # Expression
    expression_norm: float = 0.0
    
    # Tracking status
    was_reinit: bool = False
    optimization_converged: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    dot = np.dot(q0, q1)
    
    # If dot is negative, negate one quaternion to take shorter path
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(np.clip(dot, -1, 1))
    theta = theta_0 * t
    
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def rotation_angle_from_matrix(R: np.ndarray) -> float:
    """Get rotation angle in degrees from rotation matrix."""
    trace = np.trace(R)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


class TrackingStep(PipelineStep):
    """
    Week 5: Sequential tracking with temporal continuity.
    
    Processes frames in order, using previous frame's pose and expression
    as initialization for the next frame's optimization.
    """
    
    @property
    def name(self) -> str:
        return "Tracking"
    
    @property
    def description(self) -> str:
        return "Sequential face tracking with temporal continuity"
    
    def execute(self) -> StepResult:
        """Run tracking for all sequences and frames."""
        binary = Path(self.config["binary"]).resolve()
        pose_init_binary = Path(self.config.get("pose_init_binary", "build/bin/pose_init")).resolve()
        
        if not binary.exists():
            return StepResult(StepStatus.FAILED, f"Binary not found: {binary}")
        
        conversion_reports = self.config.get("conversion_reports", [])
        target_sequences = self.config.get("target_sequences", ["01", "17"])
        run_frames = self.config.get("run_frames", 5)
        
        all_reports = []
        all_metrics = []
        
        for seq_report in conversion_reports:
            seq_id = seq_report.get("sequence", "")
            
            # Filter to target sequences
            if target_sequences and seq_id not in target_sequences:
                continue
            
            self.logger.info(f"Processing sequence {seq_id} with tracking...")
            
            seq_metrics = self._process_sequence(
                seq_report, 
                binary, 
                pose_init_binary,
                run_frames
            )
            
            all_metrics.extend(seq_metrics)
            
            # Create reports for each frame
            for metrics in seq_metrics:
                all_reports.append({
                    "sequence": metrics.sequence,
                    "frame": metrics.frame_name,
                    "mesh": str(self._get_mesh_path(metrics.sequence, metrics.frame_name)),
                    "success": True,
                    "tracked": True,
                })
        
        # Save tracking summary
        self._save_tracking_summary(all_metrics)
        
        msg = f"Tracked {len(all_reports)} frames across {len(target_sequences)} sequences"
        # Convert FrameMetrics to dicts for JSON serialization
        metrics_dicts = [m.to_dict() for m in all_metrics]
        return StepResult(StepStatus.SUCCESS, msg, {"reports": all_reports, "metrics": metrics_dicts})
    
    def _process_sequence(
        self,
        seq_report: Dict[str, Any],
        binary: Path,
        pose_init_binary: Path,
        run_frames: int
    ) -> List[FrameMetrics]:
        """Process a single sequence with tracking."""
        seq_dir = Path(seq_report["output_dir"])
        seq_id = seq_report.get("sequence", seq_dir.name)
        
        rgb_dir = seq_dir / "rgb"
        depth_dir = seq_dir / "depth"
        intrinsics_path = seq_dir / "intrinsics.txt"
        
        frames = sorted(rgb_dir.glob("frame_*.png"))[:run_frames]
        
        if not frames:
            self.logger.warning(f"No frames found in {rgb_dir}")
            return []
        
        # Initialize tracking state
        state: Optional[TrackingState] = None
        metrics_list: List[FrameMetrics] = []
        
        for frame_idx, frame_path in enumerate(frames):
            frame_name = frame_path.stem
            depth_path = depth_dir / f"{frame_name}.png"
            landmarks_path = Path(self.config["landmarks_root"]) / seq_id / f"{frame_name}.txt"
            
            if not depth_path.exists():
                self.logger.warning(f"Missing depth for {frame_name}")
                continue
            
            if not landmarks_path.exists():
                self.logger.warning(f"Missing landmarks for {frame_name}")
                continue
            
            # Determine if we need to (re)initialize
            prev_prev_frame_name = frames[frame_idx - 2].stem if frame_idx >= 2 else None
            needs_init = (
                frame_idx == 0 or 
                state is None or
                self._should_reinit(frame_idx, state, seq_id=seq_id, prev_prev_frame_name=prev_prev_frame_name)
            )
            
            if needs_init:
                # Run Procrustes initialization
                self.logger.info(f"  Frame {frame_idx}: Initializing with Procrustes")
                state = self._init_from_procrustes(
                    pose_init_binary,
                    seq_id,
                    frame_name,
                    depth_path,
                    frame_path,
                    intrinsics_path,
                    landmarks_path
                )
                if state is None:
                    self.logger.warning(f"  Failed to initialize frame {frame_idx}")
                    continue
                state.reinit_count += 1
            else:
                self.logger.info(f"  Frame {frame_idx}: Warm-starting from previous frame")
            
            # Save init state for this frame
            state.frame_idx = frame_idx
            init_state_path = self._get_state_path(seq_id, frame_name, "init")
            state.save(init_state_path)
            
            # Run reconstruction with warm-start
            metrics = self._run_reconstruction(
                binary,
                seq_id,
                frame_name,
                frame_path,
                depth_path,
                intrinsics_path,
                landmarks_path,
                state,
                was_reinit=needs_init
            )
            
            if metrics is None:
                self.logger.warning(f"  Reconstruction failed for frame {frame_idx}")
                continue
            
            metrics_list.append(metrics)
            
            # Load final state from reconstruction
            final_state_path = self._get_state_path(seq_id, frame_name, "final")
            if final_state_path.exists():
                new_state = TrackingState.load(final_state_path)
                new_state.reinit_count = state.reinit_count
                
                # Apply temporal smoothing if enabled
                if self.config.get("temporal_smoothing", False) and not needs_init:
                    new_state = self._apply_temporal_smoothing(state, new_state)
                
                state = new_state
                state.last_rmse_mm = metrics.optimization_energy
                state.final_energy = metrics.final_energy
            
            # Generate overlays if enabled
            if self.config.get("save_overlays_3d", True):
                self._generate_overlay(seq_id, frame_name, depth_path, intrinsics_path)
            
            # Generate depth residual visualization if enabled
            if self.config.get("save_depth_residual_vis", True):
                self._generate_depth_residual_vis(seq_id, frame_name, depth_path, intrinsics_path)
            
            self.logger.info(f"    Final energy: {metrics.final_energy:.2f}")
        
        return metrics_list
    
    def _should_reinit(
        self,
        frame_idx: int,
        state: TrackingState,
        seq_id: Optional[str] = None,
        prev_prev_frame_name: Optional[str] = None,
    ) -> bool:
        """Check if re-initialization is needed."""
        reinit_every = self.config.get("reinit_every", 0)
        
        # Periodic re-init
        if reinit_every > 0 and frame_idx > 0 and frame_idx % reinit_every == 0:
            self.logger.info(f"    Triggering periodic re-init (every {reinit_every} frames)")
            return True
        
        # Z-range check: reinit if previous pose is outside plausible depth range (e.g. mesh drifted)
        if frame_idx >= 1 and state is not None:
            z_min = self.config.get("drift_z_range_min_m", 0.5)
            z_max = self.config.get("drift_z_range_max_m", 1.5)
            tz = state.translation[2] if len(state.translation) > 2 else 0.0
            if z_max > z_min and (tz < z_min or tz > z_max):
                self.logger.info(
                    f"    Triggering re-init: translation_z {tz:.3f} m outside [{z_min}, {z_max}] m"
                )
                return True
            # Reinit when per-pixel depth RMSE is too high (poor fit)
            depth_rmse_thresh = self.config.get("drift_depth_rmse_thresh_mm", 0.0)
            if depth_rmse_thresh > 0:
                depth_rmse = getattr(state, "depth_rmse_mm", -1.0)
                if depth_rmse >= 0 and depth_rmse >= depth_rmse_thresh:
                    self.logger.info(
                        f"    Triggering re-init: depth_rmse_mm {depth_rmse:.1f} >= {depth_rmse_thresh} mm"
                    )
                    return True
        
        # Translation-delta drift detection (when config is set)
        if frame_idx >= 2 and seq_id and prev_prev_frame_name:
            drift_z_thresh = self.config.get("drift_translation_z_thresh_m", 0.5)
            drift_norm_thresh = self.config.get("drift_translation_norm_thresh_m", 0.0)
            if drift_z_thresh <= 0 and drift_norm_thresh <= 0:
                pass
            else:
                prev_prev_state_path = self._get_state_path(seq_id, prev_prev_frame_name, "final")
                if prev_prev_state_path.exists():
                    try:
                        prev_prev_state = TrackingState.load(prev_prev_state_path)
                        delta = np.array(state.translation) - np.array(prev_prev_state.translation)
                        if drift_z_thresh > 0 and abs(delta[2]) >= drift_z_thresh:
                            self.logger.info(
                                f"    Triggering re-init: translation_z delta {delta[2]:.3f} m >= {drift_z_thresh} m"
                            )
                            return True
                        if drift_norm_thresh > 0 and np.linalg.norm(delta) >= drift_norm_thresh:
                            self.logger.info(
                                f"    Triggering re-init: translation norm delta {np.linalg.norm(delta):.3f} m >= {drift_norm_thresh} m"
                            )
                            return True
                    except Exception as e:
                        self.logger.debug(f"Could not load prev-prev state for drift check: {e}")
        
        return False
    
    def _init_from_procrustes(
        self,
        binary: Path,
        seq_id: str,
        frame_name: str,
        depth_path: Path,
        rgb_path: Path,
        intrinsics_path: Path,
        landmarks_path: Path
    ) -> Optional[TrackingState]:
        """Run pose_init to get initial pose from Procrustes."""
        model_dir = Path(self.config["model_dir"])
        mapping_path = Path(self.config.get("landmark_mapping", "data/landmark_mapping_bfm.txt"))
        pose_init_root = Path(self.config.get("pose_init_root", "outputs/pose_init"))
        
        output_mesh = pose_init_root / seq_id / f"{frame_name}_aligned.ply"
        report_path = pose_init_root / seq_id / f"{frame_name}_rigid_report.json"
        
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(binary),
            "--depth", str(depth_path),
            "--rgb", str(rgb_path),
            "--intrinsics", str(intrinsics_path),
            "--landmarks", str(landmarks_path),
            "--mapping", str(mapping_path),
            "--model-dir", str(model_dir),
            "--output", str(output_mesh),
            "--report", str(report_path),
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.config.get("timeout", 60)
            )
            
            if result.returncode != 0:
                self.logger.warning(f"pose_init failed: {result.stderr}")
                return None
            
            # Parse report for pose
            if not report_path.exists():
                return None
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            transform = report.get("transform", {})
            
            # Note: pose_init reports scale in mm->mm units, but face_reconstruction
            # expects scale in mm->meters units, so we divide by 1000
            raw_scale = transform.get("scale", 1.0)
            scale_mm_to_m = raw_scale / 1000.0
            
            state = TrackingState(
                rotation=transform.get("rotation", [[1,0,0],[0,1,0],[0,0,1]]),
                translation=transform.get("translation", [0, 0, 0]),
                scale=scale_mm_to_m,
                expression=[],  # Will be initialized to zero
                identity=[],
                frame_idx=0,
                reinit_count=0,
                last_rmse_mm=report.get("alignment_errors", {}).get("rmse_mm", 0)
            )
            
            return state
            
        except subprocess.TimeoutExpired:
            self.logger.warning("pose_init timed out")
            return None
        except Exception as e:
            self.logger.warning(f"pose_init error: {e}")
            return None
    
    def _run_reconstruction(
        self,
        binary: Path,
        seq_id: str,
        frame_name: str,
        rgb_path: Path,
        depth_path: Path,
        intrinsics_path: Path,
        landmarks_path: Path,
        init_state: TrackingState,
        was_reinit: bool
    ) -> Optional[FrameMetrics]:
        """Run face reconstruction with warm-start from init_state."""
        model_dir = Path(self.config["model_dir"])
        mapping_path = Path(self.config.get("landmark_mapping", "data/landmark_mapping_bfm.txt"))
        meshes_root = Path(self.config["meshes_root"])
        tracking_root = Path(self.config.get("tracking_root", "outputs/tracking"))
        
        output_mesh = meshes_root / seq_id / f"{frame_name}_tracked.ply"
        init_state_path = self._get_state_path(seq_id, frame_name, "init")
        final_state_path = self._get_state_path(seq_id, frame_name, "final")
        
        output_mesh.parent.mkdir(parents=True, exist_ok=True)
        
        # Use balanced weights for stable tracking
        # Too aggressive depth weight causes expression explosion
        cmd = [
            str(binary),
            "--rgb", str(rgb_path),
            "--depth", str(depth_path),
            "--intrinsics", str(intrinsics_path),
            "--model-dir", str(model_dir),
            "--landmarks", str(landmarks_path),
            "--mapping", str(mapping_path),
            "--output-mesh", str(output_mesh),
            "--init-pose-json", str(init_state_path),
            "--output-state-json", str(final_state_path),
            "--max-iter", str(self.config.get("max_iterations", 10)),
            "--lambda-landmark", str(self.config.get("lambda_landmark", 1.0)),
            "--lambda-depth", str(self.config.get("lambda_depth", 0.1)),
            "--lambda-reg", str(self.config.get("lambda_reg", 10.0)),
            "--lambda-translation-prior", str(self.config.get("lambda_translation_prior", 0.5)),
            "--max-translation-delta-m", str(self.config.get("max_translation_delta_m", 0.1)),
        ]
        
        if self.config.get("optimize", False):
            cmd.append("--optimize")
        else:
            cmd.append("--no-optimize")
        
        if self.config.get("verbose", False):
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 120)
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Reconstruction failed: {result.stderr[:500]}")
                return None
            
            # Compute metrics (pass landmarks/intrinsics for optional landmark reprojection RMSE)
            metrics = self._compute_frame_metrics(
                seq_id,
                frame_name,
                init_state.frame_idx,
                output_mesh,
                final_state_path,
                was_reinit,
                landmarks_path=landmarks_path,
                intrinsics_path=intrinsics_path,
            )
            
            return metrics
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Reconstruction timed out for {frame_name}")
            return None
        except Exception as e:
            self.logger.warning(f"Reconstruction error: {e}")
            return None
    
    def _compute_landmark_reprojection_rmse(
        self,
        mesh_path: Path,
        landmarks_path: Path,
        intrinsics_path: Path,
    ) -> float:
        """Compute 2D landmark reprojection RMSE in pixels. Returns -1 if not available."""
        mapping_path = Path(self.config.get("landmark_mapping", "data/bfm_landmark_68.txt"))
        if not mesh_path.exists() or not landmarks_path.exists() or not intrinsics_path.exists() or not mapping_path.exists():
            return -1.0
        try:
            verts = self._load_ply_vertices(mesh_path)
            if len(verts) == 0:
                return -1.0
            # Load landmarks (x y per line)
            lm_2d = []
            with open(landmarks_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        lm_2d.append([float(parts[0]), float(parts[1])])
            if not lm_2d:
                return -1.0
            lm_2d = np.array(lm_2d)
            # Load mapping: line i -> vertex index for landmark i
            vertex_indices = []
            with open(mapping_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        vertex_indices.append(int(parts[1]))
            if not vertex_indices or len(vertex_indices) > len(verts):
                return -1.0
            # Intrinsics
            with open(intrinsics_path, "r") as f:
                parts = f.read().strip().split()
            if len(parts) < 4:
                return -1.0
            fx, fy, cx, cy = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            n = min(len(vertex_indices), len(lm_2d))
            errors = []
            for i in range(n):
                vi = vertex_indices[i]
                if vi < 0 or vi >= len(verts):
                    continue
                p = verts[vi]
                if p[2] <= 0:
                    continue
                u = fx * p[0] / p[2] + cx
                v = fy * p[1] / p[2] + cy
                d = np.sqrt((u - lm_2d[i, 0]) ** 2 + (v - lm_2d[i, 1]) ** 2)
                errors.append(d)
            if not errors:
                return -1.0
            return float(np.sqrt(np.mean(np.array(errors) ** 2)))
        except Exception as e:
            self.logger.debug(f"Landmark reprojection RMSE failed: {e}")
            return -1.0

    def _compute_frame_metrics(
        self,
        seq_id: str,
        frame_name: str,
        frame_idx: int,
        mesh_path: Path,
        state_path: Path,
        was_reinit: bool,
        *,
        landmarks_path: Optional[Path] = None,
        intrinsics_path: Optional[Path] = None,
    ) -> FrameMetrics:
        """Compute metrics for a frame."""
        metrics = FrameMetrics(
            frame_idx=frame_idx,
            frame_name=frame_name,
            sequence=seq_id,
            was_reinit=was_reinit
        )
        if landmarks_path is not None and intrinsics_path is not None:
            metrics.landmark_reprojection_rmse_px = self._compute_landmark_reprojection_rmse(
                mesh_path, landmarks_path, intrinsics_path
            )
        # Load final state if available
        if state_path.exists():
            try:
                state = TrackingState.load(state_path)
                
                metrics.translation_x = state.translation[0]
                metrics.translation_y = state.translation[1]
                metrics.translation_z = state.translation[2]
                metrics.scale = state.scale
                
                # Compute rotation angle
                R = np.array(state.rotation)
                metrics.rotation_angle_deg = rotation_angle_from_matrix(R)
                
                # Expression norm
                if state.expression:
                    metrics.expression_norm = np.linalg.norm(state.expression)
                
                # Optimization energy (C++ state: final_energy; legacy last_rmse_mm is same value)
                energy_val = getattr(state, "final_energy", None) or state.last_rmse_mm
                metrics.final_energy = energy_val
                metrics.optimization_energy = energy_val
                # Per-pixel depth RMSE in mm (from C++ state when available)
                metrics.depth_rmse_mm = getattr(state, "depth_rmse_mm", -1.0)
                if metrics.depth_rmse_mm < 0:
                    metrics.depth_rmse_mm = -1.0
            except Exception as e:
                self.logger.warning(f"Error loading state: {e}")
        
        return metrics
    
    def _apply_temporal_smoothing(
        self,
        prev_state: TrackingState,
        curr_state: TrackingState
    ) -> TrackingState:
        """Apply temporal smoothing to pose and expression."""
        alpha_pose = self.config.get("smooth_pose_alpha", 0.7)
        alpha_expr = self.config.get("smooth_expr_alpha", 0.7)
        
        # Note: alpha=0.7 means 70% of previous, 30% of current (more smoothing)
        # We invert it so alpha is weight of current: t = 1 - alpha
        t_pose = 1.0 - alpha_pose
        t_expr = 1.0 - alpha_expr
        
        # Smooth translation with EMA
        prev_t = np.array(prev_state.translation)
        curr_t = np.array(curr_state.translation)
        smooth_t = prev_t * alpha_pose + curr_t * t_pose
        curr_state.translation = smooth_t.tolist()
        
        # Smooth scale with EMA
        curr_state.scale = prev_state.scale * alpha_pose + curr_state.scale * t_pose
        
        # Smooth rotation with SLERP
        prev_R = np.array(prev_state.rotation)
        curr_R = np.array(curr_state.rotation)
        
        prev_q = quaternion_from_rotation_matrix(prev_R)
        curr_q = quaternion_from_rotation_matrix(curr_R)
        smooth_q = slerp(prev_q, curr_q, t_pose)
        smooth_R = rotation_matrix_from_quaternion(smooth_q)
        curr_state.rotation = smooth_R.tolist()
        
        # Smooth expression with EMA
        if prev_state.expression and curr_state.expression:
            prev_e = np.array(prev_state.expression)
            curr_e = np.array(curr_state.expression)
            if len(prev_e) == len(curr_e):
                smooth_e = prev_e * alpha_expr + curr_e * t_expr
                curr_state.expression = smooth_e.tolist()
        
        return curr_state
    
    def _generate_overlay(
        self,
        seq_id: str,
        frame_name: str,
        depth_path: Path,
        intrinsics_path: Path
    ):
        """Generate 3D overlay for a frame."""
        overlay_binary = Path(self.config.get("overlay_binary", "build/bin/create_overlays"))
        
        if not overlay_binary.exists():
            return
        
        meshes_root = Path(self.config["meshes_root"])
        pose_init_root = Path(self.config.get("pose_init_root", "outputs/pose_init"))
        overlays_root = Path(self.config.get("overlays_3d_root", "outputs/overlays_3d"))
        
        tracked_mesh = meshes_root / seq_id / f"{frame_name}_tracked.ply"
        rigid_mesh = pose_init_root / seq_id / f"{frame_name}_aligned.ply"
        out_dir = overlays_root / seq_id
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(overlay_binary),
            "--depth", str(depth_path),
            "--intrinsics", str(intrinsics_path),
            "--mesh-rigid", str(rigid_mesh) if rigid_mesh.exists() else str(tracked_mesh),
            "--mesh-opt", str(tracked_mesh),
            "--out-dir", str(out_dir),
            "--frame-name", frame_name,
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except Exception as e:
            self.logger.debug(f"Overlay generation failed: {e}")
    
    def _generate_depth_residual_vis(
        self,
        seq_id: str,
        frame_name: str,
        depth_path: Path,
        intrinsics_path: Path
    ):
        """Generate depth residual visualization."""
        import cv2
        
        meshes_root = Path(self.config["meshes_root"])
        analysis_root = Path(self.config.get("analysis_root", "outputs/analysis"))
        
        tracked_mesh = meshes_root / seq_id / f"{frame_name}_tracked.ply"
        residual_dir = analysis_root / "depth_residual_vis" / seq_id
        residual_path = residual_dir / f"{frame_name}_residual.png"
        
        residual_dir.mkdir(parents=True, exist_ok=True)
        
        if not tracked_mesh.exists():
            return
        
        # Load depth
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            return
        
        # Load intrinsics
        try:
            with open(intrinsics_path, 'r') as f:
                parts = f.read().strip().split()
                fx, fy, cx, cy = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except:
            fx, fy, cx, cy = 525, 525, 320, 240
        
        # Load mesh vertices
        vertices = self._load_ply_vertices(tracked_mesh)
        if len(vertices) == 0:
            return
        
        # Render mesh depth (simple z-buffer)
        h, w = depth.shape
        rendered = np.zeros((h, w), dtype=np.float32)
        
        for v in vertices:
            if v[2] <= 0:
                continue
            u = int(fx * v[0] / v[2] + cx)
            pv = int(fy * v[1] / v[2] + cy)
            if 0 <= u < w and 0 <= pv < h:
                z_mm = v[2] * 1000  # Convert to mm
                if rendered[pv, u] == 0 or z_mm < rendered[pv, u]:
                    rendered[pv, u] = z_mm
        
        # Compute residual
        depth_f = depth.astype(np.float32)
        valid = (depth_f > 0) & (rendered > 0)
        
        if not np.any(valid):
            return
        
        residual = np.zeros_like(depth_f)
        residual[valid] = rendered[valid] - depth_f[valid]
        
        # Colormap: blue (mesh closer) -> white (match) -> red (mesh farther)
        # Normalize to [-50, 50] mm range
        residual_clipped = np.clip(residual, -50, 50)
        residual_norm = (residual_clipped + 50) / 100  # 0 to 1
        
        # Create colormap
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[valid, 0] = (residual_norm[valid] * 255).astype(np.uint8)  # R
        vis[valid, 2] = ((1 - residual_norm[valid]) * 255).astype(np.uint8)  # B
        vis[~valid] = [30, 30, 30]  # Gray for invalid
        
        cv2.imwrite(str(residual_path), vis)
    
    def _load_ply_vertices(self, path: Path) -> np.ndarray:
        """Load vertices from PLY file."""
        vertices = []
        try:
            with open(path, 'r') as f:
                in_header = True
                num_verts = 0
                for line in f:
                    if in_header:
                        if 'element vertex' in line:
                            num_verts = int(line.split()[-1])
                        if 'end_header' in line:
                            in_header = False
                    else:
                        if len(vertices) < num_verts:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except:
            pass
        return np.array(vertices) if vertices else np.zeros((0, 3))
    
    def _get_state_path(self, seq_id: str, frame_name: str, suffix: str) -> Path:
        """Get path for tracking state file."""
        tracking_root = Path(self.config.get("tracking_root", "outputs/tracking"))
        state_dir = tracking_root / "state" / seq_id
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / f"{frame_name}_{suffix}.json"
    
    def _get_mesh_path(self, seq_id: str, frame_name: str) -> Path:
        """Get path for tracked mesh."""
        meshes_root = Path(self.config["meshes_root"])
        return meshes_root / seq_id / f"{frame_name}_tracked.ply"
    
    def _save_tracking_summary(self, metrics_list: List[FrameMetrics]):
        """Save tracking summary to JSON and CSV."""
        if not metrics_list:
            return
        
        analysis_root = Path(self.config.get("analysis_root", "outputs/analysis"))
        
        # Group by sequence
        by_seq: Dict[str, List[FrameMetrics]] = {}
        for m in metrics_list:
            if m.sequence not in by_seq:
                by_seq[m.sequence] = []
            by_seq[m.sequence].append(m)
        
        for seq_id, seq_metrics in by_seq.items():
            # Save JSON
            json_path = analysis_root / f"tracking_summary_{seq_id}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            summary = {
                "sequence": seq_id,
                "num_frames": len(seq_metrics),
                "frames": [m.to_dict() for m in seq_metrics],
                "summary": {
                    "mean_final_energy": float(np.mean([m.final_energy for m in seq_metrics])),
                    "std_final_energy": float(np.std([m.final_energy for m in seq_metrics])),
                    "mean_depth_rmse_mm": float(np.mean([m.depth_rmse_mm for m in seq_metrics if m.depth_rmse_mm >= 0])) if any(m.depth_rmse_mm >= 0 for m in seq_metrics) else None,
                    "reinit_count": sum(1 for m in seq_metrics if m.was_reinit),
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save CSV
            csv_path = analysis_root / f"tracking_summary_{seq_id}.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "frame_idx", "frame_name", "sequence",
                    "landmark_rmse_mm", "landmark_reprojection_rmse_px", "optimization_energy", "final_energy", "depth_rmse_mm",
                    "translation_x", "translation_y", "translation_z",
                    "rotation_angle_deg", "scale", "expression_norm",
                    "was_reinit", "optimization_converged"
                ])
                writer.writeheader()
                for m in seq_metrics:
                    writer.writerow(m.to_dict())
            
            self.logger.info(f"Saved tracking summary to {json_path} and {csv_path}")
