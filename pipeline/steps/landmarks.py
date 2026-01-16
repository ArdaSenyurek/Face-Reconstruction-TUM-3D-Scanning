"""
Landmark detection step using dlib (single, fixed method).
"""

import cv2
from pathlib import Path
from typing import List, Optional, Tuple

from main import PipelineStep, StepResult, StepStatus

try:
    import dlib
    _DLIB_AVAILABLE = True
except Exception:
    _DLIB_AVAILABLE = False


class LandmarkDetectionStep(PipelineStep):
    """Detect facial landmarks in RGB images."""
    
    @property
    def name(self) -> str:
        return "Landmark Detection"
    
    @property
    def description(self) -> str:
        return "Detect facial landmarks using dlib"
    
    def execute(self) -> StepResult:
        """Detect landmarks for converted sequences."""
        if not _DLIB_AVAILABLE:
            error_msg = "dlib is not installed. Install with: pip install dlib"
            return StepResult(StepStatus.FAILED, error_msg)
        
        conversion_reports = self.config.get("conversion_reports", [])
        landmarks_root = Path(self.config["landmarks_root"])
        overlays_root = Path(self.config.get("overlays_root", landmarks_root.parent / "overlays"))
        run_frames = self.config.get("run_frames", 5)
        
        landmark_reports = []
        for seq_report in conversion_reports:
            output_dir = seq_report.get("output_dir")
            if not output_dir:
                continue  # skip failed conversions with no output dir
            seq_dir = Path(output_dir)
            rgb_dir = seq_dir / "rgb"
            
            if not rgb_dir.exists():
                continue
            
            # Process all frames (not just the first one)
            frames = sorted(rgb_dir.glob("frame_*.png"))[:run_frames]
            if not frames:
                continue
            
            seq_landmarks = []
            for frame_path in frames:
                try:
                    landmarks = self._detect_dlib(frame_path)
                    if landmarks:
                        lm_path = landmarks_root / seq_dir.name / f"{frame_path.stem}.txt"
                        self._save_landmarks(landmarks, lm_path)
                        
                        overlay_path = overlays_root / seq_dir.name / f"{frame_path.stem}_overlay.png"
                        self._save_overlay(frame_path, landmarks, overlay_path)
                        
                        seq_landmarks.append({
                            "frame": frame_path.name,
                            "landmarks_file": str(lm_path),
                            "overlay": str(overlay_path)
                        })
                    else:
                        self.logger.warning(f"No landmarks detected for {seq_dir.name}/{frame_path.name}")
                except Exception as e:
                    self.logger.warning(f"Landmark detection failed for {seq_dir.name}/{frame_path.name}: {e}")
            
            landmark_reports.append({
                "sequence": seq_dir.name,
                "landmarks": seq_landmarks,
                "count": len(seq_landmarks)
            })
        
        total_landmarks = sum(r.get("count", 0) for r in landmark_reports)
        return StepResult(StepStatus.SUCCESS, f"Detected landmarks for {total_landmarks} frames across {len(landmark_reports)} sequences",
                         {"reports": landmark_reports})
    
    def _detect_landmarks(self, image_path: Path, method: str) -> Optional[List[Tuple[int, int]]]:
        """Detect landmarks (dlib only)."""
        return self._detect_dlib(image_path)
    
    def _detect_dlib(self, image_path: Path) -> Optional[List[Tuple[int, int]]]:
        """Detect landmarks using dlib."""
        if not _DLIB_AVAILABLE:
            self.logger.warning("dlib is not available")
            return None
        
        predictor_path = Path("data/models/shape_predictor_68_face_landmarks.dat")
        if not predictor_path.exists():
            self.logger.warning(f"dlib shape predictor not found at {predictor_path}")
            self.logger.warning("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            return None
        
        # Validate file size (should be ~99MB)
        file_size = predictor_path.stat().st_size
        if file_size < 10 * 1024 * 1024:  # Less than 10MB is suspicious
            self.logger.warning(f"dlib model file is too small ({file_size} bytes), may be corrupted")
            return None
        
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(str(predictor_path))
        except Exception as e:
            self.logger.warning(f"Failed to load dlib shape predictor: {e}")
            self.logger.warning("The model file may be corrupted. Please re-download it.")
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if not faces:
                return None
            
            face = max(faces, key=lambda r: r.width() * r.height())
            shape = predictor(gray, face)
            return [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        except Exception as e:
            self.logger.warning(f"Landmark detection failed for {image_path.name}: {e}")
            return None
    
    def _save_landmarks(self, landmarks: List[Tuple[int, int]], output_path: Path):
        """Save landmarks to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for x, y in landmarks:
                f.write(f"{x} {y} -1\n")
    
    def _save_overlay(self, image_path: Path, landmarks: List[Tuple[int, int]], output_path: Path):
        """Save landmark overlay visualization."""
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(img, str(i), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

