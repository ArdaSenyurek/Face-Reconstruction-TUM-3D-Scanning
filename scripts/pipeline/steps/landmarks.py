"""
Landmark detection step using MediaPipe, dlib, or face_alignment.
"""

import cv2
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline import PipelineStep, StepResult, StepStatus

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
    _MEDIAPIPE_HAS_SOLUTIONS = hasattr(mp, "solutions")
except Exception as e:
    _MEDIAPIPE_AVAILABLE = False
    _MEDIAPIPE_HAS_SOLUTIONS = False
    # Store error for debugging (but don't print here to avoid spam)
    _MEDIAPIPE_ERROR = str(e)

try:
    import dlib
    _DLIB_AVAILABLE = True
except Exception:
    _DLIB_AVAILABLE = False

try:
    import face_alignment
    _FA_AVAILABLE = True
except Exception:
    _FA_AVAILABLE = False


class LandmarkDetectionStep(PipelineStep):
    """Detect facial landmarks in RGB images."""
    
    @property
    def name(self) -> str:
        return "Landmark Detection"
    
    @property
    def description(self) -> str:
        return "Detect facial landmarks using specified method"
    
    def execute(self) -> StepResult:
        """Detect landmarks for converted sequences."""
        method = self.config.get("method", "mediapipe")
        if method == "none":
            return StepResult(StepStatus.SKIPPED, "Landmark detection disabled")
        
        # Check if the required library is available
        if method == "mediapipe" and not (_MEDIAPIPE_AVAILABLE and _MEDIAPIPE_HAS_SOLUTIONS):
            error_msg = "MediaPipe is not installed or not available. Install with: pip install mediapipe"
            if '_MEDIAPIPE_ERROR' in globals():
                error_msg += f" (Error: {_MEDIAPIPE_ERROR})"
            self.logger.error(error_msg)
            # Suggest using dlib if available
            if _DLIB_AVAILABLE:
                self.logger.info("Note: dlib is available. You can use --landmarks dlib instead.")
            return StepResult(StepStatus.FAILED, error_msg)
        elif method == "dlib" and not _DLIB_AVAILABLE:
            error_msg = "dlib is not installed. Install with: pip install dlib"
            self.logger.error(error_msg)
            return StepResult(StepStatus.FAILED, error_msg)
        elif method == "face_alignment" and not _FA_AVAILABLE:
            error_msg = "face_alignment is not installed. Install with: pip install face-alignment"
            self.logger.error(error_msg)
            return StepResult(StepStatus.FAILED, error_msg)
        
        conversion_reports = self.config.get("conversion_reports", [])
        landmarks_root = Path(self.config["landmarks_root"])
        overlays_root = Path(self.config.get("overlays_root", landmarks_root.parent / "overlays"))
        run_frames = self.config.get("run_frames", 5)
        
        landmark_reports = []
        for seq_report in conversion_reports:
            seq_dir = Path(seq_report["output_dir"])
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
                    landmarks = self._detect_landmarks(frame_path, method)
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
        """Detect landmarks using specified method."""
        if method == "mediapipe":
            return self._detect_mediapipe(image_path)
        elif method == "dlib":
            return self._detect_dlib(image_path)
        elif method == "face_alignment":
            return self._detect_face_alignment(image_path)
        else:
            self.logger.warning(f"Unknown landmark method: {method}")
            return None
    
    def _detect_mediapipe(self, image_path: Path) -> Optional[List[Tuple[int, int]]]:
        """Detect landmarks using MediaPipe.
        
        Note: MediaPipe Face Mesh returns 468 landmarks, not 68.
        This implementation uses MediaPipe's face_oval and key facial features
        to approximate the dlib 68-point format, but results may differ.
        For accurate 68-point landmarks, use dlib or face_alignment instead.
        """
        if not (_MEDIAPIPE_AVAILABLE and _MEDIAPIPE_HAS_SOLUTIONS):
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        h, w = img.shape[:2]
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if not res.multi_face_landmarks:
            return None
        
        # MediaPipe Face Mesh has 468 landmarks
        # We need to map to dlib's 68-point format
        # Using MediaPipe's face mesh indices that correspond to dlib 68-point landmarks
        landmarks_468 = res.multi_face_landmarks[0].landmark
        
        # MediaPipe to dlib 68-point mapping
        # Based on MediaPipe face mesh topology
        # These indices correspond to key facial features in dlib format
        mp_to_dlib_indices = [
            # Jawline (0-16): 17 points
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            # Right eyebrow (17-21): 5 points  
            107, 55, 65, 52, 53,
            # Left eyebrow (22-26): 5 points
            46, 3, 41, 81, 80,
            # Nose (27-35): 9 points
            1, 2, 5, 4, 6, 19, 20, 94, 98,
            # Right eye (36-41): 6 points
            33, 7, 163, 144, 145, 153,
            # Left eye (42-47): 6 points
            362, 382, 381, 380, 374, 373,
            # Mouth outer (48-59): 12 points
            61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321,
            # Mouth inner (60-67): 8 points
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 324
        ]
        
        # Ensure we have exactly 68 indices
        if len(mp_to_dlib_indices) > 68:
            mp_to_dlib_indices = mp_to_dlib_indices[:68]
        elif len(mp_to_dlib_indices) < 68:
            # Pad with last valid index if needed
            while len(mp_to_dlib_indices) < 68:
                mp_to_dlib_indices.append(mp_to_dlib_indices[-1] if mp_to_dlib_indices else 0)
        
        pts = []
        for mp_idx in mp_to_dlib_indices:
            if mp_idx < len(landmarks_468):
                lm = landmarks_468[mp_idx]
                pts.append((int(lm.x * w), int(lm.y * h)))
            else:
                # Fallback to first landmark if index out of range
                pts.append((int(landmarks_468[0].x * w), int(landmarks_468[0].y * h)))
        
        return pts
    
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
    
    def _detect_face_alignment(self, image_path: Path) -> Optional[List[Tuple[int, int]]]:
        """Detect landmarks using face_alignment."""
        if not _FA_AVAILABLE:
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False)
        preds = fa.get_landmarks_from_image(img[..., ::-1])
        if not preds:
            return None
        
        return [(int(x), int(y)) for x, y in preds[0]]
    
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

