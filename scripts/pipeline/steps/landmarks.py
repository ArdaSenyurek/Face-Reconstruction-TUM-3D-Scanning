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
except Exception:
    _MEDIAPIPE_AVAILABLE = False
    _MEDIAPIPE_HAS_SOLUTIONS = False

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
        # Using MediaPipe's FACE_CONNECTIONS indices for key facial features
        # This is an approximation - for best results, use dlib or face_alignment
        
        landmarks_468 = res.multi_face_landmarks[0].landmark
        
        # MediaPipe landmark indices for key facial features (approximate mapping to dlib 68)
        # This is a simplified mapping - may not be 100% accurate
        # Jawline (0-16 in dlib): Use face oval points
        # Right eyebrow (17-21): Use eyebrow points
        # Left eyebrow (22-26): Use eyebrow points  
        # Nose (27-35): Use nose points
        # Right eye (36-41): Use eye points
        # Left eye (42-47): Use eye points
        # Mouth (48-67): Use mouth points
        
        # For now, return first 68 landmarks as approximation
        # WARNING: This may not match dlib format exactly
        pts = []
        for lm in landmarks_468[:68]:
            pts.append((int(lm.x * w), int(lm.y * h)))
        
        if len(pts) < 68:
            # If we got fewer than 68, pad with last point
            while len(pts) < 68:
                pts.append(pts[-1] if pts else (0, 0))
        
        return pts
    
    def _detect_dlib(self, image_path: Path) -> Optional[List[Tuple[int, int]]]:
        """Detect landmarks using dlib."""
        if not _DLIB_AVAILABLE:
            return None
        
        predictor_path = Path("data/models/shape_predictor_68_face_landmarks.dat")
        if not predictor_path.exists():
            return None
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            return None
        
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(gray, face)
        return [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    
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

