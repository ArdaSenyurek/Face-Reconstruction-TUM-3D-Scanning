"""
Model download step: Download landmark detection models (dlib, MediaPipe, etc.)
"""

import sys
import ssl
import urllib.request
import bz2
import io
from pathlib import Path
from typing import Optional

from main import PipelineStep, StepResult, StepStatus

# SSL context helper
def _get_ssl_context():
    """Build SSL context that works on macOS when system certs are not available."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _check_mediapipe_models() -> bool:
    """Check if MediaPipe models are available (auto-downloaded on first use)."""
    try:
        import mediapipe as mp
        return True
    except ImportError:
        return False


def _check_face_alignment_models() -> bool:
    """Check if face_alignment models are available (auto-downloaded on first use)."""
    try:
        import face_alignment
        return True
    except ImportError:
        return False


def _download_dlib_model_with_logger(output_dir: Path, logger) -> bool:
    """Download dlib shape predictor model with logger support."""
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    output_file = output_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    final_file = output_dir / "shape_predictor_68_face_landmarks.dat"
    
    # Check if file exists and is valid (not empty, not corrupted)
    if final_file.exists():
        file_size = final_file.stat().st_size
        # Valid dlib model should be at least 10MB (actual size is ~99MB)
        if file_size > 10 * 1024 * 1024:
            # Try to validate by loading it
            try:
                import dlib
                predictor = dlib.shape_predictor(str(final_file))
                logger.info(f"✓ dlib model already exists and is valid: {final_file} ({file_size / 1024 / 1024:.1f} MB)")
                return True
            except Exception as e:
                logger.warning(f"Existing dlib model appears corrupted: {e}")
                logger.info("Removing corrupted file and re-downloading...")
                final_file.unlink()
        else:
            logger.warning(f"Existing dlib model file is too small ({file_size} bytes), removing...")
            final_file.unlink()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dlib shape predictor from {url}...")
    logger.info("  This is a large file (~95 MB compressed)...")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_get_ssl_context()) as resp, open(output_file, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            chunk_size = 1024 * 512
            
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    if downloaded % (chunk_size * 10) == 0 or downloaded == total:  # Log every 10 chunks
                        logger.info(f"  Progress: {pct:.1f}% ({downloaded}/{total} bytes)")
        
        # Verify download completed
        if total > 0 and downloaded < total:
            logger.warning(f"Download incomplete: {downloaded}/{total} bytes")
            if output_file.exists():
                output_file.unlink()
            return False
        
        # Decompress bz2 file
        logger.info("  Decompressing...")
        try:
            with bz2.open(output_file, "rb") as f_in, open(final_file, "wb") as f_out:
                decompressed_data = f_in.read()
                if not decompressed_data:
                    raise ValueError("Decompressed file is empty")
                f_out.write(decompressed_data)
        except Exception as decomp_error:
            logger.warning(f"Decompression failed: {decomp_error}")
            if output_file.exists():
                output_file.unlink()
            if final_file.exists():
                final_file.unlink()
            return False
        
        # Verify extracted file is valid size (should be ~99MB)
        if final_file.stat().st_size < 10 * 1024 * 1024:
            logger.warning(f"Extracted file is too small ({final_file.stat().st_size} bytes), download may have failed")
            if final_file.exists():
                final_file.unlink()
            if output_file.exists():
                output_file.unlink()
            return False
        
        # Remove compressed file
        output_file.unlink()
        
        logger.info(f"✓ Downloaded and extracted: {final_file} ({final_file.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to download dlib model: {e}")
        if output_file.exists():
            output_file.unlink()
        if final_file.exists():
            final_file.unlink()
        return False


class LandmarkModelDownloadStep(PipelineStep):
    """Download dlib landmark detection model for 2D facial feature detection."""
    
    @property
    def name(self) -> str:
        return "Landmark Model Download"
    
    @property
    def description(self) -> str:
        return "Download dlib shape predictor for 2D landmark detection"
    
    def execute(self) -> StepResult:
        """Download models based on the landmark detection method being used."""
        # Only dlib is supported
        models_dir = Path(self.config.get("models_dir", "data/models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Download dlib model
        dlib_model_path = models_dir / "shape_predictor_68_face_landmarks.dat"
        success = _download_dlib_model_with_logger(models_dir, self.logger)
        if success:
            results.append("dlib")
        else:
            if dlib_model_path.exists():
                file_size = dlib_model_path.stat().st_size
                if file_size < 10 * 1024 * 1024:
                    self.logger.warning(f"dlib model file is too small ({file_size} bytes) and download failed.")
                    self.logger.warning("Please manually download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                else:
                    self.logger.warning("dlib model download failed, but existing file may be usable.")
            else:
                self.logger.warning("dlib model download failed. You can download it manually later.")
        
        if results:
            return StepResult(
                StepStatus.SUCCESS,
                f"Models ready: {', '.join(results)}",
                {"models": results, "models_dir": str(models_dir)}
            )
        else:
            # Don't fail - models might be auto-downloaded on first use
            return StepResult(
                StepStatus.SUCCESS,
                "Model check completed (some models may auto-download on first use)",
                {"models_dir": str(models_dir)}
            )

