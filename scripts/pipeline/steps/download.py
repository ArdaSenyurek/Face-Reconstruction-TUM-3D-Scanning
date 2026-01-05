"""
Download and extraction step for the Biwi dataset.
"""

import subprocess
import tarfile
from pathlib import Path
from typing import List

from pipeline import PipelineStep, StepResult, StepStatus, extract_zip_find_tgz, get_ssl_context, http_get


class DownloadStep(PipelineStep):
    """Download and extract the Biwi Kinect Head Pose dataset."""
    
    @property
    def name(self) -> str:
        return "Download & Extract"
    
    @property
    def description(self) -> str:
        return "Download and extract Biwi dataset"
    
    def execute(self) -> StepResult:
        """Download and extract the dataset."""
        download_dir = Path(self.config["download_dir"])
        dataset_urls = self.config.get("dataset_urls", [])
        kaggle_dataset = self.config.get("kaggle_dataset")
        force = self.config.get("force", False)
        skip = self.config.get("skip", False)
        
        if skip:
            # Check for existing extracted data
            extract_root = download_dir / "hpdb"
            preexisting_faces = download_dir / "faces_0"
            if extract_root.exists():
                return StepResult(StepStatus.SUCCESS, f"Using existing data at {extract_root}", 
                               {"archive_path": extract_root})
            if preexisting_faces.exists():
                return StepResult(StepStatus.SUCCESS, f"Using existing data at {preexisting_faces}",
                               {"archive_path": preexisting_faces})
            return StepResult(StepStatus.SKIPPED, "No existing data found and download skipped")
        
        # Download
        archive_path = download_dir / "kinect_head_pose_db.tgz"
        
        if archive_path.exists() and not force:
            self.logger.info(f"Dataset archive already exists at {archive_path}")
        else:
            if kaggle_dataset:
                archive_path = self._download_from_kaggle(kaggle_dataset, download_dir)
            else:
                archive_path = self._download_from_urls(dataset_urls, download_dir, force)
        
        # Extract
        extract_root = self._extract_dataset(archive_path, download_dir)
        
        return StepResult(StepStatus.SUCCESS, f"Dataset ready at {extract_root}",
                        {"archive_path": archive_path, "extract_root": str(extract_root)})
    
    def _download_from_urls(self, urls: List[str], download_dir: Path, force: bool) -> Path:
        """Download from a list of URLs."""
        download_dir.mkdir(parents=True, exist_ok=True)
        archive_path = download_dir / "kinect_head_pose_db.tgz"
        
        ssl_ctx = get_ssl_context()
        last_err = None
        
        for url in urls:
            self.logger.info(f"Downloading from {url}")
            try:
                http_get(url, archive_path, ssl_ctx)
                size_mb = archive_path.stat().st_size / 1e6
                self.logger.info(f"Download complete: {archive_path} ({size_mb:.2f} MB)")
                return archive_path
            except Exception as exc:
                last_err = exc
                self.logger.warning(f"Download failed from {url}: {exc}")
        
        raise RuntimeError(f"All dataset URLs failed: {last_err}")
    
    def _download_from_kaggle(self, dataset: str, download_dir: Path) -> Path:
        """Download from Kaggle using CLI."""
        download_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "kaggle", "datasets", "download", "-d", dataset,
            "-p", str(download_dir), "--force"
        ]
        
        try:
            self.logger.info(f"Downloading from Kaggle ({dataset})...")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("kaggle CLI not found. Install with `pip install kaggle`")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Kaggle download failed: {exc.stderr}")
        
        candidates = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("Kaggle download did not produce a zip file")
        
        return candidates[0]
    
    def _extract_dataset(self, archive_path: Path, target_dir: Path) -> Path:
        """Extract the dataset archive."""
        target_dir.mkdir(parents=True, exist_ok=True)
        extract_root = target_dir / "hpdb"
        preexisting_faces = target_dir / "faces_0"
        
        if extract_root.exists():
            self.logger.info(f"Dataset already extracted at {extract_root}")
            return extract_root
        if preexisting_faces.exists():
            self.logger.info(f"Dataset already extracted at {preexisting_faces}")
            return preexisting_faces
        
        self.logger.info(f"Extracting archive to {target_dir}")
        
        actual_archive = archive_path
        if archive_path.suffix.lower() == ".zip":
            actual_archive = extract_zip_find_tgz(archive_path, target_dir)
        
        if actual_archive.suffix.lower() in [".tgz", ".gz", ".tar"]:
            with tarfile.open(actual_archive, "r:*") as tar:
                tar.extractall(path=target_dir)
        elif actual_archive.is_dir():
            pass  # Already extracted
        else:
            raise RuntimeError(f"Unsupported archive format: {actual_archive}")
        
        # Detect root folder
        if extract_root.exists():
            return extract_root
        if preexisting_faces.exists():
            return preexisting_faces
        
        alt_faces = target_dir / "faces_0"
        if alt_faces.exists():
            return alt_faces
        
        raise RuntimeError(f"Expected extracted directory at {extract_root} or faces_0, but none was found.")

