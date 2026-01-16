"""
Download step for the Biwi dataset using kagglehub.
"""

import shutil
from pathlib import Path

from main import PipelineStep, StepResult, StepStatus


class DownloadStep(PipelineStep):
    """Download the Biwi Kinect Head Pose dataset using kagglehub."""
    
    @property
    def name(self) -> str:
        return "Download & Extract"
    
    @property
    def description(self) -> str:
        return "Download Biwi dataset using kagglehub"
    
    def execute(self) -> StepResult:
        """Download the dataset using kagglehub."""
        skip = self.config.get("skip", False)
        use_local_cache_only = self.config.get("use_local_cache_only", False)
        download_retries = max(0, int(self.config.get("download_retries", 0)))
        
        if skip or use_local_cache_only:
            # Check for existing extracted data
            download_dir = Path(self.config.get("download_dir", "data/biwi_download"))
            extract_root = download_dir / "hpdb"
            preexisting_faces = download_dir / "faces_0"
            
            if extract_root.exists():
                return StepResult(StepStatus.SUCCESS, f"Using existing data at {extract_root}", 
                               {"extract_root": str(extract_root)})
            if preexisting_faces.exists():
                return StepResult(StepStatus.SUCCESS, f"Using existing data at {preexisting_faces}",
                               {"extract_root": str(preexisting_faces)})
            if use_local_cache_only:
                return StepResult(StepStatus.FAILED, "Local cache only requested but dataset not found")
            return StepResult(StepStatus.SKIPPED, "No existing data found and download skipped")
        
        try:
            import kagglehub
        except ImportError:
            return StepResult(
                StepStatus.FAILED,
                "kagglehub not installed. Install with: pip install kagglehub",
                {}
            )
        
        # Get target directory in project
        download_dir = Path(self.config.get("download_dir", "data/biwi_download"))
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset using kagglehub
        dataset_id = self.config.get("kaggle_dataset", "kmader/biwi-kinect-head-pose-database")
        
        cache_path = None
        last_error = None
        for attempt in range(1, download_retries + 2):
            try:
                self.logger.info(f"Downloading dataset from Kaggle (attempt {attempt}): {dataset_id}")
                cache_path = kagglehub.dataset_download(dataset_id)
                self.logger.info(f"Dataset downloaded to cache: {cache_path}")
                break
            except Exception as e:
                last_error = e
                if attempt <= download_retries:
                    self.logger.warning(f"Download attempt {attempt} failed: {e}")
                else:
                    self.logger.error(f"All download attempts failed: {e}")
        
        if cache_path is None:
            return StepResult(
                StepStatus.FAILED,
                f"Failed to download dataset after {download_retries + 1} attempt(s): {last_error}",
                {"exception": str(last_error) if last_error else None}
            )

        # Convert to Path and find the actual dataset root in cache
        cache_dataset_path = Path(cache_path)
        
        # Find the dataset root (hpdb or faces_0) in cache
        cache_extract_root = None
        if (cache_dataset_path / "hpdb").exists():
            cache_extract_root = cache_dataset_path / "hpdb"
        elif (cache_dataset_path / "faces_0").exists():
            cache_extract_root = cache_dataset_path / "faces_0"
        else:
            # Check subdirectories
            for subdir in cache_dataset_path.iterdir():
                if subdir.is_dir():
                    if (subdir / "hpdb").exists():
                        cache_extract_root = subdir / "hpdb"
                        break
                    elif (subdir / "faces_0").exists():
                        cache_extract_root = subdir / "faces_0"
                        break
                    elif subdir.name == "hpdb" or subdir.name == "faces_0":
                        cache_extract_root = subdir
                        break
            
            # If still not found, use the path itself
            if cache_extract_root is None:
                cache_extract_root = cache_dataset_path
        
        # Copy dataset to project data folder
        self.logger.info(f"Copying dataset to project directory: {download_dir}")
        
        # Determine what to copy (hpdb or faces_0)
        if cache_extract_root.name == "hpdb":
            target_dir = download_dir / "hpdb"
        elif cache_extract_root.name == "faces_0":
            target_dir = download_dir / "faces_0"
        else:
            # If cache_extract_root is the dataset root, copy everything
            target_dir = download_dir
            cache_extract_root = cache_dataset_path
        
        # Remove target if it exists (to ensure clean copy)
        if target_dir.exists():
            self.logger.info(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        
        # Copy the dataset
        shutil.copytree(cache_extract_root, target_dir)
        self.logger.info(f"Dataset copied to: {target_dir}")
        
        # Set extract_root to the project-local path
        extract_root = target_dir
        
        return StepResult(
            StepStatus.SUCCESS,
            f"Dataset ready at {extract_root}",
            {"extract_root": str(extract_root), "dataset_path": str(download_dir)}
        )
