"""
Download step for the Biwi dataset using kagglehub.
"""

import shutil
from pathlib import Path

from pipeline import PipelineStep, StepResult, StepStatus


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
        
        if skip:
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
        
        try:
            self.logger.info(f"Downloading dataset from Kaggle: {dataset_id}")
            cache_path = kagglehub.dataset_download(dataset_id)
            self.logger.info(f"Dataset downloaded to cache: {cache_path}")
            
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
            
            # Check for PLY files (for model creation)
            ply_files = self._check_ply_files(extract_root)
            if ply_files:
                self.logger.info(f"Found {len(ply_files)} PLY point cloud file(s) - model can be created automatically")
            else:
                self.logger.warning("No PLY files found in dataset - model creation may require manual setup")
            
            return StepResult(
                StepStatus.SUCCESS,
                f"Dataset ready at {extract_root}",
                {"extract_root": str(extract_root), "dataset_path": str(download_dir)}
            )
            
        except Exception as e:
            error_msg = f"Failed to download dataset: {e}"
            self.logger.error(error_msg)
            return StepResult(StepStatus.FAILED, error_msg, {"exception": str(e)})
    
    def _check_ply_files(self, root: Path) -> list[Path]:
        """Check for PLY point cloud files in dataset."""
        search_patterns = [
            "hpdb/*/*.ply",
            "faces_0/*/*.ply",
            "**/*.ply",
        ]
        
        ply_files = []
        for pattern in search_patterns:
            try:
                found = list(root.glob(pattern))
                ply_files.extend(found)
            except Exception:
                # Pattern might not match, continue
                pass
        
        # Remove duplicates and sort
        ply_files = sorted(set(ply_files))
        return ply_files
