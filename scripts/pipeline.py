#!/usr/bin/env python3
"""
Unified pipeline for the Biwi Kinect Head Pose dataset.

Capabilities:
- Download and extract the Biwi dataset from ETH Zürich (Kinect v1) mirror
- Convert RGB/depth frames into a standardized layout with intrinsics
- Run landmark detection (MediaPipe or dlib, optional)
- Call the C++ reconstruction binary to produce meshes
- Persist detailed logs, per-step summaries, and intermediate artifacts

The script is intentionally self-contained so it replaces the legacy
helper scripts in this repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import ssl
import subprocess
import tarfile
import zipfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Prefer system cert bundle; fall back to certifi if available
try:
    import certifi  # type: ignore

    _CERT_PATH = certifi.where()
except Exception:
    _CERT_PATH = None

try:
    import mediapipe as mp  # type: ignore

    _MEDIAPIPE_AVAILABLE = True
    _MEDIAPIPE_HAS_SOLUTIONS = hasattr(mp, "solutions")
except Exception:
    _MEDIAPIPE_AVAILABLE = False
    _MEDIAPIPE_HAS_SOLUTIONS = False

try:
    import dlib  # type: ignore

    _DLIB_AVAILABLE = True
except Exception:
    _DLIB_AVAILABLE = False


DATASET_URL = "https://data.vision.ee.ethz.ch/cvl/gfanelli/kinect_head_pose_db.tgz"
# Alternative mirrors (first one that succeeds is used)
ALT_DATASET_URLS = [
    DATASET_URL,
    # If ETH blocks/403s, try the GitHub mirror from dataset scripts (usually slower):
    "https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose/resolve/main/kinect_head_pose_db.tgz",
]
DEFAULT_MODEL_DIR = Path("data/model_biwi")
DEFAULT_RECON_BIN = Path("build/bin/test_real_data")


# --------------------------------------------------------------------------- #
# Logging utilities
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.debug("Logging initialized")
    return log_file


# --------------------------------------------------------------------------- #
# Download and extraction
# --------------------------------------------------------------------------- #
def _http_get(url: str, target: Path, ssl_ctx: ssl.SSLContext) -> bool:
    """
    Download with a user-agent header to reduce 403s from some mirrors.
    Returns True on success.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pip-like downloader)"})
    with urllib.request.urlopen(req, context=ssl_ctx) as resp, open(target, "wb") as f:
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
                logging.info("Download progress: %.1f%% (%d/%d bytes)", pct, downloaded, total)
    return True


def download_dataset(urls: List[str], download_dir: Path, force: bool = False) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "kinect_head_pose_db.tgz"

    if archive_path.exists() and not force:
        logging.info("Dataset archive already exists at %s (use --force-download to re-fetch)", archive_path)
        return archive_path

    # Build SSL context that works on macOS when system certs are not available
    ssl_ctx = ssl.create_default_context(cafile=_CERT_PATH) if _CERT_PATH else ssl.create_default_context()

    last_err: Optional[Exception] = None
    for url in urls:
        logging.info("Downloading Biwi dataset from %s", url)
        try:
            _http_get(url, archive_path, ssl_ctx)
            logging.info("Download complete: %s (%.2f MB)", archive_path, archive_path.stat().st_size / 1e6)
            return archive_path
        except Exception as exc:
            last_err = exc
            logging.warning("Download failed from %s: %s", url, exc)

    raise RuntimeError(f"All dataset URLs failed: {last_err}")


def download_from_kaggle(dataset: str, download_dir: Path) -> Path:
    """
    Download from Kaggle using the CLI. Requires kaggle API credentials in ~/.kaggle/kaggle.json.
    The Kaggle dataset typically ships as a .zip that contains the Biwi .tgz.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "kaggle_biwi.zip"

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(download_dir),
        "--force",
    ]
    try:
        logging.info("Downloading Biwi dataset from Kaggle (%s)...", dataset)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("kaggle CLI not found. Install with `pip install kaggle` and ensure it's on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        logging.error("Kaggle download failed: %s", exc.stderr)
        raise RuntimeError("Kaggle download failed. Check credentials in ~/.kaggle/kaggle.json.") from exc

    # Find the newest zip in the download dir (Kaggle names vary)
    candidates = sorted(download_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        zip_path = candidates[0]
    else:
        raise RuntimeError("Kaggle download did not produce a zip file.")

    logging.info("Kaggle zip downloaded: %s", zip_path)
    return zip_path


def _extract_zip_find_tgz(zip_path: Path, target_dir: Path) -> Path:
    """
    Extract a Kaggle zip; if it contains a .tgz, return that path.
    Otherwise, if it contains the hpdb folder directly, return that folder.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=target_dir)
        members = zf.namelist()
    tgz_candidates = [target_dir / m for m in members if m.lower().endswith((".tgz", ".tar.gz"))]
    if tgz_candidates:
        return tgz_candidates[0]
    # If no tgz, try to detect hpdb folder
    hpdb_candidates = [target_dir / m for m in members if m.strip("/").endswith("hpdb")]
    if hpdb_candidates:
        return hpdb_candidates[0]
    faces_candidates = [target_dir / m for m in members if m.strip("/").endswith("faces_0")]
    if faces_candidates:
        return faces_candidates[0]
    raise RuntimeError("Zip extracted but no .tgz or hpdb directory found.")


def extract_dataset(archive_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    extract_root = target_dir / "hpdb"

    preexisting_faces = target_dir / "faces_0"
    if extract_root.exists():
        logging.info("Dataset already extracted at %s", extract_root)
        return extract_root
    if preexisting_faces.exists():
        logging.info("Dataset already extracted at %s", preexisting_faces)
        return preexisting_faces

    logging.info("Extracting archive to %s", target_dir)

    actual_archive = archive_path
    if archive_path.suffix.lower() == ".zip":
        actual_archive = _extract_zip_find_tgz(archive_path, target_dir)

    if actual_archive.suffix.lower() in [".tgz", ".gz", ".tar"]:
        with tarfile.open(actual_archive, "r:*") as tar:
            tar.extractall(path=target_dir)
    elif actual_archive.is_dir():
        # Already an extracted folder (e.g., hpdb)
        pass
    else:
        raise RuntimeError(f"Unsupported archive format: {actual_archive}")

    # Detect root folder
    if extract_root.exists():
        return extract_root
    if preexisting_faces.exists():
        return preexisting_faces

    # Some Kaggle zips extract to faces_0 directly
    alt_faces = target_dir / "faces_0"
    if alt_faces.exists():
        return alt_faces

    raise RuntimeError(f"Expected extracted directory at {extract_root} or faces_0, but none was found.")

    logging.info("Extraction complete: %s", extract_root)
    return extract_root


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def read_biwi_calibration(cal_file: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse Biwi calibration file (depth.cal) to intrinsics.
    """
    try:
        with open(cal_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            fx = float(lines[0].strip().split()[0])
            cx = float(lines[0].strip().split()[2])
            fy = float(lines[1].strip().split()[1])
            cy = float(lines[1].strip().split()[2])
            return fx, fy, cx, cy
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Could not parse calibration file %s: %s", cal_file, exc)
        return None


def default_intrinsics() -> Tuple[float, float, float, float]:
    # Kinect v1 defaults for 640x480
    return 525.0, 525.0, 319.5, 239.5


def read_biwi_depth_binary(depth_path: Path) -> Optional[np.ndarray]:
    """
    Decode Biwi RLE depth binary format to uint16 image.
    """
    try:
        with open(depth_path, "rb") as f:
            header = f.read(8)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return None  # It's a PNG, handled elsewhere
            f.seek(0)
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            if width < 100 or width > 2000 or height < 100 or height > 2000:
                return None

            depth_img = np.zeros((height, width), dtype=np.int16)
            total = width * height
            p = 0
            while p < total:
                num_empty = np.fromfile(f, dtype=np.int32, count=1)
                if len(num_empty) == 0:
                    break
                p += int(num_empty[0])
                if p >= total:
                    break
                num_full = np.fromfile(f, dtype=np.int32, count=1)
                if len(num_full) == 0:
                    break
                num_full_val = int(num_full[0])
                if num_full_val > 0:
                    vals = np.fromfile(f, dtype=np.int16, count=num_full_val)
                    end_idx = min(p + num_full_val, total)
                    depth_img.flat[p:end_idx] = vals[: end_idx - p]
                    p += num_full_val
                else:
                    break

            return np.clip(depth_img, 0, 65535).astype(np.uint16)
    except Exception as exc:
        logging.debug("Binary depth decode failed for %s: %s", depth_path, exc)
        return None


def convert_depth_image(depth_path: Path, output_path: Path) -> bool:
    depth = read_biwi_depth_binary(depth_path)
    if depth is None:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if depth is None:
        logging.warning("Could not read depth file %s", depth_path)
        return False
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    if depth.dtype == np.uint8:
        depth = depth.astype(np.uint16) * 256
    elif depth.dtype in (np.float32, np.float64):
        depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    elif depth.dtype in (np.int32, np.int64):
        depth = np.clip(depth, 0, 65535).astype(np.uint16)
    depth = np.clip(depth, 0, 8000).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), depth)
    return True


def copy_rgb(rgb_path: Path, output_path: Path) -> bool:
    img = cv2.imread(str(rgb_path))
    if img is None:
        logging.warning("Could not read RGB file %s", rgb_path)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return True


def find_rgb_depth_pairs(seq_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Discover matching RGB/depth files inside a Biwi sequence directory.
    """
    rgb_candidates = list(seq_dir.glob("*rgb*.png")) + list((seq_dir / "rgb").glob("*.png"))
    depth_candidates = list(seq_dir.glob("*depth*.*")) + list((seq_dir / "depth").glob("*.*"))
    rgb_candidates = [p for p in rgb_candidates if p.is_file()]
    depth_candidates = [p for p in depth_candidates if p.is_file()]

    depth_map: Dict[Tuple[str, str], Path] = {}
    for depth in depth_candidates:
        stem = depth.stem.lower()
        for suffix in ("_depth", "-depth"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        depth_map[(str(depth.parent), stem)] = depth

    pairs: List[Tuple[Path, Path]] = []
    for rgb in sorted(rgb_candidates):
        stem = rgb.stem.lower()
        for suffix in ("_rgb", "-rgb"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        key = (str(rgb.parent), stem)
        if key in depth_map:
            pairs.append((rgb, depth_map[key]))
        else:
            fallback = rgb.parent / f"{stem}_depth.bin"
            if fallback.exists():
                pairs.append((rgb, fallback))

    return pairs


def convert_sequence(seq_dir: Path, output_dir: Path, max_frames: int, intrinsics_override: Optional[Tuple[float, float, float, float]]) -> Dict:
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
    success_rgb = success_depth = 0

    for idx, (rgb_path, depth_path) in enumerate(pairs):
        rgb_out = rgb_out_dir / f"frame_{idx:05d}.png"
        depth_out = depth_out_dir / f"frame_{idx:05d}.png"
        if copy_rgb(rgb_path, rgb_out):
            success_rgb += 1
        if convert_depth_image(depth_path, depth_out):
            success_depth += 1
        if (idx + 1) % 10 == 0 or idx == len(pairs) - 1:
            logging.info("Converted %d/%d frames in %s", idx + 1, len(pairs), seq_dir.name)

    return {
        "sequence": seq_dir.name,
        "frames_total": len(pairs),
        "rgb_ok": success_rgb,
        "depth_ok": success_depth,
        "intrinsics": intrinsics,
        "output_dir": str(output_dir),
    }


# --------------------------------------------------------------------------- #
# Landmark detection
# --------------------------------------------------------------------------- #
def detect_landmarks(image_path: Path, method: str = "mediapipe") -> Optional[List[Tuple[int, int]]]:
    if method == "mediapipe":
        if not (_MEDIAPIPE_AVAILABLE and _MEDIAPIPE_HAS_SOLUTIONS):
            logging.warning("MediaPipe not available/compatible, skipping landmarks for %s", image_path)
            return None
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning("Could not read image for landmarks: %s", image_path)
            return None
        h, w = img.shape[:2]
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            logging.warning("No face detected for %s", image_path)
            return None
        pts = []
        for lm in res.multi_face_landmarks[0].landmark[:68]:
            pts.append((int(lm.x * w), int(lm.y * h)))
        return pts
    if method == "dlib":
        if not _DLIB_AVAILABLE:
            logging.warning("dlib not available, skipping landmarks for %s", image_path)
            return None
        predictor_path = Path("data/models/shape_predictor_68_face_landmarks.dat")
        if not predictor_path.exists():
            logging.warning("dlib predictor missing at %s", predictor_path)
            return None
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))
        img = cv2.imread(str(image_path))
        if img is None:
            logging.warning("Could not read image for landmarks: %s", image_path)
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            logging.warning("No face detected for %s", image_path)
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = predictor(gray, face)
        return [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    logging.warning("Unknown landmark method: %s", method)
    return None


def save_landmarks(landmarks: List[Tuple[int, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for x, y in landmarks:
            f.write(f"{x} {y} -1\n")


def save_landmark_overlay(image_path: Path, landmarks: List[Tuple[int, int]], output_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(img, str(i), (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def run_reconstruction(binary: Path, model_dir: Path, intrinsics: Path, rgb: Path, depth: Path, output_mesh: Path, timeout: int) -> bool:
    binary = binary.resolve()
    if not binary.exists():
        logging.error("Reconstruction binary not found: %s", binary)
        return False
    cmd = [
        str(binary),
        "--rgb",
        str(rgb),
        "--depth",
        str(depth),
        "--intrinsics",
        str(intrinsics),
        "--model-dir",
        str(model_dir),
        "--output-mesh",
        str(output_mesh),
    ]
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        logging.debug("Reconstruction stdout: %s", res.stdout)
        logging.debug("Reconstruction stderr: %s", res.stderr)
        return output_mesh.exists()
    except subprocess.CalledProcessError as exc:
        logging.error("Reconstruction failed for %s: %s", rgb.name, exc)
        logging.debug("Stdout: %s", exc.stdout)
        logging.debug("Stderr: %s", exc.stderr)
    except subprocess.TimeoutExpired:
        logging.error("Reconstruction timed out for %s", rgb.name)
    return False


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end Biwi pipeline with logging and artifact tracking.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root data directory.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root output directory.")
    parser.add_argument("--max-frames", type=int, default=25, help="Limit frames per sequence (0 = all).")
    parser.add_argument("--sequence", type=str, default="", help="Process only a specific sequence id (e.g. 01).")
    parser.add_argument("--dataset-url", type=str, default=DATASET_URL, help="Primary dataset URL to try first.")
    parser.add_argument("--kaggle-dataset", type=str, default="", help="Kaggle dataset id (e.g. kmader/biwi-kinect-head-pose-database).")
    parser.add_argument("--skip-download", action="store_true", help="Assume dataset already present in data/biwi_download.")
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion step.")
    parser.add_argument("--skip-reconstruct", action="store_true", help="Skip reconstruction step.")
    parser.add_argument("--run-frames", type=int, default=5, help="How many converted frames to reconstruct.")
    parser.add_argument("--force-download", action="store_true", help="Re-download dataset even if archive exists.")
    parser.add_argument("--recon-binary", type=Path, default=DEFAULT_RECON_BIN, help="Path to reconstruction binary.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Model directory for reconstruction.")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), help="Override intrinsics.")
    parser.add_argument("--landmarks", choices=["none", "mediapipe", "dlib"], default="mediapipe", help="Landmark detector.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--timeout", type=int, default=60, help="Seconds before reconstruction command is aborted.")
    return parser


def orchestrate(args: argparse.Namespace) -> Dict:
    log_file = setup_logging(args.output_root / "logs", args.log_level)
    logging.info("Pipeline started. Logs: %s", log_file)
    summary: Dict[str, object] = {"log_file": str(log_file), "steps": [], "started_at": time.time()}

    download_dir = args.data_root / "biwi_download"
    converted_root = args.output_root / "converted"
    landmarks_root = args.output_root / "landmarks"
    meshes_root = args.output_root / "meshes"
    overlays_root = args.output_root / "overlays"

    # 1. Download
    if args.skip_download:
        logging.info("Skipping download step as requested.")
        archive_path = download_dir / "kinect_head_pose_db.tgz"
        preexisting_faces = download_dir / "faces_0"
        preexisting_hpdb = download_dir / "hpdb"
        if not archive_path.exists():
            if preexisting_faces.exists():
                archive_path = preexisting_faces  # treat extracted folder as archive input
                logging.info("Using existing extracted folder at %s", preexisting_faces)
            elif preexisting_hpdb.exists():
                archive_path = preexisting_hpdb
                logging.info("Using existing extracted folder at %s", preexisting_hpdb)
            else:
                raise RuntimeError(
                    f"--skip-download was set but no archive found at {archive_path} "
                    "and no extracted folders found in biwi_download. "
                    "Place the archive there or run without --skip-download."
                )
    else:
        if args.kaggle_dataset:
            archive_path = download_from_kaggle(args.kaggle_dataset, download_dir)
        else:
            # First try user-provided URL, then fall back to known list
            urls_to_try = [args.dataset_url] + [u for u in ALT_DATASET_URLS if u != args.dataset_url]
            archive_path = download_dataset(urls_to_try, download_dir, force=args.force_download)
    summary["steps"].append({"download": str(archive_path)})

    # 2. Extract
    raw_root = extract_dataset(archive_path, download_dir)
    summary["steps"].append({"extract": str(raw_root)})

    # 3. Conversion
    sequence_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and (not args.sequence or p.name == args.sequence)])
    if not sequence_dirs:
        raise RuntimeError("No Biwi sequences found to process.")

    conversion_reports = []
    if not args.skip_convert:
        for seq in sequence_dirs:
            report = convert_sequence(seq, converted_root / seq.name, args.max_frames, args.intrinsics)
            conversion_reports.append(report)
            logging.info("Converted sequence %s: %d/%d RGB, %d/%d depth",
                         report["sequence"], report["rgb_ok"], report["frames_total"], report["depth_ok"], report["frames_total"])
    else:
        logging.info("Conversion skipped.")
    summary["steps"].append({"conversion": conversion_reports})

    # 4. Landmarks (first frame per sequence)
    landmark_reports = []
    if args.landmarks != "none":
        for seq_report in conversion_reports:
            seq_dir = Path(seq_report["output_dir"])
            first_rgb = seq_dir / "rgb" / "frame_00000.png"
            if first_rgb.exists():
                try:
                    lm = detect_landmarks(first_rgb, args.landmarks)
                except Exception as exc:  # pragma: no cover - safety
                    logging.warning("Landmark detection failed for %s: %s", seq_dir.name, exc)
                    lm = None
                if lm:
                    lm_path = landmarks_root / seq_dir.name / "frame_00000.txt"
                    save_landmarks(lm, lm_path)
                    overlay_path = overlays_root / seq_dir.name / "frame_00000_overlay.png"
                    save_landmark_overlay(first_rgb, lm, overlay_path)
                    landmark_reports.append({"sequence": seq_dir.name, "landmarks_saved": str(lm_path), "overlay": str(overlay_path)})
                else:
                    landmark_reports.append({"sequence": seq_dir.name, "landmarks_saved": None})
    summary["steps"].append({"landmarks": landmark_reports})

    # 5. Reconstruction
    recon_reports = []
    if not args.skip_reconstruct:
        for seq_report in conversion_reports:
            seq_dir = Path(seq_report["output_dir"])
            intrinsics_path = seq_dir / "intrinsics.txt"
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            frames = sorted(rgb_dir.glob("frame_*.png"))[: args.run_frames]
            for frame in frames:
                depth_frame = depth_dir / frame.name
                mesh_out = meshes_root / seq_dir.name / f"{frame.stem}.ply"
                ok = run_reconstruction(args.recon_binary, args.model_dir, intrinsics_path, frame, depth_frame, mesh_out, args.timeout)
                recon_reports.append({"sequence": seq_dir.name, "frame": frame.name, "mesh": str(mesh_out), "success": ok})
    else:
        logging.info("Reconstruction skipped.")
    summary["steps"].append({"reconstruction": recon_reports})

    summary["finished_at"] = time.time()
    summary_path = args.output_root / "logs" / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Pipeline finished. Summary saved to %s", summary_path)
    return summary


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        orchestrate(args)
        return 0
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

