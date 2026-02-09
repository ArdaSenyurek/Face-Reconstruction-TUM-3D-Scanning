#!/usr/bin/env python3
"""
Create a minimal copy of outputs/ for upload (e.g. Google Drive).

Rule:
  - Sequence 01: every frame (all frames present in outputs).
  - All other sequences: first frame only (frame_00000).

Preserves the same directory structure; omits pointclouds and runtime_meshes to keep size small.
"""

import argparse
import shutil
from pathlib import Path


def get_converted_frames(converted_root: Path, seq_id: str) -> list[str]:
    """Return sorted frame stems (e.g. frame_00000) for a sequence from converted/rgb or depth."""
    rgb_dir = converted_root / seq_id / "rgb"
    depth_dir = converted_root / seq_id / "depth"
    if rgb_dir.exists():
        stems = [p.stem for p in rgb_dir.glob("*.png")]
    elif depth_dir.exists():
        stems = [p.stem for p in depth_dir.glob("*.png")]
    else:
        return []
    return sorted(stems)


def get_sequences(converted_root: Path) -> list[str]:
    """Return sorted sequence IDs that have a converted dir with rgb or depth."""
    seqs = []
    for d in converted_root.iterdir():
        if not d.is_dir():
            continue
        if (d / "rgb").exists() or (d / "depth").exists():
            seqs.append(d.name)
    return sorted(seqs)


def frames_to_copy(converted_root: Path, seq_id: str, full_sequence_id: str = "01") -> list[str]:
    """
    Frames to include for this sequence.
    - For full_sequence_id (default "01"): all frames.
    - For others: first frame only.
    """
    all_frames = get_converted_frames(converted_root, seq_id)
    if not all_frames:
        return []
    if seq_id == full_sequence_id:
        return all_frames
    return [all_frames[0]]  # first frame only


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create minimal outputs bundle for upload: sequence 01 = all frames, others = first frame only."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_for_upload"),
        help="Output directory (default: outputs_for_upload)",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("outputs"),
        help="Source outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--full-sequence",
        type=str,
        default="01",
        help="Sequence ID that gets every frame (default: 01); others get first frame only",
    )
    parser.add_argument(
        "--optimized-only",
        action="store_true",
        help="Bundle only meshes *_optimized.ply (and fallback .ply); same other outputs",
    )
    args = parser.parse_args()

    src = args.inputs.resolve()
    dst = args.output_dir.resolve()
    if not src.exists():
        raise SystemExit(f"Input directory does not exist: {src}")

    converted_root = src / "converted"
    if not converted_root.exists():
        raise SystemExit(f"Missing {converted_root}")

    sequences = get_sequences(converted_root)
    if not sequences:
        raise SystemExit("No sequences found under outputs/converted")

    # Build (seq_id -> list of frame stems)
    seq_frames: dict[str, list[str]] = {}
    for seq_id in sequences:
        frames = frames_to_copy(converted_root, seq_id, args.full_sequence)
        if frames:
            seq_frames[seq_id] = frames

    count = 0
    optimized_only = getattr(args, "optimized_only", False)

    # analysis/: metrics.json, overlay_checks.json; depth residual/vis
    analysis_src = src / "analysis"
    analysis_dst = dst / "analysis"
    if analysis_src.exists():
        for f in ["metrics.json", "overlay_checks.json"]:
            if copy_file(analysis_src / f, analysis_dst / f):
                count += 1
        for seq_id, frames in seq_frames.items():
            # depth_residual_vis/<seq>/<frame>_residual.png
            res_dir = analysis_src / "depth_residual_vis" / seq_id
            if res_dir.exists():
                for frame_stem in frames:
                    f = f"{frame_stem}_residual.png"
                    if copy_file(res_dir / f, analysis_dst / "depth_residual_vis" / seq_id / f):
                        count += 1
            # depth_vis: every frame for full_sequence (01), first frame only for others
            depth_vis_dir = analysis_src / "depth_vis" / seq_id
            if depth_vis_dir.exists():
                for frame_stem in frames:
                    p = depth_vis_dir / f"{frame_stem}.png"
                    if p.exists():
                        if copy_file(p, analysis_dst / "depth_vis" / seq_id / f"{frame_stem}.png"):
                            count += 1

    # converted/<seq>/: rgb, depth, intrinsics; no pointclouds; one depth_vis per seq (first frame)
    converted_src = src / "converted"
    converted_dst = dst / "converted"
    for seq_id, frames in seq_frames.items():
        for frame_stem in frames:
            for sub in ["rgb", "depth"]:
                d = converted_src / seq_id / sub
                if d.exists():
                    f = f"{frame_stem}.png"
                    if copy_file(d / f, converted_dst / seq_id / sub / f):
                        count += 1
        intrinsics = converted_src / seq_id / "intrinsics.txt"
        if intrinsics.exists():
            if copy_file(intrinsics, converted_dst / seq_id / "intrinsics.txt"):
                count += 1
        depth_vis = converted_src / seq_id / "depth_vis"
        if depth_vis.exists():
            for frame_stem in frames:
                f = f"{frame_stem}.png"
                if (depth_vis / f).exists():
                    if copy_file(depth_vis / f, converted_dst / seq_id / "depth_vis" / f):
                        count += 1

    # landmarks/<seq>/<frame>.txt
    lm_src = src / "landmarks"
    lm_dst = dst / "landmarks"
    if lm_src.exists():
        for seq_id, frames in seq_frames.items():
            for frame_stem in frames:
                if copy_file(lm_src / seq_id / f"{frame_stem}.txt", lm_dst / seq_id / f"{frame_stem}.txt"):
                    count += 1

    # pose_init/<seq>/: rigid_report, aligned.ply, overlay PNG per frame
    pose_src = src / "pose_init"
    pose_dst = dst / "pose_init"
    if pose_src.exists():
        for seq_id, frames in seq_frames.items():
            for frame_stem in frames:
                for name in [f"{frame_stem}_rigid_report.json", f"{frame_stem}_aligned.ply", f"{frame_stem}_overlay.png"]:
                    if copy_file(pose_src / seq_id / name, pose_dst / seq_id / name):
                        count += 1

    # meshes/<seq>/: *_optimized.ply and fallback .ply
    mesh_src = src / "meshes"
    mesh_dst = dst / "meshes"
    if mesh_src.exists():
        for seq_id, frames in seq_frames.items():
            for frame_stem in frames:
                for name in [f"{frame_stem}_optimized.ply", f"{frame_stem}.ply"]:
                    if copy_file(mesh_src / seq_id / name, mesh_dst / seq_id / name):
                        count += 1

    # overlays_3d/<seq>/: overlay_metrics.json and overlay PLYs/scan/mesh
    ov3_src = src / "overlays_3d"
    ov3_dst = dst / "overlays_3d"
    if ov3_src.exists():
        for seq_id, frames in seq_frames.items():
            for frame_stem in frames:
                if copy_file(ov3_src / seq_id / f"{frame_stem}_overlay_metrics.json", ov3_dst / seq_id / f"{frame_stem}_overlay_metrics.json"):
                    count += 1
            for frame_stem in frames:
                for name in [f"{frame_stem}_overlay_rigid.ply", f"{frame_stem}_overlay_opt.ply", f"{frame_stem}_scan.ply", f"{frame_stem}_mesh.ply"]:
                    if copy_file(ov3_src / seq_id / name, ov3_dst / seq_id / name):
                        count += 1

    # overlays/<seq>/: 2D overlay PNGs — every frame for full_sequence, first frame only for others
    ov_src = src / "overlays"
    ov_dst = dst / "overlays"
    if ov_src.exists():
        for seq_id, frames in seq_frames.items():
            if not frames:
                continue
            seq_ov = ov_src / seq_id
            if not seq_ov.exists():
                continue
            for frame_stem in frames:
                for p in seq_ov.glob(f"{frame_stem}*.png"):
                    if copy_file(p, ov_dst / seq_id / p.name):
                        count += 1
                    break

    # logs/
    logs_src = src / "logs"
    logs_dst = dst / "logs"
    if logs_src.exists():
        for f in ["pipeline_summary.json", "conversion_reports.json"]:
            if copy_file(logs_src / f, logs_dst / f):
                count += 1
        logs = sorted(logs_src.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            if copy_file(logs[0], logs_dst / logs[0].name):
                count += 1

    # README in bundle
    readme = dst / "README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    mode_note = " (optimized-only: meshes *_optimized.ply and .ply only)" if optimized_only else ""
    readme.write_text(
        "Minimal outputs bundle for upload (e.g. Google Drive)\n"
        "====================================================\n\n"
        f"Rule: sequence {args.full_sequence} = every frame; all other sequences = first frame only.{mode_note}\n\n"
        "Generated by:\n"
        "  python scripts/create_upload_bundle.py"
        + (" --optimized-only" if optimized_only else "") + "\n\n"
        "Contents: analysis (metrics.json, overlay_checks.json, depth residual/vis), "
        "converted (rgb, depth, intrinsics), landmarks, pose_init, meshes, overlays_3d, overlays, logs.\n"
        "Omitted: pointclouds, runtime_meshes.\n\n"
        "Repository: https://github.com/ArdaSenyurek/Face-Reconstruction-TUM-3D-Scanning\n",
        encoding="utf-8",
    )
    count += 1

    print(f"Created {dst} with {count} files." + (" (optimized-only)" if optimized_only else ""))
    print(f"Sequences: {list(seq_frames.keys())}")
    for seq_id, frames in seq_frames.items():
        print(f"  {seq_id}: {len(frames)} frame(s)" + (" (all)" if seq_id == args.full_sequence else " (first only)"))


if __name__ == "__main__":
    main()
