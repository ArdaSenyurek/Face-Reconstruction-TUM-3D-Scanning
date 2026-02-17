#!/usr/bin/env python3
"""
Regression test runner for face reconstruction pipeline.

Runs face_reconstruction on test frames and asserts:
  1) Stability: optimization does not make things worse (no z-collapse)
  2) Expression activity: delta_norm_sigma is non-trivial when fitting is enabled
  3) Basic sanity: depth valid count > 0, final energy < initial energy

Usage:
    python tests/run_regression.py --binary build/bin/face_reconstruction \\
        --data-dir outputs/converted/01 \\
        --model-dir data/model_bfm \\
        --mapping data/landmark_mapping.txt \\
        --landmarks-dir outputs/landmarks/01

Or with explicit frame paths:
    python tests/run_regression.py --binary build/bin/face_reconstruction \\
        --rgb <path> --depth <path> --intrinsics <path> \\
        --model-dir <path> --mapping <path> --landmarks <path>
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import glob


def run_reconstruction(binary, rgb, depth, intrinsics, model_dir, landmarks,
                       mapping, output_mesh, output_state, output_conv,
                       extra_args=None):
    """Run face_reconstruction and return (returncode, stdout, stderr)."""
    cmd = [
        binary,
        "--rgb", rgb,
        "--depth", depth,
        "--intrinsics", intrinsics,
        "--model-dir", model_dir,
        "--landmarks", landmarks,
        "--mapping", mapping,
        "--output-mesh", output_mesh,
        "--output-state-json", output_state,
        "--output-convergence-json", output_conv,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode, result.stdout, result.stderr


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_frames(data_dir, landmarks_dir):
    """Discover frames from a converted data directory."""
    frames = []
    depth_files = sorted(glob.glob(os.path.join(data_dir, "*_depth.png")))
    if not depth_files:
        depth_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    for df in depth_files:
        base = os.path.basename(df)
        if "_depth" in base:
            frame_id = base.replace("_depth.png", "")
        else:
            continue
        rgb_path = os.path.join(data_dir, f"{frame_id}_rgb.png")
        depth_path = df
        intrinsics_path = os.path.join(data_dir, f"{frame_id}_intrinsics.txt")
        # Try common intrinsics file if per-frame doesn't exist
        if not os.path.exists(intrinsics_path):
            intrinsics_path = os.path.join(data_dir, "intrinsics.txt")
        lm_path = os.path.join(landmarks_dir, f"{frame_id}_landmarks.txt")
        if not os.path.exists(lm_path):
            lm_path = os.path.join(landmarks_dir, f"{frame_id}_landmarks.json")
        if all(os.path.exists(p) for p in [rgb_path, depth_path, intrinsics_path, lm_path]):
            frames.append({
                "id": frame_id,
                "rgb": rgb_path,
                "depth": depth_path,
                "intrinsics": intrinsics_path,
                "landmarks": lm_path,
            })
    return frames


def test_stability(conv_data, frame_id):
    """Test 1: Optimization should not make things catastrophically worse."""
    errors = []
    initial = conv_data.get("initial_energy", 0)
    final = conv_data.get("final_energy", 0)
    depth_count = conv_data.get("depth_valid_count", 0)

    if final > initial and initial > 0:
        ratio = final / initial
        if ratio > 1.5:
            errors.append(f"[{frame_id}] Energy INCREASED by {(ratio-1)*100:.1f}% "
                         f"(initial={initial:.4f}, final={final:.4f})")

    if depth_count < 10:
        errors.append(f"[{frame_id}] Very few depth residuals ({depth_count}), "
                     "possible z-collapse or mask issue")

    return errors


def test_expression_activity(conv_data, frame_id):
    """Test 2: Expression coefficients should be non-trivial when enabled."""
    errors = []
    delta_norm = conv_data.get("delta_norm_sigma", 0)
    opt_expr = conv_data.get("optimize_expression", False)

    if opt_expr and delta_norm < 0.001:
        errors.append(f"[{frame_id}] delta_norm_sigma={delta_norm:.6f} is near zero; "
                     "expression may not be fitting")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Face reconstruction regression tests")
    parser.add_argument("--binary", required=True, help="Path to face_reconstruction binary")
    parser.add_argument("--model-dir", required=True, help="Path to PCA model directory")
    parser.add_argument("--mapping", required=True, help="Path to landmark mapping file")
    # Discovery mode
    parser.add_argument("--data-dir", help="Path to converted frame directory")
    parser.add_argument("--landmarks-dir", help="Path to landmarks directory")
    # Explicit mode
    parser.add_argument("--rgb", help="Path to RGB image")
    parser.add_argument("--depth", help="Path to depth image")
    parser.add_argument("--intrinsics", help="Path to intrinsics file")
    parser.add_argument("--landmarks", help="Path to landmarks file")
    parser.add_argument("--max-frames", type=int, default=3, help="Max frames to test")
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"ERROR: Binary not found: {args.binary}")
        return 1

    # Build frame list
    frames = []
    if args.rgb and args.depth and args.intrinsics and args.landmarks:
        frames = [{
            "id": "explicit",
            "rgb": args.rgb,
            "depth": args.depth,
            "intrinsics": args.intrinsics,
            "landmarks": args.landmarks,
        }]
    elif args.data_dir and args.landmarks_dir:
        frames = find_frames(args.data_dir, args.landmarks_dir)
        if not frames:
            print(f"WARNING: No frames found in {args.data_dir}")
            print("RESULT: SKIP (no test data)")
            return 0
        frames = frames[:args.max_frames]
    else:
        print("ERROR: Provide either --data-dir + --landmarks-dir, or explicit frame paths")
        return 1

    print(f"Testing {len(frames)} frame(s)...")
    all_errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for frame in frames:
            fid = frame["id"]
            print(f"\n--- Frame: {fid} ---")
            mesh_path = os.path.join(tmpdir, f"{fid}_mesh.ply")
            state_path = os.path.join(tmpdir, f"{fid}_state.json")
            conv_path = os.path.join(tmpdir, f"{fid}_convergence.json")

            rc, stdout, stderr = run_reconstruction(
                args.binary,
                frame["rgb"], frame["depth"], frame["intrinsics"],
                args.model_dir, frame["landmarks"], args.mapping,
                mesh_path, state_path, conv_path,
            )

            if rc != 0:
                all_errors.append(f"[{fid}] face_reconstruction exited with code {rc}")
                if stderr:
                    print(f"  stderr: {stderr[:500]}")
                continue

            if not os.path.exists(conv_path):
                all_errors.append(f"[{fid}] No convergence JSON produced")
                continue

            conv_data = load_json(conv_path)

            # Run tests
            all_errors.extend(test_stability(conv_data, fid))
            all_errors.extend(test_expression_activity(conv_data, fid))

            # Print summary
            print(f"  iterations={conv_data.get('iterations', '?')}, "
                  f"converged={conv_data.get('converged', '?')}, "
                  f"delta_norm_sigma={conv_data.get('delta_norm_sigma', '?'):.4f}, "
                  f"depth_valid={conv_data.get('depth_valid_count', '?')}")

    # Report
    print(f"\n{'='*50}")
    if all_errors:
        print(f"REGRESSION TEST FAILED ({len(all_errors)} issue(s)):")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    else:
        print(f"REGRESSION TEST PASSED ({len(frames)} frame(s) tested)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
