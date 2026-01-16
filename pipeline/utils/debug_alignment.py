#!/usr/bin/env python3
"""
Debug Alignment Tool

Visualize mesh, point cloud, and landmarks to debug high RMSE values.
Creates overlay visualizations to identify alignment issues.

Usage:
    python pipeline/utils/debug_alignment.py --seq 12 --frame 0
    python pipeline/utils/debug_alignment.py --seq 03 --frame 0 --compare-seq 12
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import json


def load_ply_vertices(ply_path: Path) -> np.ndarray:
    """Load vertices from PLY file."""
    vertices = []
    in_header = True
    num_vertices = 0
    vertex_count = 0
    
    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line == 'end_header':
                    in_header = False
            else:
                if vertex_count < num_vertices:
                    parts = line.split()
                    if len(parts) >= 3:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        vertex_count += 1
    
    return np.array(vertices)


def load_intrinsics(intrinsics_path: Path) -> Tuple[float, float, float, float]:
    """Load camera intrinsics from file."""
    with open(intrinsics_path, 'r') as f:
        lines = f.readlines()
    
    fx, fy, cx, cy = 575.816, 575.816, 320.0, 240.0  # defaults
    
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=')
            key = key.strip()
            value = float(value.strip())
            if key == 'fx':
                fx = value
            elif key == 'fy':
                fy = value
            elif key == 'cx':
                cx = value
            elif key == 'cy':
                cy = value
    
    return fx, fy, cx, cy


def load_landmarks(landmarks_path: Path) -> np.ndarray:
    """Load 2D landmarks from file."""
    landmarks = []
    with open(landmarks_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    landmarks.append([float(parts[0]), float(parts[1])])
    return np.array(landmarks)


def load_landmark_mapping(mapping_path: Path) -> dict:
    """Load landmark to model vertex mapping."""
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    landmark_idx = int(parts[0])
                    vertex_idx = int(parts[1])
                    mapping[landmark_idx] = vertex_idx
    return mapping


def load_depth_as_pointcloud(depth_path: Path, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """Load depth image and convert to 3D point cloud."""
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return np.array([])
    
    # Convert to meters (assuming 16-bit PNG with 1000x scale)
    depth = depth_img.astype(np.float32) / 1000.0
    
    fx, fy, cx, cy = intrinsics
    
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            d = depth[v, u]
            if d > 0:
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                points.append([x, y, z])
    
    return np.array(points)


def project_3d_to_2d(points_3d: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """Project 3D points to 2D image coordinates."""
    fx, fy, cx, cy = intrinsics
    
    points_2d = []
    for p in points_3d:
        if p[2] > 0:  # Only project points in front of camera
            u = (p[0] * fx / p[2]) + cx
            v = (p[1] * fy / p[2]) + cy
            points_2d.append([u, v])
        else:
            points_2d.append([np.nan, np.nan])
    
    return np.array(points_2d)


def create_overlay_visualization(
    rgb_img: np.ndarray,
    mesh_vertices_2d: np.ndarray,
    landmarks_2d: np.ndarray,
    mapping: dict,
    title: str = ""
) -> np.ndarray:
    """Create visualization overlay showing mesh projection and landmarks."""
    vis = rgb_img.copy()
    h, w = vis.shape[:2]
    
    # Draw projected mesh vertices (blue, small dots)
    valid_mesh = mesh_vertices_2d[~np.isnan(mesh_vertices_2d).any(axis=1)]
    for p in valid_mesh[::10]:  # Sample every 10th vertex for clarity
        u, v = int(p[0]), int(p[1])
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(vis, (u, v), 1, (255, 100, 0), -1)
    
    # Draw detected landmarks (green circles)
    for i, lm in enumerate(landmarks_2d):
        u, v = int(lm[0]), int(lm[1])
        if 0 <= u < w and 0 <= v < h:
            color = (0, 255, 0)  # Green
            cv2.circle(vis, (u, v), 3, color, -1)
            cv2.putText(vis, str(i), (u+3, v-3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    # Draw mapping correspondences (red lines from landmark to mesh vertex)
    for lm_idx, vtx_idx in mapping.items():
        if lm_idx < len(landmarks_2d) and vtx_idx < len(mesh_vertices_2d):
            lm = landmarks_2d[lm_idx]
            vtx = mesh_vertices_2d[vtx_idx]
            
            if not np.isnan(vtx).any():
                lm_pt = (int(lm[0]), int(lm[1]))
                vtx_pt = (int(vtx[0]), int(vtx[1]))
                
                if all(0 <= p[0] < w and 0 <= p[1] < h for p in [lm_pt, vtx_pt]):
                    cv2.line(vis, lm_pt, vtx_pt, (0, 0, 255), 1)
                    cv2.circle(vis, vtx_pt, 4, (255, 0, 0), -1)  # Blue for mesh landmark
    
    # Add title
    if title:
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return vis


def compute_point_stats(points: np.ndarray, name: str) -> dict:
    """Compute and print statistics for a point set."""
    if len(points) == 0:
        return {"name": name, "count": 0}
    
    stats = {
        "name": name,
        "count": len(points),
        "x_range": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y_range": [float(points[:, 1].min()), float(points[:, 1].max())],
        "z_range": [float(points[:, 2].min()), float(points[:, 2].max())],
        "centroid": [float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 2].mean())],
    }
    
    print(f"\n=== {name} Statistics ===")
    print(f"  Points: {stats['count']}")
    print(f"  X range: [{stats['x_range'][0]:.4f}, {stats['x_range'][1]:.4f}]")
    print(f"  Y range: [{stats['y_range'][0]:.4f}, {stats['y_range'][1]:.4f}]")
    print(f"  Z range: [{stats['z_range'][0]:.4f}, {stats['z_range'][1]:.4f}]")
    print(f"  Centroid: ({stats['centroid'][0]:.4f}, {stats['centroid'][1]:.4f}, {stats['centroid'][2]:.4f})")
    
    return stats


def debug_frame(
    seq: str,
    frame: int,
    output_dir: Path,
    project_root: Path
) -> dict:
    """Debug a single frame."""
    print(f"\n{'='*60}")
    print(f"Debugging Sequence {seq}, Frame {frame}")
    print(f"{'='*60}")
    
    # Paths
    converted_dir = project_root / "outputs" / "converted" / seq
    rgb_path = converted_dir / "rgb" / f"frame_{frame:05d}.png"
    depth_path = converted_dir / "depth" / f"frame_{frame:05d}.png"
    intrinsics_path = converted_dir / "intrinsics.txt"
    landmarks_path = project_root / "outputs" / "landmarks" / seq / f"frame_{frame:05d}.txt"
    mesh_path = project_root / "outputs" / "meshes" / seq / f"frame_{frame:05d}.ply"
    pose_init_path = project_root / "outputs" / "pose_init" / seq / f"frame_{frame:05d}_aligned.ply"
    mapping_path = project_root / "data" / "landmark_mapping.txt"
    
    results = {"seq": seq, "frame": frame, "errors": []}
    
    # Check files exist
    for name, path in [("RGB", rgb_path), ("Depth", depth_path), ("Intrinsics", intrinsics_path),
                       ("Landmarks", landmarks_path), ("Mapping", mapping_path)]:
        if not path.exists():
            results["errors"].append(f"{name} not found: {path}")
            print(f"ERROR: {name} not found: {path}")
    
    if results["errors"]:
        return results
    
    # Load data
    print("\nLoading data...")
    rgb_img = cv2.imread(str(rgb_path))
    intrinsics = load_intrinsics(intrinsics_path)
    landmarks_2d = load_landmarks(landmarks_path)
    mapping = load_landmark_mapping(mapping_path)
    
    print(f"  RGB: {rgb_img.shape}")
    print(f"  Intrinsics: fx={intrinsics[0]}, fy={intrinsics[1]}, cx={intrinsics[2]}, cy={intrinsics[3]}")
    print(f"  Landmarks: {len(landmarks_2d)}")
    print(f"  Mapping: {len(mapping)} entries")
    
    # Load point cloud from depth
    print("\nLoading point cloud from depth...")
    pointcloud = load_depth_as_pointcloud(depth_path, intrinsics)
    pc_stats = compute_point_stats(pointcloud, "Point Cloud (from depth)")
    results["pointcloud_stats"] = pc_stats
    
    # Load reconstructed mesh
    if mesh_path.exists():
        print("\nLoading reconstructed mesh...")
        mesh_vertices = load_ply_vertices(mesh_path)
        mesh_stats = compute_point_stats(mesh_vertices, "Reconstructed Mesh")
        results["mesh_stats"] = mesh_stats
        
        # Compute distance between centroids
        if pc_stats["count"] > 0 and mesh_stats["count"] > 0:
            pc_centroid = np.array(pc_stats["centroid"])
            mesh_centroid = np.array(mesh_stats["centroid"])
            centroid_dist = np.linalg.norm(pc_centroid - mesh_centroid)
            print(f"\n  Centroid distance (mesh to cloud): {centroid_dist:.4f} m")
            results["centroid_distance"] = centroid_dist
    else:
        print(f"\nWARNING: Mesh not found: {mesh_path}")
        mesh_vertices = None
    
    # Load pose_init mesh for comparison
    if pose_init_path.exists():
        print("\nLoading pose_init mesh...")
        pose_init_vertices = load_ply_vertices(pose_init_path)
        pose_stats = compute_point_stats(pose_init_vertices, "Pose Init Mesh")
        results["pose_init_stats"] = pose_stats
    else:
        pose_init_vertices = None
    
    # Create visualizations
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overlay: Mesh projected onto RGB
    if mesh_vertices is not None:
        mesh_2d = project_3d_to_2d(mesh_vertices, intrinsics)
        vis_mesh = create_overlay_visualization(
            rgb_img, mesh_2d, landmarks_2d, mapping,
            f"Seq {seq} Frame {frame} - Mesh Overlay"
        )
        mesh_overlay_path = output_dir / f"seq{seq}_frame{frame}_mesh_overlay.png"
        cv2.imwrite(str(mesh_overlay_path), vis_mesh)
        print(f"\nSaved: {mesh_overlay_path}")
        results["mesh_overlay"] = str(mesh_overlay_path)
    
    # 2. Overlay: Pose init projected onto RGB
    if pose_init_vertices is not None:
        pose_2d = project_3d_to_2d(pose_init_vertices, intrinsics)
        vis_pose = create_overlay_visualization(
            rgb_img, pose_2d, landmarks_2d, mapping,
            f"Seq {seq} Frame {frame} - Pose Init Overlay"
        )
        pose_overlay_path = output_dir / f"seq{seq}_frame{frame}_pose_init_overlay.png"
        cv2.imwrite(str(pose_overlay_path), vis_pose)
        print(f"Saved: {pose_overlay_path}")
        results["pose_init_overlay"] = str(pose_overlay_path)
    
    # 3. Landmark-only visualization
    vis_landmarks = rgb_img.copy()
    for i, lm in enumerate(landmarks_2d):
        u, v = int(lm[0]), int(lm[1])
        cv2.circle(vis_landmarks, (u, v), 4, (0, 255, 0), -1)
        cv2.putText(vis_landmarks, str(i), (u+4, v-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    landmarks_path_out = output_dir / f"seq{seq}_frame{frame}_landmarks.png"
    cv2.imwrite(str(landmarks_path_out), vis_landmarks)
    print(f"Saved: {landmarks_path_out}")
    results["landmarks_image"] = str(landmarks_path_out)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Debug alignment visualization")
    parser.add_argument("--seq", type=str, required=True, help="Sequence number (e.g., 12)")
    parser.add_argument("--frame", type=int, default=0, help="Frame number (default: 0)")
    parser.add_argument("--compare-seq", type=str, help="Compare with another sequence")
    parser.add_argument("--output-dir", type=str, default="outputs/debug", help="Output directory")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    
    # Debug primary sequence
    results = [debug_frame(args.seq, args.frame, output_dir, project_root)]
    
    # Compare sequence if specified
    if args.compare_seq:
        results.append(debug_frame(args.compare_seq, args.frame, output_dir, project_root))
    
    # Save results
    results_path = output_dir / "debug_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary comparison
    if len(results) == 2 and all("centroid_distance" in r for r in results):
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"Seq {r['seq']} Frame {r['frame']}:")
            print(f"  Mesh-to-cloud centroid distance: {r['centroid_distance']:.4f} m")
            if "mesh_stats" in r:
                print(f"  Mesh centroid: {r['mesh_stats']['centroid']}")
            if "pointcloud_stats" in r:
                print(f"  Cloud centroid: {r['pointcloud_stats']['centroid']}")


if __name__ == "__main__":
    main()
