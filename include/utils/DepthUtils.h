#pragma once

#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>
#include "camera/CameraIntrinsics.h"

namespace face_reconstruction {

/**
 * Convert a single 2D pixel with depth to 3D point
 * @param u Pixel x coordinate
 * @param v Pixel y coordinate
 * @param depth Depth value in meters
 * @param intrinsics Camera intrinsics
 * @return 3D point (X, Y, Z)
 */
Eigen::Vector3d depthTo3D(int u, int v, double depth, const CameraIntrinsics& intrinsics);

/**
 * Backproject entire depth map to 3D point cloud
 * @param depth_map Depth image (CV_32F, single channel)
 * @param intrinsics Camera intrinsics
 * @param points Output vector of 3D points (only valid depths)
 * @param pixel_indices Output vector of (u, v) pixel coordinates corresponding to points
 * @param min_depth Minimum valid depth (meters)
 * @param max_depth Maximum valid depth (meters)
 */
void backprojectDepth(const cv::Mat& depth_map,
                      const CameraIntrinsics& intrinsics,
                      std::vector<Eigen::Vector3d>& points,
                      std::vector<std::pair<int, int>>& pixel_indices,
                      float min_depth = 0.1f,
                      float max_depth = 10.0f);

/**
 * Backproject depth map to organized point cloud (same size as depth map)
 * Invalid depths will be set to NaN
 * @param depth_map Depth image
 * @param intrinsics Camera intrinsics
 * @return Matrix of 3D points (height x width x 3)
 */
std::vector<std::vector<Eigen::Vector3d>> backprojectDepthOrganized(
    const cv::Mat& depth_map,
    const CameraIntrinsics& intrinsics);

} // namespace face_reconstruction
