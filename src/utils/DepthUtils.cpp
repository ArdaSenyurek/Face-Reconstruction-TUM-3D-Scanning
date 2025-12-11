#include "utils/DepthUtils.h"
#include <cmath>

namespace face_reconstruction {

Eigen::Vector3d depthTo3D(int u, int v, double depth, const CameraIntrinsics& intrinsics) {
    Eigen::Vector3d point;
    point.z() = depth;
    point.x() = (u - intrinsics.cx) * depth / intrinsics.fx;
    point.y() = (v - intrinsics.cy) * depth / intrinsics.fy;
    return point;
}

void backprojectDepth(const cv::Mat& depth_map,
                      const CameraIntrinsics& intrinsics,
                      std::vector<Eigen::Vector3d>& points,
                      std::vector<std::pair<int, int>>& pixel_indices,
                      float min_depth,
                      float max_depth) {
    points.clear();
    pixel_indices.clear();
    
    if (depth_map.empty() || depth_map.type() != CV_32F) {
        return;
    }
    
    for (int v = 0; v < depth_map.rows; ++v) {
        for (int u = 0; u < depth_map.cols; ++u) {
            float depth = depth_map.at<float>(v, u);
            
            // Skip invalid depths
            if (std::isnan(depth) || std::isinf(depth)) {
                continue;
            }
            if (depth <= min_depth || depth >= max_depth) {
                continue;
            }
            
            Eigen::Vector3d point = depthTo3D(u, v, depth, intrinsics);
            points.push_back(point);
            pixel_indices.push_back({u, v});
        }
    }
}

std::vector<std::vector<Eigen::Vector3d>> backprojectDepthOrganized(
    const cv::Mat& depth_map,
    const CameraIntrinsics& intrinsics) {
    
    std::vector<std::vector<Eigen::Vector3d>> organized_points;
    
    if (depth_map.empty() || depth_map.type() != CV_32F) {
        return organized_points;
    }
    
    organized_points.resize(depth_map.rows);
    
    for (int v = 0; v < depth_map.rows; ++v) {
        organized_points[v].resize(depth_map.cols);
        for (int u = 0; u < depth_map.cols; ++u) {
            float depth = depth_map.at<float>(v, u);
            
            if (std::isnan(depth) || std::isinf(depth)) {
                // Set to NaN point
                organized_points[v][u] = Eigen::Vector3d(
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()
                );
            } else {
                organized_points[v][u] = depthTo3D(u, v, depth, intrinsics);
            }
        }
    }
    
    return organized_points;
}

} // namespace face_reconstruction
