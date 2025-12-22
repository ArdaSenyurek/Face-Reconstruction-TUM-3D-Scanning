#include "rendering/DepthRenderer.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace face_reconstruction {

void DepthRenderer::initialize(const CameraIntrinsics& intrinsics, int width, int height) {
    intrinsics_ = intrinsics;
    width_ = width;
    height_ = height;
    initialized_ = true;
}

Eigen::Vector2d DepthRenderer::projectPoint(const Eigen::Vector3d& point) const {
    // Pinhole camera projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
    if (point.z() <= 0.0) {
        // Point behind camera
        return Eigen::Vector2d(-1, -1);
    }
    
    double u = intrinsics_.fx * point.x() / point.z() + intrinsics_.cx;
    double v = intrinsics_.fy * point.y() / point.z() + intrinsics_.cy;
    
    return Eigen::Vector2d(u, v);
}

void DepthRenderer::rasterizeTriangle(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, 
                                      const Eigen::Vector3d& v2,
                                      cv::Mat& depth_map) const {
    // Project vertices to image plane
    Eigen::Vector2d p0 = projectPoint(v0);
    Eigen::Vector2d p1 = projectPoint(v1);
    Eigen::Vector2d p2 = projectPoint(v2);
    
    // Skip if any vertex is behind camera
    if (p0.x() < 0 || p1.x() < 0 || p2.x() < 0) {
        return;
    }
    
    // Get bounding box of triangle in image space
    int min_u = static_cast<int>(std::floor(std::min({p0.x(), p1.x(), p2.x()})));
    int max_u = static_cast<int>(std::ceil(std::max({p0.x(), p1.x(), p2.x()})));
    int min_v = static_cast<int>(std::floor(std::min({p0.y(), p1.y(), p2.y()})));
    int max_v = static_cast<int>(std::ceil(std::max({p0.y(), p1.y(), p2.y()})));
    
    // Clamp to image bounds
    min_u = std::max(0, min_u);
    max_u = std::min(width_ - 1, max_u);
    min_v = std::max(0, min_v);
    max_v = std::min(height_ - 1, max_v);
    
    // Barycentric coordinates for point-in-triangle test
    double denom = (p1.y() - p2.y()) * (p0.x() - p2.x()) + (p2.x() - p1.x()) * (p0.y() - p2.y());
    if (std::abs(denom) < 1e-10) {
        return;  // Degenerate triangle
    }
    
    // Rasterize triangle
    for (int v = min_v; v <= max_v; ++v) {
        for (int u = min_u; u <= max_u; ++u) {
            // Barycentric coordinates
            double w0 = ((p1.y() - p2.y()) * (u - p2.x()) + (p2.x() - p1.x()) * (v - p2.y())) / denom;
            double w1 = ((p2.y() - p0.y()) * (u - p2.x()) + (p0.x() - p2.x()) * (v - p2.y())) / denom;
            double w2 = 1.0 - w0 - w1;
            
            // Check if point is inside triangle
            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
                // Interpolate depth using barycentric coordinates
                double depth = w0 * v0.z() + w1 * v1.z() + w2 * v2.z();
                
                // Z-buffer: only update if closer
                float& current_depth = depth_map.at<float>(v, u);
                if (current_depth <= 0.0 || depth < current_depth) {
                    current_depth = static_cast<float>(depth);
                }
            }
        }
    }
}

cv::Mat DepthRenderer::renderDepth(const Eigen::MatrixXd& vertices, 
                                    const Eigen::MatrixXi& faces) const {
    if (!initialized_) {
        throw std::runtime_error("DepthRenderer not initialized");
    }
    
    // Initialize depth map with zeros (invalid depth)
    cv::Mat depth_map(height_, width_, CV_32F, cv::Scalar(0.0f));
    
    // Rasterize each triangle
    for (int i = 0; i < faces.rows(); ++i) {
        int idx0 = faces(i, 0);
        int idx1 = faces(i, 1);
        int idx2 = faces(i, 2);
        
        // Check indices
        if (idx0 < 0 || idx0 >= vertices.rows() ||
            idx1 < 0 || idx1 >= vertices.rows() ||
            idx2 < 0 || idx2 >= vertices.rows()) {
            continue;
        }
        
        Eigen::Vector3d v0 = vertices.row(idx0);
        Eigen::Vector3d v1 = vertices.row(idx1);
        Eigen::Vector3d v2 = vertices.row(idx2);
        
        rasterizeTriangle(v0, v1, v2, depth_map);
    }
    
    // Set zero depths to NaN (invalid)
    cv::Mat mask = (depth_map == 0.0f);
    depth_map.setTo(std::numeric_limits<float>::quiet_NaN(), mask);
    
    return depth_map;
}

cv::Mat DepthRenderer::renderDepthPoints(const Eigen::MatrixXd& vertices) const {
    if (!initialized_) {
        throw std::runtime_error("DepthRenderer not initialized");
    }
    
    // Initialize depth map with large values
    cv::Mat depth_map(height_, width_, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
    
    // Project each vertex and update depth map (nearest neighbor)
    for (int i = 0; i < vertices.rows(); ++i) {
        Eigen::Vector3d point = vertices.row(i);
        
        if (point.z() <= 0.0) {
            continue;  // Behind camera
        }
        
        Eigen::Vector2d proj = projectPoint(point);
        int u = static_cast<int>(std::round(proj.x()));
        int v = static_cast<int>(std::round(proj.y()));
        
        if (isInBounds(u, v)) {
            float& current_depth = depth_map.at<float>(v, u);
            if (point.z() < current_depth) {
                current_depth = static_cast<float>(point.z());
            }
        }
    }
    
    // Set max values to NaN (invalid)
    float max_val = std::numeric_limits<float>::max();
    cv::Mat mask = (depth_map >= max_val * 0.99f);
    depth_map.setTo(std::numeric_limits<float>::quiet_NaN(), mask);
    
    return depth_map;
}

} // namespace face_reconstruction

