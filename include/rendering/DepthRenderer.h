#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "camera/CameraIntrinsics.h"
#include <vector>

namespace face_reconstruction {

/**
 * Minimal depth renderer for comparing rendered vs observed depth.
 * 
 * Projects 3D model vertices to image plane and rasterizes triangles
 * to create a synthetic depth map.
 */
class DepthRenderer {
public:
    DepthRenderer() = default;
    
    /**
     * Initialize renderer with camera intrinsics and image dimensions
     */
    void initialize(const CameraIntrinsics& intrinsics, int width, int height);
    
    /**
     * Render depth map from 3D vertices and faces.
     * @param roi If non-empty, only rasterize triangles overlapping this region (faster when face is small).
     */
    cv::Mat renderDepth(const Eigen::MatrixXd& vertices,
                        const Eigen::MatrixXi& faces,
                        const cv::Rect& roi = cv::Rect()) const;
    
    /**
     * Render depth map from 3D vertices only (point cloud, no faces)
     * Uses nearest neighbor approach
     * @param vertices N x 3 matrix of 3D vertices
     * @return Rendered depth map
     */
    cv::Mat renderDepthPoints(const Eigen::MatrixXd& vertices) const;
    
    /**
     * Get image dimensions
     */
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    
    /**
     * Check if renderer is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    CameraIntrinsics intrinsics_;
    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;
    
    /**
     * Project 3D point to image coordinates
     */
    Eigen::Vector2d projectPoint(const Eigen::Vector3d& point) const;
    
    /**
     * Rasterize a single triangle. roi_* clamp the pixel loop when set (>=0).
     */
    void rasterizeTriangle(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1,
                          const Eigen::Vector3d& v2,
                          cv::Mat& depth_map,
                          int roi_min_u = -1, int roi_max_u = -1,
                          int roi_min_v = -1, int roi_max_v = -1) const;
    
    /**
     * Check if point is within image bounds
     */
    bool isInBounds(int u, int v) const {
        return u >= 0 && u < width_ && v >= 0 && v < height_;
    }
};

} // namespace face_reconstruction

