#pragma once

#include <Eigen/Dense>
#include <string>

namespace face_reconstruction {

/**
 * Camera intrinsics parameters
 * Standard pinhole camera model
 */
struct CameraIntrinsics {
    double fx;  // Focal length in x direction
    double fy;  // Focal length in y direction
    double cx;  // Principal point x coordinate
    double cy;  // Principal point y coordinate
    
    CameraIntrinsics() : fx(0.0), fy(0.0), cx(0.0), cy(0.0) {}
    
    CameraIntrinsics(double fx_, double fy_, double cx_, double cy_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
    
    /**
     * Load intrinsics from file (format: fx fy cx cy, one line)
     */
    static CameraIntrinsics loadFromFile(const std::string& filepath);
    
    /**
     * Save intrinsics to file
     */
    void saveToFile(const std::string& filepath) const;
    
    /**
     * Get intrinsics matrix K (3x3)
     */
    Eigen::Matrix3d getIntrinsicsMatrix() const;
};

} // namespace face_reconstruction
