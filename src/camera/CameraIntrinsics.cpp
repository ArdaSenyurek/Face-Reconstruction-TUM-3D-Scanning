/**
 * Camera Intrinsics
 * 
 * Manages camera intrinsic parameters (fx, fy, cx, cy).
 * Used for 3D backprojection and projection operations.
 */

#include "camera/CameraIntrinsics.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace face_reconstruction {

Eigen::Matrix3d CameraIntrinsics::getIntrinsicsMatrix() const {
    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;
    return K;
}

CameraIntrinsics CameraIntrinsics::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open intrinsics file: " + filepath);
    }
    
    double fx_val, fy_val, cx_val, cy_val;
    file >> fx_val >> fy_val >> cx_val >> cy_val;
    
    if (file.fail()) {
        throw std::runtime_error("Failed to parse intrinsics file: " + filepath);
    }
    
    file.close();
    return CameraIntrinsics(fx_val, fy_val, cx_val, cy_val);
}

void CameraIntrinsics::saveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file << fx << " " << fy << " " << cx << " " << cy << std::endl;
    file.close();
}

} // namespace face_reconstruction
