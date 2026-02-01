/**
 * Optimization Parameters Implementation
 * 
 * Handles parameter packing/unpacking for optimization and
 * rotation matrix <-> axis-angle conversion.
 */

#include "optimization/Parameters.h"
#include <cmath>
#include <iostream>

namespace face_reconstruction {

Eigen::VectorXd OptimizationParams::pack() const {
    Eigen::VectorXd params(numParameters());
    int idx = 0;
    
    if (optimize_identity) {
        params.segment(idx, alpha.size()) = alpha;
        idx += alpha.size();
    }
    
    if (optimize_expression) {
        params.segment(idx, delta.size()) = delta;
        idx += delta.size();
    }
    
    if (optimize_rotation) {
        params.segment(idx, 3) = rotationToAxisAngle(R);
        idx += 3;
    }
    
    if (optimize_translation) {
        params.segment(idx, 3) = t;
        idx += 3;
    }
    
    return params;
}

void OptimizationParams::unpack(const Eigen::VectorXd& params) {
    int idx = 0;
    
    if (optimize_identity) {
        alpha = params.segment(idx, alpha.size());
        idx += alpha.size();
    }
    
    if (optimize_expression) {
        delta = params.segment(idx, delta.size());
        idx += delta.size();
    }
    
    if (optimize_rotation) {
        R = axisAngleToRotation(params.segment(idx, 3));
        idx += 3;
    }
    
    if (optimize_translation) {
        t = params.segment(idx, 3);
        idx += 3;
    }
}

void OptimizationParams::applyUpdate(const Eigen::VectorXd& delta_params) {
    int idx = 0;
    
    if (optimize_identity) {
        alpha += delta_params.segment(idx, alpha.size());
        idx += alpha.size();
    }
    
    if (optimize_expression) {
        delta += delta_params.segment(idx, this->delta.size());
        idx += this->delta.size();
    }
    
    if (optimize_rotation) {
        // For rotation, we need to compose with the delta rotation
        Eigen::Vector3d delta_aa = delta_params.segment(idx, 3);
        Eigen::Matrix3d delta_R = axisAngleToRotation(delta_aa);
        R = delta_R * R;  // Apply delta rotation
        idx += 3;
    }
    
    if (optimize_translation) {
        t += delta_params.segment(idx, 3);
        idx += 3;
    }
    
    // Week 6: Clamp alpha and delta to max norm (guardrails for stability)
    if (max_alpha_norm > 0 && alpha.size() > 0) {
        double n = alpha.norm();
        if (n > max_alpha_norm) {
            alpha *= (max_alpha_norm / n);
            std::cerr << "[Parameters] alpha clamped to norm " << max_alpha_norm << " (was " << n << ")" << std::endl;
        }
    }
    if (max_delta_norm > 0 && delta.size() > 0) {
        double n = delta.norm();
        if (n > max_delta_norm) {
            delta *= (max_delta_norm / n);
            std::cerr << "[Parameters] delta clamped to norm " << max_delta_norm << " (was " << n << ")" << std::endl;
        }
    }
}

Eigen::Vector3d OptimizationParams::rotationToAxisAngle(const Eigen::Matrix3d& R) {
    // Use Rodrigues' formula inverse
    // angle = arccos((trace(R) - 1) / 2)
    // axis = 1/(2*sin(angle)) * [R32-R23, R13-R31, R21-R12]
    
    double trace = R.trace();
    double cos_angle = (trace - 1.0) / 2.0;
    
    // Clamp to [-1, 1] for numerical stability
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    double angle = std::acos(cos_angle);
    
    if (std::abs(angle) < 1e-10) {
        // Nearly identity rotation
        return Eigen::Vector3d::Zero();
    }
    
    if (std::abs(angle - M_PI) < 1e-10) {
        // Nearly 180 degree rotation - need special handling
        // Find the column of (R + I) with largest norm
        Eigen::Matrix3d B = R + Eigen::Matrix3d::Identity();
        int max_col = 0;
        double max_norm = B.col(0).norm();
        for (int i = 1; i < 3; ++i) {
            double norm = B.col(i).norm();
            if (norm > max_norm) {
                max_norm = norm;
                max_col = i;
            }
        }
        Eigen::Vector3d axis = B.col(max_col).normalized();
        return axis * angle;
    }
    
    double sin_angle = std::sin(angle);
    Eigen::Vector3d axis;
    axis << R(2, 1) - R(1, 2),
            R(0, 2) - R(2, 0),
            R(1, 0) - R(0, 1);
    axis /= (2.0 * sin_angle);
    
    return axis.normalized() * angle;
}

Eigen::Matrix3d OptimizationParams::axisAngleToRotation(const Eigen::Vector3d& aa) {
    // Rodrigues' formula: R = I + sin(angle)*K + (1-cos(angle))*K^2
    // where K is the skew-symmetric matrix of the axis
    
    double angle = aa.norm();
    
    if (angle < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }
    
    Eigen::Vector3d axis = aa / angle;
    
    // Skew-symmetric matrix
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    
    double sin_angle = std::sin(angle);
    double cos_angle = std::cos(angle);
    
    return Eigen::Matrix3d::Identity() + sin_angle * K + (1.0 - cos_angle) * K * K;
}

} // namespace face_reconstruction
