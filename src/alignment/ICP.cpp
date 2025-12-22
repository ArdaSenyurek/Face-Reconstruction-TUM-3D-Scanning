#include "alignment/ICP.h"
#include "alignment/Procrustes.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace face_reconstruction {

std::vector<int> ICP::findClosestPoints(const Eigen::MatrixXd& source_points,
                                        const Eigen::MatrixXd& target_points) const {
    std::vector<int> correspondences(source_points.rows());
    
    for (int i = 0; i < source_points.rows(); ++i) {
        Eigen::Vector3d source = source_points.row(i);
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;
        
        for (int j = 0; j < target_points.rows(); ++j) {
            Eigen::Vector3d target = target_points.row(j);
            double dist = (source - target).squaredNorm();
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = j;
            }
        }
        
        correspondences[i] = closest_idx;
    }
    
    return correspondences;
}

double ICP::computeError(const Eigen::MatrixXd& source_points,
                        const Eigen::MatrixXd& target_points,
                        const std::vector<int>& correspondences) const {
    double total_error = 0.0;
    int count = 0;
    
    for (int i = 0; i < source_points.rows(); ++i) {
        if (correspondences[i] >= 0 && correspondences[i] < target_points.rows()) {
            Eigen::Vector3d diff = source_points.row(i) - target_points.row(correspondences[i]);
            total_error += diff.squaredNorm();
            count++;
        }
    }
    
    if (count > 0) {
        return std::sqrt(total_error / count);  // RMS error
    }
    
    return std::numeric_limits<double>::max();
}

void ICP::estimateRigidTransform(const Eigen::MatrixXd& source,
                                 const Eigen::MatrixXd& target,
                                 Eigen::Matrix3d& rotation,
                                 Eigen::Vector3d& translation) const {
    // Compute centroids
    Eigen::Vector3d source_centroid = source.colwise().mean();
    Eigen::Vector3d target_centroid = target.colwise().mean();
    
    // Center the point sets
    Eigen::MatrixXd source_centered = source.rowwise() - source_centroid.transpose();
    Eigen::MatrixXd target_centered = target.rowwise() - target_centroid.transpose();
    
    // Compute cross-covariance matrix
    Eigen::Matrix3d H = source_centered.transpose() * target_centered;
    
    // SVD: H = U * S * V^T
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Rotation: R = V * U^T
    rotation = V * U.transpose();
    
    // Ensure proper rotation (det(R) = 1)
    if (rotation.determinant() < 0) {
        V.col(2) *= -1;
        rotation = V * U.transpose();
    }
    
    // Translation: t = target_centroid - R * source_centroid
    translation = target_centroid - rotation * source_centroid;
}

ICP::ICPResult ICP::align(const Eigen::MatrixXd& source_points,
                          const Eigen::MatrixXd& target_points,
                          const Eigen::Matrix3d& initial_rotation,
                          const Eigen::Vector3d& initial_translation,
                          int max_iterations,
                          double convergence_threshold) {
    
    ICPResult result;
    result.iterations = 0;
    result.converged = false;
    
    // Initialize transform
    Eigen::Matrix3d R = initial_rotation;
    Eigen::Vector3d t = initial_translation;
    
    // Transform source points
    Eigen::MatrixXd transformed = (source_points * R.transpose()).rowwise() + t.transpose();
    
    double prev_error = std::numeric_limits<double>::max();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find closest points
        std::vector<int> correspondences = findClosestPoints(transformed, target_points);
        
        // Compute error
        double error = computeError(transformed, target_points, correspondences);
        
        // Check convergence
        if (std::abs(prev_error - error) < convergence_threshold) {
            result.converged = true;
            result.iterations = iter + 1;
            result.final_error = error;
            result.rotation = R;
            result.translation = t;
            return result;
        }
        
        prev_error = error;
        
        // Build correspondence sets
        std::vector<Eigen::Vector3d> source_corr, target_corr;
        for (int i = 0; i < transformed.rows(); ++i) {
            if (correspondences[i] >= 0) {
                source_corr.push_back(transformed.row(i));
                target_corr.push_back(target_points.row(correspondences[i]));
            }
        }
        
        if (source_corr.size() < 3) {
            // Not enough correspondences
            break;
        }
        
        // Convert to matrices
        Eigen::MatrixXd source_mat(source_corr.size(), 3);
        Eigen::MatrixXd target_mat(target_corr.size(), 3);
        for (size_t i = 0; i < source_corr.size(); ++i) {
            source_mat.row(i) = source_corr[i];
            target_mat.row(i) = target_corr[i];
        }
        
        // Estimate rigid transform
        Eigen::Matrix3d delta_R;
        Eigen::Vector3d delta_t;
        estimateRigidTransform(source_mat, target_mat, delta_R, delta_t);
        
        // Update transform
        R = delta_R * R;
        t = delta_R * t + delta_t;
        
        // Transform source points
        transformed = (source_points * R.transpose()).rowwise() + t.transpose();
    }
    
    // Final error
    std::vector<int> final_correspondences = findClosestPoints(transformed, target_points);
    result.final_error = computeError(transformed, target_points, final_correspondences);
    result.iterations = max_iterations;
    result.rotation = R;
    result.translation = t;
    
    return result;
}

} // namespace face_reconstruction

