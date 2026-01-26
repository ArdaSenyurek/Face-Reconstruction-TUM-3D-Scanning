#pragma once

#include <Eigen/Dense>
#include <vector>

namespace face_reconstruction {

/**
 * Result of similarity transform estimation
 */
struct SimilarityTransform {
    double scale;                          // Uniform scale factor
    Eigen::Matrix3d rotation;              // 3x3 rotation matrix
    Eigen::Vector3d translation;           // Translation vector
    
    SimilarityTransform() : scale(1.0), rotation(Eigen::Matrix3d::Identity()), 
                           translation(Eigen::Vector3d::Zero()) {}
    
    /**
     * Apply transform to a 3D point
     */
    Eigen::Vector3d apply(const Eigen::Vector3d& point) const {
        return scale * rotation * point + translation;
    }
    
    /**
     * Apply transform to multiple points
     */
    Eigen::MatrixXd apply(const Eigen::MatrixXd& points) const;
};

/**
 * Estimate similarity transform (scale, rotation, translation) using Procrustes analysis
 * Finds transform that best aligns source points to target points:
 *   target ≈ scale * R * source + t
 * 
 * @param source_points Source 3D points (N x 3 matrix)
 * @param target_points Target 3D points (N x 3 matrix, same order as source)
 * @return Similarity transform
 */
SimilarityTransform estimateSimilarityTransform(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points);

/**
 * Overloaded version using vectors of points
 */
SimilarityTransform estimateSimilarityTransform(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& target_points);

/**
 * Validate correspondences for Procrustes alignment
 * Checks for degenerate cases (collinear points, insufficient correspondences)
 * 
 * @param source_points Source 3D points
 * @param target_points Target 3D points
 * @return true if correspondences are valid, false otherwise
 */
bool validateCorrespondences(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points);

/**
 * Compute detailed alignment errors after transformation
 * 
 * @param source_points Source points (before transform)
 * @param target_points Target points
 * @param transform Transform to apply
 * @return Vector of per-correspondence errors (in meters)
 */
std::vector<double> computeAlignmentErrors(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const SimilarityTransform& transform);

} // namespace face_reconstruction
