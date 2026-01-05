/**
 * Procrustes Alignment
 * 
 * Estimates similarity transform (scale, rotation, translation) between two 3D point sets.
 * Used for pose initialization in pose_init tool.
 */

#include "alignment/Procrustes.h"
#include <cmath>
#include <algorithm>

namespace face_reconstruction {

Eigen::MatrixXd SimilarityTransform::apply(const Eigen::MatrixXd& points) const {
    Eigen::MatrixXd transformed = points * rotation.transpose();  // points * R^T
    transformed *= scale;
    transformed.rowwise() += translation.transpose();
    return transformed;
}

SimilarityTransform estimateSimilarityTransform(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points) {
    
    if (source_points.rows() != target_points.rows() || 
        source_points.cols() != 3 || target_points.cols() != 3) {
        throw std::invalid_argument("Point sets must have same size and be 3D");
    }
    
    int n = source_points.rows();
    if (n < 3) {
        throw std::invalid_argument("Need at least 3 points for Procrustes");
    }
    
    SimilarityTransform transform;
    
    // Compute centroids
    Eigen::Vector3d source_centroid = source_points.colwise().mean();
    Eigen::Vector3d target_centroid = target_points.colwise().mean();
    
    // Center the point sets
    Eigen::MatrixXd source_centered = source_points.rowwise() - source_centroid.transpose();
    Eigen::MatrixXd target_centered = target_points.rowwise() - target_centroid.transpose();
    
    // Compute scale from source (or use combined scale)
    double source_scale = 0.0;
    double target_scale = 0.0;
    
    for (int i = 0; i < n; ++i) {
        source_scale += source_centered.row(i).squaredNorm();
        target_scale += target_centered.row(i).squaredNorm();
    }
    
    source_scale = std::sqrt(source_scale / n);
    target_scale = std::sqrt(target_scale / n);
    
    // Normalize by source scale (or use ratio)
    if (source_scale > 1e-10) {
        source_centered /= source_scale;
    }
    if (target_scale > 1e-10) {
        target_centered /= target_scale;
    }
    
    // Compute cross-covariance matrix
    Eigen::Matrix3d H = source_centered.transpose() * target_centered;
    
    // SVD: H = U * S * V^T
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Rotation: R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();
    
    // Ensure proper rotation (det(R) = 1)
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }
    
    // Compute scale
    double scale_ratio = target_scale / source_scale;
    
    // Final scale is the ratio
    transform.scale = scale_ratio;
    transform.rotation = R;
    
    // Translation: t = target_centroid - scale * R * source_centroid
    transform.translation = target_centroid - transform.scale * transform.rotation * source_centroid;
    
    return transform;
}

SimilarityTransform estimateSimilarityTransform(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& target_points) {
    
    if (source_points.size() != target_points.size()) {
        throw std::invalid_argument("Point sets must have same size");
    }
    
    int n = static_cast<int>(source_points.size());
    Eigen::MatrixXd source_matrix(n, 3);
    Eigen::MatrixXd target_matrix(n, 3);
    
    for (int i = 0; i < n; ++i) {
        source_matrix.row(i) = source_points[i].transpose();
        target_matrix.row(i) = target_points[i].transpose();
    }
    
    return estimateSimilarityTransform(source_matrix, target_matrix);
}

} // namespace face_reconstruction
