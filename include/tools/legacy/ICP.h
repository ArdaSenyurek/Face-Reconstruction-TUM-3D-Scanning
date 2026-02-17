#pragma once

#include <Eigen/Dense>
#include <vector>

namespace face_reconstruction {

/**
 * ICP (Iterative Closest Point) - Point-to-point variant
 * Used ONLY for validation, not as a full optimization method.
 */
class ICP {
public:
    ICP() = default;
    
    /**
     * Run ICP alignment
     * @param source_points Source 3D points (model vertices)
     * @param target_points Target 3D points (observed depth points)
     * @param initial_transform Initial transform (from Procrustes)
     * @param max_iterations Maximum number of iterations
     * @param convergence_threshold Convergence threshold (change in error)
     * @return Final transform and error statistics
     */
    struct ICPResult {
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        double final_error;
        int iterations;
        bool converged;
    };
    
    ICPResult align(const Eigen::MatrixXd& source_points,
                    const Eigen::MatrixXd& target_points,
                    const Eigen::Matrix3d& initial_rotation,
                    const Eigen::Vector3d& initial_translation,
                    int max_iterations = 50,
                    double convergence_threshold = 1e-6);
    
    /**
     * Find closest point in target for each source point
     * @param source_points Source points
     * @param target_points Target points
     * @return Indices of closest target points for each source point
     */
    std::vector<int> findClosestPoints(const Eigen::MatrixXd& source_points,
                                       const Eigen::MatrixXd& target_points) const;
    
    /**
     * Compute mean squared error between point sets
     */
    double computeError(const Eigen::MatrixXd& source_points,
                       const Eigen::MatrixXd& target_points,
                       const std::vector<int>& correspondences) const;

private:
    /**
     * Estimate rigid transform using Procrustes (scale=1, only rotation+translation)
     */
    void estimateRigidTransform(const Eigen::MatrixXd& source,
                                 const Eigen::MatrixXd& target,
                                 Eigen::Matrix3d& rotation,
                                 Eigen::Vector3d& translation) const;
};

} // namespace face_reconstruction

