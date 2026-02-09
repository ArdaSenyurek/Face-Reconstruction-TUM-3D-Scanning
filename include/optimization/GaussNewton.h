#pragma once

#include "optimization/Parameters.h"
#include "optimization/EnergyFunction.h"
#include "model/MorphableModel.h"
#include "landmarks/LandmarkData.h"
#include "alignment/LandmarkMapping.h"
#include "camera/CameraIntrinsics.h"
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace face_reconstruction {

/**
 * Gauss-Newton optimizer for face reconstruction.
 * 
 * Minimizes the energy function E(P) = E_sparse + E_dense + E_reg
 * using iterative least squares:
 *   1. Compute residuals r(P)
 *   2. Compute Jacobian J(P)
 *   3. Solve (J^T * J) * delta = -J^T * r
 *   4. Update P = P + step_size * delta
 *   5. Repeat until convergence
 */
class GaussNewtonOptimizer {
public:
    GaussNewtonOptimizer() = default;
    
    /**
     * Initialize optimizer with model and camera parameters.
     */
    void initialize(const MorphableModel& model,
                   const CameraIntrinsics& intrinsics,
                   int image_width, int image_height);
    
    /**
     * Run optimization.
     * 
     * @param initial_params Initial parameters (pose from Procrustes, zero coefficients)
     * @param landmarks Detected 2D landmarks
     * @param mapping Landmark to vertex mapping
     * @param observed_depth Observed depth map
     * @return Optimization result with final parameters and diagnostics
     */
    OptimizationResult optimize(
        const OptimizationParams& initial_params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping,
        const cv::Mat& observed_depth);
    
    /**
     * Set verbose mode for debugging output.
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    EnergyFunction energy_func_;
    const MorphableModel* model_ = nullptr;
    bool initialized_ = false;
    bool verbose_ = false;
    
    // Damping factor for Levenberg-Marquardt style regularization
    double damping_ = 1e-4;
    
    // Translation prior (previous frame t) for tracking; used in normal equations and clipping
    Eigen::Vector3d t_prior_ = Eigen::Vector3d::Zero();
    
    /**
     * Solve (JtJ + damping) * delta = -Jtr. Used after optionally adding prior to JtJ/Jtr.
     */
    Eigen::VectorXd solveNormalEquations(const Eigen::MatrixXd& JtJ, const Eigen::VectorXd& Jtr);
    
    /**
     * Solve the linear system (J^T * J + lambda * I) * delta = -J^T * r
     * Uses Cholesky decomposition for stability.
     */
    Eigen::VectorXd solveLinearSystem(
        const Eigen::MatrixXd& J,
        const Eigen::VectorXd& r);
    
    /**
     * Check convergence based on parameter change.
     */
    bool checkConvergence(const Eigen::VectorXd& delta, double threshold);
};

} // namespace face_reconstruction
