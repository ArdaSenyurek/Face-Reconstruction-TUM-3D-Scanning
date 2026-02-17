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

    /**
     * Forward depth mask to the internal energy function.
     */
    void setDepthMask(const cv::Mat& mask) { energy_func_.setDepthMask(mask); }

    /**
     * Set depth sample step (every Nth pixel). Higher = faster optimization.
     */
    void setDepthSampleStep(int step) { energy_func_.setDepthSampleStep(step); }

private:
    EnergyFunction energy_func_;
    const MorphableModel* model_ = nullptr;
    bool initialized_ = false;
    bool verbose_ = false;
    
    // LM adaptive damping: regularizes step direction when JtJ is ill-conditioned.
    // Decreased on accepted step (/2), increased on rejected step (*10), clamped to [1e-6, 1e2].
    double damping_ = 1e-4;
    
    /**
     * Solve (JtJ + damping * diag(JtJ + I)) * delta = -Jtr (Levenberg-Marquardt style).
     */
    Eigen::VectorXd solveNormalEquations(const Eigen::MatrixXd& JtJ, const Eigen::VectorXd& Jtr, double damping);
    
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
