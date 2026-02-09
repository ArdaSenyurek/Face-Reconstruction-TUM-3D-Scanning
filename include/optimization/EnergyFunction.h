#pragma once

#include "optimization/Parameters.h"
#include "model/MorphableModel.h"
#include "landmarks/LandmarkData.h"
#include "alignment/LandmarkMapping.h"
#include "camera/CameraIntrinsics.h"
#include <opencv2/core.hpp>
#include <Eigen/Dense>

namespace face_reconstruction {

/**
 * Energy function for face reconstruction optimization.
 * 
 * Total energy: E(P) = E_sparse(P) + E_dense(P) + E_reg(P)
 * 
 * Where:
 * - E_sparse: Landmark reprojection error
 * - E_dense: Dense depth alignment error
 * - E_reg: Regularization on coefficients
 */
class EnergyFunction {
public:
    EnergyFunction() = default;
    
    /**
     * Initialize with model and camera parameters.
     */
    void initialize(const MorphableModel& model,
                   const CameraIntrinsics& intrinsics,
                   int image_width, int image_height);
    
    /**
     * Compute sparse landmark energy: E_sparse = sum_i ||pi(R*vi + t) - li||^2
     * 
     * @param params Current optimization parameters
     * @param landmarks Detected 2D landmarks
     * @param mapping Landmark to vertex mapping
     * @return Landmark energy value
     */
    double computeLandmarkEnergy(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping) const;
    
    /**
     * Compute dense depth energy: E_dense = sum_p ||D_obs(p) - D_rend(p)||^2
     * 
     * @param params Current optimization parameters
     * @param observed_depth Observed depth map (CV_32F, meters)
     * @return Depth energy value
     */
    double computeDepthEnergy(
        const OptimizationParams& params,
        const cv::Mat& observed_depth) const;
    
    /**
     * Compute regularization energy: E_reg = lambda_a*||alpha/sigma_a||^2 + lambda_d*||delta/sigma_d||^2
     * 
     * @param params Current optimization parameters
     * @return Regularization energy value
     */
    double computeRegularization(const OptimizationParams& params) const;
    
    /**
     * Compute total energy: E = lambda_l*E_sparse + lambda_d*E_dense + E_reg
     */
    double computeTotalEnergy(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping,
        const cv::Mat& observed_depth) const;
    
    /**
     * Compute residual vector for Gauss-Newton optimization.
     * Residuals are stacked: [landmark_residuals, depth_residuals, reg_residuals]
     * 
     * @param params Current optimization parameters
     * @param landmarks Detected 2D landmarks
     * @param mapping Landmark to vertex mapping
     * @param observed_depth Observed depth map
     * @return Residual vector
     */
    Eigen::VectorXd computeResiduals(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping,
        const cv::Mat& observed_depth) const;
    
    /**
     * Compute Jacobian matrix using numerical differentiation.
     * 
     * @param params Current optimization parameters
     * @param landmarks Detected 2D landmarks
     * @param mapping Landmark to vertex mapping
     * @param observed_depth Observed depth map
     * @return Jacobian matrix (num_residuals x num_params)
     */
    Eigen::MatrixXd computeJacobian(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping,
        const cv::Mat& observed_depth) const;
    
    /**
     * Week 6: Get mesh vertices in camera space (for early-stop Z range check).
     */
    Eigen::MatrixXd getTransformedVertices(const OptimizationParams& params) const;

    /**
     * Set translation prior (previous frame pose) for tracking; penalizes ||t - t_prior||^2 when lambda_translation_prior > 0.
     */
    void setTranslationPrior(const Eigen::Vector3d& t_prior);

private:
    const MorphableModel* model_ = nullptr;
    CameraIntrinsics intrinsics_;
    int image_width_ = 0;
    int image_height_ = 0;
    bool initialized_ = false;
    
    // Numerical differentiation step size
    double epsilon_ = 1e-6;
    
    // Depth sampling parameters (for efficiency)
    int depth_sample_step_ = 16;  // Sample every Nth pixel (higher = faster)

    // Translation prior for tracking (reduce drift)
    Eigen::Vector3d t_prior_ = Eigen::Vector3d::Zero();
    bool use_translation_prior_ = false;
    
    /**
     * Reconstruct face mesh with given parameters
     */
    Eigen::MatrixXd reconstructMesh(const OptimizationParams& params) const;
    
    /**
     * Apply pose transformation to vertices
     */
    Eigen::MatrixXd applyPose(const Eigen::MatrixXd& vertices,
                              const OptimizationParams& params) const;
    
    /**
     * Project 3D point to 2D image coordinates
     */
    Eigen::Vector2d projectPoint(const Eigen::Vector3d& point) const;
    
    /**
     * Render depth map from current mesh
     */
    cv::Mat renderDepth(const Eigen::MatrixXd& vertices) const;
    
    /**
     * Compute landmark residuals only (for Jacobian computation)
     */
    Eigen::VectorXd computeLandmarkResiduals(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping) const;
    
    /**
     * Compute depth residuals only (for Jacobian computation)
     */
    Eigen::VectorXd computeDepthResiduals(
        const OptimizationParams& params,
        const cv::Mat& observed_depth) const;
    
    /**
     * Compute regularization residuals
     */
    Eigen::VectorXd computeRegResiduals(const OptimizationParams& params) const;

    /**
     * Compute translation prior energy: lambda * ||t - t_prior||^2
     */
    double computeTranslationPriorEnergy(const OptimizationParams& params) const;

    /**
     * Compute translation prior residuals: sqrt(lambda) * (t - t_prior) for Gauss-Newton
     */
    Eigen::VectorXd computeTranslationPriorResiduals(const OptimizationParams& params) const;
};

} // namespace face_reconstruction
