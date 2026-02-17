#pragma once

#include "optimization/Parameters.h"
#include "model/MorphableModel.h"
#include "landmarks/LandmarkData.h"
#include "alignment/LandmarkMapping.h"
#include "camera/CameraIntrinsics.h"
#include "rendering/DepthRenderer.h"
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <utility>

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
     * Count valid depth residual terms (same sampling as computeDepthEnergy).
     * Used to report per-pixel depth RMSE.
     */
    int computeDepthValidCount(
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
     * Uses a fixed pixel set for depth residuals to guarantee consistent
     * residual vector sizes across parameter perturbations.
     */
    Eigen::MatrixXd computeJacobian(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping,
        const cv::Mat& observed_depth) const;

    /** Pixel coordinate pair (u, v). */
    using PixelCoord = std::pair<int, int>;

    /**
     * Collect the set of valid depth pixel coordinates at the current params.
     * If out_baseline_rendered is non-null, fills it with the rendered depth used (avoids re-render in computeJacobian).
     */
    std::vector<PixelCoord> collectDepthPixels(
        const OptimizationParams& params,
        const cv::Mat& observed_depth,
        cv::Mat* out_baseline_rendered = nullptr) const;

    /**
     * Compute depth residuals at a fixed set of pixel coordinates.
     * If pre_rendered is non-empty and same size as depth map, use it instead of rendering (saves one render when at baseline).
     */
    Eigen::VectorXd computeDepthResidualsFixed(
        const OptimizationParams& params,
        const cv::Mat& observed_depth,
        const std::vector<PixelCoord>& pixels,
        const cv::Mat& pre_rendered = cv::Mat()) const;

    /**
     * Compute the full stacked residual vector using a fixed depth pixel set.
     * If baseline_rendered is non-empty, use it for the depth part instead of re-rendering.
     */
    Eigen::VectorXd computeResidualsFixed(
        const OptimizationParams& params,
        const LandmarkData& landmarks,
        const LandmarkMapping& mapping,
        const cv::Mat& observed_depth,
        const std::vector<PixelCoord>& depth_pixels,
        const cv::Mat& baseline_rendered = cv::Mat()) const;
    
    /**
     * Week 6: Get mesh vertices in camera space (for early-stop Z range check).
     */
    Eigen::MatrixXd getTransformedVertices(const OptimizationParams& params) const;

    /**
     * Set depth mask (CV_8U). Only pixels where mask != 0 are used for depth residuals.
     */
    void setDepthMask(const cv::Mat& mask);

    /**
     * Set depth sampling step (sample every Nth pixel). Higher = faster, less coverage. Default 8.
     */
    void setDepthSampleStep(int step) { depth_sample_step_ = (step > 0) ? step : 1; }

    /**
     * Build a rectangular ROI mask from landmark bounding box, expanded by margin pixels.
     * Returns a CV_8U mask (255 inside ROI, 0 outside).
     */
    static cv::Mat buildLandmarkRoiMask(
        const LandmarkData& landmarks, int W, int H, int margin = 40);

private:
    const MorphableModel* model_ = nullptr;
    CameraIntrinsics intrinsics_;
    int image_width_ = 0;
    int image_height_ = 0;
    bool initialized_ = false;
    
    // Numerical differentiation step size
    double epsilon_ = 1e-6;
    
    // Depth sampling parameters (for efficiency)
    int depth_sample_step_ = 8;  // Sample every Nth pixel (higher = faster, lower = more coverage)

    // Face ROI mask for depth residuals (always on when set)
    cv::Mat depth_mask_;
    bool has_mask_ = false;
    
    // Reused depth renderer (avoid creating 100+ per iteration)
    mutable DepthRenderer depth_renderer_;
    
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
};

} // namespace face_reconstruction
