#pragma once

#include <Eigen/Dense>
#include <vector>

namespace face_reconstruction {

/**
 * Optimization parameters for face reconstruction.
 * 
 * Contains all parameters that can be optimized:
 * - Identity coefficients (alpha)
 * - Expression coefficients (delta)
 * - Rotation matrix (R)
 * - Translation vector (t)
 * 
 * Also contains regularization weights and optimizer settings.
 */
struct OptimizationParams {
    // Model coefficients
    Eigen::VectorXd alpha;      // Identity coefficients
    Eigen::VectorXd delta;      // Expression coefficients
    
    // Pose parameters (similarity transform: v' = scale * R * v + t)
    Eigen::Matrix3d R;          // Rotation matrix
    Eigen::Vector3d t;          // Translation vector
    double scale = 1.0;         // Scale factor (BFM mm -> camera meters)
    
    // Regularization weights
    double lambda_alpha = 1.0;   // Weight for identity regularization
    double lambda_delta = 1.0;   // Weight for expression regularization
    double lambda_landmark = 1.0; // Weight for landmark term
    double lambda_depth = 0.1;   // Weight for depth term (lower than landmarks for stability)
    
    // Optimizer settings (fewer iterations, stricter early stopping)
    int max_iterations = 5;
    double convergence_threshold = 1e-4;
    double step_size = 1.0;      // Gauss-Newton step scale (fixed)
    /** LM initial damping (higher = smaller steps at start; adaptive schedule then adjusts) */
    double initial_damping = 1e-2;
    
    // Per-coefficient sigma for clipping (set from MorphableModel at init)
    Eigen::VectorXd identity_stddev;
    Eigen::VectorXd expression_stddev;
    /** Clip alpha/delta to ± coeff_clip_sigma * sigma (default 2.0; lower = smoother, less spikes) */
    double coeff_clip_sigma = 2.0;
    
    // Which parameters to optimize (pose R, t, scale are always fixed from Procrustes)
    bool optimize_identity = true;
    bool optimize_expression = true;
    
    /**
     * Default constructor - initializes with identity pose
     */
    OptimizationParams() 
        : R(Eigen::Matrix3d::Identity())
        , t(Eigen::Vector3d::Zero()) {}
    
    /**
     * Initialize with specified dimensions
     */
    OptimizationParams(int num_identity, int num_expression)
        : alpha(Eigen::VectorXd::Zero(num_identity))
        , delta(Eigen::VectorXd::Zero(num_expression))
        , R(Eigen::Matrix3d::Identity())
        , t(Eigen::Vector3d::Zero()) {}
    
    /**
     * Get total number of optimizable parameters
     */
    int numParameters() const {
        int n = 0;
        if (optimize_identity) n += alpha.size();
        if (optimize_expression) n += delta.size();
        return n;
    }
    
    /**
     * Pack parameters into a single vector for optimization
     */
    Eigen::VectorXd pack() const;
    
    /**
     * Unpack parameters from a single vector
     */
    void unpack(const Eigen::VectorXd& params);
    
    /**
     * Apply a delta update to parameters
     */
    void applyUpdate(const Eigen::VectorXd& delta_params);
};

/**
 * Result of optimization
 */
struct OptimizationResult {
    OptimizationParams final_params;
    double initial_energy;
    double final_energy;
    int iterations;
    bool converged;
    std::vector<double> energy_history;
    std::vector<double> step_norms;  // Week 6: per-iteration step norm for convergence.json
    
    // Per-term energies for diagnostics
    double landmark_energy;
    double depth_energy;
    double regularization_energy;
    int depth_valid_count = 0;     // Number of valid depth pixels (for per-pixel RMSE)
    double final_step_norm = 0.0;  // Week 6: last applied step norm
};

} // namespace face_reconstruction
