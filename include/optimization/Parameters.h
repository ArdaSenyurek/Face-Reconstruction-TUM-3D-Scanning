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
    
    // Optimizer settings
    int max_iterations = 50;
    double convergence_threshold = 1e-6;
    double step_size = 1.0;      // Gauss-Newton step size (can reduce for stability)
    
    // Week 6: Coefficient clamps (0 = no clamp). Log when exceeded.
    double max_alpha_norm = 0.0;   // Max L2 norm for identity (0 = disabled)
    double max_delta_norm = 3.0;   // Max L2 norm for expression (0 = disabled; 3.0 = safe default)
    
    // Which parameters to optimize (useful for multi-frame tracking)
    // Default: pose-only for speed (identity/expression disabled)
    bool optimize_identity = true;    // Enabled for performance
    bool optimize_expression = true;  // Enabled for performance  
    bool optimize_rotation = true;
    bool optimize_translation = true;
    
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
        if (optimize_rotation) n += 3;  // Rotation as axis-angle (3 params)
        if (optimize_translation) n += 3;
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
    
    /**
     * Convert rotation matrix to axis-angle representation
     */
    static Eigen::Vector3d rotationToAxisAngle(const Eigen::Matrix3d& R);
    
    /**
     * Convert axis-angle to rotation matrix
     */
    static Eigen::Matrix3d axisAngleToRotation(const Eigen::Vector3d& aa);
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
    double final_step_norm = 0.0;  // Week 6: last applied step norm
    double damping_used = 1e-4;    // Week 6: damping value used (for convergence.json)
};

} // namespace face_reconstruction
