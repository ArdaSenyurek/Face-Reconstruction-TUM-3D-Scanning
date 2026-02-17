/**
 * Gauss-Newton Optimizer Implementation
 *
 * Iterative least-squares optimizer for face reconstruction.
 * Uses numerical differentiation for Jacobian computation.
 * Pure Gauss-Newton (no LM damping); minimal diagonal regularization only if JtJ is singular.
 * Stricter early stopping: stop on small step or minimal energy change, or on first line-search failure.
 *
 * In-memory: The entire optimization loop runs in memory. No file or disk I/O is performed
 * during optimize(); only the inputs (params, landmarks, mapping, observed_depth) and
 * internal matrices (residuals, Jacobian, JtJ, etc.) are used. Ensure sufficient RAM so
 * the process does not swap.
 */

#include "optimization/GaussNewton.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace face_reconstruction {

// Minimal regularization only when Cholesky fails (numerical stability, not LM)
static constexpr double JTJ_EPS = 1e-10;

// Z-range for centroid sanity check (soft: reject step, do not terminate)
static constexpr double Z_CENTROID_MIN = 0.3;
static constexpr double Z_CENTROID_MAX = 2.0;

void GaussNewtonOptimizer::initialize(const MorphableModel& model,
                                      const CameraIntrinsics& intrinsics,
                                      int image_width, int image_height) {
    model_ = &model;
    energy_func_.initialize(model, intrinsics, image_width, image_height);
    initialized_ = true;
}

Eigen::VectorXd GaussNewtonOptimizer::solveNormalEquations(
    const Eigen::MatrixXd& JtJ,
    const Eigen::VectorXd& Jtr) {
    Eigen::MatrixXd M = JtJ;
    int n = M.rows();
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    if (llt.info() == Eigen::Success) {
        return llt.solve(-Jtr);
    }
    // Fallback: add minimal diagonal for singularity (not LM damping)
    for (int i = 0; i < n; ++i) {
        M(i, i) += JTJ_EPS * (JtJ(i, i) + 1.0);
    }
    Eigen::LLT<Eigen::MatrixXd> llt2(M);
    if (llt2.info() == Eigen::Success) {
        return llt2.solve(-Jtr);
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt(M);
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(-Jtr);
    }
    if (verbose_) {
        std::cout << "Warning: Using pseudoinverse for linear solve" << std::endl;
    }
    return M.completeOrthogonalDecomposition().solve(-Jtr);
}

Eigen::VectorXd GaussNewtonOptimizer::solveLinearSystem(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& r) {
    Eigen::MatrixXd JtJ = J.transpose() * J;
    Eigen::VectorXd Jtr = J.transpose() * r;
    return solveNormalEquations(JtJ, Jtr);
}

bool GaussNewtonOptimizer::checkConvergence(const Eigen::VectorXd& delta, 
                                            double threshold) {
    return delta.norm() < threshold;
}

OptimizationResult GaussNewtonOptimizer::optimize(
    const OptimizationParams& initial_params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping,
    const cv::Mat& observed_depth) {
    
    OptimizationResult result;
    result.converged = false;
    result.iterations = 0;
    
    if (!initialized_ || !model_) {
        std::cerr << "Optimizer not initialized!" << std::endl;
        return result;
    }
    
    OptimizationParams params = initial_params;
    
    // Ensure coefficient vectors have correct size
    if (params.alpha.size() == 0 && model_->num_identity_components > 0) {
        params.alpha = Eigen::VectorXd::Zero(model_->num_identity_components);
    }
    if (params.delta.size() == 0 && model_->num_expression_components > 0) {
        params.delta = Eigen::VectorXd::Zero(model_->num_expression_components);
    }
    
    // Compute initial energy
    result.initial_energy = energy_func_.computeTotalEnergy(
        params, landmarks, mapping, observed_depth);
    result.energy_history.push_back(result.initial_energy);
    
    if (verbose_) {
        std::cout << "=== Gauss-Newton Optimization (pure GN, strict early stop) ===" << std::endl;
        std::cout << "Initial energy: " << std::fixed << std::setprecision(6) 
                  << result.initial_energy << std::endl;
        std::cout << "Parameters: " << params.numParameters() << std::endl;
        std::cout << "  Identity coeffs: " << params.alpha.size() << std::endl;
        std::cout << "  Expression coeffs: " << params.delta.size() << std::endl;
        std::cout << "  Optimize identity: " << params.optimize_identity << std::endl;
        std::cout << "  Optimize expression: " << params.optimize_expression << std::endl;
        std::cout << "  Pose: fixed from Procrustes" << std::endl;
        std::cout << "  Max iterations: " << params.max_iterations 
                  << ", convergence threshold: " << params.convergence_threshold << std::endl;
    }
    
    double prev_energy = result.initial_energy;
    
    for (int iter = 0; iter < params.max_iterations; ++iter) {
        result.iterations = iter + 1;
        
        // Compute residuals and Jacobian
        Eigen::VectorXd residuals = energy_func_.computeResiduals(
            params, landmarks, mapping, observed_depth);
        
        if (residuals.size() == 0) {
            if (verbose_) {
                std::cout << "Warning: Empty residuals, stopping" << std::endl;
            }
            break;
        }
        
        Eigen::MatrixXd J = energy_func_.computeJacobian(
            params, landmarks, mapping, observed_depth);
        
        if (J.rows() == 0 || J.cols() == 0) {
            if (verbose_) {
                std::cout << "Warning: Empty Jacobian, stopping" << std::endl;
            }
            break;
        }
        
        // Build normal equations
        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::VectorXd Jtr = J.transpose() * residuals;
        Eigen::VectorXd delta = solveNormalEquations(JtJ, Jtr);
        
        delta *= params.step_size;
        
        // Check convergence based on delta norm
        if (checkConvergence(delta, params.convergence_threshold)) {
            result.converged = true;
            if (verbose_) {
                std::cout << "Converged (small delta norm) at iteration " << iter + 1 << std::endl;
            }
            break;
        }
        
        // Line search: try full step, halve up to 5 times
        double best_energy = prev_energy;
        OptimizationParams best_params = params;
        bool step_accepted = false;
        double step = 1.0;
        
        for (int ls = 0; ls < 5; ++ls) {
            OptimizationParams test_params = params;
            test_params.applyUpdate(delta * step);
            
            // Soft Z-range check on mesh centroid (reject step, don't terminate)
            Eigen::MatrixXd verts = energy_func_.getTransformedVertices(test_params);
            if (verts.rows() > 0) {
                double z_centroid = verts.col(2).mean();
                if (z_centroid < Z_CENTROID_MIN || z_centroid > Z_CENTROID_MAX) {
                    if (verbose_) {
                        std::cout << "  Line search step " << ls << ": rejecting (centroid Z="
                                  << z_centroid << " outside [" << Z_CENTROID_MIN << ", " << Z_CENTROID_MAX << "])" << std::endl;
                    }
                    step *= 0.5;
                    continue;
                }
            }
            
            double test_energy = energy_func_.computeTotalEnergy(
                test_params, landmarks, mapping, observed_depth);
            
            if (test_energy < best_energy) {
                best_energy = test_energy;
                best_params = test_params;
                step_accepted = true;
                break;
            }
            
            step *= 0.5;
        }
        
        if (step_accepted) {
            params = best_params;
            result.energy_history.push_back(best_energy);
            result.step_norms.push_back((delta * step).norm());
        } else {
            // Stricter early stopping: stop on first line-search failure (no retries)
            if (verbose_) {
                std::cout << "  Line search failed; stopping (strict early stop)" << std::endl;
            }
            result.converged = false;
            result.energy_history.push_back(prev_energy);
            result.step_norms.push_back(0.0);
            break;
        }
        
        // Compute per-term energies for logging
        double lm_energy = energy_func_.computeLandmarkEnergy(params, landmarks, mapping);
        double depth_energy = energy_func_.computeDepthEnergy(params, observed_depth);
        double reg_energy = energy_func_.computeRegularization(params);
        
        if (verbose_ || iter < 10 || iter % 5 == 0) {
            std::cout << "Iter " << std::setw(3) << iter + 1 << ": "
                      << "total=" << std::fixed << std::setprecision(6) << best_energy
                      << " (lm=" << std::setprecision(4) << lm_energy
                      << ", depth=" << depth_energy
                      << ", reg=" << reg_energy << "), "
                      << "step_norm=" << std::setprecision(4) << (delta * step).norm()
                      << std::endl;
        }
        
        // Check for minimal relative energy progress
        double energy_change = std::abs(prev_energy - best_energy);
        if (prev_energy > 0 && energy_change < params.convergence_threshold * prev_energy) {
            result.converged = true;
            if (verbose_) {
                std::cout << "Converged (minimal energy change) at iteration " 
                          << iter + 1 << std::endl;
            }
            break;
        }
        
        prev_energy = best_energy;
    }
    
    // Compute final energy terms for diagnostics
    result.final_params = params;
    result.final_energy = energy_func_.computeTotalEnergy(
        params, landmarks, mapping, observed_depth);
    result.landmark_energy = energy_func_.computeLandmarkEnergy(
        params, landmarks, mapping);
    result.depth_energy = energy_func_.computeDepthEnergy(params, observed_depth);
    result.depth_valid_count = energy_func_.computeDepthValidCount(params, observed_depth);
    result.regularization_energy = energy_func_.computeRegularization(params);
    if (!result.step_norms.empty()) {
        result.final_step_norm = result.step_norms.back();
    }
    
    if (verbose_) {
        std::cout << "=== Optimization Complete ===" << std::endl;
        std::cout << "Final energy: " << result.final_energy << std::endl;
        std::cout << "  Landmark: " << result.landmark_energy << std::endl;
        std::cout << "  Depth: " << result.depth_energy << std::endl;
        std::cout << "  Regularization: " << result.regularization_energy << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
        // Interpretable metrics: landmark RMSE (px), depth scale
        int num_mapped = 0;
        for (size_t i = 0; i < landmarks.size(); ++i) {
            if (mapping.hasMapping(static_cast<int>(i))) ++num_mapped;
        }
        if (num_mapped > 0 && result.landmark_energy >= 0) {
            const double LM_SIGMA_PX = 2.0;
            double lm_rmse_px = LM_SIGMA_PX * std::sqrt(result.landmark_energy / (2 * num_mapped));
            std::cout << "  Landmark RMSE (approx): " << std::fixed << std::setprecision(2) 
                      << lm_rmse_px << " px (over " << num_mapped << " landmarks, sigma=" << LM_SIGMA_PX << " px)" << std::endl;
        }
        if (result.depth_valid_count > 0) {
            std::cout << "  Depth: " << result.depth_valid_count << " valid samples, raw energy " 
                      << std::fixed << std::setprecision(2) << result.depth_energy << " (Huber; lambda_depth scales contribution)" << std::endl;
        }
        if (result.initial_energy > 0) {
            std::cout << "Energy reduction: " 
                      << (1.0 - result.final_energy / result.initial_energy) * 100.0 
                      << "%" << std::endl;
        }
    }
    
    return result;
}

} // namespace face_reconstruction
