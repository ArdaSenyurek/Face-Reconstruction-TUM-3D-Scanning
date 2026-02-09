/**
 * Gauss-Newton Optimizer Implementation
 * 
 * Iterative least-squares optimizer for face reconstruction.
 * Uses numerical differentiation for Jacobian computation.
 */

#include "optimization/GaussNewton.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace face_reconstruction {

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
    Eigen::MatrixXd JtJ_damped = JtJ;
    int n = JtJ_damped.rows();
    for (int i = 0; i < n; ++i) {
        JtJ_damped(i, i) += damping_ * (JtJ(i, i) + 1.0);
    }
    Eigen::LLT<Eigen::MatrixXd> llt(JtJ_damped);
    if (llt.info() == Eigen::Success) {
        return llt.solve(-Jtr);
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt(JtJ_damped);
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(-Jtr);
    }
    if (verbose_) {
        std::cout << "Warning: Using pseudoinverse for linear solve" << std::endl;
    }
    return JtJ_damped.completeOrthogonalDecomposition().solve(-Jtr);
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
    
    // Copy initial parameters
    OptimizationParams params = initial_params;
    
    // Set translation prior (previous frame pose) for tracking; reduces drift when lambda_translation_prior > 0
    energy_func_.setTranslationPrior(initial_params.t);
    if (params.lambda_translation_prior > 0) {
        t_prior_ = initial_params.t;
        if (damping_ < 5e-4) {
            damping_ = 5e-4;
        }
    } else {
        t_prior_ = Eigen::Vector3d::Zero();
    }
    
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
        std::cout << "=== Gauss-Newton Optimization ===" << std::endl;
        std::cout << "Initial energy: " << std::fixed << std::setprecision(6) 
                  << result.initial_energy << std::endl;
        std::cout << "Parameters: " << params.numParameters() << std::endl;
        std::cout << "  Identity coeffs: " << params.alpha.size() << std::endl;
        std::cout << "  Expression coeffs: " << params.delta.size() << std::endl;
        std::cout << "  Optimize identity: " << params.optimize_identity << std::endl;
        std::cout << "  Optimize expression: " << params.optimize_expression << std::endl;
        std::cout << "  Optimize rotation: " << params.optimize_rotation << std::endl;
        std::cout << "  Optimize translation: " << params.optimize_translation << std::endl;
    }
    
    double prev_energy = result.initial_energy;
    double prev_depth_energy = energy_func_.computeDepthEnergy(params, observed_depth);
    int depth_not_improving_count = 0;
    const double z_min_m = 0.5;
    const double z_max_m = 1.5;
    
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
        
        // Build normal equations and add translation prior in the linear system when used
        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::VectorXd Jtr = J.transpose() * residuals;
        if (params.lambda_translation_prior > 0 && params.numParameters() >= 3) {
            int ti = params.numParameters() - 3;
            JtJ(ti, ti) += params.lambda_translation_prior;
            JtJ(ti + 1, ti + 1) += params.lambda_translation_prior;
            JtJ(ti + 2, ti + 2) += params.lambda_translation_prior;
            Jtr.segment(ti, 3) += params.lambda_translation_prior * (params.t - t_prior_);
        }
        Eigen::VectorXd delta = solveNormalEquations(JtJ, Jtr);
        
        // Apply step size
        delta *= params.step_size;
        
        // Check convergence before update
        if (checkConvergence(delta, params.convergence_threshold)) {
            result.converged = true;
            if (verbose_) {
                std::cout << "Converged at iteration " << iter + 1 << std::endl;
            }
            break;
        }
        
        // Line search: try full step, reduce if energy increases
        double best_energy = prev_energy;
        OptimizationParams best_params = params;
        double step = 1.0;
        
        for (int ls = 0; ls < 5; ++ls) {
            OptimizationParams test_params = params;
            test_params.applyUpdate(delta * step);
            
            double test_energy = energy_func_.computeTotalEnergy(
                test_params, landmarks, mapping, observed_depth);
            
            if (test_energy < best_energy) {
                best_energy = test_energy;
                best_params = test_params;
                break;
            }
            
            step *= 0.5;
        }
        
        // Translation clipping: hard-bound drift when prior is used
        OptimizationParams candidate_params = best_params;
        if (candidate_params.lambda_translation_prior > 0 && candidate_params.max_translation_delta_m > 0) {
            double md = candidate_params.max_translation_delta_m;
            for (int i = 0; i < 3; ++i) {
                candidate_params.t(i) = std::max(t_prior_(i) - md, std::min(t_prior_(i) + md, candidate_params.t(i)));
            }
        }
        // Reject step if mesh Z would go out of range [0.5, 1.5] m (keep previous params)
        Eigen::MatrixXd verts = energy_func_.getTransformedVertices(candidate_params);
        if (verts.rows() > 0) {
            double z_min = verts.col(2).minCoeff();
            double z_max = verts.col(2).maxCoeff();
            if (z_min < z_min_m || z_max > z_max_m) {
                if (verbose_) {
                    std::cout << "Early stop: rejecting step (mesh Z [" << z_min << ", " << z_max
                              << "] outside [" << z_min_m << ", " << z_max_m << "] m); keeping previous pose at iteration "
                              << iter + 1 << std::endl;
                }
                result.converged = true;
                break;
            }
        }
        params = candidate_params;
        result.energy_history.push_back(best_energy);
        result.step_norms.push_back((delta * step).norm());
        // Compute per-term energies for logging
        double lm_energy = energy_func_.computeLandmarkEnergy(params, landmarks, mapping);
        double depth_energy = energy_func_.computeDepthEnergy(params, observed_depth);
        double reg_energy = energy_func_.computeRegularization(params);
        // Week 6: Early stop if depth term not improving (2 consecutive increases)
        if (depth_energy > prev_depth_energy) {
            depth_not_improving_count++;
            if (depth_not_improving_count >= 2) {
                if (verbose_) {
                    std::cout << "Early stop: depth term not improving at iteration " << iter + 1 << std::endl;
                }
                result.converged = true;
                break;
            }
        } else {
            depth_not_improving_count = 0;
        }
        prev_depth_energy = depth_energy;
        
        if (verbose_ || iter < 10 || iter % 5 == 0) {
            std::cout << "Iter " << std::setw(3) << iter + 1 << ": "
                      << "total=" << std::fixed << std::setprecision(6) << best_energy
                      << " (lm=" << std::setprecision(4) << lm_energy
                      << ", depth=" << depth_energy
                      << ", reg=" << reg_energy << "), "
                      << "step_norm=" << std::setprecision(4) << (delta * step).norm()
                      << ", damping=" << std::setprecision(2) << damping_
                      << std::endl;
        }
        
        // Check for minimal progress
        double energy_change = std::abs(prev_energy - best_energy);
        if (energy_change < params.convergence_threshold * prev_energy) {
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
    result.damping_used = damping_;
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
        std::cout << "Energy reduction: " 
                  << (1.0 - result.final_energy / result.initial_energy) * 100.0 
                  << "%" << std::endl;
    }
    
    return result;
}

} // namespace face_reconstruction
