/**
 * Gauss-Newton Optimizer Implementation
 * 
 * Iterative least-squares optimizer for face reconstruction.
 * Uses numerical differentiation for Jacobian computation.
 */

#include "optimization/GaussNewton.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace face_reconstruction {

void GaussNewtonOptimizer::initialize(const MorphableModel& model,
                                      const CameraIntrinsics& intrinsics,
                                      int image_width, int image_height) {
    model_ = &model;
    energy_func_.initialize(model, intrinsics, image_width, image_height);
    initialized_ = true;
}

Eigen::VectorXd GaussNewtonOptimizer::solveLinearSystem(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& r) {
    
    // Compute J^T * J and J^T * r
    Eigen::MatrixXd JtJ = J.transpose() * J;
    Eigen::VectorXd Jtr = J.transpose() * r;
    
    // Add damping for numerical stability (Levenberg-Marquardt style)
    int n = JtJ.rows();
    for (int i = 0; i < n; ++i) {
        JtJ(i, i) += damping_ * (JtJ(i, i) + 1.0);
    }
    
    // Solve using Cholesky decomposition (more stable than direct inverse)
    Eigen::LLT<Eigen::MatrixXd> llt(JtJ);
    
    if (llt.info() == Eigen::Success) {
        return llt.solve(-Jtr);
    }
    
    // Fallback to LDLT if Cholesky fails
    Eigen::LDLT<Eigen::MatrixXd> ldlt(JtJ);
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(-Jtr);
    }
    
    // Last resort: pseudoinverse
    if (verbose_) {
        std::cout << "Warning: Using pseudoinverse for linear solve" << std::endl;
    }
    return JtJ.completeOrthogonalDecomposition().solve(-Jtr);
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
        
        // Solve for parameter update
        Eigen::VectorXd delta = solveLinearSystem(J, residuals);
        
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
        
        // Update parameters
        params = best_params;
        result.energy_history.push_back(best_energy);
        
        if (verbose_ && (iter % 5 == 0 || iter < 5)) {
            std::cout << "Iter " << std::setw(3) << iter + 1 
                      << ": energy = " << std::setprecision(6) << best_energy
                      << ", delta_norm = " << std::setprecision(4) << delta.norm()
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
    result.regularization_energy = energy_func_.computeRegularization(params);
    
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
