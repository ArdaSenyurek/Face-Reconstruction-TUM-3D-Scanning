/**
 * Energy Function Implementation
 * 
 * Computes energy terms for face reconstruction optimization:
 * - Sparse landmark reprojection error
 * - Dense depth alignment error
 * - Coefficient regularization
 */

#include "optimization/EnergyFunction.h"
#include "rendering/DepthRenderer.h"
#include <cmath>
#include <iostream>

namespace face_reconstruction {

void EnergyFunction::initialize(const MorphableModel& model,
                                const CameraIntrinsics& intrinsics,
                                int image_width, int image_height) {
    model_ = &model;
    intrinsics_ = intrinsics;
    image_width_ = image_width;
    image_height_ = image_height;
    initialized_ = true;
}

void EnergyFunction::setTranslationPrior(const Eigen::Vector3d& t_prior) {
    t_prior_ = t_prior;
    use_translation_prior_ = true;
}

Eigen::MatrixXd EnergyFunction::reconstructMesh(const OptimizationParams& params) const {
    if (!model_ || !model_->isValid()) {
        return Eigen::MatrixXd();
    }
    return model_->reconstructFace(params.alpha, params.delta);
}

Eigen::MatrixXd EnergyFunction::applyPose(const Eigen::MatrixXd& vertices,
                                          const OptimizationParams& params) const {
    // Apply similarity transform: v' = scale * R * v + t
    // Scale converts BFM millimeters to camera meters
    Eigen::MatrixXd transformed(vertices.rows(), 3);
    for (int i = 0; i < vertices.rows(); ++i) {
        Eigen::Vector3d v = vertices.row(i).transpose();
        Eigen::Vector3d v_transformed = params.scale * (params.R * v) + params.t;
        transformed.row(i) = v_transformed.transpose();
    }
    return transformed;
}

Eigen::MatrixXd EnergyFunction::getTransformedVertices(const OptimizationParams& params) const {
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return vertices;
    return applyPose(vertices, params);
}

Eigen::Vector2d EnergyFunction::projectPoint(const Eigen::Vector3d& point) const {
    // Pinhole projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
    if (point.z() <= 0.0) {
        return Eigen::Vector2d(-1, -1);
    }
    double u = intrinsics_.fx * point.x() / point.z() + intrinsics_.cx;
    double v = intrinsics_.fy * point.y() / point.z() + intrinsics_.cy;
    return Eigen::Vector2d(u, v);
}

cv::Mat EnergyFunction::renderDepth(const Eigen::MatrixXd& vertices) const {
    if (!model_ || model_->faces.rows() == 0) {
        // No faces - use point-based rendering
        cv::Mat depth(image_height_, image_width_, CV_32F, cv::Scalar(0.0f));
        for (int i = 0; i < vertices.rows(); ++i) {
            Eigen::Vector3d point = vertices.row(i);
            if (point.z() <= 0) continue;
            Eigen::Vector2d proj = projectPoint(point);
            int u = static_cast<int>(std::round(proj.x()));
            int v = static_cast<int>(std::round(proj.y()));
            if (u >= 0 && u < image_width_ && v >= 0 && v < image_height_) {
                float& current = depth.at<float>(v, u);
                if (current <= 0 || point.z() < current) {
                    current = static_cast<float>(point.z());
                }
            }
        }
        return depth;
    }
    
    // Use DepthRenderer for mesh rendering
    DepthRenderer renderer;
    renderer.initialize(intrinsics_, image_width_, image_height_);
    return renderer.renderDepth(vertices, model_->faces);
}

double EnergyFunction::computeLandmarkEnergy(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping) const {
    
    if (!initialized_ || !model_) return 0.0;
    
    // Reconstruct and transform mesh
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return 0.0;
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    
    double energy = 0.0;
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        int lm_idx = static_cast<int>(i);
        if (!mapping.hasMapping(lm_idx)) continue;
        
        int vertex_idx = mapping.getModelVertex(lm_idx);
        if (vertex_idx < 0 || vertex_idx >= transformed.rows()) continue;
        
        // Get 3D vertex and project to 2D
        Eigen::Vector3d vertex_3d = transformed.row(vertex_idx);
        Eigen::Vector2d projected = projectPoint(vertex_3d);
        
        if (projected.x() < 0) continue;  // Behind camera
        
        // Compute reprojection error
        const auto& lm = landmarks[i];
        double dx = projected.x() - lm.x;
        double dy = projected.y() - lm.y;
        energy += dx * dx + dy * dy;
    }
    
    return energy;
}

double EnergyFunction::computeDepthEnergy(
    const OptimizationParams& params,
    const cv::Mat& observed_depth) const {
    
    if (!initialized_ || !model_) return 0.0;
    if (observed_depth.empty()) return 0.0;
    
    // Reconstruct and transform mesh
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return 0.0;
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    
    // Render depth
    cv::Mat rendered = renderDepth(transformed);
    
    double energy = 0.0;
    
    // Compare rendered vs observed depth
    for (int v = 0; v < image_height_; v += depth_sample_step_) {
        for (int u = 0; u < image_width_; u += depth_sample_step_) {
            float obs_d = observed_depth.at<float>(v, u);
            float rend_d = rendered.at<float>(v, u);
            
            // Skip invalid pixels
            if (obs_d <= 0 || rend_d <= 0) continue;
            if (std::isnan(obs_d) || std::isnan(rend_d)) continue;
            
            double diff = obs_d - rend_d;
            energy += diff * diff;
        }
    }
    
    return energy;
}

double EnergyFunction::computeRegularization(const OptimizationParams& params) const {
    if (!initialized_ || !model_) return 0.0;
    
    double energy = 0.0;
    
    // Identity regularization: ||alpha / sigma_alpha||^2
    if (params.alpha.size() > 0 && model_->identity_stddev.size() > 0) {
        for (int i = 0; i < params.alpha.size() && i < model_->identity_stddev.size(); ++i) {
            double sigma = model_->identity_stddev(i);
            if (sigma > 1e-10) {
                double normalized = params.alpha(i) / sigma;
                energy += params.lambda_alpha * normalized * normalized;
            }
        }
    }
    
    // Expression regularization: ||delta / sigma_delta||^2
    if (params.delta.size() > 0 && model_->expression_stddev.size() > 0) {
        for (int i = 0; i < params.delta.size() && i < model_->expression_stddev.size(); ++i) {
            double sigma = model_->expression_stddev(i);
            if (sigma > 1e-10) {
                double normalized = params.delta(i) / sigma;
                energy += params.lambda_delta * normalized * normalized;
            }
        }
    }
    
    return energy;
}

double EnergyFunction::computeTotalEnergy(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping,
    const cv::Mat& observed_depth) const {
    
    double e_landmark = computeLandmarkEnergy(params, landmarks, mapping);
    double e_depth = computeDepthEnergy(params, observed_depth);
    double e_reg = computeRegularization(params);
    double e_prior = computeTranslationPriorEnergy(params);
    
    return params.lambda_landmark * e_landmark + 
           params.lambda_depth * e_depth + 
           e_reg + e_prior;
}

Eigen::VectorXd EnergyFunction::computeLandmarkResiduals(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping) const {
    
    if (!initialized_ || !model_) return Eigen::VectorXd();
    
    // Reconstruct and transform mesh
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return Eigen::VectorXd();
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    
    // Count valid landmarks
    int num_valid = 0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        if (mapping.hasMapping(static_cast<int>(i))) {
            int vertex_idx = mapping.getModelVertex(static_cast<int>(i));
            if (vertex_idx >= 0 && vertex_idx < transformed.rows()) {
                num_valid++;
            }
        }
    }
    
    // Compute residuals (2 per landmark: x and y)
    Eigen::VectorXd residuals(num_valid * 2);
    int idx = 0;
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        int lm_idx = static_cast<int>(i);
        if (!mapping.hasMapping(lm_idx)) continue;
        
        int vertex_idx = mapping.getModelVertex(lm_idx);
        if (vertex_idx < 0 || vertex_idx >= transformed.rows()) continue;
        
        Eigen::Vector3d vertex_3d = transformed.row(vertex_idx);
        Eigen::Vector2d projected = projectPoint(vertex_3d);
        
        const auto& lm = landmarks[i];
        double weight = std::sqrt(params.lambda_landmark);
        
        if (projected.x() >= 0) {
            residuals(idx * 2) = weight * (projected.x() - lm.x);
            residuals(idx * 2 + 1) = weight * (projected.y() - lm.y);
        } else {
            residuals(idx * 2) = 0;
            residuals(idx * 2 + 1) = 0;
        }
        idx++;
    }
    
    return residuals;
}

Eigen::VectorXd EnergyFunction::computeDepthResiduals(
    const OptimizationParams& params,
    const cv::Mat& observed_depth) const {
    
    if (!initialized_ || !model_ || observed_depth.empty()) {
        return Eigen::VectorXd();
    }
    
    // Reconstruct and transform mesh
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return Eigen::VectorXd();
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    cv::Mat rendered = renderDepth(transformed);
    
    // Count valid depth points
    std::vector<double> residual_list;
    double weight = std::sqrt(params.lambda_depth);
    
    for (int v = 0; v < image_height_; v += depth_sample_step_) {
        for (int u = 0; u < image_width_; u += depth_sample_step_) {
            float obs_d = observed_depth.at<float>(v, u);
            float rend_d = rendered.at<float>(v, u);
            
            if (obs_d > 0 && rend_d > 0 && 
                !std::isnan(obs_d) && !std::isnan(rend_d)) {
                residual_list.push_back(weight * (obs_d - rend_d));
            }
        }
    }
    
    Eigen::VectorXd residuals(residual_list.size());
    for (size_t i = 0; i < residual_list.size(); ++i) {
        residuals(i) = residual_list[i];
    }
    
    return residuals;
}

Eigen::VectorXd EnergyFunction::computeRegResiduals(const OptimizationParams& params) const {
    if (!initialized_ || !model_) return Eigen::VectorXd();
    
    int num_alpha = params.alpha.size();
    int num_delta = params.delta.size();
    
    Eigen::VectorXd residuals(num_alpha + num_delta);
    
    // Identity regularization residuals
    for (int i = 0; i < num_alpha; ++i) {
        double sigma = (i < model_->identity_stddev.size()) ? 
                       model_->identity_stddev(i) : 1.0;
        if (sigma < 1e-10) sigma = 1.0;
        residuals(i) = std::sqrt(params.lambda_alpha) * params.alpha(i) / sigma;
    }
    
    // Expression regularization residuals
    for (int i = 0; i < num_delta; ++i) {
        double sigma = (i < model_->expression_stddev.size()) ? 
                       model_->expression_stddev(i) : 1.0;
        if (sigma < 1e-10) sigma = 1.0;
        residuals(num_alpha + i) = std::sqrt(params.lambda_delta) * params.delta(i) / sigma;
    }
    
    return residuals;
}

double EnergyFunction::computeTranslationPriorEnergy(const OptimizationParams& params) const {
    if (!use_translation_prior_ || params.lambda_translation_prior <= 0.0) {
        return 0.0;
    }
    return params.lambda_translation_prior * (params.t - t_prior_).squaredNorm();
}

Eigen::VectorXd EnergyFunction::computeTranslationPriorResiduals(const OptimizationParams& params) const {
    if (!use_translation_prior_ || params.lambda_translation_prior <= 0.0) {
        return Eigen::VectorXd(0);
    }
    double w = std::sqrt(params.lambda_translation_prior);
    Eigen::VectorXd r(3);
    r(0) = w * (params.t(0) - t_prior_(0));
    r(1) = w * (params.t(1) - t_prior_(1));
    r(2) = w * (params.t(2) - t_prior_(2));
    return r;
}

Eigen::VectorXd EnergyFunction::computeResiduals(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping,
    const cv::Mat& observed_depth) const {
    
    Eigen::VectorXd lm_residuals = computeLandmarkResiduals(params, landmarks, mapping);
    Eigen::VectorXd depth_residuals = computeDepthResiduals(params, observed_depth);
    Eigen::VectorXd reg_residuals = computeRegResiduals(params);
    // Prior is applied in the normal equations (GaussNewton), not as residuals, to avoid scale imbalance
    int total_size = lm_residuals.size() + depth_residuals.size() + reg_residuals.size();
    Eigen::VectorXd residuals(total_size);
    
    int idx = 0;
    if (lm_residuals.size() > 0) {
        residuals.segment(idx, lm_residuals.size()) = lm_residuals;
        idx += lm_residuals.size();
    }
    if (depth_residuals.size() > 0) {
        residuals.segment(idx, depth_residuals.size()) = depth_residuals;
        idx += depth_residuals.size();
    }
    if (reg_residuals.size() > 0) {
        residuals.segment(idx, reg_residuals.size()) = reg_residuals;
    }
    return residuals;
}

Eigen::MatrixXd EnergyFunction::computeJacobian(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping,
    const cv::Mat& observed_depth) const {
    
    // Compute residuals at current point
    Eigen::VectorXd r0 = computeResiduals(params, landmarks, mapping, observed_depth);
    int num_residuals = r0.size();
    int num_params = params.numParameters();
    
    if (num_residuals == 0 || num_params == 0) {
        return Eigen::MatrixXd();
    }
    
    Eigen::MatrixXd J(num_residuals, num_params);
    
    // Numerical differentiation: J_ij = (r_i(p + eps*e_j) - r_i(p)) / eps
    Eigen::VectorXd p0 = params.pack();
    
    for (int j = 0; j < num_params; ++j) {
        // Perturb parameter j
        Eigen::VectorXd p_plus = p0;
        p_plus(j) += epsilon_;
        
        // Create perturbed params
        OptimizationParams params_plus = params;
        params_plus.unpack(p_plus);
        
        // Compute perturbed residuals
        Eigen::VectorXd r_plus = computeResiduals(params_plus, landmarks, mapping, observed_depth);
        
        // Finite difference
        J.col(j) = (r_plus - r0) / epsilon_;
    }
    
    return J;
}

} // namespace face_reconstruction
