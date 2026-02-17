/**
 * Energy Function Implementation
 * 
 * Computes energy terms for face reconstruction optimization:
 * - Sparse landmark reprojection error (noise-normalized)
 * - Dense depth alignment error (masked + gated + robust Huber)
 * - Coefficient regularization
 */

#include "optimization/EnergyFunction.h"
#include "rendering/DepthRenderer.h"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace face_reconstruction {

// Hardcoded robust constants (no CLI flags)
static constexpr float DEPTH_SIGMA_M = 0.01f;   // 10 mm depth noise for normalization
static constexpr float HUBER_K       = 1.345f;   // Huber threshold in sigma units
static constexpr float LM_SIGMA_PX   = 2.0f;    // Landmark detection noise in pixels

void EnergyFunction::initialize(const MorphableModel& model,
                                const CameraIntrinsics& intrinsics,
                                int image_width, int image_height) {
    model_ = &model;
    intrinsics_ = intrinsics;
    image_width_ = image_width;
    image_height_ = image_height;
    depth_renderer_.initialize(intrinsics_, image_width_, image_height_);
    initialized_ = true;
}

void EnergyFunction::setDepthMask(const cv::Mat& mask) {
    depth_mask_ = mask.clone();
    has_mask_ = true;
}

cv::Mat EnergyFunction::buildLandmarkRoiMask(
    const LandmarkData& landmarks, int W, int H, int margin) {
    cv::Mat mask = cv::Mat::zeros(H, W, CV_8U);
    if (landmarks.size() == 0) return mask;

    double min_x = W, max_x = 0, min_y = H, max_y = 0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const auto& lm = landmarks[i];
        if (lm.x >= 0 && lm.x < W && lm.y >= 0 && lm.y < H) {
            min_x = std::min(min_x, lm.x);
            max_x = std::max(max_x, lm.x);
            min_y = std::min(min_y, lm.y);
            max_y = std::max(max_y, lm.y);
        }
    }

    int x0 = std::max(0, static_cast<int>(min_x) - margin);
    int y0 = std::max(0, static_cast<int>(min_y) - margin);
    int x1 = std::min(W - 1, static_cast<int>(max_x) + margin);
    int y1 = std::min(H - 1, static_cast<int>(max_y) + margin);

    cv::rectangle(mask, cv::Point(x0, y0), cv::Point(x1, y1),
                  cv::Scalar(255), cv::FILLED);
    return mask;
}

Eigen::MatrixXd EnergyFunction::reconstructMesh(const OptimizationParams& params) const {
    if (!model_ || !model_->isValid()) {
        return Eigen::MatrixXd();
    }
    return model_->reconstructFace(params.alpha, params.delta);
}

Eigen::MatrixXd EnergyFunction::applyPose(const Eigen::MatrixXd& vertices,
                                          const OptimizationParams& params) const {
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
    if (point.z() <= 0.0) {
        return Eigen::Vector2d(-1, -1);
    }
    double u = intrinsics_.fx * point.x() / point.z() + intrinsics_.cx;
    double v = intrinsics_.fy * point.y() / point.z() + intrinsics_.cy;
    return Eigen::Vector2d(u, v);
}

cv::Mat EnergyFunction::renderDepth(const Eigen::MatrixXd& vertices) const {
    if (!model_ || model_->faces.rows() == 0) {
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
    
    cv::Rect roi;
    if (has_mask_ && !depth_mask_.empty()) {
        roi = cv::boundingRect(depth_mask_);
    }
    return depth_renderer_.renderDepth(vertices, model_->faces, roi);
}

// ---------------------------------------------------------------------------
// Landmark energy (noise-normalized by LM_SIGMA_PX)
// ---------------------------------------------------------------------------

double EnergyFunction::computeLandmarkEnergy(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping) const {
    
    if (!initialized_ || !model_) return 0.0;
    
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return 0.0;
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    
    double energy = 0.0;
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        int lm_idx = static_cast<int>(i);
        if (!mapping.hasMapping(lm_idx)) continue;
        
        int vertex_idx = mapping.getModelVertex(lm_idx);
        if (vertex_idx < 0 || vertex_idx >= transformed.rows()) continue;
        
        Eigen::Vector3d vertex_3d = transformed.row(vertex_idx);
        Eigen::Vector2d projected = projectPoint(vertex_3d);
        
        if (projected.x() < 0) continue;
        
        const auto& lm = landmarks[i];
        double dx = projected.x() - lm.x;
        double dy = projected.y() - lm.y;
        energy += (dx * dx + dy * dy) / (LM_SIGMA_PX * LM_SIGMA_PX);
    }
    
    return energy;
}

// ---------------------------------------------------------------------------
// Depth energy (masked + gated + Huber robust)
// ---------------------------------------------------------------------------

double EnergyFunction::computeDepthEnergy(
    const OptimizationParams& params,
    const cv::Mat& observed_depth) const {
    
    if (!initialized_ || !model_) return 0.0;
    if (observed_depth.empty()) return 0.0;
    
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return 0.0;
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    cv::Mat rendered = renderDepth(transformed);
    
    double energy = 0.0;
    
    for (int v = 0; v < image_height_; v += depth_sample_step_) {
        for (int u = 0; u < image_width_; u += depth_sample_step_) {
            float obs_d = observed_depth.at<float>(v, u);
            float rend_d = rendered.at<float>(v, u);
            
            if (obs_d <= 0 || rend_d <= 0) continue;
            if (std::isnan(obs_d) || std::isnan(rend_d)) continue;
            if (has_mask_ && depth_mask_.at<uchar>(v, u) == 0) continue;
            
            float diff = obs_d - rend_d;
            double e = diff / DEPTH_SIGMA_M;
            double abs_e = std::abs(e);
            if (abs_e <= HUBER_K) {
                energy += 0.5 * e * e;
            } else {
                energy += HUBER_K * abs_e - 0.5 * HUBER_K * HUBER_K;
            }
        }
    }
    
    return energy;
}

int EnergyFunction::computeDepthValidCount(
    const OptimizationParams& params,
    const cv::Mat& observed_depth) const {
    if (!initialized_ || !model_) return 0;
    if (observed_depth.empty()) return 0;
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return 0;
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    cv::Mat rendered = renderDepth(transformed);
    int count = 0;
    for (int v = 0; v < image_height_; v += depth_sample_step_) {
        for (int u = 0; u < image_width_; u += depth_sample_step_) {
            float obs_d = observed_depth.at<float>(v, u);
            float rend_d = rendered.at<float>(v, u);
            if (obs_d <= 0 || rend_d <= 0) continue;
            if (std::isnan(obs_d) || std::isnan(rend_d)) continue;
            if (has_mask_ && depth_mask_.at<uchar>(v, u) == 0) continue;
            count++;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// Regularization (unchanged)
// ---------------------------------------------------------------------------

double EnergyFunction::computeRegularization(const OptimizationParams& params) const {
    if (!initialized_ || !model_) return 0.0;
    
    double energy = 0.0;
    
    if (params.alpha.size() > 0 && model_->identity_stddev.size() > 0) {
        for (int i = 0; i < params.alpha.size() && i < model_->identity_stddev.size(); ++i) {
            double sigma = model_->identity_stddev(i);
            if (sigma > 1e-10) {
                double normalized = params.alpha(i) / sigma;
                energy += params.lambda_alpha * normalized * normalized;
            }
        }
    }
    
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
    
    return params.lambda_landmark * e_landmark + 
           params.lambda_depth * e_depth + 
           e_reg;
}

// ---------------------------------------------------------------------------
// Landmark residuals (noise-normalized by LM_SIGMA_PX)
// ---------------------------------------------------------------------------

Eigen::VectorXd EnergyFunction::computeLandmarkResiduals(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping) const {
    
    if (!initialized_ || !model_) return Eigen::VectorXd();
    
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return Eigen::VectorXd();
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    
    int num_valid = 0;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        if (mapping.hasMapping(static_cast<int>(i))) {
            int vertex_idx = mapping.getModelVertex(static_cast<int>(i));
            if (vertex_idx >= 0 && vertex_idx < transformed.rows()) {
                num_valid++;
            }
        }
    }
    
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
            residuals(idx * 2)     = weight * (projected.x() - lm.x) / LM_SIGMA_PX;
            residuals(idx * 2 + 1) = weight * (projected.y() - lm.y) / LM_SIGMA_PX;
        } else {
            residuals(idx * 2) = 0;
            residuals(idx * 2 + 1) = 0;
        }
        idx++;
    }
    
    return residuals;
}

// ---------------------------------------------------------------------------
// Depth residuals (masked + Huber IRLS weight, noise-normalized, no hard gate)
// ---------------------------------------------------------------------------

Eigen::VectorXd EnergyFunction::computeDepthResiduals(
    const OptimizationParams& params,
    const cv::Mat& observed_depth) const {
    
    if (!initialized_ || !model_ || observed_depth.empty()) {
        return Eigen::VectorXd();
    }
    
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return Eigen::VectorXd();
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    cv::Mat rendered = renderDepth(transformed);
    
    std::vector<double> residual_list;
    double weight = std::sqrt(params.lambda_depth);
    
    for (int v = 0; v < image_height_; v += depth_sample_step_) {
        for (int u = 0; u < image_width_; u += depth_sample_step_) {
            float obs_d = observed_depth.at<float>(v, u);
            float rend_d = rendered.at<float>(v, u);
            
            if (obs_d <= 0 || rend_d <= 0) continue;
            if (std::isnan(obs_d) || std::isnan(rend_d)) continue;
            if (has_mask_ && depth_mask_.at<uchar>(v, u) == 0) continue;
            
            float diff = obs_d - rend_d;
            double e = diff / DEPTH_SIGMA_M;
            double abs_e = std::abs(e);
            double w = (abs_e <= HUBER_K) ? 1.0 : (HUBER_K / abs_e);
            residual_list.push_back(weight * std::sqrt(w) * e);
        }
    }
    
    Eigen::VectorXd residuals(residual_list.size());
    for (size_t i = 0; i < residual_list.size(); ++i) {
        residuals(i) = residual_list[i];
    }
    
    return residuals;
}

// ---------------------------------------------------------------------------
// Regularization residuals (unchanged)
// ---------------------------------------------------------------------------

Eigen::VectorXd EnergyFunction::computeRegResiduals(const OptimizationParams& params) const {
    if (!initialized_ || !model_) return Eigen::VectorXd();
    
    int num_alpha = params.alpha.size();
    int num_delta = params.delta.size();
    
    Eigen::VectorXd residuals(num_alpha + num_delta);
    
    for (int i = 0; i < num_alpha; ++i) {
        double sigma = (i < model_->identity_stddev.size()) ? 
                       model_->identity_stddev(i) : 1.0;
        if (sigma < 1e-10) sigma = 1.0;
        residuals(i) = std::sqrt(params.lambda_alpha) * params.alpha(i) / sigma;
    }
    
    for (int i = 0; i < num_delta; ++i) {
        double sigma = (i < model_->expression_stddev.size()) ? 
                       model_->expression_stddev(i) : 1.0;
        if (sigma < 1e-10) sigma = 1.0;
        residuals(num_alpha + i) = std::sqrt(params.lambda_delta) * params.delta(i) / sigma;
    }
    
    return residuals;
}

// ---------------------------------------------------------------------------
// Fixed-pixel-set depth methods (for stable Jacobian computation)
// ---------------------------------------------------------------------------

std::vector<EnergyFunction::PixelCoord> EnergyFunction::collectDepthPixels(
    const OptimizationParams& params,
    const cv::Mat& observed_depth,
    cv::Mat* out_baseline_rendered) const {
    
    std::vector<PixelCoord> pixels;
    if (!initialized_ || !model_ || observed_depth.empty()) return pixels;
    
    Eigen::MatrixXd vertices = reconstructMesh(params);
    if (vertices.rows() == 0) return pixels;
    
    Eigen::MatrixXd transformed = applyPose(vertices, params);
    cv::Mat rendered = renderDepth(transformed);
    if (out_baseline_rendered && rendered.rows == image_height_ && rendered.cols == image_width_) {
        rendered.copyTo(*out_baseline_rendered);
    }
    
    for (int v = 0; v < image_height_; v += depth_sample_step_) {
        for (int u = 0; u < image_width_; u += depth_sample_step_) {
            float obs_d = observed_depth.at<float>(v, u);
            float rend_d = rendered.at<float>(v, u);
            if (obs_d <= 0 || rend_d <= 0) continue;
            if (std::isnan(obs_d) || std::isnan(rend_d)) continue;
            if (has_mask_ && depth_mask_.at<uchar>(v, u) == 0) continue;
            pixels.emplace_back(u, v);
        }
    }
    return pixels;
}

Eigen::VectorXd EnergyFunction::computeDepthResidualsFixed(
    const OptimizationParams& params,
    const cv::Mat& observed_depth,
    const std::vector<PixelCoord>& pixels,
    const cv::Mat& pre_rendered) const {
    
    if (!initialized_ || !model_ || observed_depth.empty() || pixels.empty()) {
        return Eigen::VectorXd(static_cast<int>(pixels.size()));
    }
    
    cv::Mat rendered;
    if (pre_rendered.rows == image_height_ && pre_rendered.cols == image_width_) {
        rendered = pre_rendered;
    } else {
        Eigen::MatrixXd vertices = reconstructMesh(params);
        if (vertices.rows() == 0) {
            return Eigen::VectorXd::Zero(static_cast<int>(pixels.size()));
        }
        Eigen::MatrixXd transformed = applyPose(vertices, params);
        rendered = renderDepth(transformed);
    }
    
    double weight = std::sqrt(params.lambda_depth);
    Eigen::VectorXd residuals(static_cast<int>(pixels.size()));
    
    for (size_t i = 0; i < pixels.size(); ++i) {
        int u = pixels[i].first;
        int v = pixels[i].second;
        float obs_d = observed_depth.at<float>(v, u);
        float rend_d = rendered.at<float>(v, u);
        
        if (obs_d <= 0 || rend_d <= 0 || std::isnan(obs_d) || std::isnan(rend_d)) {
            residuals(static_cast<int>(i)) = 0.0;
            continue;
        }
        
        float diff = obs_d - rend_d;
        double e = diff / DEPTH_SIGMA_M;
        double abs_e = std::abs(e);
        double w = (abs_e <= HUBER_K) ? 1.0 : (HUBER_K / abs_e);
        residuals(static_cast<int>(i)) = weight * std::sqrt(w) * e;
    }
    
    return residuals;
}

Eigen::VectorXd EnergyFunction::computeResidualsFixed(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping,
    const cv::Mat& observed_depth,
    const std::vector<PixelCoord>& depth_pixels,
    const cv::Mat& baseline_rendered) const {
    
    Eigen::VectorXd lm_residuals = computeLandmarkResiduals(params, landmarks, mapping);
    Eigen::VectorXd depth_residuals = computeDepthResidualsFixed(params, observed_depth, depth_pixels, baseline_rendered);
    Eigen::VectorXd reg_residuals = computeRegResiduals(params);
    
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

// ---------------------------------------------------------------------------
// Combined residuals + Jacobian
// ---------------------------------------------------------------------------

Eigen::VectorXd EnergyFunction::computeResiduals(
    const OptimizationParams& params,
    const LandmarkData& landmarks,
    const LandmarkMapping& mapping,
    const cv::Mat& observed_depth) const {
    
    Eigen::VectorXd lm_residuals = computeLandmarkResiduals(params, landmarks, mapping);
    Eigen::VectorXd depth_residuals = computeDepthResiduals(params, observed_depth);
    Eigen::VectorXd reg_residuals = computeRegResiduals(params);
    
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
    
    // Collect the valid depth pixel set ONCE at the baseline parameters and reuse the baseline render for r0.
    cv::Mat baseline_rendered;
    std::vector<PixelCoord> depth_pixels = collectDepthPixels(params, observed_depth, &baseline_rendered);
    
    Eigen::VectorXd r0 = computeResidualsFixed(params, landmarks, mapping, observed_depth, depth_pixels, baseline_rendered);
    int num_residuals = r0.size();
    int num_params = params.numParameters();
    
    if (num_residuals == 0 || num_params == 0) {
        return Eigen::MatrixXd();
    }
    
    Eigen::MatrixXd J(num_residuals, num_params);
    Eigen::VectorXd p0 = params.pack();
    
    for (int j = 0; j < num_params; ++j) {
        Eigen::VectorXd p_plus = p0;
        p_plus(j) += epsilon_;
        
        OptimizationParams params_plus = params;
        params_plus.unpack(p_plus);
        
        Eigen::VectorXd r_plus = computeResidualsFixed(params_plus, landmarks, mapping, observed_depth, depth_pixels);
        
        J.col(j) = (r_plus - r0) / epsilon_;
    }
    
    return J;
}

} // namespace face_reconstruction
