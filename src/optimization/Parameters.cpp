/**
 * Optimization Parameters Implementation
 * 
 * Handles parameter packing/unpacking for optimization and
 * rotation matrix <-> axis-angle conversion.
 */

#include "optimization/Parameters.h"
#include <cmath>
#include <iostream>

namespace face_reconstruction {

Eigen::VectorXd OptimizationParams::pack() const {
    Eigen::VectorXd params(numParameters());
    int idx = 0;
    
    if (optimize_identity) {
        params.segment(idx, alpha.size()) = alpha;
        idx += alpha.size();
    }
    
    if (optimize_expression) {
        params.segment(idx, delta.size()) = delta;
        idx += delta.size();
    }
    
    return params;
}

void OptimizationParams::unpack(const Eigen::VectorXd& params) {
    int idx = 0;
    
    if (optimize_identity) {
        alpha = params.segment(idx, alpha.size());
        idx += alpha.size();
    }
    
    if (optimize_expression) {
        delta = params.segment(idx, delta.size());
        idx += delta.size();
    }
}

void OptimizationParams::applyUpdate(const Eigen::VectorXd& delta_params) {
    int idx = 0;
    
    if (optimize_identity) {
        alpha += delta_params.segment(idx, alpha.size());
        idx += alpha.size();
    }
    
    if (optimize_expression) {
        delta += delta_params.segment(idx, this->delta.size());
        idx += this->delta.size();
    }
    
    // Per-coefficient clipping: clamp to ± coeff_clip_sigma * sigma (reduces spikes in under-constrained regions)
    if (identity_stddev.size() > 0) {
        for (int i = 0; i < alpha.size() && i < identity_stddev.size(); ++i) {
            double limit = coeff_clip_sigma * identity_stddev(i);
            if (limit > 0) {
                alpha(i) = std::max(-limit, std::min(limit, alpha(i)));
            }
        }
    }
    if (expression_stddev.size() > 0) {
        for (int i = 0; i < delta.size() && i < expression_stddev.size(); ++i) {
            double limit = coeff_clip_sigma * expression_stddev(i);
            if (limit > 0) {
                delta(i) = std::max(-limit, std::min(limit, delta(i)));
            }
        }
    }
}

} // namespace face_reconstruction
