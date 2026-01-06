/**
 * STEP 4: Dense Depth Residual Computation
 * 
 * Goal: Compute dense depth residuals between observed and rendered depth.
 * 
 * Usage:
 *   bin/test_depth_residuals <observed_depth> <rendered_depth> [output_residual_image]
 * 
 * Example:
 *   bin/test_depth_residuals data/biwi_person01/depth/frame_00000.png \
 *                            build/rendered_depth.png \
 *                            build/residual_heatmap.png
 */

#include "data/RGBDFrame.h"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace face_reconstruction;

/**
 * Compute depth residuals and statistics
 */
void computeResiduals(const cv::Mat& observed_depth, const cv::Mat& rendered_depth,
                      cv::Mat& residual_map, cv::Mat& residual_mask,
                      double& mean_abs_error, double& max_error, double& min_error,
                      int& valid_pixels) {
    
    residual_map = cv::Mat::zeros(observed_depth.size(), CV_32F);
    residual_mask = cv::Mat::zeros(observed_depth.size(), CV_8U);
    
    valid_pixels = 0;
    mean_abs_error = 0.0;
    max_error = std::numeric_limits<double>::lowest();
    min_error = std::numeric_limits<double>::max();
    
    double sum_abs_error = 0.0;
    
    for (int v = 0; v < observed_depth.rows; ++v) {
        for (int u = 0; u < observed_depth.cols; ++u) {
            float obs = observed_depth.at<float>(v, u);
            float ren = rendered_depth.at<float>(v, u);
            
            // Both must be valid
            bool obs_valid = !std::isnan(obs) && obs > 0.0;
            bool ren_valid = !std::isnan(ren) && ren > 0.0;
            
            if (obs_valid && ren_valid) {
                float residual = obs - ren;  // observed - rendered
                residual_map.at<float>(v, u) = residual;
                residual_mask.at<uchar>(v, u) = 255;
                
                double abs_error = std::abs(residual);
                sum_abs_error += abs_error;
                valid_pixels++;
                
                if (residual > max_error) max_error = residual;
                if (residual < min_error) min_error = residual;
            }
        }
    }
    
    if (valid_pixels > 0) {
        mean_abs_error = sum_abs_error / valid_pixels;
    }
}

/**
 * Create heatmap visualization of residuals
 */
cv::Mat createResidualHeatmap(const cv::Mat& residual_map, const cv::Mat& mask,
                              double min_val, double max_val) {
    cv::Mat heatmap = cv::Mat::zeros(residual_map.size(), CV_8UC3);
    
    // Normalize to [0, 1] then map to color
    double range = max_val - min_val;
    if (range < 1e-10) {
        range = 1.0;
    }
    
    for (int v = 0; v < residual_map.rows; ++v) {
        for (int u = 0; u < residual_map.cols; ++u) {
            if (mask.at<uchar>(v, u) > 0) {
                float residual = residual_map.at<float>(v, u);
                double normalized = (residual - min_val) / range;
                
                // Color mapping: blue (negative) -> green (zero) -> red (positive)
                cv::Vec3b color;
                if (normalized < 0.5) {
                    // Blue to green
                    double t = normalized * 2.0;
                    color = cv::Vec3b(0, static_cast<uchar>(t * 255), static_cast<uchar>((1.0 - t) * 255));
                } else {
                    // Green to red
                    double t = (normalized - 0.5) * 2.0;
                    color = cv::Vec3b(0, static_cast<uchar>((1.0 - t) * 255), static_cast<uchar>(t * 255));
                }
                
                heatmap.at<cv::Vec3b>(v, u) = color;
            }
        }
    }
    
    return heatmap;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <observed_depth> <rendered_depth> [output_residual_image]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/biwi_person01/depth/frame_00000.png \\" << std::endl;
        std::cerr << "     build/rendered_depth.png \\" << std::endl;
        std::cerr << "     build/residual_heatmap.png" << std::endl;
        return 1;
    }
    
    std::string observed_depth_path = argv[1];
    std::string rendered_depth_path = argv[2];
    std::string output_residual = (argc > 3) ? argv[3] : "residual_heatmap.png";
    
    std::cout << "=== STEP 4: Dense Depth Residual Computation ===" << std::endl;
    std::cout << "Observed depth: " << observed_depth_path << std::endl;
    std::cout << "Rendered depth: " << rendered_depth_path << std::endl;
    std::cout << "Output residual: " << output_residual << std::endl;
    std::cout << std::endl;
    
    // Load observed depth
    RGBDFrame frame;
    if (!frame.loadDepth(observed_depth_path, 1000.0)) {
        std::cerr << "Failed to load observed depth" << std::endl;
        return 1;
    }
    const cv::Mat& observed_depth = frame.getDepth();
    
    // Try to load raw float depth first (more accurate)
    std::string float_depth_path = rendered_depth_path;
    size_t last_dot = float_depth_path.find_last_of(".");
    if (last_dot != std::string::npos) {
        float_depth_path = float_depth_path.substr(0, last_dot) + "_float.bin";
    } else {
        float_depth_path += "_float.bin";
    }
    
    cv::Mat rendered_depth;
    std::ifstream float_file(float_depth_path, std::ios::binary);
    if (float_file.is_open()) {
        // Load from binary file
        int width, height;
        double min_depth, max_depth;
        float_file.read(reinterpret_cast<char*>(&width), sizeof(int));
        float_file.read(reinterpret_cast<char*>(&height), sizeof(int));
        float_file.read(reinterpret_cast<char*>(&min_depth), sizeof(double));
        float_file.read(reinterpret_cast<char*>(&max_depth), sizeof(double));
        
        rendered_depth = cv::Mat::zeros(height, width, CV_32F);
        for (int v = 0; v < height; ++v) {
            for (int u = 0; u < width; ++u) {
                float depth;
                float_file.read(reinterpret_cast<char*>(&depth), sizeof(float));
                rendered_depth.at<float>(v, u) = depth;
            }
        }
        float_file.close();
        std::cout << "Loaded raw float depth from: " << float_depth_path << std::endl;
        std::cout << "  Range: [" << min_depth << ", " << max_depth << "] meters" << std::endl;
    } else {
        // Fallback: Load from PNG (may be normalized)
        cv::Mat rendered_16bit = cv::imread(rendered_depth_path, cv::IMREAD_ANYDEPTH);
        if (rendered_16bit.empty()) {
            std::cerr << "Failed to load rendered depth" << std::endl;
            return 1;
        }
        
        // Check if image is normalized (max value close to 65535)
        double max_val = 0.0;
        cv::minMaxLoc(rendered_16bit, nullptr, &max_val);
        
        if (max_val > 10000) {
            // Image is normalized, use observed depth range as estimate
            double obs_min = std::numeric_limits<double>::max();
            double obs_max = std::numeric_limits<double>::lowest();
            for (int v = 0; v < observed_depth.rows; ++v) {
                for (int u = 0; u < observed_depth.cols; ++u) {
                    float d = observed_depth.at<float>(v, u);
                    if (frame.isValidDepth(d)) {
                        if (d < obs_min) obs_min = d;
                        if (d > obs_max) obs_max = d;
                    }
                }
            }
            
            // Use observed range as estimate for rendered depth
            double range_m = obs_max - obs_min;
            rendered_16bit.convertTo(rendered_depth, CV_32F, range_m / 65535.0, obs_min);
            
            std::cout << "Note: Rendered depth appears normalized, using observed range [" 
                      << obs_min << ", " << obs_max << "] meters" << std::endl;
        } else {
            // Image is in mm, convert to meters
            rendered_16bit.convertTo(rendered_depth, CV_32F, 1.0 / 1000.0);
        }
        
        // Set zero values to NaN
        cv::Mat mask_zero = (rendered_depth == 0.0f);
        rendered_depth.setTo(std::numeric_limits<float>::quiet_NaN(), mask_zero);
    }
    
    std::cout << "Image dimensions: " << observed_depth.cols << " x " << observed_depth.rows << std::endl;
    std::cout << std::endl;
    
    // Compute residuals
    std::cout << "--- Computing Depth Residuals ---" << std::endl;
    cv::Mat residual_map, residual_mask;
    double mean_abs_error, max_error, min_error;
    int valid_pixels;
    
    computeResiduals(observed_depth, rendered_depth, residual_map, residual_mask,
                     mean_abs_error, max_error, min_error, valid_pixels);
    
    std::cout << "Valid pixels for residual computation: " << valid_pixels << std::endl;
    std::cout << std::endl;
    
    if (valid_pixels == 0) {
        std::cerr << "ERROR: No valid pixels found for residual computation!" << std::endl;
        std::cerr << "Make sure rendered depth overlaps with observed depth." << std::endl;
        return 1;
    }
    
    // Print statistics
    std::cout << "--- Residual Statistics ---" << std::endl;
    std::cout << "Valid pixels: " << valid_pixels << std::endl;
    std::cout << "Mean absolute error: " << std::fixed << std::setprecision(4) 
              << mean_abs_error << " meters (" << mean_abs_error * 1000.0 << " mm)" << std::endl;
    std::cout << "Min residual: " << min_error << " meters (" << min_error * 1000.0 << " mm)" << std::endl;
    std::cout << "Max residual: " << max_error << " meters (" << max_error * 1000.0 << " mm)" << std::endl;
    std::cout << "Residual range: [" << min_error << ", " << max_error << "] meters" << std::endl;
    std::cout << std::endl;
    
    // Compute histogram (optional)
    std::cout << "--- Residual Distribution ---" << std::endl;
    std::vector<double> residuals;
    residuals.reserve(valid_pixels);
    
    for (int v = 0; v < residual_map.rows; ++v) {
        for (int u = 0; u < residual_map.cols; ++u) {
            if (residual_mask.at<uchar>(v, u) > 0) {
                residuals.push_back(residual_map.at<float>(v, u));
            }
        }
    }
    
    std::sort(residuals.begin(), residuals.end());
    
    if (!residuals.empty()) {
        double median = residuals[residuals.size() / 2];
        double q25 = residuals[residuals.size() / 4];
        double q75 = residuals[residuals.size() * 3 / 4];
        
        std::cout << "Median residual: " << std::fixed << std::setprecision(4) 
                  << median << " meters (" << median * 1000.0 << " mm)" << std::endl;
        std::cout << "Q25 (25th percentile): " << q25 << " meters (" << q25 * 1000.0 << " mm)" << std::endl;
        std::cout << "Q75 (75th percentile): " << q75 << " meters (" << q75 * 1000.0 << " mm)" << std::endl;
    }
    std::cout << std::endl;
    
    // Create heatmap visualization
    std::cout << "--- Creating Residual Heatmap ---" << std::endl;
    cv::Mat heatmap = createResidualHeatmap(residual_map, residual_mask, min_error, max_error);
    
    if (cv::imwrite(output_residual, heatmap)) {
        std::cout << "✓ Successfully saved residual heatmap to: " << output_residual << std::endl;
        std::cout << "  Color coding: Blue (negative) -> Green (zero) -> Red (positive)" << std::endl;
    } else {
        std::cerr << "Failed to save residual heatmap" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== STEP 4 Summary ===" << std::endl;
    std::cout << "✓ Depth residuals computed" << std::endl;
    std::cout << "✓ Mean absolute error: " << mean_abs_error * 1000.0 << " mm" << std::endl;
    std::cout << "✓ Residual range: [" << min_error * 1000.0 << ", " << max_error * 1000.0 << "] mm" << std::endl;
    std::cout << "✓ Residual heatmap saved: " << output_residual << std::endl;
    std::cout << std::endl;
    
    // Interpretation
    if (mean_abs_error < 0.01) {  // < 1cm
        std::cout << "✓ Excellent consistency (mean error < 1cm)" << std::endl;
    } else if (mean_abs_error < 0.02) {  // < 2cm
        std::cout << "✓ Good consistency (mean error < 2cm)" << std::endl;
    } else if (mean_abs_error < 0.05) {  // < 5cm
        std::cout << "⚠ Moderate consistency (mean error < 5cm)" << std::endl;
        std::cout << "  Consider improving alignment or checking depth quality" << std::endl;
    } else {
        std::cout << "⚠ High residual error (> 5cm)" << std::endl;
        std::cout << "  Alignment may need improvement" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "=== STEP 4 Complete ===" << std::endl;
    std::cout << "✓ Dense depth residual computation completed!" << std::endl;
    std::cout << "  Next: ICP validation (STEP 5)" << std::endl;
    
    return 0;
}

