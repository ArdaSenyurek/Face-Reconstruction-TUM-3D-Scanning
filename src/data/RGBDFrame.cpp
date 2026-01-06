/**
 * RGB-D Frame
 * 
 * Handles loading and management of RGB and depth images.
 * Core data structure for RGB-D processing pipeline.
 */

#include "data/RGBDFrame.h"
#include <iostream>
#include <limits>
#include <cmath>

namespace face_reconstruction {

bool RGBDFrame::loadRGB(const std::string& rgb_path) {
    rgb_image_ = cv::imread(rgb_path, cv::IMREAD_COLOR);
    if (rgb_image_.empty()) {
        std::cerr << "Failed to load RGB image: " << rgb_path << std::endl;
        return false;
    }
    return true;
}

bool RGBDFrame::loadDepth(const std::string& depth_path, double scale_factor) {
    // Try loading as 16-bit PNG first (common format for depth maps)
    cv::Mat depth_16bit = cv::imread(depth_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    
    if (depth_16bit.empty()) {
        std::cerr << "Failed to load depth image: " << depth_path << std::endl;
        return false;
    }
    
    // Convert to single channel if needed
    if (depth_16bit.channels() > 1) {
        cv::cvtColor(depth_16bit, depth_16bit, cv::COLOR_BGR2GRAY);
    }
    
    // Convert to float and apply scale factor
    depth_16bit.convertTo(depth_image_, CV_32F, 1.0 / scale_factor);
    
    // Set zero values to NaN (invalid depth)
    cv::Mat mask = (depth_image_ == 0);
    depth_image_.setTo(std::numeric_limits<float>::quiet_NaN(), mask);
    
    return true;
}

bool RGBDFrame::isValidDepth(float depth_value, float min_depth, float max_depth) const {
    if (std::isnan(depth_value) || std::isinf(depth_value)) {
        return false;
    }
    if (depth_value <= min_depth || depth_value >= max_depth) {
        return false;
    }
    return true;
}

cv::Mat RGBDFrame::createValidDepthMask(float min_depth, float max_depth) const {
    if (depth_image_.empty()) {
        return cv::Mat();
    }
    
    cv::Mat mask(depth_image_.size(), CV_8UC1, cv::Scalar(0));
    
    for (int y = 0; y < depth_image_.rows; ++y) {
        for (int x = 0; x < depth_image_.cols; ++x) {
            float d = depth_image_.at<float>(y, x);
            if (isValidDepth(d, min_depth, max_depth)) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    
    return mask;
}

void RGBDFrame::printStats() const {
    std::cout << "=== RGB-D Frame Statistics ===" << std::endl;
    std::cout << "RGB: " << width() << "x" << height() << std::endl;
    
    if (!depth_image_.empty()) {
        std::cout << "Depth: " << depth_image_.cols << "x" << depth_image_.rows << std::endl;
        
        // Calculate depth statistics
        cv::Mat valid_mask = createValidDepthMask();
        int valid_pixels = cv::countNonZero(valid_mask);
        int total_pixels = depth_image_.rows * depth_image_.cols;
        
        std::cout << "Valid depth pixels: " << valid_pixels << " / " << total_pixels 
                  << " (" << (100.0 * valid_pixels / total_pixels) << "%)" << std::endl;
        
        if (valid_pixels > 0) {
            cv::Scalar mean_val, stddev_val;
            cv::Mat depth_masked;
            depth_image_.copyTo(depth_masked, valid_mask);
            cv::meanStdDev(depth_masked, mean_val, stddev_val, valid_mask);
            
            double min_val, max_val;
            cv::minMaxLoc(depth_masked, &min_val, &max_val, nullptr, nullptr, valid_mask);
            std::cout << "Depth range: [" << min_val << ", " << max_val << "] meters" << std::endl;
            std::cout << "Mean depth: " << mean_val[0] << " meters" << std::endl;
            std::cout << "Std dev: " << stddev_val[0] << " meters" << std::endl;
        }
    } else {
        std::cout << "Depth: Not loaded" << std::endl;
    }
    std::cout << "==============================" << std::endl;
}

} // namespace face_reconstruction
