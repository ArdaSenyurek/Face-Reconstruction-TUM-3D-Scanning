#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace face_reconstruction {

/**
 * Container for RGB-D frame data
 */
class RGBDFrame {
public:
    RGBDFrame() = default;
    
    /**
     * Load RGB image from file
     * @param rgb_path Path to RGB image file (e.g., PNG, JPEG)
     * @return true if successful, false otherwise
     */
    bool loadRGB(const std::string& rgb_path);
    
    /**
     * Load depth map from file
     * Supports: PNG (16-bit), binary float files, or OpenCV formats
     * @param depth_path Path to depth image file
     * @param scale_factor Scale factor to convert depth values to meters (default: 1000.0 for mm->m)
     * @return true if successful, false otherwise
     */
    bool loadDepth(const std::string& depth_path, double scale_factor = 1000.0);
    
    /**
     * Get RGB image (const)
     */
    const cv::Mat& getRGB() const { return rgb_image_; }
    
    /**
     * Get RGB image (non-const)
     */
    cv::Mat& getRGB() { return rgb_image_; }
    
    /**
     * Get depth map (const)
     */
    const cv::Mat& getDepth() const { return depth_image_; }
    
    /**
     * Get depth map (non-const)
     */
    cv::Mat& getDepth() { return depth_image_; }
    
    /**
     * Check if depth value is valid (not NaN, not zero, within reasonable range)
     */
    bool isValidDepth(float depth_value, float min_depth = 0.1f, float max_depth = 10.0f) const;
    
    /**
     * Create a mask for valid depth pixels
     */
    cv::Mat createValidDepthMask(float min_depth = 0.1f, float max_depth = 10.0f) const;
    
    /**
     * Get frame width
     */
    int width() const { return rgb_image_.cols; }
    
    /**
     * Get frame height
     */
    int height() const { return rgb_image_.rows; }
    
    /**
     * Print basic statistics about the frame
     */
    void printStats() const;

private:
    cv::Mat rgb_image_;   // RGB image (BGR format from OpenCV)
    cv::Mat depth_image_; // Depth map (single channel, float)
};

} // namespace face_reconstruction
