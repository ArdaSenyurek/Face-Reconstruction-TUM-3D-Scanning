#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace face_reconstruction {

/**
 * 2D landmark point
 */
struct Landmark2D {
    double x;  // Pixel x coordinate
    double y;  // Pixel y coordinate
    int model_index;  // Index in the morphable model (which vertex this corresponds to)
    
    Landmark2D() : x(0.0), y(0.0), model_index(-1) {}
    Landmark2D(double x_, double y_, int idx = -1) : x(x_), y(y_), model_index(idx) {}
};

/**
 * Container for 2D landmarks detected in an image
 */
class LandmarkData {
public:
    LandmarkData() = default;
    
    /**
     * Load landmarks from JSON file
     * Expected format:
     * {
     *   "landmarks": [
     *     {"x": 100.5, "y": 200.3, "model_index": 0},
     *     ...
     *   ]
     * }
     */
    bool loadFromJSON(const std::string& filepath);
    
    /**
     * Load landmarks from simple text file
     * Format: one landmark per line
     * x y model_index
     */
    bool loadFromTXT(const std::string& filepath);
    
    /**
     * Save landmarks to JSON file
     */
    bool saveToJSON(const std::string& filepath) const;
    
    /**
     * Save landmarks to TXT file
     */
    bool saveToTXT(const std::string& filepath) const;
    
    /**
     * Add a landmark
     */
    void addLandmark(double x, double y, int model_index = -1) {
        landmarks_.emplace_back(x, y, model_index);
    }
    
    /**
     * Get all landmarks
     */
    const std::vector<Landmark2D>& getLandmarks() const { return landmarks_; }
    
    /**
     * Get number of landmarks
     */
    size_t size() const { return landmarks_.size(); }
    
    /**
     * Get landmark at index
     */
    const Landmark2D& operator[](size_t idx) const { return landmarks_[idx]; }
    
    /**
     * Clear all landmarks
     */
    void clear() { landmarks_.clear(); }
    
    /**
     * Convert to Eigen matrix (N x 2) for easier computation
     */
    Eigen::MatrixXd toMatrix() const;
    
    /**
     * Get 3D model vertex indices that correspond to these landmarks
     * (useful for Procrustes alignment)
     */
    std::vector<int> getModelIndices() const;

private:
    std::vector<Landmark2D> landmarks_;
};

} // namespace face_reconstruction
