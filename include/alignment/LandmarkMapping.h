#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace face_reconstruction {

/**
 * Mapping from landmark indices to model vertex indices
 * Used for sparse alignment in pose initialization
 */
class LandmarkMapping {
public:
    LandmarkMapping() = default;
    
    /**
     * Load mapping from file
     * Format: landmark_index model_vertex_index
     * One line per correspondence
     * 
     * Example:
     *   0 1234
     *   1 5678
     *   ...
     */
    bool loadFromFile(const std::string& filepath);
    
    /**
     * Save mapping to file
     */
    bool saveToFile(const std::string& filepath) const;
    
    /**
     * Add a mapping
     */
    void addMapping(int landmark_index, int model_vertex_index) {
        mapping_[landmark_index] = model_vertex_index;
    }
    
    /**
     * Get model vertex index for a landmark index
     * Returns -1 if not found
     */
    int getModelVertex(int landmark_index) const {
        auto it = mapping_.find(landmark_index);
        if (it != mapping_.end()) {
            return it->second;
        }
        return -1;
    }
    
    /**
     * Get all landmark indices that have mappings
     */
    std::vector<int> getMappedLandmarks() const;
    
    /**
     * Get number of mappings
     */
    size_t size() const { return mapping_.size(); }
    
    /**
     * Check if a landmark has a mapping
     */
    bool hasMapping(int landmark_index) const {
        return mapping_.find(landmark_index) != mapping_.end();
    }
    
    /**
     * Clear all mappings
     */
    void clear() { mapping_.clear(); }

private:
    std::map<int, int> mapping_;  // landmark_index -> model_vertex_index
};

} // namespace face_reconstruction

