/**
 * Landmark Mapping
 * 
 * Manages correspondence between 2D landmark indices and 3D model vertex indices.
 * Used by pose_init and validate_mapping tools.
 */

#include "alignment/LandmarkMapping.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace face_reconstruction {

bool LandmarkMapping::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open landmark mapping file: " << filepath << std::endl;
        return false;
    }
    
    mapping_.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
        
        std::istringstream iss(line);
        int landmark_idx, model_vertex_idx;
        
        if (iss >> landmark_idx >> model_vertex_idx) {
            mapping_[landmark_idx] = model_vertex_idx;
        }
    }
    
    file.close();
    
    if (mapping_.empty()) {
        std::cerr << "Warning: No mappings loaded from file" << std::endl;
        return false;
    }
    
    std::cout << "Loaded " << mapping_.size() << " landmark-to-model mappings" << std::endl;
    return true;
}

bool LandmarkMapping::saveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    file << "# Landmark to Model Vertex Mapping" << std::endl;
    file << "# Format: landmark_index model_vertex_index" << std::endl;
    file << "# One line per correspondence" << std::endl;
    file << std::endl;
    
    for (const auto& pair : mapping_) {
        file << pair.first << " " << pair.second << std::endl;
    }
    
    file.close();
    return true;
}

std::vector<int> LandmarkMapping::getMappedLandmarks() const {
    std::vector<int> landmarks;
    landmarks.reserve(mapping_.size());
    for (const auto& pair : mapping_) {
        landmarks.push_back(pair.first);
    }
    return landmarks;
}

} // namespace face_reconstruction

