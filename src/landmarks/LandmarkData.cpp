#include "landmarks/LandmarkData.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace face_reconstruction {

bool LandmarkData::loadFromTXT(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open landmark file: " << filepath << std::endl;
        return false;
    }
    
    landmarks_.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
        
        std::istringstream iss(line);
        double x, y;
        int model_index = -1;
        
        iss >> x >> y;
        if (!iss.fail()) {
            iss >> model_index;  // Optional model index
            landmarks_.emplace_back(x, y, model_index);
        }
    }
    
    file.close();
    return true;
}

bool LandmarkData::loadFromJSON(const std::string& filepath) {
    // Simple JSON parser for landmark format
    // For a more robust solution, consider using a JSON library (nlohmann/json)
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file: " << filepath << std::endl;
        return false;
    }
    
    // For now, fall back to TXT format or implement simple JSON parsing
    // This is a placeholder - in production, use a proper JSON library
    std::cerr << "Warning: JSON loading not fully implemented. "
              << "Please use TXT format for now or add a JSON library." << std::endl;
    
    file.close();
    return false;
}

bool LandmarkData::saveToTXT(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    for (const auto& lm : landmarks_) {
        file << std::fixed << std::setprecision(2) 
             << lm.x << " " << lm.y << " " << lm.model_index << std::endl;
    }
    
    file.close();
    return true;
}

bool LandmarkData::saveToJSON(const std::string& filepath) const {
    // Simple JSON writer
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    file << "{\n";
    file << "  \"landmarks\": [\n";
    
    for (size_t i = 0; i < landmarks_.size(); ++i) {
        const auto& lm = landmarks_[i];
        file << "    {\"x\": " << lm.x 
             << ", \"y\": " << lm.y
             << ", \"model_index\": " << lm.model_index << "}";
        
        if (i < landmarks_.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    return true;
}

Eigen::MatrixXd LandmarkData::toMatrix() const {
    Eigen::MatrixXd matrix(landmarks_.size(), 2);
    for (size_t i = 0; i < landmarks_.size(); ++i) {
        matrix(i, 0) = landmarks_[i].x;
        matrix(i, 1) = landmarks_[i].y;
    }
    return matrix;
}

std::vector<int> LandmarkData::getModelIndices() const {
    std::vector<int> indices;
    indices.reserve(landmarks_.size());
    for (const auto& lm : landmarks_) {
        indices.push_back(lm.model_index);
    }
    return indices;
}

} // namespace face_reconstruction
