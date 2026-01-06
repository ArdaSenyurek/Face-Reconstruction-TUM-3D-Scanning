/**
 * STEP 1: Landmark-to-Model Vertex Mapping Validation
 * 
 * Goal: Validate that the landmark-to-model vertex mapping is correct
 * and visualize the mapped 3D model points for inspection.
 * 
 * Usage:
 *   bin/test_landmark_mapping <model_dir> <mapping_file> [output_ply]
 * 
 * Example:
 *   bin/test_landmark_mapping data/model data/landmark_mapping.txt mapped_landmarks.ply
 */

#include "model/MorphableModel.h"
#include "alignment/LandmarkMapping.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

using namespace face_reconstruction;

/**
 * Export mapped landmark points to PLY file for visualization
 */
bool saveMappedLandmarksPLY(const std::vector<Eigen::Vector3d>& points,
                            const std::vector<int>& landmark_indices,
                            const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "comment Mapped landmark points from dlib 68-point model\n";
    file << "end_header\n";
    
    // Write vertices with colors (red for visibility)
    for (const auto& point : points) {
        file << std::fixed << std::setprecision(6)
             << point.x() << " " 
             << point.y() << " " 
             << point.z() << " "
             << "255 0 0\n";  // Red color for landmarks
    }
    
    file.close();
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <model_dir> <mapping_file> [output_ply]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/model data/landmark_mapping.txt mapped_landmarks.ply" << std::endl;
        return 1;
    }
    
    std::string model_dir = argv[1];
    std::string mapping_file = argv[2];
    std::string output_ply = (argc > 3) ? argv[3] : "mapped_landmarks.ply";
    
    std::cout << "=== Landmark-to-Model Vertex Mapping Validation ===" << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Mapping file: " << mapping_file << std::endl;
    std::cout << "Output PLY: " << output_ply << std::endl;
    std::cout << std::endl;
    
    // Load model
    MorphableModel model;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "Failed to load model from: " << model_dir << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded:" << std::endl;
    model.printStats();
    std::cout << std::endl;
    
    // Load mapping
    LandmarkMapping mapping;
    if (!mapping.loadFromFile(mapping_file)) {
        std::cerr << "Failed to load mapping from: " << mapping_file << std::endl;
        return 1;
    }
    
    std::cout << "Mapping loaded: " << mapping.size() << " correspondences" << std::endl;
    std::cout << std::endl;
    
    // Validate mapping
    std::cout << "--- Validating Mapping ---" << std::endl;
    std::vector<int> mapped_landmarks = mapping.getMappedLandmarks();
    std::vector<int> invalid_vertices;
    
    for (int landmark_idx : mapped_landmarks) {
        int vertex_idx = mapping.getModelVertex(landmark_idx);
        if (vertex_idx < 0 || vertex_idx >= model.num_vertices) {
            std::cerr << "ERROR: Invalid vertex index " << vertex_idx 
                      << " for landmark " << landmark_idx << std::endl;
            invalid_vertices.push_back(vertex_idx);
        }
    }
    
    if (!invalid_vertices.empty()) {
        std::cerr << "ERROR: " << invalid_vertices.size() 
                  << " invalid vertex indices found!" << std::endl;
        return 1;
    }
    
    std::cout << "✓ All vertex indices are valid" << std::endl;
    std::cout << std::endl;
    
    // Get mean shape
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
    // Extract mapped 3D points
    std::cout << "--- Extracting Mapped 3D Points ---" << std::endl;
    std::vector<Eigen::Vector3d> mapped_points;
    std::vector<int> landmark_indices;
    
    for (int landmark_idx : mapped_landmarks) {
        int vertex_idx = mapping.getModelVertex(landmark_idx);
        Eigen::Vector3d point = mean_shape.row(vertex_idx);
        mapped_points.push_back(point);
        landmark_indices.push_back(landmark_idx);
        
        std::cout << "  Landmark " << std::setw(2) << landmark_idx 
                  << " -> Vertex " << std::setw(4) << vertex_idx
                  << ": (" << std::fixed << std::setprecision(4)
                  << point.x() << ", " << point.y() << ", " << point.z() << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Calculate statistics
    std::cout << "--- Mapping Statistics ---" << std::endl;
    Eigen::Vector3d min_point = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d max_point = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    
    for (const auto& point : mapped_points) {
        for (int i = 0; i < 3; ++i) {
            if (point(i) < min_point(i)) min_point(i) = point(i);
            if (point(i) > max_point(i)) max_point(i) = point(i);
        }
        centroid += point;
    }
    centroid /= static_cast<double>(mapped_points.size());
    
    std::cout << "Number of mapped landmarks: " << mapped_points.size() << std::endl;
    std::cout << "Bounding box:" << std::endl;
    std::cout << "  X: [" << min_point.x() << ", " << max_point.x() << "]" << std::endl;
    std::cout << "  Y: [" << min_point.y() << ", " << max_point.y() << "]" << std::endl;
    std::cout << "  Z: [" << min_point.z() << ", " << max_point.z() << "]" << std::endl;
    std::cout << "Centroid: (" << centroid.x() << ", " << centroid.y() << ", " << centroid.z() << ")" << std::endl;
    std::cout << std::endl;
    
    // Check if we have enough correspondences
    if (mapped_points.size() < 6) {
        std::cerr << "WARNING: Only " << mapped_points.size() 
                  << " correspondences. Need at least 6 for stable Procrustes alignment." << std::endl;
    } else {
        std::cout << "✓ Sufficient correspondences (" << mapped_points.size() 
                  << " >= 6) for Procrustes alignment" << std::endl;
    }
    std::cout << std::endl;
    
    // Save to PLY for visualization
    std::cout << "--- Saving Mapped Points ---" << std::endl;
    if (saveMappedLandmarksPLY(mapped_points, landmark_indices, output_ply)) {
        std::cout << "✓ Successfully saved mapped landmarks to: " << output_ply << std::endl;
        std::cout << "  You can visualize it with: meshlab " << output_ply << std::endl;
    } else {
        std::cerr << "Failed to save mapped landmarks" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== Validation Summary ===" << std::endl;
    std::cout << "✓ Mapping file loaded: " << mapping.size() << " correspondences" << std::endl;
    std::cout << "✓ All vertex indices valid" << std::endl;
    std::cout << "✓ Mapped points extracted and saved" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Test Complete ===" << std::endl;
    std::cout << "✓ Landmark-to-model vertex mapping validated successfully!" << std::endl;
    std::cout << "  Next: Use this mapping in pose initialization (STEP 2)" << std::endl;
    
    return 0;
}

