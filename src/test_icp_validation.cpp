/**
 * STEP 5: ICP as Validation Tool
 * 
 * Goal: Use ICP only to validate alignment quality, not as a full optimizer.
 * 
 * Usage:
 *   bin/test_icp_validation <aligned_mesh_ply> <observed_pointcloud_ply> [max_iterations]
 * 
 * Example:
 *   bin/test_icp_validation build/aligned_mesh_step2.ply \
 *                            build/output_pointcloud.ply \
 *                            50
 */

#include "alignment/ICP.h"
#include "alignment/Procrustes.h"
#include "utils/DepthUtils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>

using namespace face_reconstruction;

/**
 * Load vertices from PLY file (simple parser)
 */
bool loadPLYVertices(const std::string& filepath, Eigen::MatrixXd& vertices) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    int num_vertices = 0;
    bool in_header = true;
    int vertex_count = 0;
    
    std::vector<Eigen::Vector3d> verts;
    
    while (std::getline(file, line)) {
        if (in_header) {
            if (line.find("element vertex") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_vertices;
            } else if (line.find("end_header") != std::string::npos) {
                in_header = false;
                verts.reserve(num_vertices);
            }
        } else {
            if (vertex_count < num_vertices) {
                std::istringstream iss(line);
                double x, y, z;
                if (iss >> x >> y >> z) {
                    verts.emplace_back(x, y, z);
                    vertex_count++;
                }
            } else {
                break;  // Skip faces
            }
        }
    }
    
    file.close();
    
    if (verts.empty()) {
        std::cerr << "No vertices loaded from PLY file" << std::endl;
        return false;
    }
    
    vertices.resize(verts.size(), 3);
    for (size_t i = 0; i < verts.size(); ++i) {
        vertices.row(i) = verts[i];
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <aligned_mesh_ply> <observed_pointcloud_ply> [max_iterations]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " build/aligned_mesh_step2.ply \\" << std::endl;
        std::cerr << "     build/output_pointcloud.ply \\" << std::endl;
        std::cerr << "     50" << std::endl;
        return 1;
    }
    
    std::string aligned_mesh_ply = argv[1];
    std::string observed_pointcloud_ply = argv[2];
    int max_iterations = (argc > 3) ? std::stoi(argv[3]) : 50;
    
    std::cout << "=== STEP 5: ICP Validation Tool ===" << std::endl;
    std::cout << "Aligned mesh: " << aligned_mesh_ply << std::endl;
    std::cout << "Observed point cloud: " << observed_pointcloud_ply << std::endl;
    std::cout << "Max iterations: " << max_iterations << std::endl;
    std::cout << std::endl;
    
    // Load aligned mesh (source)
    std::cout << "--- Loading Aligned Mesh ---" << std::endl;
    Eigen::MatrixXd source_points;
    if (!loadPLYVertices(aligned_mesh_ply, source_points)) {
        std::cerr << "Failed to load aligned mesh" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << source_points.rows() << " source points" << std::endl;
    std::cout << std::endl;
    
    // Load observed point cloud (target)
    std::cout << "--- Loading Observed Point Cloud ---" << std::endl;
    Eigen::MatrixXd target_points;
    if (!loadPLYVertices(observed_pointcloud_ply, target_points)) {
        std::cerr << "Failed to load observed point cloud" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << target_points.rows() << " target points" << std::endl;
    std::cout << std::endl;
    
    // Initial alignment (identity - mesh is already aligned)
    std::cout << "--- Initial Alignment (from Procrustes) ---" << std::endl;
    Eigen::Matrix3d initial_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d initial_t = Eigen::Vector3d::Zero();
    
    // Compute initial error
    ICP icp;
    std::vector<int> initial_correspondences = icp.findClosestPoints(source_points, target_points);
    double initial_error = icp.computeError(source_points, target_points, initial_correspondences);
    
    std::cout << "Initial error (before ICP): " << std::fixed << std::setprecision(4) 
              << initial_error << " meters (" << initial_error * 1000.0 << " mm)" << std::endl;
    std::cout << std::endl;
    
    // Run ICP
    std::cout << "--- Running ICP (Validation Only) ---" << std::endl;
    std::cout << "ICP is used ONLY for validation, not as a full optimizer." << std::endl;
    std::cout << "Starting ICP iterations..." << std::endl;
    std::cout << std::endl;
    
    ICP::ICPResult result = icp.align(source_points, target_points, 
                                      initial_R, initial_t, 
                                      max_iterations, 1e-6);
    
    std::cout << "ICP completed:" << std::endl;
    std::cout << "  Iterations: " << result.iterations << " / " << max_iterations << std::endl;
    std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "  Final error: " << std::fixed << std::setprecision(4) 
              << result.final_error << " meters (" << result.final_error * 1000.0 << " mm)" << std::endl;
    std::cout << std::endl;
    
    // Compare before and after
    std::cout << "--- Alignment Quality Comparison ---" << std::endl;
    double error_reduction = initial_error - result.final_error;
    double error_reduction_percent = (error_reduction / initial_error) * 100.0;
    
    std::cout << "Error before ICP: " << std::fixed << std::setprecision(4) 
              << initial_error * 1000.0 << " mm" << std::endl;
    std::cout << "Error after ICP:  " << result.final_error * 1000.0 << " mm" << std::endl;
    std::cout << "Error reduction:  " << error_reduction * 1000.0 << " mm (" 
              << std::setprecision(1) << error_reduction_percent << "%)" << std::endl;
    std::cout << std::endl;
    
    // Interpretation
    std::cout << "--- Validation Results ---" << std::endl;
    if (result.final_error < 0.01) {  // < 1cm
        std::cout << "✓ Excellent alignment quality (error < 1cm)" << std::endl;
    } else if (result.final_error < 0.02) {  // < 2cm
        std::cout << "✓ Good alignment quality (error < 2cm)" << std::endl;
    } else if (result.final_error < 0.05) {  // < 5cm
        std::cout << "⚠ Moderate alignment quality (error < 5cm)" << std::endl;
    } else {
        std::cout << "⚠ Alignment quality needs improvement (error > 5cm)" << std::endl;
    }
    
    if (error_reduction_percent > 10.0) {
        std::cout << "✓ ICP provided significant improvement (" 
                  << std::setprecision(1) << error_reduction_percent << "% reduction)" << std::endl;
    } else if (error_reduction_percent > 0.0) {
        std::cout << "→ ICP provided minor improvement (" 
                  << std::setprecision(1) << error_reduction_percent << "% reduction)" << std::endl;
    } else {
        std::cout << "→ ICP did not improve alignment (initial alignment was already good)" << std::endl;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== STEP 5 Summary ===" << std::endl;
    std::cout << "✓ ICP validation completed" << std::endl;
    std::cout << "✓ Initial error: " << initial_error * 1000.0 << " mm" << std::endl;
    std::cout << "✓ Final error: " << result.final_error * 1000.0 << " mm" << std::endl;
    std::cout << "✓ Error reduction: " << error_reduction_percent << "%" << std::endl;
    std::cout << "✓ ICP converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== STEP 5 Complete ===" << std::endl;
    std::cout << "✓ ICP validation tool completed successfully!" << std::endl;
    std::cout << "  Note: ICP is used ONLY for validation, not as a full optimizer." << std::endl;
    std::cout << "  Week 3 milestones completed!" << std::endl;
    
    return 0;
}

