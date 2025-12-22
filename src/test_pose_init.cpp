/**
 * STEP 2: Rigid Pose Initialization (Procrustes)
 * 
 * Goal: Re-run rigid alignment using the finalized landmark mapping.
 * 
 * Pipeline:
 *   1) Use depth lifting to obtain 3D landmark points from the depth map
 *   2) Apply Procrustes alignment using valid correspondences
 *   3) Skip landmarks with invalid depth
 *   4) Print alignment statistics (correspondences, scale, mean alignment error)
 *   5) Export aligned mesh
 * 
 * Usage:
 *   bin/test_pose_init <rgb_path> <depth_path> <intrinsics_path> <landmarks_path> <model_dir> <mapping_file> [output_ply]
 * 
 * Example:
 *   bin/test_pose_init data/biwi_person01/rgb/frame_00000.png \
 *                      data/biwi_person01/depth/frame_00000.png \
 *                      data/biwi_person01/intrinsics.txt \
 *                      build/landmarks_frame_00000.txt \
 *                      data/model \
 *                      data/landmark_mapping.txt \
 *                      build/aligned_mesh.ply
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include "landmarks/LandmarkData.h"
#include "model/MorphableModel.h"
#include "alignment/Procrustes.h"
#include "alignment/LandmarkMapping.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace face_reconstruction;

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] 
                  << " <rgb_path> <depth_path> <intrinsics_path> <landmarks_path> <model_dir> <mapping_file> [output_ply]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/biwi_person01/rgb/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/depth/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/intrinsics.txt \\" << std::endl;
        std::cerr << "     build/landmarks_frame_00000.txt \\" << std::endl;
        std::cerr << "     data/model \\" << std::endl;
        std::cerr << "     data/landmark_mapping.txt \\" << std::endl;
        std::cerr << "     build/aligned_mesh.ply" << std::endl;
        std::cerr << "\nNote: mapping_file is REQUIRED for STEP 2." << std::endl;
        return 1;
    }
    
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    std::string intrinsics_path = argv[3];
    std::string landmarks_path = argv[4];
    std::string model_dir = argv[5];
    std::string mapping_file = argv[6];  // REQUIRED
    std::string output_ply = (argc > 7) ? argv[7] : "aligned_mesh.ply";
    
    std::cout << "=== STEP 2: Rigid Pose Initialization ===" << std::endl;
    std::cout << "RGB: " << rgb_path << std::endl;
    std::cout << "Depth: " << depth_path << std::endl;
    std::cout << "Intrinsics: " << intrinsics_path << std::endl;
    std::cout << "Landmarks: " << landmarks_path << std::endl;
    std::cout << "Model: " << model_dir << std::endl;
    std::cout << "Mapping: " << mapping_file << std::endl;
    std::cout << "Output: " << output_ply << std::endl;
    std::cout << std::endl;
    
    // Load RGB-D frame
    RGBDFrame frame;
    if (!frame.loadRGB(rgb_path)) {
        std::cerr << "Failed to load RGB image" << std::endl;
        return 1;
    }
    
    if (!frame.loadDepth(depth_path, 1000.0)) {
        std::cerr << "Failed to load depth image" << std::endl;
        return 1;
    }
    
    // Load camera intrinsics
    CameraIntrinsics intrinsics;
    try {
        intrinsics = CameraIntrinsics::loadFromFile(intrinsics_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load intrinsics: " << e.what() << std::endl;
        return 1;
    }
    
    // Load landmarks
    LandmarkData landmarks;
    std::string ext = landmarks_path.substr(landmarks_path.find_last_of(".") + 1);
    bool loaded = false;
    
    if (ext == "json" || ext == "JSON") {
        loaded = landmarks.loadFromJSON(landmarks_path);
    } else {
        loaded = landmarks.loadFromTXT(landmarks_path);
    }
    
    if (!loaded || landmarks.size() == 0) {
        std::cerr << "Failed to load landmarks or no landmarks found" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << landmarks.size() << " landmarks" << std::endl;
    
    // Load model
    MorphableModel model;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "Failed to load model from: " << model_dir << std::endl;
        return 1;
    }
    
    std::cout << "Loaded model with " << model.num_vertices << " vertices" << std::endl;
    std::cout << std::endl;
    
    // Load landmark mapping (REQUIRED for STEP 2)
    LandmarkMapping mapping;
    if (!mapping.loadFromFile(mapping_file)) {
        std::cerr << "ERROR: Failed to load mapping file: " << mapping_file << std::endl;
        std::cerr << "Mapping file is required for STEP 2. Create it using:" << std::endl;
        std::cerr << "  python scripts/create_landmark_mapping.py data/model data/landmark_mapping.txt" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << mapping.size() << " landmark-to-model mappings" << std::endl;
    std::cout << std::endl;
    
    // Step 1: Get 3D points from depth at landmark locations
    std::cout << "--- Step 1: Extracting 3D points from depth at landmarks ---" << std::endl;
    std::cout << "Checking depth validity for " << landmarks.size() << " landmarks..." << std::endl;
    const cv::Mat& depth_map = frame.getDepth();
    
    std::vector<Eigen::Vector3d> observed_3d_points;
    std::vector<int> valid_landmark_indices;
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const auto& lm = landmarks[i];
        int u = static_cast<int>(std::round(lm.x));
        int v = static_cast<int>(std::round(lm.y));
        
        // Check if pixel is within bounds
        if (u < 0 || u >= depth_map.cols || v < 0 || v >= depth_map.rows) {
            continue;
        }
        
        // Get depth value
        float depth = depth_map.at<float>(v, u);
        
        // Check if depth is valid
        if (!frame.isValidDepth(depth)) {
            continue;
        }
        
        // Backproject to 3D
        Eigen::Vector3d point_3d = depthTo3D(u, v, static_cast<double>(depth), intrinsics);
        observed_3d_points.push_back(point_3d);
        valid_landmark_indices.push_back(static_cast<int>(i));
    }
    
    std::cout << "Found " << observed_3d_points.size() << " valid 3D points from " 
              << landmarks.size() << " landmarks" << std::endl;
    
    if (observed_3d_points.size() < 6) {
        std::cerr << "ERROR: Not enough valid depth points (" << observed_3d_points.size() 
                  << "). Need at least 6 for Procrustes alignment." << std::endl;
        std::cerr << "This frame may not be suitable for pose initialization." << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Step 2: Get corresponding model vertices using finalized mapping
    std::cout << "--- Step 2: Getting corresponding model vertices (using finalized mapping) ---" << std::endl;
    std::vector<Eigen::Vector3d> model_3d_points;
    std::vector<int> valid_correspondences;
    std::vector<int> mapped_landmark_indices;  // Track which landmarks were used
    
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
    for (size_t i = 0; i < valid_landmark_indices.size(); ++i) {
        int landmark_idx = valid_landmark_indices[i];
        int model_vertex_idx = -1;
        
        // Use mapping file (REQUIRED for STEP 2)
        if (mapping.hasMapping(landmark_idx)) {
            model_vertex_idx = mapping.getModelVertex(landmark_idx);
        } else {
            // Skip if not in mapping
            continue;
        }
        
        // Check if model vertex index is valid
        if (model_vertex_idx < 0 || model_vertex_idx >= model.num_vertices) {
            std::cerr << "Warning: Invalid vertex index " << model_vertex_idx 
                      << " for landmark " << landmark_idx << std::endl;
            continue;
        }
        
        // Get 3D point from mean shape
        Eigen::Vector3d model_point = mean_shape.row(model_vertex_idx);
        model_3d_points.push_back(model_point);
        valid_correspondences.push_back(static_cast<int>(i));
        mapped_landmark_indices.push_back(landmark_idx);
    }
    
    std::cout << "Found " << model_3d_points.size() << " valid correspondences" << std::endl;
    
    if (model_3d_points.size() < 6) {
        std::cerr << "ERROR: Not enough valid correspondences (" << model_3d_points.size() 
                  << "). Need at least 6 for Procrustes alignment." << std::endl;
        std::cerr << "Possible reasons:" << std::endl;
        std::cerr << "  - Not enough landmarks have valid depth" << std::endl;
        std::cerr << "  - Mapping file doesn't cover enough landmarks" << std::endl;
        std::cerr << "  - Landmark indices don't match mapping file" << std::endl;
        return 1;
    }
    
    // Print correspondence details
    std::cout << "Correspondences:" << std::endl;
    for (size_t i = 0; i < valid_correspondences.size(); ++i) {
        int corr_idx = valid_correspondences[i];
        int landmark_idx = mapped_landmark_indices[i];
        std::cout << "  Landmark " << std::setw(2) << landmark_idx 
                  << " -> Observed: (" << std::fixed << std::setprecision(4)
                  << observed_3d_points[corr_idx].x() << ", "
                  << observed_3d_points[corr_idx].y() << ", "
                  << observed_3d_points[corr_idx].z() << ")"
                  << " Model: (" << model_3d_points[i].x() << ", "
                  << model_3d_points[i].y() << ", " << model_3d_points[i].z() << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Match up the correspondences
    std::vector<Eigen::Vector3d> matched_observed;
    std::vector<Eigen::Vector3d> matched_model;
    
    for (size_t i = 0; i < valid_correspondences.size(); ++i) {
        int corr_idx = valid_correspondences[i];
        matched_observed.push_back(observed_3d_points[corr_idx]);
        matched_model.push_back(model_3d_points[i]);
    }
    
    std::cout << "Using " << matched_observed.size() << " correspondences for Procrustes alignment" << std::endl;
    std::cout << std::endl;
    
    // Step 3: Estimate similarity transform with Procrustes
    std::cout << "--- Step 3: Estimating similarity transform (Procrustes) ---" << std::endl;
    std::cout << "Computing scale, rotation, and translation..." << std::endl;
    
    // Convert to matrices
    Eigen::MatrixXd source_points(matched_model.size(), 3);
    Eigen::MatrixXd target_points(matched_observed.size(), 3);
    
    for (size_t i = 0; i < matched_model.size(); ++i) {
        source_points.row(i) = matched_model[i].transpose();
        target_points.row(i) = matched_observed[i].transpose();
    }
    
    SimilarityTransform transform;
    try {
        transform = estimateSimilarityTransform(source_points, target_points);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to estimate transform: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Estimated transform:" << std::endl;
    std::cout << "  Scale: " << std::fixed << std::setprecision(6) << transform.scale << std::endl;
    std::cout << "  Rotation matrix:" << std::endl;
    std::cout << "    [" << std::setw(10) << transform.rotation(0,0) << " " 
              << std::setw(10) << transform.rotation(0,1) << " " 
              << std::setw(10) << transform.rotation(0,2) << "]" << std::endl;
    std::cout << "    [" << std::setw(10) << transform.rotation(1,0) << " " 
              << std::setw(10) << transform.rotation(1,1) << " " 
              << std::setw(10) << transform.rotation(1,2) << "]" << std::endl;
    std::cout << "    [" << std::setw(10) << transform.rotation(2,0) << " " 
              << std::setw(10) << transform.rotation(2,1) << " " 
              << std::setw(10) << transform.rotation(2,2) << "]" << std::endl;
    std::cout << "  Translation: (" << std::setprecision(4) 
              << transform.translation.x() << ", " 
              << transform.translation.y() << ", " 
              << transform.translation.z() << ") meters" << std::endl;
    std::cout << std::endl;
    
    // Step 4: Apply transform to whole model mesh
    std::cout << "--- Step 4: Applying transform to model mesh ---" << std::endl;
    
    // Get mean shape (or reconstructed shape with zero coefficients)
    Eigen::VectorXd alpha_zero = Eigen::VectorXd::Zero(model.num_identity_components);
    Eigen::VectorXd beta_zero = Eigen::VectorXd::Zero(model.num_expression_components);
    Eigen::MatrixXd model_vertices = model.reconstructFace(alpha_zero, beta_zero);
    
    // Apply transform
    Eigen::MatrixXd aligned_vertices = transform.apply(model_vertices);
    
    std::cout << "Transformed " << aligned_vertices.rows() << " vertices" << std::endl;
    
    // Calculate bounding boxes
    Eigen::Vector3d model_min = model_vertices.colwise().minCoeff();
    Eigen::Vector3d model_max = model_vertices.colwise().maxCoeff();
    Eigen::Vector3d aligned_min = aligned_vertices.colwise().minCoeff();
    Eigen::Vector3d aligned_max = aligned_vertices.colwise().maxCoeff();
    
    std::cout << "Original model bounds:" << std::endl;
    std::cout << "  [" << model_min.x() << ", " << model_max.x() << "] x "
              << "[" << model_min.y() << ", " << model_max.y() << "] x "
              << "[" << model_min.z() << ", " << model_max.z() << "]" << std::endl;
    std::cout << "Aligned model bounds:" << std::endl;
    std::cout << "  [" << aligned_min.x() << ", " << aligned_max.x() << "] x "
              << "[" << aligned_min.y() << ", " << aligned_max.y() << "] x "
              << "[" << aligned_min.z() << ", " << aligned_max.z() << "]" << std::endl;
    
    // Calculate alignment error
    double alignment_error = 0.0;
    double max_error = 0.0;
    double min_error = std::numeric_limits<double>::max();
    
    std::cout << "Per-correspondence alignment errors:" << std::endl;
    for (size_t i = 0; i < matched_observed.size(); ++i) {
        int landmark_idx = mapped_landmark_indices[i];
        int model_vertex_idx = mapping.getModelVertex(landmark_idx);
        
        if (model_vertex_idx >= 0 && model_vertex_idx < aligned_vertices.rows()) {
            Eigen::Vector3d transformed_point = aligned_vertices.row(model_vertex_idx);
            double error = (transformed_point - matched_observed[i]).norm();
            alignment_error += error;
            
            if (error > max_error) max_error = error;
            if (error < min_error) min_error = error;
            
            std::cout << "  Landmark " << std::setw(2) << landmark_idx 
                      << ": " << std::fixed << std::setprecision(4) << error * 1000.0 
                      << " mm" << std::endl;
        }
    }
    alignment_error /= matched_observed.size();
    
    std::cout << std::endl;
    std::cout << "Alignment error statistics:" << std::endl;
    std::cout << "  Mean error: " << std::fixed << std::setprecision(4) 
              << alignment_error << " meters (" << alignment_error * 1000.0 << " mm)" << std::endl;
    std::cout << "  Min error:  " << min_error << " meters (" << min_error * 1000.0 << " mm)" << std::endl;
    std::cout << "  Max error:  " << max_error << " meters (" << max_error * 1000.0 << " mm)" << std::endl;
    std::cout << std::endl;
    
    // Step 5: Export aligned mesh
    std::cout << "--- Step 5: Exporting aligned mesh ---" << std::endl;
    if (model.saveMeshPLY(aligned_vertices, output_ply)) {
        std::cout << "Successfully saved aligned mesh to: " << output_ply << std::endl;
    } else {
        std::cerr << "Failed to save aligned mesh" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== STEP 2 Summary ===" << std::endl;
    std::cout << "✓ Valid depth points: " << observed_3d_points.size() << " / " << landmarks.size() << " landmarks" << std::endl;
    std::cout << "✓ Valid correspondences: " << matched_observed.size() << " (using finalized mapping)" << std::endl;
    std::cout << "✓ Estimated scale: " << std::fixed << std::setprecision(4) << transform.scale << std::endl;
    std::cout << "✓ Mean alignment error: " << alignment_error * 1000.0 << " mm" << std::endl;
    std::cout << "✓ Aligned mesh saved to: " << output_ply << std::endl;
    std::cout << std::endl;
    
    // Check if alignment is good
    if (alignment_error < 0.01) {  // < 1cm
        std::cout << "✓ Excellent alignment (error < 1cm)" << std::endl;
    } else if (alignment_error < 0.02) {  // < 2cm
        std::cout << "⚠ Good alignment (error < 2cm)" << std::endl;
    } else {
        std::cout << "⚠ Alignment error is high (> 2cm). Consider:" << std::endl;
        std::cout << "  - Checking mapping accuracy" << std::endl;
        std::cout << "  - Verifying depth quality" << std::endl;
        std::cout << "  - Adding more correspondences" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "=== STEP 2 Complete ===" << std::endl;
    std::cout << "✓ Rigid pose initialization completed successfully!" << std::endl;
    std::cout << "  Next: Implement depth renderer (STEP 3)" << std::endl;
    std::cout << "  Visualize: meshlab " << output_ply << std::endl;
    
    return 0;
}

