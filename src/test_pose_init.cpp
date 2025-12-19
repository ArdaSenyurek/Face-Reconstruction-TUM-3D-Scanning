/**
 * STEP D: Initial Sparse Alignment + Pose Initialization Pipeline
 * 
 * Goal: Reliable initial pose/scale/translation for the face model.
 * 
 * Pipeline:
 *   1) Choose a small stable set of landmarks (e.g., eye corners, nose tip, mouth corners)
 *   2) Get corresponding 3D points from depth lifting at landmark pixels (if valid)
 *   3) Get corresponding 3D vertices from the mean face model (predefined mapping file)
 *   4) Estimate similarity transform with Procrustes (scale,R,t)
 *   5) Apply transform to whole model mesh and export aligned mesh
 * 
 * Usage:
 *   bin/test_pose_init <rgb_path> <depth_path> <intrinsics_path> <landmarks_path> <model_dir> [mapping_file] [output_ply]
 * 
 * Example:
 *   bin/test_pose_init data/biwi_person01/rgb/frame_00000.png \
 *                      data/biwi_person01/depth/frame_00000.png \
 *                      data/biwi_person01/intrinsics.txt \
 *                      data/biwi_person01/landmarks/frame_00000.txt \
 *                      data/model \
 *                      data/landmark_mapping.txt \
 *                      aligned_mesh.ply
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
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] 
                  << " <rgb_path> <depth_path> <intrinsics_path> <landmarks_path> <model_dir> [mapping_file] [output_ply]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/biwi_person01/rgb/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/depth/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/intrinsics.txt \\" << std::endl;
        std::cerr << "     data/biwi_person01/landmarks/frame_00000.txt \\" << std::endl;
        std::cerr << "     data/model \\" << std::endl;
        std::cerr << "     data/landmark_mapping.txt \\" << std::endl;
        std::cerr << "     aligned_mesh.ply" << std::endl;
        std::cerr << "\nNote: mapping_file is optional. If not provided, will use model_index from landmarks." << std::endl;
        return 1;
    }
    
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    std::string intrinsics_path = argv[3];
    std::string landmarks_path = argv[4];
    std::string model_dir = argv[5];
    std::string mapping_file = (argc > 6) ? argv[6] : "";
    std::string output_ply = (argc > 7) ? argv[7] : "aligned_mesh.ply";
    
    std::cout << "=== Pose Initialization Test ===" << std::endl;
    std::cout << "RGB: " << rgb_path << std::endl;
    std::cout << "Depth: " << depth_path << std::endl;
    std::cout << "Intrinsics: " << intrinsics_path << std::endl;
    std::cout << "Landmarks: " << landmarks_path << std::endl;
    std::cout << "Model: " << model_dir << std::endl;
    if (!mapping_file.empty()) {
        std::cout << "Mapping: " << mapping_file << std::endl;
    }
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
    
    // Load landmark mapping (if provided)
    LandmarkMapping mapping;
    if (!mapping_file.empty()) {
        if (!mapping.loadFromFile(mapping_file)) {
            std::cerr << "Warning: Failed to load mapping file, will use model_index from landmarks" << std::endl;
        }
    }
    
    // Step 1: Get 3D points from depth at landmark locations
    std::cout << "--- Step 1: Extracting 3D points from depth at landmarks ---" << std::endl;
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
    
    // Step 2: Get corresponding model vertices
    std::cout << "--- Step 2: Getting corresponding model vertices ---" << std::endl;
    std::vector<Eigen::Vector3d> model_3d_points;
    std::vector<int> valid_correspondences;
    
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
    for (size_t i = 0; i < valid_landmark_indices.size(); ++i) {
        int landmark_idx = valid_landmark_indices[i];
        int model_vertex_idx = -1;
        
        // Try to get model vertex index from mapping or landmark data
        if (mapping.hasMapping(landmark_idx)) {
            model_vertex_idx = mapping.getModelVertex(landmark_idx);
        } else if (landmarks[landmark_idx].model_index >= 0) {
            model_vertex_idx = landmarks[landmark_idx].model_index;
        }
        
        // Check if model vertex index is valid
        if (model_vertex_idx < 0 || model_vertex_idx >= model.num_vertices) {
            continue;
        }
        
        // Get 3D point from mean shape
        Eigen::Vector3d model_point = mean_shape.row(model_vertex_idx);
        model_3d_points.push_back(model_point);
        valid_correspondences.push_back(static_cast<int>(i));
    }
    
    std::cout << "Found " << model_3d_points.size() << " valid correspondences" << std::endl;
    
    if (model_3d_points.size() < 6) {
        std::cerr << "ERROR: Not enough valid correspondences (" << model_3d_points.size() 
                  << "). Need at least 6 for Procrustes alignment." << std::endl;
        std::cerr << "You may need to create a landmark mapping file." << std::endl;
        return 1;
    }
    
    // Match up the correspondences
    std::vector<Eigen::Vector3d> matched_observed;
    std::vector<Eigen::Vector3d> matched_model;
    
    for (int corr_idx : valid_correspondences) {
        matched_observed.push_back(observed_3d_points[corr_idx]);
        matched_model.push_back(model_3d_points[corr_idx]);
    }
    
    std::cout << "Using " << matched_observed.size() << " correspondences for alignment" << std::endl;
    std::cout << std::endl;
    
    // Step 3: Estimate similarity transform with Procrustes
    std::cout << "--- Step 3: Estimating similarity transform (Procrustes) ---" << std::endl;
    
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
    std::cout << "  Scale: " << transform.scale << std::endl;
    std::cout << "  Rotation:" << std::endl;
    std::cout << "    " << transform.rotation.row(0) << std::endl;
    std::cout << "    " << transform.rotation.row(1) << std::endl;
    std::cout << "    " << transform.rotation.row(2) << std::endl;
    std::cout << "  Translation: (" << transform.translation.x() << ", " 
              << transform.translation.y() << ", " << transform.translation.z() << ")" << std::endl;
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
    for (size_t i = 0; i < matched_observed.size(); ++i) {
        int corr_idx = valid_correspondences[i];
        int landmark_idx = valid_landmark_indices[corr_idx];
        int model_vertex_idx = -1;
        
        if (mapping.hasMapping(landmark_idx)) {
            model_vertex_idx = mapping.getModelVertex(landmark_idx);
        } else if (landmarks[landmark_idx].model_index >= 0) {
            model_vertex_idx = landmarks[landmark_idx].model_index;
        }
        
        if (model_vertex_idx >= 0 && model_vertex_idx < aligned_vertices.rows()) {
            Eigen::Vector3d transformed_point = aligned_vertices.row(model_vertex_idx);
            double error = (transformed_point - matched_observed[i]).norm();
            alignment_error += error;
        }
    }
    alignment_error /= matched_observed.size();
    
    std::cout << "Mean alignment error: " << alignment_error << " meters" << std::endl;
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
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "✓ Valid depth points: " << observed_3d_points.size() << std::endl;
    std::cout << "✓ Valid correspondences: " << matched_observed.size() << std::endl;
    std::cout << "✓ Estimated scale: " << transform.scale << std::endl;
    std::cout << "✓ Mean alignment error: " << alignment_error << " meters" << std::endl;
    std::cout << "✓ Aligned mesh saved to: " << output_ply << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Test Complete ===" << std::endl;
    std::cout << "✓ Pose initialization pipeline completed successfully!" << std::endl;
    std::cout << "  You can visualize the aligned mesh with: meshlab " << output_ply << std::endl;
    
    return 0;
}

