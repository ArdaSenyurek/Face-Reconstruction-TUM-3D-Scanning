/**
 * Pose Initialization Tool
 * 
 * Uses Procrustes alignment to initialize model pose from landmarks and depth.
 * 
 * Pipeline:
 *   1) Load RGB-D frame and landmarks
 *   2) Extract 3D points from depth at landmark locations
 *   3) Map landmarks to model vertices using LandmarkMapping
 *   4) Estimate similarity transform (scale, rotation, translation) using Procrustes
 *   5) Apply transform to model and save aligned mesh
 * 
 * Usage:
 *   build/bin/pose_init --rgb <path> --depth <path> --intrinsics <path> \
 *                       --landmarks <path> --model-dir <path> \
 *                       --mapping <path> --output <path>
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include "landmarks/LandmarkData.h"
#include "model/MorphableModel.h"
#include "alignment/Procrustes.h"
#include "alignment/LandmarkMapping.h"
#include "alignment/ICP.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

// OpenCV is required (included via RGBDFrame.h)

using namespace face_reconstruction;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --rgb <path>              Path to RGB image\n"
              << "  --depth <path>            Path to depth image\n"
              << "  --intrinsics <path>       Path to camera intrinsics file\n"
              << "  --landmarks <path>         Path to landmarks file (TXT or JSON)\n"
              << "  --model-dir <path>         Directory containing PCA model files\n"
              << "  --mapping <path>           Path to landmark mapping file\n"
              << "  --output <path>            Output aligned mesh path (PLY)\n"
              << "  --report <path>            Output JSON report path (optional)\n"
              << "  --depth-scale <value>     Depth scale factor (default: 1000.0)\n"
              << "  --bfm-scale <value>       BFM model scale factor mm->m (default: 0.001)\n"
              << "  --help                     Show this help\n";
}

// Helper to write JSON (simple, no external dependency)
void writeJSONReport(const std::string& filepath,
                     const SimilarityTransform& transform,
                     double rmse, double mean_error, double median_error,
                     const std::vector<Eigen::Vector3d>& observed_points,
                     const Eigen::MatrixXd& aligned_vertices) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open JSON report file: " << filepath << std::endl;
        return;
    }
    
    // Compute depth z-range
    double obs_z_min = std::numeric_limits<double>::max();
    double obs_z_max = std::numeric_limits<double>::lowest();
    for (const auto& p : observed_points) {
        obs_z_min = std::min(obs_z_min, p.z());
        obs_z_max = std::max(obs_z_max, p.z());
    }
    
    double mesh_z_min = std::numeric_limits<double>::max();
    double mesh_z_max = std::numeric_limits<double>::lowest();
    for (int i = 0; i < aligned_vertices.rows(); ++i) {
        double z = aligned_vertices(i, 2);
        mesh_z_min = std::min(mesh_z_min, z);
        mesh_z_max = std::max(mesh_z_max, z);
    }
    
    file << std::fixed << std::setprecision(6);
    file << "{\n";
    file << "  \"transform\": {\n";
    file << "    \"scale\": " << transform.scale << ",\n";
    file << "    \"translation\": [" << transform.translation.x() << ", "
         << transform.translation.y() << ", " << transform.translation.z() << "],\n";
    file << "    \"translation_norm\": " << transform.translation.norm() << ",\n";
    file << "    \"rotation_det\": " << transform.rotation.determinant() << ",\n";
    file << "    \"rotation\": [\n";
    for (int i = 0; i < 3; ++i) {
        file << "      [" << transform.rotation(i, 0) << ", "
             << transform.rotation(i, 1) << ", " << transform.rotation(i, 2) << "]";
        if (i < 2) file << ",";
        file << "\n";
    }
    file << "    ]\n";
    file << "  },\n";
    file << "  \"alignment_errors\": {\n";
    file << "    \"rmse_m\": " << rmse << ",\n";
    file << "    \"rmse_mm\": " << rmse * 1000.0 << ",\n";
    file << "    \"mean_error_m\": " << mean_error << ",\n";
    file << "    \"mean_error_mm\": " << mean_error * 1000.0 << ",\n";
    file << "    \"median_error_m\": " << median_error << ",\n";
    file << "    \"median_error_mm\": " << median_error * 1000.0 << "\n";
    file << "  },\n";
    file << "  \"depth_z_range\": {\n";
    file << "    \"observed_min_m\": " << obs_z_min << ",\n";
    file << "    \"observed_max_m\": " << obs_z_max << ",\n";
    file << "    \"mesh_min_m\": " << mesh_z_min << ",\n";
    file << "    \"mesh_max_m\": " << mesh_z_max << ",\n";
    file << "    \"mesh_in_range\": " << (mesh_z_min >= obs_z_min - 0.3 && mesh_z_max <= obs_z_max + 0.3 ? "true" : "false") << "\n";
    file << "  },\n";
    file << "  \"sanity_checks\": {\n";
    file << "    \"rotation_det_valid\": " << (std::abs(transform.rotation.determinant() - 1.0) < 0.01 ? "true" : "false") << ",\n";
    file << "    \"scale_in_range\": " << (transform.scale > 0.5 && transform.scale < 2.0 ? "true" : "false") << ",\n";
    file << "    \"rmse_acceptable\": " << (rmse < 0.05 ? "true" : "false") << "\n";
    file << "  }\n";
    file << "}\n";
    file.close();
}

int main(int argc, char* argv[]) {
    std::string rgb_path, depth_path, intrinsics_path, landmarks_path, 
                model_dir, mapping_path, output_path, report_path;
    double depth_scale = 1000.0;
    double bfm_scale = 0.001;  // BFM is in mm, convert to meters
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--rgb" && i + 1 < argc) {
            rgb_path = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depth_path = argv[++i];
        } else if (arg == "--intrinsics" && i + 1 < argc) {
            intrinsics_path = argv[++i];
        } else if (arg == "--landmarks" && i + 1 < argc) {
            landmarks_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--mapping" && i + 1 < argc) {
            mapping_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        } else if (arg == "--depth-scale" && i + 1 < argc) {
            depth_scale = std::stod(argv[++i]);
        } else if (arg == "--bfm-scale" && i + 1 < argc) {
            bfm_scale = std::stod(argv[++i]);
        }
    }
    
    // Validate required arguments
    if (rgb_path.empty() || depth_path.empty() || intrinsics_path.empty() ||
        landmarks_path.empty() || model_dir.empty() || mapping_path.empty() ||
        output_path.empty()) {
        std::cerr << "Error: Missing required arguments\n";
        printUsage(argv[0]);
        return 1;
    }
    
    std::cout << "=== Pose Initialization (Procrustes Alignment) ===\n" << std::endl;
    
    // Load RGB-D frame
    RGBDFrame frame;
    std::cout << "[1] Loading RGB-D frame..." << std::endl;
    if (!frame.loadRGB(rgb_path)) {
        std::cerr << "Error: Failed to load RGB image" << std::endl;
        return 1;
    }
    if (!frame.loadDepth(depth_path, depth_scale)) {
        std::cerr << "Error: Failed to load depth image" << std::endl;
        return 1;
    }
    std::cout << "    RGB: " << frame.width() << "x" << frame.height() << std::endl;
    std::cout << "    Depth: " << frame.getDepth().cols << "x" << frame.getDepth().rows << std::endl;
    
    // Load camera intrinsics
    CameraIntrinsics intrinsics;
    std::cout << "[2] Loading camera intrinsics..." << std::endl;
    try {
        intrinsics = CameraIntrinsics::loadFromFile(intrinsics_path);
        std::cout << "    fx=" << intrinsics.fx << ", fy=" << intrinsics.fy 
                  << ", cx=" << intrinsics.cx << ", cy=" << intrinsics.cy << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Load landmarks
    LandmarkData landmarks;
    std::cout << "[3] Loading landmarks..." << std::endl;
    std::string ext = landmarks_path.substr(landmarks_path.find_last_of(".") + 1);
    bool loaded = false;
    if (ext == "json" || ext == "JSON") {
        loaded = landmarks.loadFromJSON(landmarks_path);
    } else {
        loaded = landmarks.loadFromTXT(landmarks_path);
    }
    if (!loaded || landmarks.size() == 0) {
        std::cerr << "Error: Failed to load landmarks" << std::endl;
        return 1;
    }
    std::cout << "    Loaded " << landmarks.size() << " landmarks" << std::endl;
    
    // Load model
    MorphableModel model;
    std::cout << "[4] Loading PCA model..." << std::endl;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return 1;
    }
    std::cout << "    Model: " << model.num_vertices << " vertices" << std::endl;
    
    // Load landmark mapping
    LandmarkMapping mapping;
    std::cout << "[5] Loading landmark mapping..." << std::endl;
    if (!mapping.loadFromFile(mapping_path)) {
        std::cerr << "Error: Failed to load mapping file" << std::endl;
        return 1;
    }
    std::cout << "    Loaded " << mapping.size() << " mappings" << std::endl;
    
    // Step 1: Extract 3D points from depth at landmark locations
    std::cout << "\n[6] Extracting 3D points from depth at landmarks..." << std::endl;
    const cv::Mat& depth_map = frame.getDepth();
    std::vector<Eigen::Vector3d> observed_3d_points;
    std::vector<int> valid_landmark_indices;
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const auto& lm = landmarks[i];
        int u = static_cast<int>(std::round(lm.x));
        int v = static_cast<int>(std::round(lm.y));
        
        if (u < 0 || u >= depth_map.cols || v < 0 || v >= depth_map.rows) {
            continue;
        }
        
        float depth = depth_map.at<float>(v, u);
        if (!frame.isValidDepth(depth)) {
            continue;
        }
        
        Eigen::Vector3d point_3d = depthTo3D(u, v, static_cast<double>(depth), intrinsics);
        observed_3d_points.push_back(point_3d);
        valid_landmark_indices.push_back(static_cast<int>(i));
    }
    
    std::cout << "    Found " << observed_3d_points.size() << " valid 3D points" << std::endl;
    
    if (observed_3d_points.size() < 6) {
        std::cerr << "Error: Not enough valid depth points (" << observed_3d_points.size() 
                  << "). Need at least 6 for Procrustes alignment." << std::endl;
        return 1;
    }
    
    // Step 2: Get corresponding model vertices using mapping
    // Build matched pairs: only keep landmarks that have both valid depth AND valid mapping
    // CRITICAL: Convert BFM model from mm to meters before Procrustes
    // CRITICAL: Apply coordinate system transformation (BFM -> Camera)
    std::cout << "[7] Getting corresponding model vertices..." << std::endl;
    std::cout << "    Converting BFM model from mm to meters (scale=" << bfm_scale << ")..." << std::endl;
    std::cout << "    Applying BFM->Camera coordinate transform (flip Y and Z)..." << std::endl;
    
    std::vector<Eigen::Vector3d> matched_observed;
    std::vector<Eigen::Vector3d> matched_model;
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
    // BFM coordinate system: X right, Y up, Z out of face (towards viewer)
    // Camera coordinate system: X right, Y down, Z forward (into scene)
    // Transform: flip Y (up->down) and flip Z (face towards camera -> face away)
    Eigen::Matrix3d bfm_to_camera;
    bfm_to_camera << 1,  0,  0,
                     0, -1,  0,
                     0,  0, -1;
    
    // Verify same ordering: ensure matched pairs have same indices
    for (size_t i = 0; i < valid_landmark_indices.size(); ++i) {
        int landmark_idx = valid_landmark_indices[i];
        
        if (!mapping.hasMapping(landmark_idx)) {
            continue;
        }
        
        int model_vertex_idx = mapping.getModelVertex(landmark_idx);
        if (model_vertex_idx < 0 || model_vertex_idx >= model.num_vertices) {
            continue;
        }
        
        // Both observed 3D point AND model point are valid - add as matched pair
        // CRITICAL: Convert BFM from mm to meters AND transform coordinate system
        Eigen::Vector3d model_point = mean_shape.row(model_vertex_idx).transpose() * bfm_scale;
        model_point = bfm_to_camera * model_point;  // Apply coordinate transform
        matched_model.push_back(model_point);
        matched_observed.push_back(observed_3d_points[i]);  // Use same index i - same ordering
    }
    
    std::cout << "    Found " << matched_model.size() << " valid correspondences" << std::endl;
    
    // Sanity check: same size
    if (matched_model.size() != matched_observed.size()) {
        std::cerr << "Error: Mismatch in correspondence sizes: model=" << matched_model.size()
                  << ", observed=" << matched_observed.size() << std::endl;
        return 1;
    }
    
    if (matched_model.size() < 6) {
        std::cerr << "Error: Not enough valid correspondences (" << matched_model.size() 
                  << "). Need at least 6." << std::endl;
        return 1;
    }
    
    // Step 3: Estimate similarity transform with Procrustes
    std::cout << "[8] Estimating similarity transform (Procrustes)..." << std::endl;
    
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
        std::cerr << "Error: Procrustes alignment failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "    Scale: " << std::fixed << std::setprecision(6) << transform.scale << std::endl;
    std::cout << "    Translation: (" << transform.translation.x() << ", "
              << transform.translation.y() << ", " << transform.translation.z() << ")" << std::endl;
    std::cout << "    Translation norm: " << transform.translation.norm() << " m" << std::endl;
    
    // Sanity checks
    double rot_det = transform.rotation.determinant();
    std::cout << "    Rotation det: " << rot_det << " (should be ~1.0)" << std::endl;
    if (std::abs(rot_det - 1.0) > 0.01) {
        std::cerr << "    WARNING: Rotation determinant is not ~1.0! Possible reflection." << std::endl;
    }
    
    if (transform.scale < 0.5 || transform.scale > 2.0) {
        std::cerr << "    WARNING: Scale factor " << transform.scale 
                  << " is outside expected range [0.5, 2.0]. Possible unit mismatch." << std::endl;
    }
    
    // Compute alignment error
    Eigen::MatrixXd transformed_model = transform.apply(source_points);
    std::vector<double> errors;
    double total_error = 0.0;
    for (int i = 0; i < transformed_model.rows(); ++i) {
        double err = (transformed_model.row(i) - target_points.row(i)).norm();
        errors.push_back(err);
        total_error += err * err;
    }
    double rmse = std::sqrt(total_error / transformed_model.rows());
    std::sort(errors.begin(), errors.end());
    double mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double median_error = errors[errors.size() / 2];
    
    std::cout << "    Alignment RMSE (after Procrustes): " << rmse << " m (" << rmse * 1000.0 << " mm)" << std::endl;
    std::cout << "    Mean error: " << mean_error << " m (" << mean_error * 1000.0 << " mm)" << std::endl;
    std::cout << "    Median error: " << median_error << " m (" << median_error * 1000.0 << " mm)" << std::endl;
    
    // Step 4: Refine with ICP on landmark correspondences only
    std::cout << "[9] Refining alignment with ICP (landmark correspondences only)..." << std::endl;
    ICP icp;
    
    // Use only landmark correspondences for ICP refinement
    // This ensures we refine on known good correspondences
    ICP::ICPResult icp_result = icp.align(
        source_points,  // Model landmark vertices
        target_points,  // Observed landmark points
        transform.rotation,
        transform.translation,
        5,   // max iterations (fewer for landmark-only)
        1e-4  // convergence threshold
    );
    
    // Update transform with ICP result (keep scale from Procrustes)
    transform.rotation = icp_result.rotation;
    transform.translation = icp_result.translation;
    
    std::cout << "    ICP iterations: " << icp_result.iterations << std::endl;
    std::cout << "    ICP converged: " << (icp_result.converged ? "Yes" : "No") << std::endl;
    std::cout << "    ICP final error: " << icp_result.final_error << " m (" 
              << icp_result.final_error * 1000.0 << " mm)" << std::endl;
    
    // Recompute alignment error after ICP
    Eigen::MatrixXd refined_model_landmarks = transform.apply(source_points);
    errors.clear();
    total_error = 0.0;
    for (int i = 0; i < refined_model_landmarks.rows(); ++i) {
        double err = (refined_model_landmarks.row(i) - target_points.row(i)).norm();
        errors.push_back(err);
        total_error += err * err;
    }
    rmse = std::sqrt(total_error / refined_model_landmarks.rows());
    std::sort(errors.begin(), errors.end());
    mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    median_error = errors[errors.size() / 2];
    
    std::cout << "    Alignment RMSE (after ICP): " << rmse << " m (" << rmse * 1000.0 << " mm)" << std::endl;
    std::cout << "    Mean error: " << mean_error << " m (" << mean_error * 1000.0 << " mm)" << std::endl;
    std::cout << "    Median error: " << median_error << " m (" << median_error * 1000.0 << " mm)" << std::endl;
    
    // Step 5: Apply final transform to full model
    std::cout << "[10] Applying final transform to model and saving..." << std::endl;
    // CRITICAL: Convert BFM from mm to meters AND apply coordinate transform before Procrustes transform
    Eigen::MatrixXd mean_vertices = model.getMeanShapeMatrix() * bfm_scale;  // Convert to meters
    
    // Apply BFM->Camera coordinate transform to all vertices (reuse bfm_to_camera from above)
    // BFM: X right, Y up, Z out | Camera: X right, Y down, Z forward
    mean_vertices = (bfm_to_camera * mean_vertices.transpose()).transpose();
    
    Eigen::MatrixXd aligned_vertices = transform.apply(mean_vertices);
    
    // Depth z-range check
    double obs_z_min = std::numeric_limits<double>::max();
    double obs_z_max = std::numeric_limits<double>::lowest();
    for (const auto& p : observed_3d_points) {
        obs_z_min = std::min(obs_z_min, p.z());
        obs_z_max = std::max(obs_z_max, p.z());
    }
    
    double mesh_z_min = std::numeric_limits<double>::max();
    double mesh_z_max = std::numeric_limits<double>::lowest();
    for (int i = 0; i < aligned_vertices.rows(); ++i) {
        double z = aligned_vertices(i, 2);
        mesh_z_min = std::min(mesh_z_min, z);
        mesh_z_max = std::max(mesh_z_max, z);
    }
    
    std::cout << "    Observed depth z-range: [" << obs_z_min << ", " << obs_z_max << "] m" << std::endl;
    std::cout << "    Mesh z-range: [" << mesh_z_min << ", " << mesh_z_max << "] m" << std::endl;
    
    bool z_in_range = (mesh_z_min >= obs_z_min - 0.3 && mesh_z_max <= obs_z_max + 0.3);
    if (!z_in_range) {
        std::cerr << "    WARNING: Mesh z-range does not overlap well with observed depth range!" << std::endl;
    } else {
        std::cout << "    ✓ Mesh z-range is within observed depth range" << std::endl;
    }
    
    if (!model.saveMeshPLY(aligned_vertices, output_path)) {
        std::cerr << "Error: Failed to save aligned mesh" << std::endl;
        return 1;
    }
    
    std::cout << "    Saved aligned mesh to: " << output_path << std::endl;
    
    // Write JSON report if requested
    if (!report_path.empty()) {
        std::cout << "[11] Writing JSON report..." << std::endl;
        writeJSONReport(report_path, transform, rmse, mean_error, median_error, 
                       observed_3d_points, aligned_vertices);
        std::cout << "    Report saved to: " << report_path << std::endl;
    }
    
    std::cout << "\n=== Pose initialization completed successfully ===" << std::endl;
    
    return 0;
}

