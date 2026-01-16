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
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

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
              << "  --depth-scale <value>     Depth scale factor (default: 1000.0)\n"
              << "  --help                     Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string rgb_path, depth_path, intrinsics_path, landmarks_path, 
                model_dir, mapping_path, output_path;
    double depth_scale = 1000.0;
    
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
        } else if (arg == "--depth-scale" && i + 1 < argc) {
            depth_scale = std::stod(argv[++i]);
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
    std::cout << "[7] Getting corresponding model vertices..." << std::endl;
    std::vector<Eigen::Vector3d> matched_observed;
    std::vector<Eigen::Vector3d> matched_model;
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
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
        Eigen::Vector3d model_point = mean_shape.row(model_vertex_idx);
        matched_model.push_back(model_point);
        matched_observed.push_back(observed_3d_points[i]);  // Use same index i
    }
    
    std::cout << "    Found " << matched_model.size() << " valid correspondences" << std::endl;
    
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
    
    // Compute alignment error
    Eigen::MatrixXd transformed_model = transform.apply(source_points);
    double total_error = 0.0;
    for (int i = 0; i < transformed_model.rows(); ++i) {
        total_error += (transformed_model.row(i) - target_points.row(i)).squaredNorm();
    }
    double rmse = std::sqrt(total_error / transformed_model.rows());
    std::cout << "    Alignment RMSE: " << rmse << " m" << std::endl;
    
    // Step 4: Apply transform to full model and save
    std::cout << "[9] Applying transform to model and saving..." << std::endl;
    
    Eigen::MatrixXd mean_vertices = model.getMeanShapeMatrix();
    Eigen::MatrixXd aligned_vertices = transform.apply(mean_vertices);
    
    if (!model.saveMeshPLY(aligned_vertices, output_path)) {
        std::cerr << "Error: Failed to save aligned mesh" << std::endl;
        return 1;
    }
    
    std::cout << "    Saved aligned mesh to: " << output_path << std::endl;
    std::cout << "\n=== Pose initialization completed successfully ===" << std::endl;
    
    return 0;
}

