/**
 * Face Reconstruction Tool
 * 
 * Main 3D face reconstruction executable called by Python pipeline.
 * Reconstructs 3D face meshes from RGB-D data using a PCA morphable model.
 * 
 * Pipeline Step: ReconstructionStep (Python)
 * 
 * Usage:
 *   build/bin/face_reconstruction --rgb <path> --depth <path> \
 *                                  --intrinsics <path> --model-dir <path> \
 *                                  --output-mesh <path> [--output-pointcloud <path>]
 * 
 * Example:
 *   build/bin/face_reconstruction --rgb outputs/converted/01/rgb/frame_00000.png \
 *                                  --depth outputs/converted/01/depth/frame_00000.png \
 *                                  --intrinsics outputs/converted/01/intrinsics.txt \
 *                                  --model-dir data/model_biwi \
 *                                  --output-mesh outputs/meshes/01/frame_00000.ply
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include "model/MorphableModel.h"
#include "landmarks/LandmarkData.h"
#include "alignment/Procrustes.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace face_reconstruction;

/**
 * Export point cloud to PLY file
 */
bool savePointCloudPLY(const std::vector<Eigen::Vector3d>& points,
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
    file << "end_header\n";
    
    // Write vertices
    for (const auto& point : points) {
        file << std::fixed << std::setprecision(6)
             << point.x() << " " 
             << point.y() << " " 
             << point.z() << "\n";
    }
    
    file.close();
    return true;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --rgb <path>              Path to RGB image file\n"
              << "  --depth <path>            Path to depth image file\n"
              << "  --depth-scale <value>     Depth scale factor (default: 1000.0)\n"
              << "  --intrinsics <path>       Path to camera intrinsics file (fx fy cx cy)\n"
              << "  --model-dir <path>        Directory containing PCA model files\n"
              << "  --landmarks <path>        Path to landmarks file (TXT or JSON)\n"
              << "  --output-mesh <path>      Output mesh file path (PLY or OBJ)\n"
              << "  --output-pointcloud <path> Output point cloud file path (PLY)\n"
              << "  --help                    Show this help message\n"
              << "\n"
              << "Example:\n"
              << "  " << program_name << " --rgb data/rgb.png --depth data/depth.png \\\n"
              << "     --intrinsics data/intrinsics.txt --model-dir data/model \\\n"
              << "     --landmarks data/landmarks.txt --output-mesh output/face.ply \\\n"
              << "     --output-pointcloud output/cloud.ply\n";
}

int main(int argc, char* argv[]) {
    std::string rgb_path, depth_path, intrinsics_path, model_dir, landmarks_path, output_mesh, output_pointcloud;
    double depth_scale = 1000.0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--rgb" && i + 1 < argc) {
            rgb_path = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depth_path = argv[++i];
        } else if (arg == "--depth-scale" && i + 1 < argc) {
            depth_scale = std::stod(argv[++i]);
        } else if (arg == "--intrinsics" && i + 1 < argc) {
            intrinsics_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--landmarks" && i + 1 < argc) {
            landmarks_path = argv[++i];
        } else if (arg == "--output-mesh" && i + 1 < argc) {
            output_mesh = argv[++i];
        } else if (arg == "--output-pointcloud" && i + 1 < argc) {
            output_pointcloud = argv[++i];
        }
    }
    
    std::cout << "=== 3D Face Reconstruction ===\n" << std::endl;
    
    // Load RGB-D frame
    RGBDFrame frame;
    if (!rgb_path.empty()) {
        std::cout << "[1] Loading RGB image: " << rgb_path << std::endl;
        if (!frame.loadRGB(rgb_path)) {
            std::cerr << "Error: Failed to load RGB image" << std::endl;
            return 1;
        }
        std::cout << "    RGB loaded: " << frame.width() << "x" << frame.height() << std::endl;
    }
    
    if (!depth_path.empty()) {
        std::cout << "[2] Loading depth image: " << depth_path << std::endl;
        if (!frame.loadDepth(depth_path, depth_scale)) {
            std::cerr << "Error: Failed to load depth image" << std::endl;
            return 1;
        }
        std::cout << "    Depth loaded: " << frame.getDepth().cols 
                  << "x" << frame.getDepth().rows << std::endl;
        frame.printStats();
    }
    
    // Load camera intrinsics
    CameraIntrinsics intrinsics;
    if (!intrinsics_path.empty()) {
        std::cout << "[3] Loading camera intrinsics: " << intrinsics_path << std::endl;
        try {
            intrinsics = CameraIntrinsics::loadFromFile(intrinsics_path);
            std::cout << "    fx=" << intrinsics.fx << ", fy=" << intrinsics.fy 
                      << ", cx=" << intrinsics.cx << ", cy=" << intrinsics.cy << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "[3] Using default intrinsics (525, 525, 320, 240)" << std::endl;
        intrinsics = CameraIntrinsics(525.0, 525.0, 320.0, 240.0);
    }
    
    // Backproject depth to 3D
    std::vector<Eigen::Vector3d> points_3d;
    std::vector<std::pair<int, int>> pixel_indices;
    if (!depth_path.empty()) {
        std::cout << "[4] Backprojecting depth to 3D..." << std::endl;
        backprojectDepth(frame.getDepth(), intrinsics, points_3d, pixel_indices);
        std::cout << "    Generated " << points_3d.size() << " 3D points" << std::endl;
        
        // Save point cloud if requested
        if (!output_pointcloud.empty()) {
            std::cout << "[4a] Saving point cloud to: " << output_pointcloud << std::endl;
            if (savePointCloudPLY(points_3d, output_pointcloud)) {
                std::cout << "    Point cloud saved successfully!" << std::endl;
            } else {
                std::cerr << "Error: Failed to save point cloud" << std::endl;
                return 1;
            }
        }
    }
    
    // Load PCA model
    MorphableModel model;
    if (!model_dir.empty()) {
        std::cout << "[5] Loading PCA model from: " << model_dir << std::endl;
        if (!model.loadFromFiles(model_dir)) {
            std::cerr << "Error: Failed to load PCA model" << std::endl;
            return 1;
        }
        model.printStats();
        
        if (!model.isValid()) {
            std::cerr << "Error: Model is not valid" << std::endl;
            return 1;
        }
    } else {
        std::cout << "[5] No model directory specified, skipping model loading" << std::endl;
    }
    
    // Load landmarks
    LandmarkData landmarks;
    if (!landmarks_path.empty()) {
        std::cout << "[6] Loading landmarks: " << landmarks_path << std::endl;
        std::string ext = landmarks_path.substr(landmarks_path.find_last_of(".") + 1);
        bool loaded = false;
        if (ext == "json" || ext == "JSON") {
            loaded = landmarks.loadFromJSON(landmarks_path);
        } else {
            loaded = landmarks.loadFromTXT(landmarks_path);
        }
        
        if (!loaded) {
            std::cerr << "Error: Failed to load landmarks" << std::endl;
            return 1;
        }
        std::cout << "    Loaded " << landmarks.size() << " landmarks" << std::endl;
    }
    
    // Reconstruct face mesh (if model is loaded)
    if (model.isValid()) {
        std::cout << "[7] Reconstructing face mesh..." << std::endl;
        
        // For now, use zero coefficients (mean shape)
        // In Week 2+, we'll optimize these coefficients based on landmarks and depth
        Eigen::VectorXd identity_coeffs = Eigen::VectorXd::Zero(model.num_identity_components);
        Eigen::VectorXd expression_coeffs = Eigen::VectorXd::Zero(model.num_expression_components);
        
        Eigen::MatrixXd vertices = model.reconstructFace(identity_coeffs, expression_coeffs);
        std::cout << "    Reconstructed mesh with " << vertices.rows() << " vertices" << std::endl;
        
        // Save mesh to file
        if (!output_mesh.empty()) {
            std::cout << "[8] Saving mesh to: " << output_mesh << std::endl;
            std::string ext = output_mesh.substr(output_mesh.find_last_of(".") + 1);
            
            bool saved = false;
            if (ext == "ply" || ext == "PLY") {
                saved = model.saveMeshPLY(vertices, output_mesh);
            } else if (ext == "obj" || ext == "OBJ") {
                saved = model.saveMeshOBJ(vertices, output_mesh);
            } else {
                std::cerr << "Error: Unsupported mesh format. Use .ply or .obj" << std::endl;
                return 1;
            }
            
            if (saved) {
                std::cout << "    Mesh saved successfully!" << std::endl;
            } else {
                std::cerr << "Error: Failed to save mesh" << std::endl;
                return 1;
            }
        } else {
            std::cout << "[8] No output mesh path specified (use --output-mesh)" << std::endl;
        }
    }
    
    std::cout << "\n=== Reconstruction completed successfully ===" << std::endl;
    return 0;
}
