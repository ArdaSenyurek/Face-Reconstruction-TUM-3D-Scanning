/**
 * Face Reconstruction Tool (Week 4: With Optimization)
 * 
 * Main 3D face reconstruction executable called by Python pipeline.
 * Reconstructs 3D face meshes from RGB-D data using a PCA morphable model
 * and Gauss-Newton optimization.
 * 
 * Pipeline Step: ReconstructionStep (Python)
 * 
 * Usage:
 *   build/bin/face_reconstruction --rgb <path> --depth <path> \
 *                                  --intrinsics <path> --model-dir <path> \
 *                                  --landmarks <path> --mapping <path> \
 *                                  --output-mesh <path> [options]
 * 
 * Optimization modes:
 *   --optimize         Enable Gauss-Newton optimization (default: off for mean shape only)
 *   --no-optimize      Output mean shape (zero coefficients)
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include "model/MorphableModel.h"
#include "landmarks/LandmarkData.h"
#include "alignment/Procrustes.h"
#include "alignment/LandmarkMapping.h"
#include "optimization/Parameters.h"
#include "optimization/EnergyFunction.h"
#include "optimization/GaussNewton.h"
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

/**
 * Apply pose transformation to vertices (with scale)
 */
Eigen::MatrixXd applyPoseToVertices(const Eigen::MatrixXd& vertices,
                                    const Eigen::Matrix3d& R,
                                    const Eigen::Vector3d& t,
                                    double scale = 1.0) {
    Eigen::MatrixXd transformed(vertices.rows(), 3);
    for (int i = 0; i < vertices.rows(); ++i) {
        Eigen::Vector3d v = vertices.row(i).transpose();
        Eigen::Vector3d v_transformed = scale * (R * v) + t;
        transformed.row(i) = v_transformed.transpose();
    }
    return transformed;
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
              << "  --mapping <path>          Path to landmark mapping file\n"
              << "  --output-mesh <path>      Output mesh file path (PLY or OBJ)\n"
              << "  --output-pointcloud <path> Output point cloud file path (PLY)\n"
              << "  --optimize                Enable Gauss-Newton optimization\n"
              << "  --no-optimize             Output mean shape only (default)\n"
              << "  --pose-only               Optimize pose only (no shape/expression)\n"
              << "  --max-iter <n>            Max optimization iterations (default: 50)\n"
              << "  --lambda-landmark <w>     Landmark weight (default: 1.0)\n"
              << "  --lambda-depth <w>        Depth weight (default: 0.1)\n"
              << "  --lambda-reg <w>          Regularization weight (default: 1.0)\n"
              << "  --verbose                 Print detailed optimization output\n"
              << "  --help                    Show this help message\n"
              << "\n"
              << "Example:\n"
              << "  " << program_name << " --rgb data/rgb.png --depth data/depth.png \\\n"
              << "     --intrinsics data/intrinsics.txt --model-dir data/model_bfm \\\n"
              << "     --landmarks data/landmarks.txt --mapping data/landmark_mapping.txt \\\n"
              << "     --output-mesh output/face.ply --optimize\n";
}

int main(int argc, char* argv[]) {
    std::string rgb_path, depth_path, intrinsics_path, model_dir;
    std::string landmarks_path, mapping_path, output_mesh, output_pointcloud;
    double depth_scale = 1000.0;
    bool optimize = false;
    bool pose_only = false;
    bool verbose = false;
    int max_iterations = 50;
    double lambda_landmark = 1.0;
    double lambda_depth = 0.1;
    double lambda_reg = 1.0;
    
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
        } else if (arg == "--mapping" && i + 1 < argc) {
            mapping_path = argv[++i];
        } else if (arg == "--output-mesh" && i + 1 < argc) {
            output_mesh = argv[++i];
        } else if (arg == "--output-pointcloud" && i + 1 < argc) {
            output_pointcloud = argv[++i];
        } else if (arg == "--optimize") {
            optimize = true;
        } else if (arg == "--no-optimize") {
            optimize = false;
        } else if (arg == "--pose-only") {
            pose_only = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--max-iter" && i + 1 < argc) {
            max_iterations = std::stoi(argv[++i]);
        } else if (arg == "--lambda-landmark" && i + 1 < argc) {
            lambda_landmark = std::stod(argv[++i]);
        } else if (arg == "--lambda-depth" && i + 1 < argc) {
            lambda_depth = std::stod(argv[++i]);
        } else if (arg == "--lambda-reg" && i + 1 < argc) {
            lambda_reg = std::stod(argv[++i]);
        }
    }
    
    std::cout << "=== 3D Face Reconstruction ===" << std::endl;
    std::cout << "Mode: " << (optimize ? "Optimized" : "Mean Shape Only") << std::endl;
    std::cout << std::endl;
    
    // Load RGB-D frame
    RGBDFrame frame;
    int image_width = 640, image_height = 480;  // Default
    
    if (!rgb_path.empty()) {
        std::cout << "[1] Loading RGB image: " << rgb_path << std::endl;
        if (!frame.loadRGB(rgb_path)) {
            std::cerr << "Error: Failed to load RGB image" << std::endl;
            return 1;
        }
        image_width = frame.width();
        image_height = frame.height();
        std::cout << "    RGB loaded: " << image_width << "x" << image_height << std::endl;
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
        return 1;
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
    
    // Load landmark mapping
    LandmarkMapping mapping;
    if (!mapping_path.empty()) {
        std::cout << "[7] Loading landmark mapping: " << mapping_path << std::endl;
        if (!mapping.loadFromFile(mapping_path)) {
            std::cerr << "Error: Failed to load landmark mapping" << std::endl;
            return 1;
        }
        std::cout << "    Loaded " << mapping.size() << " mappings" << std::endl;
    }
    
    // Initialize optimization parameters
    OptimizationParams params(model.num_identity_components, model.num_expression_components);
    params.max_iterations = max_iterations;
    params.lambda_landmark = lambda_landmark;
    params.lambda_depth = lambda_depth;
    params.lambda_alpha = lambda_reg;
    params.lambda_delta = lambda_reg;
    
    // Apply pose-only mode (faster but less accurate)
    if (pose_only) {
        params.optimize_identity = false;
        params.optimize_expression = false;
    }
    
    // Scale factor from Procrustes (BFM mm -> camera meters)
    double pose_scale = 1.0;
    
    Eigen::MatrixXd final_vertices;
    
    if (optimize && landmarks.size() > 0 && mapping.size() > 0) {
        // =========================================
        // Week 4: Gauss-Newton Optimization
        // =========================================
        std::cout << "\n[8] Running Gauss-Newton optimization..." << std::endl;
        
        // Initialize pose from Procrustes if we have depth and landmarks
        if (points_3d.size() > 0) {
            std::cout << "    Initializing pose with Procrustes..." << std::endl;
            
            // Extract 3D points at landmark locations
            std::vector<Eigen::Vector3d> landmark_points_3d;
            std::vector<Eigen::Vector3d> model_points_3d;
            
            for (size_t i = 0; i < landmarks.size(); ++i) {
                int lm_idx = static_cast<int>(i);
                if (!mapping.hasMapping(lm_idx)) continue;
                
                int vertex_idx = mapping.getModelVertex(lm_idx);
                if (vertex_idx < 0 || vertex_idx >= model.num_vertices) continue;
                
                // Get landmark position
                const auto& lm = landmarks[i];
                int u = static_cast<int>(lm.x);
                int v = static_cast<int>(lm.y);
                
                // Get depth at landmark location
                if (u >= 0 && u < frame.getDepth().cols && 
                    v >= 0 && v < frame.getDepth().rows) {
                    float d = frame.getDepth().at<float>(v, u);
                    if (d > 0) {
                        // Backproject to 3D
                        double X = (u - intrinsics.cx) * d / intrinsics.fx;
                        double Y = (v - intrinsics.cy) * d / intrinsics.fy;
                        double Z = d;
                        landmark_points_3d.push_back(Eigen::Vector3d(X, Y, Z));
                        
                        // Get corresponding model point
                        model_points_3d.push_back(model.getMeanShapeMatrix().row(vertex_idx));
                    }
                }
            }
            
            if (landmark_points_3d.size() >= 4) {
                // Estimate similarity transform
                SimilarityTransform transform = estimateSimilarityTransform(
                    model_points_3d, landmark_points_3d);
                
                // Initialize params with Procrustes result
                params.R = transform.rotation;
                params.t = transform.translation;
                params.scale = transform.scale;  // Critical: BFM mm -> camera meters
                pose_scale = transform.scale;    // Also store for final output
                
                std::cout << "    Procrustes init: " << landmark_points_3d.size() 
                          << " correspondences, scale=" << pose_scale << std::endl;
            }
        }
        
        // Run optimization
        GaussNewtonOptimizer optimizer;
        optimizer.initialize(model, intrinsics, image_width, image_height);
        optimizer.setVerbose(verbose);
        
        OptimizationResult result = optimizer.optimize(
            params, landmarks, mapping, frame.getDepth());
        
        // Report results
        std::cout << "\n    Optimization results:" << std::endl;
        std::cout << "      Iterations: " << result.iterations << std::endl;
        std::cout << "      Converged: " << (result.converged ? "Yes" : "No") << std::endl;
        std::cout << "      Initial energy: " << result.initial_energy << std::endl;
        std::cout << "      Final energy: " << result.final_energy << std::endl;
        std::cout << "      Landmark energy: " << result.landmark_energy << std::endl;
        std::cout << "      Depth energy: " << result.depth_energy << std::endl;
        std::cout << "      Regularization: " << result.regularization_energy << std::endl;
        
        // Reconstruct with optimized coefficients
        final_vertices = model.reconstructFace(
            result.final_params.alpha, result.final_params.delta);
        
        // Apply final pose (including scale from optimization result)
        final_vertices = applyPoseToVertices(
            final_vertices, result.final_params.R, result.final_params.t, 
            result.final_params.scale);
        
    } else {
        // =========================================
        // Mean Shape Only (no optimization)
        // =========================================
        std::cout << "\n[8] Reconstructing mean shape (no optimization)..." << std::endl;
        
        Eigen::VectorXd identity_coeffs = Eigen::VectorXd::Zero(model.num_identity_components);
        Eigen::VectorXd expression_coeffs = Eigen::VectorXd::Zero(model.num_expression_components);
        
        final_vertices = model.reconstructFace(identity_coeffs, expression_coeffs);
        
        // If we have landmarks and mapping, try to align with Procrustes
        if (landmarks.size() > 0 && mapping.size() > 0 && points_3d.size() > 0) {
            std::cout << "    Applying Procrustes alignment..." << std::endl;
            
            std::vector<Eigen::Vector3d> landmark_points_3d;
            std::vector<Eigen::Vector3d> model_points_3d;
            
            for (size_t i = 0; i < landmarks.size(); ++i) {
                int lm_idx = static_cast<int>(i);
                if (!mapping.hasMapping(lm_idx)) continue;
                
                int vertex_idx = mapping.getModelVertex(lm_idx);
                if (vertex_idx < 0 || vertex_idx >= model.num_vertices) continue;
                
                const auto& lm = landmarks[i];
                int u = static_cast<int>(lm.x);
                int v = static_cast<int>(lm.y);
                
                if (u >= 0 && u < frame.getDepth().cols && 
                    v >= 0 && v < frame.getDepth().rows) {
                    float d = frame.getDepth().at<float>(v, u);
                    if (d > 0) {
                        double X = (u - intrinsics.cx) * d / intrinsics.fx;
                        double Y = (v - intrinsics.cy) * d / intrinsics.fy;
                        double Z = d;
                        landmark_points_3d.push_back(Eigen::Vector3d(X, Y, Z));
                        model_points_3d.push_back(final_vertices.row(vertex_idx));
                    }
                }
            }
            
            if (landmark_points_3d.size() >= 4) {
                SimilarityTransform transform = estimateSimilarityTransform(
                    model_points_3d, landmark_points_3d);
                final_vertices = applyPoseToVertices(
                    final_vertices, transform.rotation, transform.translation, transform.scale);
                std::cout << "    Aligned using " << landmark_points_3d.size() 
                          << " correspondences, scale=" << transform.scale << std::endl;
            }
        }
    }
    
    std::cout << "    Final mesh: " << final_vertices.rows() << " vertices" << std::endl;
        
        // Save mesh to file
        if (!output_mesh.empty()) {
        std::cout << "\n[9] Saving mesh to: " << output_mesh << std::endl;
            std::string ext = output_mesh.substr(output_mesh.find_last_of(".") + 1);
            
            bool saved = false;
            if (ext == "ply" || ext == "PLY") {
            saved = model.saveMeshPLY(final_vertices, output_mesh);
            } else if (ext == "obj" || ext == "OBJ") {
            saved = model.saveMeshOBJ(final_vertices, output_mesh);
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
        std::cout << "\n[9] No output mesh path specified (use --output-mesh)" << std::endl;
    }
    
    std::cout << "\n=== Reconstruction completed successfully ===" << std::endl;
    return 0;
}
