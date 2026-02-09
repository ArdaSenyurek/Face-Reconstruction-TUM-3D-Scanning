/**
 * Face Reconstruction Tool (Week 5: With Tracking Support)
 * 
 * Main 3D face reconstruction executable called by Python pipeline.
 * Reconstructs 3D face meshes from RGB-D data using a PCA morphable model
 * and Gauss-Newton optimization.
 * 
 * Week 5 additions:
 *   --init-pose-json    Load initial pose/expression from JSON (for tracking warm-start)
 *   --output-state-json Save final pose/expression to JSON (for next frame)
 * 
 * Pipeline Step: ReconstructionStep / TrackingStep (Python)
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
#include <sstream>

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

/**
 * Week 5/6: Simple JSON parsing for tracking state
 * Parses a minimal JSON format for pose, identity (alpha), and expression (delta) coefficients
 */
bool loadTrackingStateJSON(const std::string& filepath,
                           Eigen::Matrix3d& R,
                           Eigen::Vector3d& t,
                           double& scale,
                           Eigen::VectorXd& expression,
                           Eigen::VectorXd* identity_out = nullptr) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking state file: " << filepath << std::endl;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    // Simple JSON parsing (not a full parser, but sufficient for our format)
    auto findValue = [&content](const std::string& key) -> std::string {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = content.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n')) pos++;
        
        size_t start = pos;
        if (content[pos] == '[') {
            // Find matching bracket
            int depth = 1;
            pos++;
            while (pos < content.size() && depth > 0) {
                if (content[pos] == '[') depth++;
                else if (content[pos] == ']') depth--;
                pos++;
            }
            return content.substr(start, pos - start);
        } else {
            // Find number or string end
            while (pos < content.size() && content[pos] != ',' && content[pos] != '}' && content[pos] != '\n') pos++;
            return content.substr(start, pos - start);
        }
    };
    
    auto parseDoubleArray = [](const std::string& arr) -> std::vector<double> {
        std::vector<double> result;
        std::string cleaned;
        for (char c : arr) {
            if (c == '[' || c == ']' || c == ' ' || c == '\n') continue;
            cleaned += c;
        }
        std::stringstream ss(cleaned);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                try {
                    result.push_back(std::stod(token));
                } catch (...) {}
            }
        }
        return result;
    };
    
    // Parse scale
    std::string scale_str = findValue("scale");
    if (!scale_str.empty()) {
        try {
            scale = std::stod(scale_str);
        } catch (...) {
            scale = 1.0;
        }
    }
    
    // Parse translation
    std::string trans_str = findValue("translation");
    if (!trans_str.empty()) {
        auto trans = parseDoubleArray(trans_str);
        if (trans.size() >= 3) {
            t = Eigen::Vector3d(trans[0], trans[1], trans[2]);
        }
    }
    
    // Parse rotation (3x3 matrix as nested array)
    std::string rot_str = findValue("rotation");
    if (!rot_str.empty()) {
        auto rot = parseDoubleArray(rot_str);
        if (rot.size() >= 9) {
            R << rot[0], rot[1], rot[2],
                 rot[3], rot[4], rot[5],
                 rot[6], rot[7], rot[8];
        }
    }
    
    // Parse expression coefficients
    std::string expr_str = findValue("expression");
    if (!expr_str.empty()) {
        auto expr = parseDoubleArray(expr_str);
        expression.resize(expr.size());
        for (size_t i = 0; i < expr.size(); ++i) {
            expression(i) = expr[i];
        }
    }
    
    // Week 6: Parse identity coefficients (alpha) for Stage 2
    if (identity_out) {
        std::string id_str = findValue("identity");
        if (!id_str.empty()) {
            auto id_arr = parseDoubleArray(id_str);
            identity_out->resize(id_arr.size());
            for (size_t i = 0; i < id_arr.size(); ++i) {
                (*identity_out)(i) = id_arr[i];
            }
        }
    }
    
    return true;
}

/**
 * Week 5/6: Save tracking state to JSON (with optional identity/alpha for Stage 1 handoff)
 */
bool saveTrackingStateJSON(const std::string& filepath,
                           const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& t,
                           double scale,
                           const Eigen::VectorXd& expression,
                           double last_rmse_mm = 0.0,
                           const Eigen::VectorXd* identity = nullptr) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    file << std::fixed << std::setprecision(8);
    file << "{\n";
    
    // Rotation matrix
    file << "  \"rotation\": [\n";
    for (int i = 0; i < 3; ++i) {
        file << "    [" << R(i, 0) << ", " << R(i, 1) << ", " << R(i, 2) << "]";
        if (i < 2) file << ",";
        file << "\n";
    }
    file << "  ],\n";
    
    // Translation
    file << "  \"translation\": [" << t(0) << ", " << t(1) << ", " << t(2) << "],\n";
    
    // Scale
    file << "  \"scale\": " << scale << ",\n";
    
    // Expression coefficients
    file << "  \"expression\": [";
    for (int i = 0; i < expression.size(); ++i) {
        if (i > 0) file << ", ";
        file << expression(i);
    }
    file << "],\n";
    
    // Week 6: Identity (alpha) for Stage 1 -> Stage 2 handoff
    file << "  \"identity\": [";
    if (identity && identity->size() > 0) {
        for (int i = 0; i < identity->size(); ++i) {
            if (i > 0) file << ", ";
            file << (*identity)(i);
        }
    }
    file << "],\n";
    
    // Metadata
    // last_rmse_mm: kept for backward compat; value is actually optimization final_energy (not geometric RMSE in mm).
    file << "  \"frame_idx\": 0,\n";
    file << "  \"reinit_count\": 0,\n";
    file << "  \"last_rmse_mm\": " << last_rmse_mm << ",\n";
    file << "  \"final_energy\": " << last_rmse_mm << "\n";
    
    file << "}\n";
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
              << "  --mapping <path>          Path to landmark mapping file\n"
              << "  --output-mesh <path>      Output mesh file path (PLY or OBJ)\n"
              << "  --output-pointcloud <path> Output point cloud file path (PLY)\n"
              << "  --optimize                Enable Gauss-Newton optimization\n"
              << "  --no-optimize             Output mean shape only (default)\n"
              << "  --max-iter <n>            Max optimization iterations (default: 50)\n"
              << "  --lambda-landmark <w>     Landmark weight (default: 1.0)\n"
              << "  --lambda-depth <w>        Depth weight (default: 0.1)\n"
              << "  --lambda-reg <w>          Regularization weight (default: 1.0)\n"
              << "  --lambda-alpha <w>        Identity regularization (Week 6, overrides lambda-reg)\n"
              << "  --lambda-delta <w>        Expression regularization (Week 6, overrides lambda-reg)\n"
              << "  --lambda-translation-prior <w>  Translation prior weight in tracking (default: 0.5, 0=off)\n"
              << "  --verbose                 Print detailed optimization output\n"
              << "  --help                    Show this help message\n"
              << "\n"
              << "Week 5 Tracking Options:\n"
              << "  --init-pose-json <path>   Load initial pose/expression from JSON\n"
              << "  --output-state-json <path> Save final pose/expression to JSON\n"
              << "\n"
              << "Week 6 Evaluation Protocol (3-stage):\n"
              << "  --stage <id|expr|full>    id=identity only, expr=expression only, full=tracking (default: full)\n"
              << "  --init-identity-json <path> Load identity (alpha) and pose from Stage 1 (for Stage 2)\n"
              << "  --output-convergence-json <path> Save energy_history, step_norms, iterations\n"
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
    std::string init_pose_json, output_state_json;  // Week 5: Tracking support
    std::string init_identity_json, output_convergence_json;  // Week 6: Stage 2 + convergence
    std::string stage_mode = "full";  // Week 6: "id" | "expr" | "full" (default full = backward compat)
    double depth_scale = 1000.0;
    bool optimize = false;
    bool verbose = false;
    int max_iterations = 50;
    double lambda_landmark = 1.0;
    double lambda_depth = 0.1;
    double lambda_reg = 1.0;
    double lambda_alpha_sep = 0.0;  // Week 6: if > 0 use for identity; else lambda_reg
    double lambda_delta_sep = 0.0;  // Week 6: if > 0 use for expression; else lambda_reg
    double lambda_translation_prior = 0.5;  // Translation prior in tracking (0 = disabled)
    
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
        } else if (arg == "--init-pose-json" && i + 1 < argc) {
            init_pose_json = argv[++i];  // Week 5
        } else if (arg == "--output-state-json" && i + 1 < argc) {
            output_state_json = argv[++i];  // Week 5
        } else if (arg == "--stage" && i + 1 < argc) {
            stage_mode = argv[++i];  // Week 6: id | expr | full
        } else if (arg == "--init-identity-json" && i + 1 < argc) {
            init_identity_json = argv[++i];  // Week 6: Stage 2
        } else if (arg == "--output-convergence-json" && i + 1 < argc) {
            output_convergence_json = argv[++i];  // Week 6
        } else if (arg == "--optimize") {
            optimize = true;
        } else if (arg == "--no-optimize") {
            optimize = false;
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
        } else if (arg == "--lambda-alpha" && i + 1 < argc) {
            lambda_alpha_sep = std::stod(argv[++i]);
        } else if (arg == "--lambda-delta" && i + 1 < argc) {
            lambda_delta_sep = std::stod(argv[++i]);
        } else if (arg == "--lambda-translation-prior" && i + 1 < argc) {
            lambda_translation_prior = std::stod(argv[++i]);
        }
    }
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
    params.lambda_alpha = (lambda_alpha_sep > 0) ? lambda_alpha_sep : lambda_reg;
    params.lambda_delta = (lambda_delta_sep > 0) ? lambda_delta_sep : lambda_reg;
    
    // Week 6: Set which parameters to optimize from --stage (id | expr | full)
    // id = identity only (delta=0, pose fixed); expr = expression only (alpha fixed); full = tracking (expression + pose warm-start)
    if (optimize && stage_mode == "id") {
        params.optimize_identity = true;
        params.optimize_expression = false;
        params.optimize_rotation = false;
        params.optimize_translation = false;
        params.delta.setZero();
    } else if (optimize && stage_mode == "expr") {
        params.optimize_identity = false;
        params.optimize_expression = true;
        params.optimize_rotation = false;
        params.optimize_translation = false;
    } else {
        // Week 4/5: full or legacy — optimize expression and pose so tracking can change over time
        params.optimize_expression = optimize;
        params.optimize_identity = false;
        params.optimize_rotation = optimize;
        params.optimize_translation = optimize;
    }
    
    // Scale factor from Procrustes (BFM mm -> camera meters)
    double pose_scale = 1.0;
    
    Eigen::MatrixXd final_vertices;
    
    if (optimize && landmarks.size() > 0 && mapping.size() > 0) {
        // =========================================
        // Week 4/5: Gauss-Newton Optimization with Tracking Support
        // =========================================
        std::cout << "\n[8] Running Gauss-Newton optimization..." << std::endl;
        
        bool loaded_from_json = false;
        
        // Week 6: Stage 2 or Stage 3 — load identity (alpha) from Stage 1 when init_identity_json is set
        if (!init_identity_json.empty()) {
            std::cout << "    Loading identity from Stage 1: " << init_identity_json << std::endl;
            Eigen::VectorXd init_expression;
            if (loadTrackingStateJSON(init_identity_json, params.R, params.t, params.scale, init_expression, &params.alpha)) {
                pose_scale = params.scale;
                loaded_from_json = true;
                std::cout << "    Loaded identity (alpha size=" << params.alpha.size() << ")" << std::endl;
            }
        }
        // Week 6 Stage 3: If both init_identity_json and init_pose_json set, overwrite pose with previous frame (keep identity from above)
        if (!init_pose_json.empty()) {
            std::cout << "    Loading pose (and optionally expression) from: " << init_pose_json << std::endl;
            Eigen::VectorXd prev_expression;
            Eigen::Matrix3d R_pose;
            Eigen::Vector3d t_pose;
            double scale_pose;
            if (loadTrackingStateJSON(init_pose_json, R_pose, t_pose, scale_pose, prev_expression, nullptr)) {
                params.R = R_pose;
                params.t = t_pose;
                params.scale = scale_pose;
                pose_scale = scale_pose;
                loaded_from_json = true;
                params.lambda_translation_prior = lambda_translation_prior;  // Reduce drift in tracking
                params.max_translation_delta_m = 0.3;  // Hard-bound translation change per frame
                std::cout << "    Loaded pose: scale=" << pose_scale << ", t=[" << params.t.transpose() << "]" << std::endl;
            }
        }
        
        // Initialize pose from Procrustes if not loaded from JSON
        if (!loaded_from_json && points_3d.size() > 0) {
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
        
        // Week 5/6: Save final state to JSON for tracking (Stage 1 "id" saves identity for Stage 2)
        if (!output_state_json.empty()) {
            std::cout << "    Saving final state to: " << output_state_json << std::endl;
            double final_energy = result.final_energy;
            const Eigen::VectorXd* identity_to_save = (stage_mode == "id" && result.final_params.alpha.size() > 0)
                ? &result.final_params.alpha : nullptr;
            if (saveTrackingStateJSON(output_state_json,
                                      result.final_params.R,
                                      result.final_params.t,
                                      result.final_params.scale,
                                      result.final_params.delta,
                                      final_energy,
                                      identity_to_save)) {
                std::cout << "    State saved successfully" << std::endl;
            }
        }
        // Week 6: Save convergence data for evaluation (energy_history, step_norms, iterations, damping)
        if (!output_convergence_json.empty()) {
            std::ofstream conv_file(output_convergence_json);
            if (conv_file.is_open()) {
                conv_file << std::fixed << std::setprecision(8);
                conv_file << "{\n  \"iterations\": " << result.iterations
                          << ",\n  \"converged\": " << (result.converged ? "true" : "false")
                          << ",\n  \"initial_energy\": " << result.initial_energy
                          << ",\n  \"final_energy\": " << result.final_energy
                          << ",\n  \"final_step_norm\": " << result.final_step_norm
                          << ",\n  \"damping_used\": " << result.damping_used
                          << ",\n  \"energy_history\": [";
                for (size_t k = 0; k < result.energy_history.size(); ++k) {
                    if (k > 0) conv_file << ", ";
                    conv_file << result.energy_history[k];
                }
                conv_file << "],\n  \"step_norms\": [";
                for (size_t k = 0; k < result.step_norms.size(); ++k) {
                    if (k > 0) conv_file << ", ";
                    conv_file << result.step_norms[k];
                }
                conv_file << "]\n}\n";
                conv_file.close();
                std::cout << "    Convergence data saved to: " << output_convergence_json << std::endl;
            }
        }
        
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
