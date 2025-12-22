/**
 * STEP 3: Minimal Depth Renderer Test
 * 
 * Goal: Implement a minimal depth renderer to compare rendered depth with observed depth.
 * 
 * Usage:
 *   bin/test_depth_renderer <rgb_path> <depth_path> <intrinsics_path> <aligned_mesh_ply> [output_rendered_depth]
 * 
 * Example:
 *   bin/test_depth_renderer data/biwi_person01/rgb/frame_00000.png \
 *                           data/biwi_person01/depth/frame_00000.png \
 *                           data/biwi_person01/intrinsics.txt \
 *                           build/aligned_mesh_step2.ply \
 *                           build/rendered_depth.png
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "rendering/DepthRenderer.h"
#include "model/MorphableModel.h"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace face_reconstruction;

/**
 * Load vertices and faces from PLY file
 * Simple PLY parser for aligned mesh
 */
bool loadPLY(const std::string& filepath, Eigen::MatrixXd& vertices, Eigen::MatrixXi& faces) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    int num_vertices = 0;
    int num_faces = 0;
    bool in_header = true;
    int vertex_count = 0;
    int face_count = 0;
    
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> face_list;
    
    while (std::getline(file, line)) {
        if (in_header) {
            if (line.find("element vertex") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_vertices;
            } else if (line.find("element face") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_faces;
            } else if (line.find("end_header") != std::string::npos) {
                in_header = false;
                verts.reserve(num_vertices);
                face_list.reserve(num_faces);
            }
        } else {
            if (vertex_count < num_vertices) {
                // Read vertex
                std::istringstream iss(line);
                double x, y, z;
                if (iss >> x >> y >> z) {
                    verts.emplace_back(x, y, z);
                    vertex_count++;
                }
            } else if (face_count < num_faces) {
                // Read face
                std::istringstream iss(line);
                int n, v0, v1, v2;
                if (iss >> n >> v0 >> v1 >> v2 && n == 3) {
                    face_list.emplace_back(v0, v1, v2);
                    face_count++;
                }
            }
        }
    }
    
    file.close();
    
    if (verts.empty()) {
        std::cerr << "No vertices loaded from PLY file" << std::endl;
        return false;
    }
    
    // Convert to Eigen matrices
    vertices.resize(verts.size(), 3);
    for (size_t i = 0; i < verts.size(); ++i) {
        vertices.row(i) = verts[i];
    }
    
    if (!face_list.empty()) {
        faces.resize(face_list.size(), 3);
        for (size_t i = 0; i < face_list.size(); ++i) {
            faces.row(i) = face_list[i];
        }
    } else {
        faces.resize(0, 3);
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <rgb_path> <depth_path> <intrinsics_path> <aligned_mesh_ply> [output_rendered_depth]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/biwi_person01/rgb/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/depth/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/intrinsics.txt \\" << std::endl;
        std::cerr << "     build/aligned_mesh_step2.ply \\" << std::endl;
        std::cerr << "     build/rendered_depth.png" << std::endl;
        return 1;
    }
    
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    std::string intrinsics_path = argv[3];
    std::string aligned_mesh_ply = argv[4];
    std::string output_rendered = (argc > 5) ? argv[5] : "rendered_depth.png";
    
    std::cout << "=== STEP 3: Minimal Depth Renderer Test ===" << std::endl;
    std::cout << "RGB: " << rgb_path << std::endl;
    std::cout << "Observed depth: " << depth_path << std::endl;
    std::cout << "Intrinsics: " << intrinsics_path << std::endl;
    std::cout << "Aligned mesh: " << aligned_mesh_ply << std::endl;
    std::cout << "Output rendered depth: " << output_rendered << std::endl;
    std::cout << std::endl;
    
    // Load RGB-D frame to get dimensions
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
    
    std::cout << "Image dimensions: " << frame.width() << " x " << frame.height() << std::endl;
    std::cout << std::endl;
    
    // Load aligned mesh
    std::cout << "--- Loading Aligned Mesh ---" << std::endl;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    
    if (!loadPLY(aligned_mesh_ply, vertices, faces)) {
        std::cerr << "Failed to load aligned mesh from PLY" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded mesh: " << vertices.rows() << " vertices, " 
              << faces.rows() << " faces" << std::endl;
    std::cout << std::endl;
    
    // Initialize depth renderer
    std::cout << "--- Initializing Depth Renderer ---" << std::endl;
    DepthRenderer renderer;
    renderer.initialize(intrinsics, frame.width(), frame.height());
    std::cout << "Renderer initialized: " << renderer.getWidth() << " x " 
              << renderer.getHeight() << std::endl;
    std::cout << std::endl;
    
    // Render depth
    std::cout << "--- Rendering Depth Map ---" << std::endl;
    cv::Mat rendered_depth;
    
    if (faces.rows() > 0) {
        std::cout << "Rendering with triangle rasterization..." << std::endl;
        rendered_depth = renderer.renderDepth(vertices, faces);
    } else {
        std::cout << "No faces found, rendering as point cloud..." << std::endl;
        rendered_depth = renderer.renderDepthPoints(vertices);
    }
    
    std::cout << "Rendering complete" << std::endl;
    std::cout << std::endl;
    
    // Calculate statistics
    std::cout << "--- Rendered Depth Statistics ---" << std::endl;
    const cv::Mat& observed_depth = frame.getDepth();
    
    int rendered_valid = 0;
    int observed_valid = 0;
    double rendered_min = std::numeric_limits<double>::max();
    double rendered_max = std::numeric_limits<double>::lowest();
    double observed_min = std::numeric_limits<double>::max();
    double observed_max = std::numeric_limits<double>::lowest();
    
    for (int v = 0; v < rendered_depth.rows; ++v) {
        for (int u = 0; u < rendered_depth.cols; ++u) {
            float r_depth = rendered_depth.at<float>(v, u);
            float o_depth = observed_depth.at<float>(v, u);
            
            if (!std::isnan(r_depth) && r_depth > 0.0) {
                rendered_valid++;
                if (r_depth < rendered_min) rendered_min = r_depth;
                if (r_depth > rendered_max) rendered_max = r_depth;
            }
            
            if (frame.isValidDepth(o_depth)) {
                observed_valid++;
                if (o_depth < observed_min) observed_min = o_depth;
                if (o_depth > observed_max) observed_max = o_depth;
            }
        }
    }
    
    std::cout << "Rendered depth:" << std::endl;
    std::cout << "  Valid pixels: " << rendered_valid << " / " 
              << (rendered_depth.rows * rendered_depth.cols) << std::endl;
    if (rendered_valid > 0) {
        std::cout << "  Range: [" << rendered_min << ", " << rendered_max << "] meters" << std::endl;
    }
    
    std::cout << "Observed depth:" << std::endl;
    std::cout << "  Valid pixels: " << observed_valid << " / " 
              << (observed_depth.rows * observed_depth.cols) << std::endl;
    if (observed_valid > 0) {
        std::cout << "  Range: [" << observed_min << ", " << observed_max << "] meters" << std::endl;
    }
    std::cout << std::endl;
    
    // Save rendered depth as 16-bit PNG (for visualization)
    std::cout << "--- Saving Rendered Depth ---" << std::endl;
    
    // Convert to 16-bit for saving
    cv::Mat rendered_16bit;
    rendered_depth.convertTo(rendered_16bit, CV_16U, 1000.0);  // meters to mm
    
    // Set NaN to 0
    cv::Mat mask = cv::Mat::zeros(rendered_depth.size(), CV_8U);
    for (int v = 0; v < rendered_depth.rows; ++v) {
        for (int u = 0; u < rendered_depth.cols; ++u) {
            if (std::isnan(rendered_depth.at<float>(v, u))) {
                rendered_16bit.at<uint16_t>(v, u) = 0;
            }
        }
    }
    
    if (cv::imwrite(output_rendered, rendered_16bit)) {
        std::cout << "✓ Successfully saved rendered depth to: " << output_rendered << std::endl;
    } else {
        std::cerr << "Failed to save rendered depth" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== STEP 3 Summary ===" << std::endl;
    std::cout << "✓ Depth renderer initialized and tested" << std::endl;
    std::cout << "✓ Rendered depth map created: " << output_rendered << std::endl;
    std::cout << "✓ Rendered depth range: [" << rendered_min << ", " << rendered_max << "] meters" << std::endl;
    std::cout << "✓ Observed depth range: [" << observed_min << ", " << observed_max << "] meters" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== STEP 3 Complete ===" << std::endl;
    std::cout << "✓ Minimal depth renderer implemented successfully!" << std::endl;
    std::cout << "  Next: Compute dense depth residuals (STEP 4)" << std::endl;
    std::cout << "  Compare rendered vs observed depth visually" << std::endl;
    
    return 0;
}

