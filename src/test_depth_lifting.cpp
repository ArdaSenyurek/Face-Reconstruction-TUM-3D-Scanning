/**
 * STEP A: Depth Lifting Test
 * 
 * Goal: Verify that depth maps are correctly backprojected to 3D point clouds
 * on Biwi frames. This is the FIRST PRIORITY milestone.
 * 
 * Usage:
 *   bin/test_depth_lifting <rgb_path> <depth_path> <intrinsics_path> [output_ply]
 * 
 * Example:
 *   bin/test_depth_lifting data/biwi_person01/rgb/frame_00000.png \
 *                          data/biwi_person01/depth/frame_00000.png \
 *                          data/biwi_person01/intrinsics.txt \
 *                          output_pointcloud.ply
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>

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

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <rgb_path> <depth_path> <intrinsics_path> [output_ply]" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " data/biwi_person01/rgb/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/depth/frame_00000.png \\" << std::endl;
        std::cerr << "     data/biwi_person01/intrinsics.txt \\" << std::endl;
        std::cerr << "     output_pointcloud.ply" << std::endl;
        return 1;
    }
    
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    std::string intrinsics_path = argv[3];
    std::string output_ply = (argc > 4) ? argv[4] : "output_pointcloud.ply";
    
    std::cout << "=== Depth Lifting Test ===" << std::endl;
    std::cout << "RGB: " << rgb_path << std::endl;
    std::cout << "Depth: " << depth_path << std::endl;
    std::cout << "Intrinsics: " << intrinsics_path << std::endl;
    std::cout << "Output PLY: " << output_ply << std::endl;
    std::cout << std::endl;
    
    // Load RGB-D frame
    RGBDFrame frame;
    if (!frame.loadRGB(rgb_path)) {
        std::cerr << "Failed to load RGB image" << std::endl;
        return 1;
    }
    
    // Load depth (Biwi depth is typically in mm, so scale_factor = 1000.0 converts to meters)
    // However, if depth is already in meters, use scale_factor = 1.0
    // Try with 1000.0 first (mm -> m), but user can adjust if needed
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
    
    // Print frame statistics
    std::cout << "--- Frame Statistics ---" << std::endl;
    frame.printStats();
    std::cout << std::endl;
    
    // Print intrinsics
    std::cout << "--- Camera Intrinsics ---" << std::endl;
    std::cout << "fx: " << intrinsics.fx << ", fy: " << intrinsics.fy << std::endl;
    std::cout << "cx: " << intrinsics.cx << ", cy: " << intrinsics.cy << std::endl;
    std::cout << std::endl;
    
    // Get depth map
    const cv::Mat& depth_map = frame.getDepth();
    
    // Calculate depth statistics
    double min_depth = std::numeric_limits<double>::max();
    double max_depth = std::numeric_limits<double>::lowest();
    int valid_pixels = 0;
    int total_pixels = depth_map.rows * depth_map.cols;
    
    for (int v = 0; v < depth_map.rows; ++v) {
        for (int u = 0; u < depth_map.cols; ++u) {
            float depth = depth_map.at<float>(v, u);
            if (frame.isValidDepth(depth)) {
                valid_pixels++;
                if (depth < min_depth) min_depth = depth;
                if (depth > max_depth) max_depth = depth;
            }
        }
    }
    
    std::cout << "--- Depth Statistics ---" << std::endl;
    std::cout << "Image size: " << depth_map.cols << " x " << depth_map.rows << std::endl;
    std::cout << "Total pixels: " << total_pixels << std::endl;
    std::cout << "Valid depth pixels: " << valid_pixels 
              << " (" << (100.0 * valid_pixels / total_pixels) << "%)" << std::endl;
    if (valid_pixels > 0) {
        std::cout << "Min depth: " << min_depth << " meters" << std::endl;
        std::cout << "Max depth: " << max_depth << " meters" << std::endl;
    } else {
        std::cerr << "ERROR: No valid depth pixels found!" << std::endl;
        std::cerr << "Check depth image format and scale factor." << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    // Backproject depth to point cloud
    std::vector<Eigen::Vector3d> points;
    std::vector<std::pair<int, int>> pixel_indices;
    
    std::cout << "--- Backprojecting Depth ---" << std::endl;
    backprojectDepth(depth_map, intrinsics, points, pixel_indices, 0.1f, 10.0f);
    
    std::cout << "Generated " << points.size() << " 3D points" << std::endl;
    
    if (points.empty()) {
        std::cerr << "ERROR: No points generated from depth backprojection!" << std::endl;
        return 1;
    }
    
    // Calculate point cloud statistics
    Eigen::Vector3d min_point(std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::max());
    Eigen::Vector3d max_point(std::numeric_limits<double>::lowest(),
                              std::numeric_limits<double>::lowest(),
                              std::numeric_limits<double>::lowest());
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    
    for (const auto& point : points) {
        for (int i = 0; i < 3; ++i) {
            if (point(i) < min_point(i)) min_point(i) = point(i);
            if (point(i) > max_point(i)) max_point(i) = point(i);
        }
        centroid += point;
    }
    centroid /= static_cast<double>(points.size());
    
    std::cout << "--- Point Cloud Statistics ---" << std::endl;
    std::cout << "Number of points: " << points.size() << std::endl;
    std::cout << "Bounding box:" << std::endl;
    std::cout << "  X: [" << min_point.x() << ", " << max_point.x() << "]" << std::endl;
    std::cout << "  Y: [" << min_point.y() << ", " << max_point.y() << "]" << std::endl;
    std::cout << "  Z: [" << min_point.z() << ", " << max_point.z() << "]" << std::endl;
    std::cout << "Centroid: (" << centroid.x() << ", " << centroid.y() << ", " << centroid.z() << ")" << std::endl;
    std::cout << std::endl;
    
    // Save point cloud to PLY
    std::cout << "--- Saving Point Cloud ---" << std::endl;
    if (savePointCloudPLY(points, output_ply)) {
        std::cout << "Successfully saved point cloud to: " << output_ply << std::endl;
    } else {
        std::cerr << "Failed to save point cloud" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
    std::cout << "✓ Depth lifting verified successfully!" << std::endl;
    std::cout << "  Point cloud saved to: " << output_ply << std::endl;
    std::cout << "  You can visualize it with: meshlab " << output_ply << std::endl;
    
    return 0;
}

