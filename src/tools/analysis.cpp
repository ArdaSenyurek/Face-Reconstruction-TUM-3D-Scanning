/**
 * Analysis Tool
 * 
 * Computes 3D metrics and visualizations:
 * - Cloud-to-mesh RMSE
 * - Depth statistics and visualization
 * 
 * This replaces Python analysis computation logic.
 * 
 * Usage:
 *   build/bin/analysis --pointcloud <ply> --mesh <ply> [--depth <png>] [--output-vis <png>] [--output-json <json>]
 */

#include "data/RGBDFrame.h"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>

#ifdef OPENCV_NOT_FOUND
#error "OpenCV is required for analysis"
#endif

using namespace face_reconstruction;
using Eigen::Vector3d;

/**
 * Load vertices from PLY file
 */
bool loadPLYVertices(const std::string& filepath, std::vector<Vector3d>& vertices) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    int num_vertices = 0;
    bool in_header = true;
    int vertex_count = 0;
    
    while (std::getline(file, line)) {
        if (in_header) {
            if (line.find("element vertex") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_vertices;
            } else if (line.find("end_header") != std::string::npos) {
                in_header = false;
                vertices.reserve(num_vertices);
            }
        } else {
            if (vertex_count < num_vertices) {
                std::istringstream iss(line);
                double x, y, z;
                if (iss >> x >> y >> z) {
                    vertices.emplace_back(Vector3d(x, y, z));
                    vertex_count++;
                }
            }
        }
    }
    
    return !vertices.empty();
}

/**
 * Compute RMSE between point cloud and mesh
 */
double computeCloudToMeshRMSE(const std::vector<Vector3d>& cloud, 
                               const std::vector<Vector3d>& mesh,
                               int max_samples = 20000) {
    if (cloud.empty() || mesh.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Sample cloud if too large
    std::vector<Vector3d> sampled_cloud = cloud;
    if (static_cast<int>(cloud.size()) > max_samples) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, cloud.size() - 1);
        sampled_cloud.clear();
        sampled_cloud.reserve(max_samples);
        for (int i = 0; i < max_samples; ++i) {
            sampled_cloud.push_back(cloud[dis(gen)]);
        }
    }
    
    // Find nearest mesh vertex for each cloud point
    double sum_sq_error = 0.0;
    int valid_count = 0;
    
    for (const auto& point : sampled_cloud) {
        double min_dist_sq = std::numeric_limits<double>::max();
        for (const auto& vertex : mesh) {
            double dist_sq = (point - vertex).squaredNorm();
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        sum_sq_error += min_dist_sq;
        valid_count++;
    }
    
    if (valid_count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    return std::sqrt(sum_sq_error / valid_count);
}

/**
 * Compute depth statistics and create visualization
 */
bool processDepth(const std::string& depth_path, const std::string& output_vis,
                  double& min_depth, double& max_depth, double& mean_depth, double& std_depth) {
    RGBDFrame frame;
    if (!frame.loadDepth(depth_path, 1000.0)) {
        return false;
    }
    
    const cv::Mat& depth = frame.getDepth();
    cv::Mat mask = depth > 0.0f;
    
    if (cv::countNonZero(mask) == 0) {
        return false;
    }
    
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(depth, mean_val, std_val, mask);
    
    cv::minMaxLoc(depth, &min_depth, &max_depth, nullptr, nullptr, mask);
    mean_depth = mean_val[0];
    std_depth = std_val[0];
    
    // Create visualization
    if (!output_vis.empty()) {
        cv::Mat normalized;
        cv::normalize(depth, normalized, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
        cv::Mat colored;
        cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
        cv::imwrite(output_vis, colored);
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    std::string pointcloud_path;
    std::string mesh_path;
    std::string depth_path;
    std::string output_vis;
    std::string output_json;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pointcloud" && i + 1 < argc) {
            pointcloud_path = argv[++i];
        } else if (arg == "--mesh" && i + 1 < argc) {
            mesh_path = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depth_path = argv[++i];
        } else if (arg == "--output-vis" && i + 1 < argc) {
            output_vis = argv[++i];
        } else if (arg == "--output-json" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "--help") {
            std::cerr << "Usage: " << argv[0] 
                      << " --pointcloud <ply> --mesh <ply> [--depth <png>] [--output-vis <png>] [--output-json <json>]" << std::endl;
            return 1;
        }
    }
    
    std::map<std::string, double> metrics;
    
    // Compute cloud-to-mesh RMSE
    if (!pointcloud_path.empty() && !mesh_path.empty()) {
        std::vector<Vector3d> cloud, mesh;
        if (loadPLYVertices(pointcloud_path, cloud) && loadPLYVertices(mesh_path, mesh)) {
            double rmse = computeCloudToMeshRMSE(cloud, mesh);
            if (!std::isnan(rmse)) {
                metrics["rmse_cloud_mesh_m"] = rmse;
                metrics["cloud_points"] = static_cast<double>(cloud.size());
            }
        }
    }
    
    // Process depth
    if (!depth_path.empty()) {
        double min_d, max_d, mean_d, std_d;
        if (processDepth(depth_path, output_vis, min_d, max_d, mean_d, std_d)) {
            metrics["depth_min"] = min_d;
            metrics["depth_max"] = max_d;
            metrics["depth_mean"] = mean_d;
            metrics["depth_std"] = std_d;
        }
    }
    
    // Output JSON
    if (!output_json.empty() && !metrics.empty()) {
        std::ofstream json_file(output_json);
        json_file << "{\n";
        bool first = true;
        for (const auto& pair : metrics) {
            if (!first) json_file << ",\n";
            json_file << "  \"" << pair.first << "\": " << std::fixed << std::setprecision(6) << pair.second;
            first = false;
        }
        json_file << "\n}\n";
    }
    
    // Output to stdout for Python to parse
    for (const auto& pair : metrics) {
        std::cout << pair.first << "=" << std::fixed << std::setprecision(6) << pair.second << std::endl;
    }
    
    return 0;
}

