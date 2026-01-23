/**
 * Overlay Generation Tool (Week 4)
 * 
 * Creates mesh-scan overlays after rigid alignment:
 * 1. 3D PLY overlay (cyan scan points + red mesh)
 * 2. 2D PNG overlay (RGB with projected mesh/scan)
 * 3. Depth comparison visualization
 * 4. Standalone scan and mesh PLY files
 * 5. Quantitative sanity check metrics
 * 
 * Usage:
 *   build/bin/create_overlays --mesh-rigid <rigid_aligned.ply> --depth <depth.png> \
 *                             --intrinsics <intrinsics.txt> --out-dir <output_dir> \
 *                             [--mesh-opt <optimized.ply>] [--rgb <rgb.png>] \
 *                             [--output-metrics <metrics.json>]
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include "rendering/DepthRenderer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <cmath>
#include <sys/stat.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace face_reconstruction;

// ============================================================================
// Overlay Metrics Structure
// ============================================================================
struct OverlayMetrics {
    double scale_ratio;           // mesh_bbox_diag / scan_bbox_diag
    double axis_dot_product;      // PCA main axis alignment
    double bbox_overlap_x;        // X-axis bbox overlap ratio
    double bbox_overlap_y;        // Y-axis bbox overlap ratio
    double bbox_overlap_z;        // Z-axis bbox overlap ratio
    double cloud_z_min;
    double cloud_z_max;
    double mesh_z_min;
    double mesh_z_max;
    int num_scan_points;
    int num_mesh_vertices;
    double nn_rmse_m;             // Nearest neighbor RMSE (sampled)
};

// ============================================================================
// Bounding Box Helper
// ============================================================================
struct BoundingBox {
    Eigen::Vector3d min_pt;
    Eigen::Vector3d max_pt;
    
    BoundingBox() : min_pt(Eigen::Vector3d::Constant(std::numeric_limits<double>::max())),
                    max_pt(Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest())) {}
    
    void extend(const Eigen::Vector3d& pt) {
        min_pt = min_pt.cwiseMin(pt);
        max_pt = max_pt.cwiseMax(pt);
    }
    
    double diagonal() const {
        return (max_pt - min_pt).norm();
    }
    
    Eigen::Vector3d center() const {
        return (min_pt + max_pt) * 0.5;
    }
    
    Eigen::Vector3d size() const {
        return max_pt - min_pt;
    }
};

// Compute overlap ratio for a single axis
double computeAxisOverlap(double min1, double max1, double min2, double max2) {
    double overlap_min = std::max(min1, min2);
    double overlap_max = std::min(max1, max2);
    if (overlap_max <= overlap_min) return 0.0;
    double overlap = overlap_max - overlap_min;
    double union_size = std::max(max1, max2) - std::min(min1, min2);
    return (union_size > 0) ? overlap / union_size : 0.0;
}

// ============================================================================
// PCA Main Axis Computation
// ============================================================================
Eigen::Vector3d computePCAMainAxis(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) {
        return Eigen::Vector3d(0, 0, 1);
    }
    
    // Compute centroid
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= static_cast<double>(points.size());
    
    // Build covariance matrix
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        Eigen::Vector3d d = p - centroid;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(points.size());
    
    // Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    // Return eigenvector with largest eigenvalue (last one)
    return solver.eigenvectors().col(2).normalized();
}

// ============================================================================
// Nearest Neighbor RMSE (sampled for speed)
// ============================================================================
double computeNNRMSE(const std::vector<Eigen::Vector3d>& scan_points,
                     const std::vector<Eigen::Vector3d>& mesh_vertices,
                     int max_samples = 5000) {
    if (scan_points.empty() || mesh_vertices.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Subsample scan points if needed
    std::vector<size_t> indices(scan_points.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    size_t num_samples = std::min(static_cast<size_t>(max_samples), scan_points.size());
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
        const Eigen::Vector3d& scan_pt = scan_points[indices[i]];
        
        // Find nearest neighbor in mesh
        double min_dist_sq = std::numeric_limits<double>::max();
        for (const auto& mesh_pt : mesh_vertices) {
            double dist_sq = (scan_pt - mesh_pt).squaredNorm();
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        sum_sq += min_dist_sq;
    }
    
    return std::sqrt(sum_sq / static_cast<double>(num_samples));
}

// ============================================================================
// Compute Overlay Metrics
// ============================================================================
OverlayMetrics computeOverlayMetrics(const std::vector<Eigen::Vector3d>& scan_points,
                                      const std::vector<Eigen::Vector3d>& mesh_vertices) {
    OverlayMetrics metrics;
    metrics.num_scan_points = static_cast<int>(scan_points.size());
    metrics.num_mesh_vertices = static_cast<int>(mesh_vertices.size());
    
    // Compute bounding boxes
    BoundingBox scan_bbox, mesh_bbox;
    for (const auto& p : scan_points) {
        scan_bbox.extend(p);
    }
    for (const auto& v : mesh_vertices) {
        mesh_bbox.extend(v);
    }
    
    // Z ranges
    metrics.cloud_z_min = scan_bbox.min_pt.z();
    metrics.cloud_z_max = scan_bbox.max_pt.z();
    metrics.mesh_z_min = mesh_bbox.min_pt.z();
    metrics.mesh_z_max = mesh_bbox.max_pt.z();
    
    // Scale ratio (bbox diagonal)
    double scan_diag = scan_bbox.diagonal();
    double mesh_diag = mesh_bbox.diagonal();
    metrics.scale_ratio = (scan_diag > 0) ? mesh_diag / scan_diag : 0.0;
    
    // Bbox overlap ratios per axis
    metrics.bbox_overlap_x = computeAxisOverlap(scan_bbox.min_pt.x(), scan_bbox.max_pt.x(),
                                                 mesh_bbox.min_pt.x(), mesh_bbox.max_pt.x());
    metrics.bbox_overlap_y = computeAxisOverlap(scan_bbox.min_pt.y(), scan_bbox.max_pt.y(),
                                                 mesh_bbox.min_pt.y(), mesh_bbox.max_pt.y());
    metrics.bbox_overlap_z = computeAxisOverlap(scan_bbox.min_pt.z(), scan_bbox.max_pt.z(),
                                                 mesh_bbox.min_pt.z(), mesh_bbox.max_pt.z());
    
    // PCA main axis dot product
    Eigen::Vector3d scan_axis = computePCAMainAxis(scan_points);
    Eigen::Vector3d mesh_axis = computePCAMainAxis(mesh_vertices);
    metrics.axis_dot_product = std::abs(scan_axis.dot(mesh_axis));
    
    // NN-RMSE (sampled)
    metrics.nn_rmse_m = computeNNRMSE(scan_points, mesh_vertices);
    
    return metrics;
}

// ============================================================================
// Directory creation helper
// ============================================================================
bool createDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    #ifdef _WIN32
    return mkdir(path.c_str()) == 0;
    #else
    return mkdir(path.c_str(), 0755) == 0;
    #endif
}

// Load PLY vertices
bool loadPLYVertices(const std::string& ply_path, std::vector<Eigen::Vector3d>& vertices) {
    std::ifstream file(ply_path);
    if (!file.is_open()) {
        return false;
    }
    
    vertices.clear();
    std::string line;
    bool in_header = true;
    int num_vertices = 0;
    int vertex_count = 0;
    
    while (std::getline(file, line)) {
        if (in_header) {
            if (line.find("element vertex") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_vertices;
            }
            if (line.find("end_header") != std::string::npos) {
                in_header = false;
            }
        } else {
            if (vertex_count < num_vertices) {
                std::istringstream iss(line);
                double x, y, z;
                if (iss >> x >> y >> z) {
                    vertices.push_back(Eigen::Vector3d(x, y, z));
                    vertex_count++;
                }
            }
        }
    }
    return true;
}

// Load PLY faces
bool loadPLYFaces(const std::string& ply_path, std::vector<std::vector<int>>& faces) {
    std::ifstream file(ply_path);
    if (!file.is_open()) {
        return false;
    }
    
    faces.clear();
    std::string line;
    bool in_header = true;
    int num_faces = 0;
    int face_count = 0;
    int vertex_count = 0;
    int num_vertices = 0;
    
    while (std::getline(file, line)) {
        if (in_header) {
            if (line.find("element vertex") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_vertices;
            }
            if (line.find("element face") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy1, dummy2;
                iss >> dummy1 >> dummy2 >> num_faces;
            }
            if (line.find("end_header") != std::string::npos) {
                in_header = false;
            }
        } else {
            if (vertex_count < num_vertices) {
                vertex_count++;
            } else if (face_count < num_faces) {
                std::istringstream iss(line);
                int n;
                if (iss >> n && n == 3) {
                    int v0, v1, v2;
                    if (iss >> v0 >> v1 >> v2) {
                        faces.push_back({v0, v1, v2});
                        face_count++;
                    }
                }
            }
        }
    }
    return true;
}

// ============================================================================
// Save scan-only PLY (point cloud, no faces)
// ============================================================================
bool saveScanOnlyPLY(const std::string& filepath,
                     const std::vector<Eigen::Vector3d>& scan_points,
                     int r = 0, int g = 255, int b = 255) {  // Cyan default
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << scan_points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    for (const auto& p : scan_points) {
        file << std::fixed << std::setprecision(6)
             << p.x() << " " << p.y() << " " << p.z() << " "
             << r << " " << g << " " << b << "\n";
    }
    
    return true;
}

// ============================================================================
// Save mesh-only PLY (with faces)
// ============================================================================
bool saveMeshOnlyPLY(const std::string& filepath,
                     const std::vector<Eigen::Vector3d>& vertices,
                     const std::vector<std::vector<int>>& faces,
                     int r = 255, int g = 0, int b = 0) {  // Red default
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << faces.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    
    for (const auto& v : vertices) {
        file << std::fixed << std::setprecision(6)
             << v.x() << " " << v.y() << " " << v.z() << " "
             << r << " " << g << " " << b << "\n";
    }
    
    for (const auto& face : faces) {
        if (face.size() == 3) {
            file << "3 " << face[0] << " " << face[1] << " " << face[2] << "\n";
        }
    }
    
    return true;
}

// ============================================================================
// Save combined overlay PLY (scan + mesh)
// ============================================================================
bool saveColoredPLY(const std::string& filepath,
                    const std::vector<Eigen::Vector3d>& scan_points,
                    const std::vector<Eigen::Vector3d>& mesh_vertices,
                    const std::vector<std::vector<int>>& mesh_faces,
                    const Eigen::Vector3d& scan_color = Eigen::Vector3d(0, 1, 1),  // Cyan
                    const Eigen::Vector3d& mesh_color = Eigen::Vector3d(1, 0, 0)) {  // Red
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    int total_vertices = scan_points.size() + mesh_vertices.size();
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << total_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << mesh_faces.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    
    // Write scan points (cyan)
    for (const auto& p : scan_points) {
        file << std::fixed << std::setprecision(6)
             << p.x() << " " << p.y() << " " << p.z() << " "
             << static_cast<int>(scan_color.x() * 255) << " "
             << static_cast<int>(scan_color.y() * 255) << " "
             << static_cast<int>(scan_color.z() * 255) << "\n";
    }
    
    // Write mesh vertices (red)
    int mesh_offset = scan_points.size();
    for (const auto& v : mesh_vertices) {
        file << std::fixed << std::setprecision(6)
             << v.x() << " " << v.y() << " " << v.z() << " "
             << static_cast<int>(mesh_color.x() * 255) << " "
             << static_cast<int>(mesh_color.y() * 255) << " "
             << static_cast<int>(mesh_color.z() * 255) << "\n";
    }
    
    // Write mesh faces (with offset)
    for (const auto& face : mesh_faces) {
        if (face.size() == 3) {
            file << "3 " << (face[0] + mesh_offset) << " "
                 << (face[1] + mesh_offset) << " " << (face[2] + mesh_offset) << "\n";
        }
    }
    
    return true;
}

// ============================================================================
// Write overlay metrics to JSON
// ============================================================================
void writeOverlayMetricsJSON(const std::string& filepath,
                              const std::string& label,
                              const OverlayMetrics& metrics) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not write metrics to " << filepath << std::endl;
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    file << "{\n";
    file << "  \"" << label << "\": {\n";
    file << "    \"scale_ratio\": " << metrics.scale_ratio << ",\n";
    file << "    \"axis_dot_product\": " << metrics.axis_dot_product << ",\n";
    file << "    \"bbox_overlap_x\": " << metrics.bbox_overlap_x << ",\n";
    file << "    \"bbox_overlap_y\": " << metrics.bbox_overlap_y << ",\n";
    file << "    \"bbox_overlap_z\": " << metrics.bbox_overlap_z << ",\n";
    file << "    \"cloud_z_range\": [" << metrics.cloud_z_min << ", " << metrics.cloud_z_max << "],\n";
    file << "    \"mesh_z_range\": [" << metrics.mesh_z_min << ", " << metrics.mesh_z_max << "],\n";
    file << "    \"num_scan_points\": " << metrics.num_scan_points << ",\n";
    file << "    \"num_mesh_vertices\": " << metrics.num_mesh_vertices << ",\n";
    file << "    \"nn_rmse_m\": " << metrics.nn_rmse_m << "\n";
    file << "  }\n";
    file << "}\n";
}

// Project 3D points to 2D and draw on RGB
cv::Mat create2DOverlay(const cv::Mat& rgb,
                        const std::vector<Eigen::Vector3d>& mesh_vertices,
                        const std::vector<Eigen::Vector3d>& scan_points,
                        const CameraIntrinsics& intrinsics) {
    cv::Mat overlay = rgb.clone();
    
    // Project scan points (cyan)
    for (const auto& p : scan_points) {
        if (p.z() <= 0) continue;
        double u = intrinsics.fx * p.x() / p.z() + intrinsics.cx;
        double v = intrinsics.fy * p.y() / p.z() + intrinsics.cy;
        int u_int = static_cast<int>(std::round(u));
        int v_int = static_cast<int>(std::round(v));
        if (u_int >= 0 && u_int < overlay.cols && v_int >= 0 && v_int < overlay.rows) {
            cv::circle(overlay, cv::Point(u_int, v_int), 2, cv::Scalar(255, 255, 0), -1);  // Cyan
        }
    }
    
    // Project mesh vertices (red)
    for (const auto& v : mesh_vertices) {
        if (v.z() <= 0) continue;
        double u = intrinsics.fx * v.x() / v.z() + intrinsics.cx;
        double v_coord = intrinsics.fy * v.y() / v.z() + intrinsics.cy;
        int u_int = static_cast<int>(std::round(u));
        int v_int = static_cast<int>(std::round(v_coord));
        if (u_int >= 0 && u_int < overlay.cols && v_int >= 0 && v_int < overlay.rows) {
            cv::circle(overlay, cv::Point(u_int, v_int), 2, cv::Scalar(0, 0, 255), -1);  // Red
        }
    }
    
    return overlay;
}

// Create depth comparison visualization
cv::Mat createDepthComparison(const cv::Mat& observed_depth,
                              const cv::Mat& rendered_depth,
                              int width, int height) {
    // Normalize observed depth for visualization
    cv::Mat obs_vis, rend_vis, residual;
    observed_depth.copyTo(obs_vis);
    rendered_depth.copyTo(rend_vis);
    
    // Get valid depth ranges
    double obs_min, obs_max, rend_min, rend_max;
    cv::minMaxLoc(obs_vis, &obs_min, &obs_max, nullptr, nullptr, obs_vis > 0);
    cv::minMaxLoc(rend_vis, &rend_min, &rend_max, nullptr, nullptr, rend_vis > 0);
    
    // Normalize to 0-255
    obs_vis.convertTo(obs_vis, CV_8U, 255.0 / (obs_max - obs_min + 1e-6), -obs_min * 255.0 / (obs_max - obs_min + 1e-6));
    rend_vis.convertTo(rend_vis, CV_8U, 255.0 / (rend_max - rend_min + 1e-6), -rend_min * 255.0 / (rend_max - rend_min + 1e-6));
    
    // Apply colormap
    cv::applyColorMap(obs_vis, obs_vis, cv::COLORMAP_JET);
    cv::applyColorMap(rend_vis, rend_vis, cv::COLORMAP_JET);
    
    // Compute residual (absolute difference in mm)
    residual = cv::Mat::zeros(observed_depth.size(), CV_32F);
    for (int v = 0; v < observed_depth.rows; ++v) {
        for (int u = 0; u < observed_depth.cols; ++u) {
            float obs = observed_depth.at<float>(v, u);
            float rend = rendered_depth.at<float>(v, u);
            if (obs > 0 && rend > 0) {
                residual.at<float>(v, u) = std::abs(obs - rend) * 1000.0;  // Convert to mm
            }
        }
    }
    
    // Normalize residual for visualization
    double res_min, res_max;
    cv::minMaxLoc(residual, &res_min, &res_max, nullptr, nullptr, residual > 0);
    cv::Mat res_vis;
    residual.convertTo(res_vis, CV_8U, 255.0 / (res_max - res_min + 1e-6), -res_min * 255.0 / (res_max - res_min + 1e-6));
    cv::applyColorMap(res_vis, res_vis, cv::COLORMAP_HOT);
    
    // Combine side-by-side
    cv::Mat combined(height, width * 3, CV_8UC3);
    int w = width / 3;
    cv::resize(obs_vis, combined(cv::Rect(0, 0, w, height)), cv::Size(w, height));
    cv::resize(rend_vis, combined(cv::Rect(w, 0, w, height)), cv::Size(w, height));
    cv::resize(res_vis, combined(cv::Rect(w * 2, 0, w, height)), cv::Size(w, height));
    
    // Add labels
    cv::putText(combined, "Observed", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Rendered", cv::Point(w + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Residual (mm)", cv::Point(w * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    return combined;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --mesh-rigid <path>       Rigid-aligned mesh PLY file (required)\n"
              << "  --mesh-opt <path>         Optimized mesh PLY file (optional)\n"
              << "  --depth <path>            Observed depth image (required)\n"
              << "  --rgb <path>              RGB image (optional, for 2D overlay)\n"
              << "  --intrinsics <path>       Camera intrinsics file (required)\n"
              << "  --out-dir <path>          Output directory for all files (required)\n"
              << "  --frame-name <name>       Frame name prefix (default: frame_00000)\n"
              << "  --output-metrics <path>   Output JSON metrics file (optional)\n"
              << "  --depth-scale <value>     Depth scale factor (default: 1000.0)\n"
              << "  --max-scan-points <n>     Max scan points to use (default: 50000)\n"
              << "\n"
              << "Output files (in out-dir):\n"
              << "  <frame>_scan.ply          Cyan point cloud from depth\n"
              << "  <frame>_mesh_rigid.ply    Red rigid-aligned mesh\n"
              << "  <frame>_overlay_rigid.ply Combined scan + rigid mesh\n"
              << "  <frame>_mesh_opt.ply      Red optimized mesh (if --mesh-opt)\n"
              << "  <frame>_overlay_opt.ply   Combined scan + opt mesh (if --mesh-opt)\n"
              << "  <frame>_overlay_2d.png    2D overlay on RGB (if --rgb)\n";
}

int main(int argc, char* argv[]) {
    std::string mesh_rigid_path, mesh_opt_path, depth_path, rgb_path, intrinsics_path;
    std::string out_dir, frame_name = "frame_00000", output_metrics;
    double depth_scale = 1000.0;
    int max_scan_points = 50000;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--mesh-rigid" && i + 1 < argc) {
            mesh_rigid_path = argv[++i];
        } else if (arg == "--mesh-opt" && i + 1 < argc) {
            mesh_opt_path = argv[++i];
        } else if (arg == "--mesh" && i + 1 < argc) {
            // Legacy support
            mesh_rigid_path = argv[++i];
        } else if (arg == "--depth" && i + 1 < argc) {
            depth_path = argv[++i];
        } else if (arg == "--rgb" && i + 1 < argc) {
            rgb_path = argv[++i];
        } else if (arg == "--intrinsics" && i + 1 < argc) {
            intrinsics_path = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--frame-name" && i + 1 < argc) {
            frame_name = argv[++i];
        } else if (arg == "--output-metrics" && i + 1 < argc) {
            output_metrics = argv[++i];
        } else if (arg == "--output-3d" && i + 1 < argc) {
            // Legacy support - derive out_dir from path
            std::string legacy_path = argv[++i];
            size_t last_slash = legacy_path.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                out_dir = legacy_path.substr(0, last_slash);
            }
        } else if (arg == "--output-2d" && i + 1 < argc) {
            // Legacy support - ignore
            ++i;
        } else if (arg == "--output-depth" && i + 1 < argc) {
            // Legacy support - ignore
            ++i;
        } else if (arg == "--depth-scale" && i + 1 < argc) {
            depth_scale = std::stod(argv[++i]);
        } else if (arg == "--max-scan-points" && i + 1 < argc) {
            max_scan_points = std::stoi(argv[++i]);
        }
    }
    
    if (mesh_rigid_path.empty() || depth_path.empty() || intrinsics_path.empty() || out_dir.empty()) {
        std::cerr << "Error: Missing required arguments\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Create output directory
    createDirectory(out_dir);
    
    std::cout << "=== Creating Mesh-Scan Overlays ===\n" << std::endl;
    std::cout << "Output directory: " << out_dir << std::endl;
    std::cout << "Frame name: " << frame_name << std::endl;
    
    // ========================================================================
    // [1] Load rigid-aligned mesh
    // ========================================================================
    std::cout << "\n[1] Loading rigid-aligned mesh..." << std::endl;
    std::vector<Eigen::Vector3d> mesh_rigid_vertices;
    std::vector<std::vector<int>> mesh_faces;
    if (!loadPLYVertices(mesh_rigid_path, mesh_rigid_vertices)) {
        std::cerr << "Error: Failed to load rigid mesh vertices from " << mesh_rigid_path << std::endl;
        return 1;
    }
    loadPLYFaces(mesh_rigid_path, mesh_faces);
    std::cout << "    Loaded " << mesh_rigid_vertices.size() << " vertices, " << mesh_faces.size() << " faces" << std::endl;
    
    // ========================================================================
    // [2] Load optimized mesh (if provided)
    // ========================================================================
    std::vector<Eigen::Vector3d> mesh_opt_vertices;
    std::vector<std::vector<int>> mesh_opt_faces;
    bool has_opt_mesh = false;
    if (!mesh_opt_path.empty()) {
        std::cout << "\n[2] Loading optimized mesh..." << std::endl;
        if (loadPLYVertices(mesh_opt_path, mesh_opt_vertices)) {
            loadPLYFaces(mesh_opt_path, mesh_opt_faces);
            has_opt_mesh = true;
            std::cout << "    Loaded " << mesh_opt_vertices.size() << " vertices, " << mesh_opt_faces.size() << " faces" << std::endl;
        } else {
            std::cerr << "    Warning: Failed to load optimized mesh, skipping" << std::endl;
        }
    }
    
    // ========================================================================
    // [3] Load depth and create point cloud
    // ========================================================================
    std::cout << "\n[3] Loading depth and creating point cloud..." << std::endl;
    RGBDFrame frame;
    if (!frame.loadDepth(depth_path, depth_scale)) {
        std::cerr << "Error: Failed to load depth" << std::endl;
        return 1;
    }
    
    CameraIntrinsics intrinsics = CameraIntrinsics::loadFromFile(intrinsics_path);
    std::vector<Eigen::Vector3d> scan_points;
    std::vector<std::pair<int, int>> pixel_indices;
    backprojectDepth(frame.getDepth(), intrinsics, scan_points, pixel_indices);
    
    std::cout << "    Generated " << scan_points.size() << " scan points from depth" << std::endl;
    
    // Subsample if too large
    std::vector<Eigen::Vector3d> scan_points_full = scan_points;  // Keep full for metrics
    if (static_cast<int>(scan_points.size()) > max_scan_points) {
        std::vector<Eigen::Vector3d> subsampled;
        std::vector<size_t> indices(scan_points.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(42);  // Fixed seed for reproducibility
        std::shuffle(indices.begin(), indices.end(), g);
        size_t num_samples = static_cast<size_t>(max_scan_points);
        for (size_t i = 0; i < num_samples && i < indices.size(); ++i) {
            subsampled.push_back(scan_points[indices[i]]);
        }
        scan_points = subsampled;
        std::cout << "    Subsampled to " << scan_points.size() << " points for PLY output" << std::endl;
    }
    
    // ========================================================================
    // [4] Compute overlay metrics
    // ========================================================================
    std::cout << "\n[4] Computing overlay metrics..." << std::endl;
    OverlayMetrics rigid_metrics = computeOverlayMetrics(scan_points_full, mesh_rigid_vertices);
    
    std::cout << "    Rigid alignment metrics:" << std::endl;
    std::cout << "      Scale ratio (mesh/scan bbox): " << std::fixed << std::setprecision(3) << rigid_metrics.scale_ratio << std::endl;
    std::cout << "      PCA axis dot product: " << rigid_metrics.axis_dot_product << std::endl;
    std::cout << "      BBox overlap (X/Y/Z): " << rigid_metrics.bbox_overlap_x << " / " 
              << rigid_metrics.bbox_overlap_y << " / " << rigid_metrics.bbox_overlap_z << std::endl;
    std::cout << "      Scan Z-range: [" << rigid_metrics.cloud_z_min << ", " << rigid_metrics.cloud_z_max << "] m" << std::endl;
    std::cout << "      Mesh Z-range: [" << rigid_metrics.mesh_z_min << ", " << rigid_metrics.mesh_z_max << "] m" << std::endl;
    std::cout << "      NN-RMSE: " << rigid_metrics.nn_rmse_m * 1000.0 << " mm" << std::endl;
    
    OverlayMetrics opt_metrics;
    if (has_opt_mesh) {
        opt_metrics = computeOverlayMetrics(scan_points_full, mesh_opt_vertices);
        std::cout << "\n    Optimized mesh metrics:" << std::endl;
        std::cout << "      Scale ratio: " << opt_metrics.scale_ratio << std::endl;
        std::cout << "      NN-RMSE: " << opt_metrics.nn_rmse_m * 1000.0 << " mm" << std::endl;
    }
    
    // ========================================================================
    // [5] Export standalone scan PLY (cyan)
    // ========================================================================
    std::cout << "\n[5] Exporting standalone PLY files..." << std::endl;
    std::string scan_ply = out_dir + "/" + frame_name + "_scan.ply";
    if (saveScanOnlyPLY(scan_ply, scan_points)) {
        std::cout << "    Saved scan: " << scan_ply << std::endl;
    }
    
    // ========================================================================
    // [6] Export mesh_rigid PLY (red)
    // ========================================================================
    std::string mesh_rigid_ply = out_dir + "/" + frame_name + "_mesh_rigid.ply";
    if (saveMeshOnlyPLY(mesh_rigid_ply, mesh_rigid_vertices, mesh_faces)) {
        std::cout << "    Saved mesh_rigid: " << mesh_rigid_ply << std::endl;
    }
    
    // ========================================================================
    // [7] Export overlay_rigid PLY (combined)
    // ========================================================================
    std::string overlay_rigid_ply = out_dir + "/" + frame_name + "_overlay_rigid.ply";
    if (saveColoredPLY(overlay_rigid_ply, scan_points, mesh_rigid_vertices, mesh_faces)) {
        std::cout << "    Saved overlay_rigid: " << overlay_rigid_ply << std::endl;
    }
    
    // ========================================================================
    // [8] Export optimized mesh files (if available)
    // ========================================================================
    if (has_opt_mesh) {
        std::string mesh_opt_ply = out_dir + "/" + frame_name + "_mesh_opt.ply";
        if (saveMeshOnlyPLY(mesh_opt_ply, mesh_opt_vertices, mesh_opt_faces)) {
            std::cout << "    Saved mesh_opt: " << mesh_opt_ply << std::endl;
        }
        
        std::string overlay_opt_ply = out_dir + "/" + frame_name + "_overlay_opt.ply";
        if (saveColoredPLY(overlay_opt_ply, scan_points, mesh_opt_vertices, mesh_opt_faces)) {
            std::cout << "    Saved overlay_opt: " << overlay_opt_ply << std::endl;
        }
    }
    
    // ========================================================================
    // [9] Export 2D overlay (if RGB provided)
    // ========================================================================
    if (!rgb_path.empty()) {
        std::cout << "\n[9] Creating 2D overlay..." << std::endl;
        cv::Mat rgb = cv::imread(rgb_path);
        if (!rgb.empty()) {
            cv::Mat overlay = create2DOverlay(rgb, mesh_rigid_vertices, scan_points, intrinsics);
            std::string overlay_2d_png = out_dir + "/" + frame_name + "_overlay_2d.png";
            cv::imwrite(overlay_2d_png, overlay);
            std::cout << "    Saved: " << overlay_2d_png << std::endl;
        }
    }
    
    // ========================================================================
    // [10] Write metrics JSON
    // ========================================================================
    std::string metrics_path = output_metrics.empty() ? 
                               (out_dir + "/" + frame_name + "_overlay_metrics.json") : output_metrics;
    std::cout << "\n[10] Writing metrics JSON..." << std::endl;
    
    std::ofstream metrics_file(metrics_path);
    if (metrics_file.is_open()) {
        metrics_file << std::fixed << std::setprecision(6);
        metrics_file << "{\n";
        metrics_file << "  \"rigid\": {\n";
        metrics_file << "    \"scale_ratio\": " << rigid_metrics.scale_ratio << ",\n";
        metrics_file << "    \"axis_dot_product\": " << rigid_metrics.axis_dot_product << ",\n";
        metrics_file << "    \"bbox_overlap_x\": " << rigid_metrics.bbox_overlap_x << ",\n";
        metrics_file << "    \"bbox_overlap_y\": " << rigid_metrics.bbox_overlap_y << ",\n";
        metrics_file << "    \"bbox_overlap_z\": " << rigid_metrics.bbox_overlap_z << ",\n";
        metrics_file << "    \"cloud_z_range\": [" << rigid_metrics.cloud_z_min << ", " << rigid_metrics.cloud_z_max << "],\n";
        metrics_file << "    \"mesh_z_range\": [" << rigid_metrics.mesh_z_min << ", " << rigid_metrics.mesh_z_max << "],\n";
        metrics_file << "    \"num_scan_points\": " << rigid_metrics.num_scan_points << ",\n";
        metrics_file << "    \"num_mesh_vertices\": " << rigid_metrics.num_mesh_vertices << ",\n";
        metrics_file << "    \"nn_rmse_m\": " << rigid_metrics.nn_rmse_m << "\n";
        metrics_file << "  }";
        
        if (has_opt_mesh) {
            metrics_file << ",\n";
            metrics_file << "  \"optimized\": {\n";
            metrics_file << "    \"scale_ratio\": " << opt_metrics.scale_ratio << ",\n";
            metrics_file << "    \"axis_dot_product\": " << opt_metrics.axis_dot_product << ",\n";
            metrics_file << "    \"bbox_overlap_x\": " << opt_metrics.bbox_overlap_x << ",\n";
            metrics_file << "    \"bbox_overlap_y\": " << opt_metrics.bbox_overlap_y << ",\n";
            metrics_file << "    \"bbox_overlap_z\": " << opt_metrics.bbox_overlap_z << ",\n";
            metrics_file << "    \"cloud_z_range\": [" << opt_metrics.cloud_z_min << ", " << opt_metrics.cloud_z_max << "],\n";
            metrics_file << "    \"mesh_z_range\": [" << opt_metrics.mesh_z_min << ", " << opt_metrics.mesh_z_max << "],\n";
            metrics_file << "    \"num_scan_points\": " << opt_metrics.num_scan_points << ",\n";
            metrics_file << "    \"num_mesh_vertices\": " << opt_metrics.num_mesh_vertices << ",\n";
            metrics_file << "    \"nn_rmse_m\": " << opt_metrics.nn_rmse_m << "\n";
            metrics_file << "  }";
        }
        
        metrics_file << "\n}\n";
        std::cout << "    Saved: " << metrics_path << std::endl;
    }
    
    std::cout << "\n=== Overlay generation completed ===" << std::endl;
    std::cout << "\nGenerated files in " << out_dir << ":" << std::endl;
    std::cout << "  - " << frame_name << "_scan.ply (cyan point cloud)" << std::endl;
    std::cout << "  - " << frame_name << "_mesh_rigid.ply (red mesh)" << std::endl;
    std::cout << "  - " << frame_name << "_overlay_rigid.ply (combined)" << std::endl;
    if (has_opt_mesh) {
        std::cout << "  - " << frame_name << "_mesh_opt.ply (red optimized mesh)" << std::endl;
        std::cout << "  - " << frame_name << "_overlay_opt.ply (combined optimized)" << std::endl;
    }
    std::cout << "  - " << frame_name << "_overlay_metrics.json" << std::endl;
    
    return 0;
}
