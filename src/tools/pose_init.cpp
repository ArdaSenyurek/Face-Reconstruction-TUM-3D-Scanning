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
#include <limits>
#include <map>
#include <set>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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

// ============================================================================
// Mapping Quality Metrics
// ============================================================================
struct MappingQualityMetrics {
    int num_valid_mappings = 0;
    double coverage_percent = 0.0;
    double pre_alignment_rmse_mm = 0.0;
    std::vector<double> per_landmark_errors;
    std::vector<int> outlier_indices;  // Outliers detected before alignment
    std::vector<int> valid_landmark_indices;
    
    // Outlier detection statistics
    int num_outliers_iqr = 0;
    int num_outliers_zscore = 0;
    int num_outliers_median = 0;
    double outlier_threshold_iqr = 0.0;
    double outlier_threshold_zscore = 0.0;
    double outlier_threshold_median = 0.0;
};

// ============================================================================
// Robust Outlier Detection
// ============================================================================

/**
 * Detect outliers using IQR (Interquartile Range) method
 * More robust than simple threshold methods
 */
std::vector<int> detectOutliersIQR(const std::vector<double>& errors,
                                    const std::vector<int>& indices,
                                    double multiplier = 1.5) {
    std::vector<int> outliers;
    if (errors.empty() || indices.empty() || errors.size() != indices.size()) {
        return outliers;
    }
    
    std::vector<double> sorted_errors = errors;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    
    size_t n = sorted_errors.size();
    size_t q1_idx = n / 4;
    size_t q3_idx = (3 * n) / 4;
    
    double q1 = sorted_errors[q1_idx];
    double q3 = sorted_errors[q3_idx];
    double iqr = q3 - q1;
    
    double lower_bound = q1 - multiplier * iqr;
    double upper_bound = q3 + multiplier * iqr;
    
    for (size_t i = 0; i < errors.size(); ++i) {
        if (errors[i] < lower_bound || errors[i] > upper_bound) {
            outliers.push_back(indices[i]);
        }
    }
    
    return outliers;
}

/**
 * Detect outliers using Z-score method
 * Flags points more than N standard deviations from the mean
 */
std::vector<int> detectOutliersZScore(const std::vector<double>& errors,
                                      const std::vector<int>& indices,
                                      double z_threshold = 2.5) {
    std::vector<int> outliers;
    if (errors.empty() || indices.empty() || errors.size() != indices.size()) {
        return outliers;
    }
    
    // Compute mean and standard deviation
    double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double variance = 0.0;
    for (double err : errors) {
        variance += (err - mean) * (err - mean);
    }
    variance /= errors.size();
    double std_dev = std::sqrt(variance);
    
    if (std_dev < 1e-6) return outliers;  // All errors are the same
    
    // Find outliers
    for (size_t i = 0; i < errors.size(); ++i) {
        double z_score = std::abs((errors[i] - mean) / std_dev);
        if (z_score > z_threshold) {
            outliers.push_back(indices[i]);
        }
    }
    
    return outliers;
}

/**
 * Detect outliers using median absolute deviation (MAD) method
 * Very robust to outliers
 */
std::vector<int> detectOutliersMAD(const std::vector<double>& errors,
                                   const std::vector<int>& indices,
                                   double threshold = 3.0) {
    std::vector<int> outliers;
    if (errors.empty() || indices.empty() || errors.size() != indices.size()) {
        return outliers;
    }
    
    // Compute median
    std::vector<double> sorted_errors = errors;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    double median = sorted_errors[sorted_errors.size() / 2];
    
    // Compute MAD (Median Absolute Deviation)
    std::vector<double> deviations;
    for (double err : errors) {
        deviations.push_back(std::abs(err - median));
    }
    std::sort(deviations.begin(), deviations.end());
    double mad = deviations[deviations.size() / 2];
    
    // Use modified Z-score with MAD
    // Modified Z-score = 0.6745 * (x - median) / MAD
    // Threshold of 3.5 is commonly used
    double mad_threshold = threshold * mad / 0.6745;
    
    for (size_t i = 0; i < errors.size(); ++i) {
        if (std::abs(errors[i] - median) > mad_threshold) {
            outliers.push_back(indices[i]);
        }
    }
    
    return outliers;
}

/**
 * Combined robust outlier detection using multiple methods
 * A landmark is flagged as outlier if detected by at least 2 methods
 */
std::vector<int> detectOutliersRobust(const std::vector<double>& errors,
                                      const std::vector<int>& indices,
                                      std::map<std::string, int>& method_counts) {
    // Safety check: ensure inputs are valid
    if (errors.empty() || indices.empty() || errors.size() != indices.size()) {
        method_counts["iqr"] = 0;
        method_counts["zscore"] = 0;
        method_counts["mad"] = 0;
        return std::vector<int>();
    }
    
    std::vector<int> outliers_iqr = detectOutliersIQR(errors, indices, 1.5);
    std::vector<int> outliers_zscore = detectOutliersZScore(errors, indices, 2.5);
    std::vector<int> outliers_mad = detectOutliersMAD(errors, indices, 3.0);
    
    method_counts["iqr"] = outliers_iqr.size();
    method_counts["zscore"] = outliers_zscore.size();
    method_counts["mad"] = outliers_mad.size();
    
    // Count how many methods flag each landmark
    std::map<int, int> outlier_votes;
    for (int idx : outliers_iqr) outlier_votes[idx]++;
    for (int idx : outliers_zscore) outlier_votes[idx]++;
    for (int idx : outliers_mad) outlier_votes[idx]++;
    
    // Landmarks flagged by at least 2 methods are considered outliers
    std::vector<int> robust_outliers;
    for (const auto& pair : outlier_votes) {
        if (pair.second >= 2) {
            robust_outliers.push_back(pair.first);
        }
    }
    
    return robust_outliers;
}

MappingQualityMetrics computeMappingQuality(
    const std::vector<Eigen::Vector3d>& observed_points,
    const std::vector<Eigen::Vector3d>& model_points,
    const std::vector<int>& valid_landmark_indices,
    int total_landmarks) {
    
    MappingQualityMetrics metrics;
    metrics.valid_landmark_indices = valid_landmark_indices;
    
    if (observed_points.size() != model_points.size() || observed_points.empty()) {
        return metrics;
    }
    
    metrics.num_valid_mappings = static_cast<int>(observed_points.size());
    metrics.coverage_percent = (static_cast<double>(metrics.num_valid_mappings) / total_landmarks) * 100.0;
    
    // Compute per-landmark 3D distances (before alignment)
    double total_error_sq = 0.0;
    for (size_t i = 0; i < observed_points.size(); ++i) {
        double dist = (observed_points[i] - model_points[i]).norm();
        metrics.per_landmark_errors.push_back(dist * 1000.0);  // Convert to mm
        total_error_sq += dist * dist;
    }
    
    metrics.pre_alignment_rmse_mm = std::sqrt(total_error_sq / observed_points.size()) * 1000.0;
    
    // Robust outlier detection using multiple methods
    if (!metrics.per_landmark_errors.empty()) {
        // Simple median-based method (for comparison)
        std::vector<double> sorted_errors = metrics.per_landmark_errors;
        std::sort(sorted_errors.begin(), sorted_errors.end());
        double median_error = sorted_errors[sorted_errors.size() / 2];
        metrics.outlier_threshold_median = 2.0 * median_error;
        
        std::vector<int> outliers_median;
        for (size_t i = 0; i < metrics.per_landmark_errors.size(); ++i) {
            if (metrics.per_landmark_errors[i] > metrics.outlier_threshold_median) {
                outliers_median.push_back(valid_landmark_indices[i]);
            }
        }
        metrics.num_outliers_median = outliers_median.size();
        
        // IQR method
        std::vector<int> outliers_iqr = detectOutliersIQR(
            metrics.per_landmark_errors, valid_landmark_indices, 1.5);
        metrics.num_outliers_iqr = outliers_iqr.size();
        
        // Compute IQR threshold for reporting
        size_t n = sorted_errors.size();
        size_t q1_idx = n / 4;
        size_t q3_idx = (3 * n) / 4;
        double q1 = sorted_errors[q1_idx];
        double q3 = sorted_errors[q3_idx];
        double iqr = q3 - q1;
        metrics.outlier_threshold_iqr = q3 + 1.5 * iqr;
        
        // Z-score method
        double mean = std::accumulate(metrics.per_landmark_errors.begin(), 
                                     metrics.per_landmark_errors.end(), 0.0) / metrics.per_landmark_errors.size();
        double variance = 0.0;
        for (double err : metrics.per_landmark_errors) {
            variance += (err - mean) * (err - mean);
        }
        variance /= metrics.per_landmark_errors.size();
        double std_dev = std::sqrt(variance);
        metrics.outlier_threshold_zscore = mean + 2.5 * std_dev;
        
        std::vector<int> outliers_zscore = detectOutliersZScore(
            metrics.per_landmark_errors, valid_landmark_indices, 2.5);
        metrics.num_outliers_zscore = outliers_zscore.size();
        
        // Combined robust detection (at least 2 methods must agree)
        std::map<std::string, int> method_counts;
        metrics.outlier_indices = detectOutliersRobust(
            metrics.per_landmark_errors, valid_landmark_indices, method_counts);
    }
    
    return metrics;
}

// ============================================================================
// Visualization Functions
// ============================================================================

// Save colored PLY with landmark points
bool saveLandmarkPLY(const std::string& filepath,
                     const std::vector<Eigen::Vector3d>& observed_points,
                     const std::vector<Eigen::Vector3d>& model_points,
                     const Eigen::Vector3d& observed_color = Eigen::Vector3d(0, 1, 1),  // Cyan
                     const Eigen::Vector3d& model_color = Eigen::Vector3d(1, 0, 0)) {  // Red
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    int total_vertices = observed_points.size() + model_points.size();
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << total_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Write observed points (cyan)
    for (const auto& p : observed_points) {
        file << std::fixed << std::setprecision(6)
             << p.x() << " " << p.y() << " " << p.z() << " "
             << static_cast<int>(observed_color.x() * 255) << " "
             << static_cast<int>(observed_color.y() * 255) << " "
             << static_cast<int>(observed_color.z() * 255) << "\n";
    }
    
    // Write model points (red or green)
    for (const auto& p : model_points) {
        file << std::fixed << std::setprecision(6)
             << p.x() << " " << p.y() << " " << p.z() << " "
             << static_cast<int>(model_color.x() * 255) << " "
             << static_cast<int>(model_color.y() * 255) << " "
             << static_cast<int>(model_color.z() * 255) << "\n";
    }
    
    file.close();
    return true;
}

// Save PLY with error-colored points
bool saveErrorPLY(const std::string& filepath,
                  const std::vector<Eigen::Vector3d>& observed_points,
                  const std::vector<Eigen::Vector3d>& aligned_points,
                  const std::vector<double>& errors) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    if (observed_points.size() != aligned_points.size() || errors.size() != observed_points.size()) {
        return false;
    }
    
    // Find error range for color mapping
    double min_error = *std::min_element(errors.begin(), errors.end());
    double max_error = *std::max_element(errors.begin(), errors.end());
    double error_range = max_error - min_error;
    if (error_range < 1e-6) error_range = 1.0;
    
    int total_vertices = observed_points.size() + aligned_points.size();
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << total_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Write observed points (cyan)
    for (const auto& p : observed_points) {
        file << std::fixed << std::setprecision(6)
             << p.x() << " " << p.y() << " " << p.z() << " "
             << "0 255 255\n";  // Cyan
    }
    
    // Write aligned points with error-based coloring (green to red)
    for (size_t i = 0; i < aligned_points.size(); ++i) {
        double normalized_error = (errors[i] - min_error) / error_range;
        // Color: green (low error) to red (high error)
        int r = static_cast<int>(normalized_error * 255);
        int g = static_cast<int>((1.0 - normalized_error) * 255);
        int b = 0;
        
        file << std::fixed << std::setprecision(6)
             << aligned_points[i].x() << " " << aligned_points[i].y() << " " << aligned_points[i].z() << " "
             << r << " " << g << " " << b << "\n";
    }
    
    file.close();
    return true;
}

// Create 2D overlay PNG showing landmarks before/after alignment
bool createLandmarkOverlayPNG(const std::string& filepath,
                               const cv::Mat& rgb,
                               const LandmarkData& landmarks,
                               const std::vector<Eigen::Vector3d>& model_points_before,
                               const std::vector<Eigen::Vector3d>& model_points_after,
                               const std::vector<int>& valid_landmark_indices,
                               const CameraIntrinsics& intrinsics) {
    cv::Mat overlay = rgb.clone();
    
    // Create a set of valid landmark indices for quick lookup
    std::set<int> valid_set(valid_landmark_indices.begin(), valid_landmark_indices.end());
    
    // Project and draw detected landmarks
    // Yellow for landmarks WITH correspondences, Cyan for landmarks WITHOUT correspondences
    for (size_t i = 0; i < landmarks.size(); ++i) {
        const auto& lm = landmarks[i];
        int u = static_cast<int>(std::round(lm.x));
        int v = static_cast<int>(std::round(lm.y));
        if (u >= 0 && u < overlay.cols && v >= 0 && v < overlay.rows) {
            if (valid_set.find(static_cast<int>(i)) != valid_set.end()) {
                // Has correspondence - draw larger yellow circle
                cv::circle(overlay, cv::Point(u, v), 4, cv::Scalar(0, 255, 255), -1);  // Yellow
                cv::circle(overlay, cv::Point(u, v), 4, cv::Scalar(0, 0, 0), 1);  // Black outline
            } else {
                // No correspondence - draw smaller cyan circle
                cv::circle(overlay, cv::Point(u, v), 2, cv::Scalar(255, 255, 0), -1);  // Cyan
            }
        }
    }
    
    // Project and draw model landmarks before alignment (red, smaller)
    for (const auto& p : model_points_before) {
        if (p.z() <= 0) continue;
        double u = intrinsics.fx * p.x() / p.z() + intrinsics.cx;
        double v = intrinsics.fy * p.y() / p.z() + intrinsics.cy;
        int u_int = static_cast<int>(std::round(u));
        int v_int = static_cast<int>(std::round(v));
        if (u_int >= 0 && u_int < overlay.cols && v_int >= 0 && v_int < overlay.rows) {
            cv::circle(overlay, cv::Point(u_int, v_int), 2, cv::Scalar(0, 0, 255), -1);  // Red
        }
    }
    
    // Project and draw model landmarks after alignment (green, larger)
    for (const auto& p : model_points_after) {
        if (p.z() <= 0) continue;
        double u = intrinsics.fx * p.x() / p.z() + intrinsics.cx;
        double v = intrinsics.fy * p.y() / p.z() + intrinsics.cy;
        int u_int = static_cast<int>(std::round(u));
        int v_int = static_cast<int>(std::round(v));
        if (u_int >= 0 && u_int < overlay.cols && v_int >= 0 && v_int < overlay.rows) {
            cv::circle(overlay, cv::Point(u_int, v_int), 3, cv::Scalar(0, 255, 0), -1);  // Green
            cv::circle(overlay, cv::Point(u_int, v_int), 3, cv::Scalar(0, 0, 0), 1);  // Black outline
        }
    }
    
    // Draw error vectors (lines from detected to projected after alignment)
    // Use thicker, more visible lines with color based on error magnitude
    for (size_t i = 0; i < valid_landmark_indices.size() && i < model_points_after.size(); ++i) {
        int lm_idx = valid_landmark_indices[i];
        if (lm_idx < 0 || lm_idx >= static_cast<int>(landmarks.size())) continue;
        
        const auto& lm = landmarks[lm_idx];
        Eigen::Vector3d p = model_points_after[i];
        if (p.z() <= 0) continue;
        
        double u_proj = intrinsics.fx * p.x() / p.z() + intrinsics.cx;
        double v_proj = intrinsics.fy * p.y() / p.z() + intrinsics.cy;
        
        int u1 = static_cast<int>(std::round(lm.x));
        int v1 = static_cast<int>(std::round(lm.y));
        int u2 = static_cast<int>(std::round(u_proj));
        int v2 = static_cast<int>(std::round(v_proj));
        
        if (u1 >= 0 && u1 < overlay.cols && v1 >= 0 && v1 < overlay.rows &&
            u2 >= 0 && u2 < overlay.cols && v2 >= 0 && v2 < overlay.rows) {
            // Compute error magnitude in pixels
            double error_px = std::sqrt((u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2));
            
            // Color based on error: green (good) -> yellow (medium) -> red (bad)
            cv::Scalar line_color;
            if (error_px < 5.0) {
                line_color = cv::Scalar(0, 255, 0);  // Green - good alignment
            } else if (error_px < 15.0) {
                line_color = cv::Scalar(0, 255, 255);  // Yellow - medium error
            } else {
                line_color = cv::Scalar(0, 0, 255);  // Red - large error
            }
            
            cv::line(overlay, cv::Point(u1, v1), cv::Point(u2, v2), line_color, 2);  // Thicker line
        }
    }
    
    return cv::imwrite(filepath, overlay);
}

// Helper to write JSON (simple, no external dependency)
void writeJSONReport(const std::string& filepath,
                     const SimilarityTransform& transform,
                     double rmse_after_icp, double mean_error_after_icp, double median_error_after_icp,
                     const std::vector<Eigen::Vector3d>& observed_points,
                     const Eigen::MatrixXd& aligned_vertices,
                     const MappingQualityMetrics& mapping_quality,
                     double rmse_pre_alignment,
                     double rmse_after_procrustes,
                     const std::vector<double>& errors_after_procrustes,
                     const std::vector<double>& errors_after_icp,
                     const std::vector<int>& worst_landmarks,
                     const std::vector<int>& outliers_after_alignment,
                     const std::map<std::string, std::string>& visualization_files) {
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
    
    // Mapping quality section
    file << "  \"mapping_quality\": {\n";
    file << "    \"num_valid_mappings\": " << mapping_quality.num_valid_mappings << ",\n";
    file << "    \"coverage_percent\": " << mapping_quality.coverage_percent << ",\n";
    file << "    \"pre_alignment_rmse_mm\": " << mapping_quality.pre_alignment_rmse_mm << ",\n";
    file << "    \"per_landmark_errors_mm\": [";
    for (size_t i = 0; i < mapping_quality.per_landmark_errors.size(); ++i) {
        file << mapping_quality.per_landmark_errors[i];
        if (i < mapping_quality.per_landmark_errors.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "    \"outlier_detection\": {\n";
    file << "      \"robust_outliers\": [";
    for (size_t i = 0; i < mapping_quality.outlier_indices.size(); ++i) {
        file << mapping_quality.outlier_indices[i];
        if (i < mapping_quality.outlier_indices.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "      \"num_robust_outliers\": " << mapping_quality.outlier_indices.size() << ",\n";
    file << "      \"median_method\": {\n";
    file << "        \"num_outliers\": " << mapping_quality.num_outliers_median << ",\n";
    file << "        \"threshold_mm\": " << mapping_quality.outlier_threshold_median << "\n";
    file << "      },\n";
    file << "      \"iqr_method\": {\n";
    file << "        \"num_outliers\": " << mapping_quality.num_outliers_iqr << ",\n";
    file << "        \"threshold_mm\": " << mapping_quality.outlier_threshold_iqr << "\n";
    file << "      },\n";
    file << "      \"zscore_method\": {\n";
    file << "        \"num_outliers\": " << mapping_quality.num_outliers_zscore << ",\n";
    file << "        \"threshold_mm\": " << mapping_quality.outlier_threshold_zscore << "\n";
    file << "      }\n";
    file << "    }\n";
    file << "  },\n";
    
    // Procrustes analysis section
    double improvement_percent = 0.0;
    if (rmse_pre_alignment > 1e-6) {
        improvement_percent = ((rmse_pre_alignment - rmse_after_icp * 1000.0) / rmse_pre_alignment) * 100.0;
    }
    
    file << "  \"procrustes_analysis\": {\n";
    file << "    \"num_correspondences\": " << mapping_quality.num_valid_mappings << ",\n";
    file << "    \"pre_alignment_rmse_mm\": " << rmse_pre_alignment << ",\n";
    file << "    \"post_procrustes_rmse_mm\": " << rmse_after_procrustes * 1000.0 << ",\n";
    file << "    \"post_icp_rmse_mm\": " << rmse_after_icp * 1000.0 << ",\n";
    file << "    \"improvement_percent\": " << improvement_percent << ",\n";
    file << "    \"per_landmark_errors_after_procrustes_mm\": [";
    for (size_t i = 0; i < errors_after_procrustes.size(); ++i) {
        file << errors_after_procrustes[i] * 1000.0;
        if (i < errors_after_procrustes.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "    \"per_landmark_errors_after_icp_mm\": [";
    for (size_t i = 0; i < errors_after_icp.size(); ++i) {
        file << errors_after_icp[i] * 1000.0;
        if (i < errors_after_icp.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "    \"worst_landmarks\": [";
    for (size_t i = 0; i < worst_landmarks.size(); ++i) {
        file << worst_landmarks[i];
        if (i < worst_landmarks.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "    \"outliers_after_alignment\": [";
    for (size_t i = 0; i < outliers_after_alignment.size(); ++i) {
        file << outliers_after_alignment[i];
        if (i < outliers_after_alignment.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "    \"num_outliers_after_alignment\": " << outliers_after_alignment.size() << "\n";
    file << "  },\n";
    
    // Transform section
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
    
    // Alignment errors section (final after ICP)
    file << "  \"alignment_errors\": {\n";
    file << "    \"rmse_m\": " << rmse_after_icp << ",\n";
    file << "    \"rmse_mm\": " << rmse_after_icp * 1000.0 << ",\n";
    file << "    \"mean_error_m\": " << mean_error_after_icp << ",\n";
    file << "    \"mean_error_mm\": " << mean_error_after_icp * 1000.0 << ",\n";
    file << "    \"median_error_m\": " << median_error_after_icp << ",\n";
    file << "    \"median_error_mm\": " << median_error_after_icp * 1000.0 << "\n";
    file << "  },\n";
    
    // Depth z-range section
    file << "  \"depth_z_range\": {\n";
    file << "    \"observed_min_m\": " << obs_z_min << ",\n";
    file << "    \"observed_max_m\": " << obs_z_max << ",\n";
    file << "    \"mesh_min_m\": " << mesh_z_min << ",\n";
    file << "    \"mesh_max_m\": " << mesh_z_max << ",\n";
    file << "    \"mesh_in_range\": " << (mesh_z_min >= obs_z_min - 0.3 && mesh_z_max <= obs_z_max + 0.3 ? "true" : "false") << "\n";
    file << "  },\n";
    
    // Sanity checks section
    file << "  \"sanity_checks\": {\n";
    file << "    \"rotation_det_valid\": " << (std::abs(transform.rotation.determinant() - 1.0) < 0.01 ? "true" : "false") << ",\n";
    file << "    \"scale_in_range\": " << (transform.scale > 0.5 && transform.scale < 2.0 ? "true" : "false") << ",\n";
    file << "    \"rmse_acceptable\": " << (rmse_after_icp < 0.05 ? "true" : "false") << "\n";
    file << "  },\n";
    
    // Visualization files section
    file << "  \"visualization_files\": {\n";
    bool first = true;
    for (const auto& pair : visualization_files) {
        if (!first) file << ",\n";
        file << "    \"" << pair.first << "\": \"" << pair.second << "\"";
        first = false;
    }
    file << "\n  }\n";
    
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
    
    // Compute mapping quality metrics (before alignment)
    std::cout << "[7a] Computing mapping quality metrics..." << std::endl;
    MappingQualityMetrics mapping_quality = computeMappingQuality(
        matched_observed, matched_model, valid_landmark_indices, static_cast<int>(landmarks.size()));
    std::cout << "    Valid mappings: " << mapping_quality.num_valid_mappings 
              << " (" << mapping_quality.coverage_percent << "% coverage)" << std::endl;
    std::cout << "    Pre-alignment RMSE: " << mapping_quality.pre_alignment_rmse_mm << " mm" << std::endl;
    
    // Report outlier detection results
    if (!mapping_quality.per_landmark_errors.empty()) {
        std::cout << "    Outlier detection:" << std::endl;
        std::cout << "      - Median method: " << mapping_quality.num_outliers_median 
                  << " outliers (threshold: " << mapping_quality.outlier_threshold_median << " mm)" << std::endl;
        std::cout << "      - IQR method: " << mapping_quality.num_outliers_iqr 
                  << " outliers (threshold: " << mapping_quality.outlier_threshold_iqr << " mm)" << std::endl;
        std::cout << "      - Z-score method: " << mapping_quality.num_outliers_zscore 
                  << " outliers (threshold: " << mapping_quality.outlier_threshold_zscore << " mm)" << std::endl;
        std::cout << "      - Robust (combined): " << mapping_quality.outlier_indices.size() 
                  << " outliers (agreement of ≥2 methods)" << std::endl;
        
        if (!mapping_quality.outlier_indices.empty()) {
            std::cout << "      - Outlier landmark indices: [";
            for (size_t i = 0; i < std::min(10UL, mapping_quality.outlier_indices.size()); ++i) {
                std::cout << mapping_quality.outlier_indices[i];
                if (i < std::min(10UL, mapping_quality.outlier_indices.size()) - 1) std::cout << ", ";
            }
            if (mapping_quality.outlier_indices.size() > 10) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
    }
    
    // Save pre-alignment visualization
    std::string pre_alignment_ply = output_path;
    size_t last_dot = pre_alignment_ply.find_last_of(".");
    if (last_dot != std::string::npos) {
        pre_alignment_ply = pre_alignment_ply.substr(0, last_dot) + "_pre_alignment.ply";
    } else {
        pre_alignment_ply += "_pre_alignment.ply";
    }
    std::cout << "[7b] Saving pre-alignment visualization..." << std::endl;
    if (saveLandmarkPLY(pre_alignment_ply, matched_observed, matched_model)) {
        std::cout << "    Saved: " << pre_alignment_ply << std::endl;
    }
    
    // Step 3: Estimate similarity transform with Procrustes
    std::cout << "[8] Estimating similarity transform (Procrustes)..." << std::endl;
    
    Eigen::MatrixXd source_points(matched_model.size(), 3);
    Eigen::MatrixXd target_points(matched_observed.size(), 3);
    
    for (size_t i = 0; i < matched_model.size(); ++i) {
        source_points.row(i) = matched_model[i].transpose();
        target_points.row(i) = matched_observed[i].transpose();
    }
    
    // Validate correspondences
    if (!validateCorrespondences(source_points, target_points)) {
        std::cerr << "Warning: Correspondences may be degenerate (collinear points)" << std::endl;
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
    
    // Compute alignment error after Procrustes
    std::vector<double> errors_after_procrustes = computeAlignmentErrors(
        source_points, target_points, transform);
    Eigen::MatrixXd transformed_model = transform.apply(source_points);
    double total_error = 0.0;
    for (double err : errors_after_procrustes) {
        total_error += err * err;
    }
    double rmse_after_procrustes = std::sqrt(total_error / errors_after_procrustes.size());
    std::vector<double> sorted_errors = errors_after_procrustes;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    double mean_error_procrustes = std::accumulate(errors_after_procrustes.begin(), errors_after_procrustes.end(), 0.0) / errors_after_procrustes.size();
    double median_error_procrustes = sorted_errors[sorted_errors.size() / 2];
    
    std::cout << "    Alignment RMSE (after Procrustes): " << rmse_after_procrustes << " m (" << rmse_after_procrustes * 1000.0 << " mm)" << std::endl;
    std::cout << "    Mean error: " << mean_error_procrustes << " m (" << mean_error_procrustes * 1000.0 << " mm)" << std::endl;
    std::cout << "    Median error: " << median_error_procrustes << " m (" << median_error_procrustes * 1000.0 << " mm)" << std::endl;
    
    // Save post-Procrustes visualization
    std::string post_procrustes_ply = output_path;
    size_t last_dot2 = post_procrustes_ply.find_last_of(".");
    if (last_dot2 != std::string::npos) {
        post_procrustes_ply = post_procrustes_ply.substr(0, last_dot2) + "_post_procrustes.ply";
    } else {
        post_procrustes_ply += "_post_procrustes.ply";
    }
    std::cout << "[8a] Saving post-Procrustes visualization..." << std::endl;
    std::vector<Eigen::Vector3d> aligned_model_points;
    for (int i = 0; i < transformed_model.rows(); ++i) {
        aligned_model_points.push_back(transformed_model.row(i));
    }
    if (saveErrorPLY(post_procrustes_ply, matched_observed, aligned_model_points, errors_after_procrustes)) {
        std::cout << "    Saved: " << post_procrustes_ply << std::endl;
    }
    
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
    std::vector<double> errors_after_icp = computeAlignmentErrors(
        source_points, target_points, transform);
    total_error = 0.0;
    for (double err : errors_after_icp) {
        total_error += err * err;
    }
    double rmse_after_icp = std::sqrt(total_error / errors_after_icp.size());
    sorted_errors = errors_after_icp;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    double mean_error_after_icp = std::accumulate(errors_after_icp.begin(), errors_after_icp.end(), 0.0) / errors_after_icp.size();
    double median_error_after_icp = sorted_errors[sorted_errors.size() / 2];
    
    std::cout << "    Alignment RMSE (after ICP): " << rmse_after_icp << " m (" << rmse_after_icp * 1000.0 << " mm)" << std::endl;
    std::cout << "    Mean error: " << mean_error_after_icp << " m (" << mean_error_after_icp * 1000.0 << " mm)" << std::endl;
    std::cout << "    Median error: " << median_error_after_icp << " m (" << median_error_after_icp * 1000.0 << " mm)" << std::endl;
    
    // Find worst landmarks (top 5 with largest errors)
    std::vector<std::pair<double, int>> error_landmark_pairs;
    for (size_t i = 0; i < errors_after_icp.size() && i < valid_landmark_indices.size(); ++i) {
        error_landmark_pairs.push_back({errors_after_icp[i], valid_landmark_indices[i]});
    }
    std::sort(error_landmark_pairs.begin(), error_landmark_pairs.end(), std::greater<std::pair<double, int>>());
    std::vector<int> worst_landmarks;
    for (size_t i = 0; i < std::min(5UL, error_landmark_pairs.size()); ++i) {
        worst_landmarks.push_back(error_landmark_pairs[i].second);
    }
    
    // Detect outliers after alignment (using robust methods)
    std::vector<int> outliers_after_alignment;
    if (!errors_after_icp.empty() && !valid_landmark_indices.empty() && 
        errors_after_icp.size() == valid_landmark_indices.size()) {
        std::vector<double> errors_after_icp_mm;
        for (double err : errors_after_icp) {
            errors_after_icp_mm.push_back(err * 1000.0);  // Convert to mm
        }
        std::map<std::string, int> method_counts_after;
        outliers_after_alignment = detectOutliersRobust(
            errors_after_icp_mm, valid_landmark_indices, method_counts_after);
    }
    
    std::cout << "    Outliers after alignment: " << outliers_after_alignment.size() << " landmarks" << std::endl;
    if (!outliers_after_alignment.empty()) {
        std::cout << "      - Outlier indices: [";
        for (size_t i = 0; i < std::min(10UL, outliers_after_alignment.size()); ++i) {
            std::cout << outliers_after_alignment[i];
            if (i < std::min(10UL, outliers_after_alignment.size()) - 1) std::cout << ", ";
        }
        if (outliers_after_alignment.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    
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
    
    // Save aligned mesh with highlighted landmarks
    std::string aligned_with_landmarks_ply = output_path;
    size_t last_dot3 = aligned_with_landmarks_ply.find_last_of(".");
    if (last_dot3 != std::string::npos) {
        aligned_with_landmarks_ply = aligned_with_landmarks_ply.substr(0, last_dot3) + "_with_landmarks.ply";
    } else {
        aligned_with_landmarks_ply += "_with_landmarks.ply";
    }
    // For now, just save the regular aligned mesh (can be enhanced later to highlight landmarks)
    
    // Create 2D overlay PNG
    std::string landmarks_overlay_png = output_path;
    size_t last_dot4 = landmarks_overlay_png.find_last_of(".");
    if (last_dot4 != std::string::npos) {
        landmarks_overlay_png = landmarks_overlay_png.substr(0, last_dot4) + "_landmarks_overlay.png";
    } else {
        landmarks_overlay_png += "_landmarks_overlay.png";
    }
    
    std::cout << "[11] Creating 2D landmark overlay..." << std::endl;
    std::vector<Eigen::Vector3d> model_points_before;
    for (const auto& p : matched_model) {
        model_points_before.push_back(p);
    }
    std::vector<Eigen::Vector3d> model_points_after;
    for (int i = 0; i < refined_model_landmarks.rows(); ++i) {
        model_points_after.push_back(refined_model_landmarks.row(i));
    }
    
    // Create improved visualization with better correspondence visualization
    if (createLandmarkOverlayPNG(landmarks_overlay_png, frame.getRGB(), landmarks,
                                  model_points_before, model_points_after,
                                  valid_landmark_indices, intrinsics)) {
        std::cout << "    Saved: " << landmarks_overlay_png << std::endl;
        
        // Print diagnostic info
        std::cout << "    Visualization info:" << std::endl;
        std::cout << "      - Yellow circles: " << landmarks.size() << " detected landmarks" << std::endl;
        std::cout << "      - Red circles: " << model_points_before.size() << " model landmarks (before alignment)" << std::endl;
        std::cout << "      - Green circles: " << model_points_after.size() << " model landmarks (after alignment)" << std::endl;
        std::cout << "      - Yellow lines: Error vectors from detected to projected" << std::endl;
        std::cout << "      - Coverage: " << mapping_quality.coverage_percent << "% of landmarks have correspondences" << std::endl;
    }
    
    // Write JSON report if requested
    if (!report_path.empty()) {
        std::cout << "[12] Writing JSON report..." << std::endl;
        std::map<std::string, std::string> viz_files;
        viz_files["pre_alignment_ply"] = pre_alignment_ply;
        viz_files["post_procrustes_ply"] = post_procrustes_ply;
        viz_files["landmarks_overlay_png"] = landmarks_overlay_png;
        
        writeJSONReport(report_path, transform, rmse_after_icp, mean_error_after_icp, median_error_after_icp,
                       observed_3d_points, aligned_vertices, mapping_quality,
                       mapping_quality.pre_alignment_rmse_mm,
                       rmse_after_procrustes,
                       errors_after_procrustes,
                       errors_after_icp,
                       worst_landmarks,
                       outliers_after_alignment,
                       viz_files);
        std::cout << "    Report saved to: " << report_path << std::endl;
    }
    
    std::cout << "\n=== Pose initialization completed successfully ===" << std::endl;
    
    return 0;
}

