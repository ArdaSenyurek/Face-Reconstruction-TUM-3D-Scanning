/**
 * Component Demonstration Tool
 * 
 * Demonstrates individual library components (not used in main pipeline).
 * Useful for testing and understanding each component in isolation.
 * 
 * Usage:
 *   build/bin/FaceReconstruction
 * 
 * Note: This is a demo/example tool, not part of the production pipeline.
 */

#include "data/RGBDFrame.h"
#include "camera/CameraIntrinsics.h"
#include "utils/DepthUtils.h"
#include "model/MorphableModel.h"
#include "landmarks/LandmarkData.h"
#include "alignment/Procrustes.h"
#include <iostream>
#include <string>

using namespace face_reconstruction;

int main(int argc, char* argv[]) {
    std::cout << "=== 3D Face Reconstruction - Component Demo ===" << std::endl;
    
    // Example 1: Load RGB-D frame
    std::cout << "\n[1] Testing RGB-D Frame Loading..." << std::endl;
    RGBDFrame frame;
    
    // Uncomment when you have actual data:
    // if (!frame.loadRGB("data/rgb_image.png")) {
    //     std::cerr << "Failed to load RGB image" << std::endl;
    //     return 1;
    // }
    // if (!frame.loadDepth("data/depth_image.png", 1000.0)) {
    //     std::cerr << "Failed to load depth image" << std::endl;
    //     return 1;
    // }
    // frame.printStats();
    
    std::cout << "RGB-D frame loading test (skipped - no data files)" << std::endl;
    
    // Example 2: Camera intrinsics
    std::cout << "\n[2] Testing Camera Intrinsics..." << std::endl;
    CameraIntrinsics intrinsics(525.0, 525.0, 320.0, 240.0);  // Example values
    std::cout << "Camera intrinsics: fx=" << intrinsics.fx 
              << ", fy=" << intrinsics.fy 
              << ", cx=" << intrinsics.cx 
              << ", cy=" << intrinsics.cy << std::endl;
    
    Eigen::Matrix3d K = intrinsics.getIntrinsicsMatrix();
    std::cout << "Intrinsics matrix K:\n" << K << std::endl;
    
    // Example 3: Depth to 3D conversion
    std::cout << "\n[3] Testing Depth to 3D Conversion..." << std::endl;
    Eigen::Vector3d point = depthTo3D(100, 200, 1.5, intrinsics);
    std::cout << "3D point at pixel (100, 200) with depth 1.5m: "
              << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")" << std::endl;
    
    // Example 4: Morphable Model
    std::cout << "\n[4] Testing Morphable Model..." << std::endl;
    MorphableModel model;
    // model.loadFromFiles("data/model/");  // Uncomment when you have model files
    model.printStats();
    
    if (model.isValid()) {
        // Create dummy coefficients
        Eigen::VectorXd identity_coeffs = Eigen::VectorXd::Zero(model.num_identity_components);
        Eigen::VectorXd expression_coeffs = Eigen::VectorXd::Zero(model.num_expression_components);
        
        Eigen::MatrixXd vertices = model.reconstructFace(identity_coeffs, expression_coeffs);
        std::cout << "Reconstructed face vertices shape: " 
                  << vertices.rows() << " x " << vertices.cols() << std::endl;
    }
    
    // Example 5: Landmarks
    std::cout << "\n[5] Testing Landmark Data..." << std::endl;
    LandmarkData landmarks;
    landmarks.addLandmark(100.0, 150.0, 0);
    landmarks.addLandmark(200.0, 150.0, 1);
    landmarks.addLandmark(150.0, 200.0, 2);
    
    std::cout << "Created " << landmarks.size() << " landmarks" << std::endl;
    Eigen::MatrixXd lm_matrix = landmarks.toMatrix();
    std::cout << "Landmark matrix shape: " << lm_matrix.rows() << " x " << lm_matrix.cols() << std::endl;
    
    // Example 6: Procrustes alignment
    std::cout << "\n[6] Testing Procrustes Alignment..." << std::endl;
    std::vector<Eigen::Vector3d> source_pts = {
        Eigen::Vector3d(0, 0, 0),
        Eigen::Vector3d(1, 0, 0),
        Eigen::Vector3d(0, 1, 0)
    };
    
    std::vector<Eigen::Vector3d> target_pts = {
        Eigen::Vector3d(1, 1, 0),
        Eigen::Vector3d(2, 1, 0),
        Eigen::Vector3d(1, 2, 0)
    };
    
    SimilarityTransform transform = estimateSimilarityTransform(source_pts, target_pts);
    std::cout << "Estimated transform:" << std::endl;
    std::cout << "  Scale: " << transform.scale << std::endl;
    std::cout << "  Translation: (" << transform.translation.x() 
              << ", " << transform.translation.y() 
              << ", " << transform.translation.z() << ")" << std::endl;
    
    // Test transform
    Eigen::Vector3d test_point(0.5, 0.5, 0);
    Eigen::Vector3d transformed = transform.apply(test_point);
    std::cout << "  Test: (" << test_point.transpose() << ") -> (" 
              << transformed.transpose() << ")" << std::endl;
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    
    return 0;
}
