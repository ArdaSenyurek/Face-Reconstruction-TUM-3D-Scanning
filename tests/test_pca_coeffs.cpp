/**
 * STEP B: PCA Coefficient Evaluation Test
 * 
 * Goal: Validate that PCA coefficients actually reconstruct plausible faces.
 * 
 * Tests:
 *   1) alpha=0, beta=0 returns mean shape exactly (within eps)
 *   2) Varying one coefficient changes shape smoothly (no NaNs, no explosions)
 *   3) Coefficient regularization weights consistent with stddev (if stddev present)
 *   4) Reconstructed mesh has correct vertex count and reasonable bounds
 * 
 * Usage:
 *   bin/test_pca_coeffs <model_dir>
 * 
 * Example:
 *   bin/test_pca_coeffs data/model
 */

#include "model/MorphableModel.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

using namespace face_reconstruction;

const double EPS = 1e-6;

bool testMeanShapeReconstruction(const MorphableModel& model) {
    std::cout << "Test 1: alpha=0, beta=0 should return mean shape..." << std::endl;
    
    if (!model.isValid()) {
        std::cerr << "  ERROR: Model is not valid" << std::endl;
        return false;
    }
    
    // Create zero coefficients
    Eigen::VectorXd alpha_zero = Eigen::VectorXd::Zero(model.num_identity_components);
    Eigen::VectorXd beta_zero = Eigen::VectorXd::Zero(model.num_expression_components);
    
    // Reconstruct with zero coefficients
    Eigen::MatrixXd reconstructed = model.reconstructFace(alpha_zero, beta_zero);
    
    // Get mean shape matrix
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
    // Compare
    double max_diff = 0.0;
    for (int i = 0; i < reconstructed.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            double diff = std::abs(reconstructed(i, j) - mean_shape(i, j));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    std::cout << "  Max difference: " << max_diff << std::endl;
    
    if (max_diff < EPS) {
        std::cout << "  ✓ PASS: Mean shape matches (diff < " << EPS << ")" << std::endl;
        return true;
    } else {
        std::cerr << "  ✗ FAIL: Mean shape mismatch (diff = " << max_diff << ")" << std::endl;
        return false;
    }
}

bool testSmoothCoefficientVariation(const MorphableModel& model) {
    std::cout << "Test 2: Varying coefficients should change shape smoothly..." << std::endl;
    
    if (!model.isValid() || model.num_identity_components == 0) {
        std::cout << "  SKIP: No identity components available" << std::endl;
        return true;
    }
    
    // Test varying first identity coefficient
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(model.num_identity_components);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(model.num_expression_components);
    
    std::vector<double> test_values = {-2.0, -1.0, 0.0, 1.0, 2.0};
    Eigen::MatrixXd prev_vertices;
    bool first = true;
    
    for (double val : test_values) {
        alpha(0) = val;
        Eigen::MatrixXd vertices = model.reconstructFace(alpha, beta);
        
        // Check for NaNs or infinities
        bool has_nan = false;
        bool has_inf = false;
        for (int i = 0; i < vertices.rows(); ++i) {
            for (int j = 0; j < 3; ++j) {
                if (std::isnan(vertices(i, j))) {
                    has_nan = true;
                }
                if (std::isinf(vertices(i, j))) {
                    has_inf = true;
                }
            }
        }
        
        if (has_nan || has_inf) {
            std::cerr << "  ✗ FAIL: NaN or Inf detected for alpha[0]=" << val << std::endl;
            return false;
        }
        
        // Check smoothness (compare with previous)
        if (!first) {
            double max_change = 0.0;
            for (int i = 0; i < vertices.rows(); ++i) {
                double change = (vertices.row(i) - prev_vertices.row(i)).norm();
                if (change > max_change) {
                    max_change = change;
                }
            }
            
            // Change should be reasonable (not exploding)
            if (max_change > 1.0) {  // Arbitrary threshold - adjust based on model scale
                std::cerr << "  ✗ FAIL: Shape change too large (" << max_change 
                          << ") for alpha[0]=" << val << std::endl;
                return false;
            }
        }
        
        prev_vertices = vertices;
        first = false;
    }
    
    std::cout << "  ✓ PASS: Smooth variation, no NaNs, no explosions" << std::endl;
    return true;
}

bool testStddevConsistency(const MorphableModel& model) {
    std::cout << "Test 3: Coefficient regularization weights consistent with stddev..." << std::endl;
    
    if (!model.isValid()) {
        std::cerr << "  ERROR: Model is not valid" << std::endl;
        return false;
    }
    
    bool all_ok = true;
    
    // Check identity stddev
    if (model.num_identity_components > 0) {
        if (model.identity_stddev.size() != model.num_identity_components) {
            std::cerr << "  ✗ FAIL: identity_stddev size mismatch" << std::endl;
            all_ok = false;
        } else {
            // Check that stddev values are positive
            for (int i = 0; i < model.num_identity_components; ++i) {
                if (model.identity_stddev(i) <= 0) {
                    std::cerr << "  ✗ FAIL: identity_stddev[" << i << "] = " 
                              << model.identity_stddev(i) << " (should be > 0)" << std::endl;
                    all_ok = false;
                }
            }
            
            if (all_ok) {
                std::cout << "  Identity stddev: [" << model.identity_stddev.minCoeff() 
                          << ", " << model.identity_stddev.maxCoeff() << "]" << std::endl;
            }
        }
    }
    
    // Check expression stddev
    if (model.num_expression_components > 0) {
        if (model.expression_stddev.size() != model.num_expression_components) {
            std::cerr << "  ✗ FAIL: expression_stddev size mismatch" << std::endl;
            all_ok = false;
        } else {
            // Check that stddev values are positive
            for (int i = 0; i < model.num_expression_components; ++i) {
                if (model.expression_stddev(i) <= 0) {
                    std::cerr << "  ✗ FAIL: expression_stddev[" << i << "] = " 
                              << model.expression_stddev(i) << " (should be > 0)" << std::endl;
                    all_ok = false;
                }
            }
            
            if (all_ok) {
                std::cout << "  Expression stddev: [" << model.expression_stddev.minCoeff() 
                          << ", " << model.expression_stddev.maxCoeff() << "]" << std::endl;
            }
        }
    }
    
    if (all_ok) {
        std::cout << "  ✓ PASS: Stddev consistency verified" << std::endl;
    }
    
    return all_ok;
}

bool testMeshBounds(const MorphableModel& model) {
    std::cout << "Test 4: Reconstructed mesh has correct vertex count and reasonable bounds..." << std::endl;
    
    if (!model.isValid()) {
        std::cerr << "  ERROR: Model is not valid" << std::endl;
        return false;
    }
    
    // Create random coefficients (within reasonable range)
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(model.num_identity_components);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(model.num_expression_components);
    
    // Set coefficients to small random values (within 2 stddev)
    for (int i = 0; i < model.num_identity_components; ++i) {
        if (model.identity_stddev.size() > i && model.identity_stddev(i) > 0) {
            alpha(i) = 0.5 * model.identity_stddev(i);  // Small variation
        }
    }
    
    for (int i = 0; i < model.num_expression_components; ++i) {
        if (model.expression_stddev.size() > i && model.expression_stddev(i) > 0) {
            beta(i) = 0.5 * model.expression_stddev(i);  // Small variation
        }
    }
    
    // Reconstruct
    Eigen::MatrixXd vertices = model.reconstructFace(alpha, beta);
    
    // Check vertex count
    if (vertices.rows() != model.num_vertices) {
        std::cerr << "  ✗ FAIL: Vertex count mismatch (" << vertices.rows() 
                  << " != " << model.num_vertices << ")" << std::endl;
        return false;
    }
    
    if (vertices.cols() != 3) {
        std::cerr << "  ✗ FAIL: Vertices should be N x 3, got N x " << vertices.cols() << std::endl;
        return false;
    }
    
    // Calculate bounding box
    Eigen::Vector3d min_point = vertices.colwise().minCoeff();
    Eigen::Vector3d max_point = vertices.colwise().maxCoeff();
    Eigen::Vector3d centroid = vertices.colwise().mean();
    
    std::cout << "  Vertex count: " << vertices.rows() << " (expected: " << model.num_vertices << ")" << std::endl;
    std::cout << "  Bounding box:" << std::endl;
    std::cout << "    X: [" << min_point.x() << ", " << max_point.x() << "]" << std::endl;
    std::cout << "    Y: [" << min_point.y() << ", " << max_point.y() << "]" << std::endl;
    std::cout << "    Z: [" << min_point.z() << ", " << max_point.z() << "]" << std::endl;
    std::cout << "  Centroid: (" << centroid.x() << ", " << centroid.y() << ", " << centroid.z() << ")" << std::endl;
    
    // Check that bounds are reasonable (not too large, not all zeros)
    double max_extent = (max_point - min_point).maxCoeff();
    if (max_extent < 1e-6) {
        std::cerr << "  ✗ FAIL: Mesh extent too small (likely all zeros)" << std::endl;
        return false;
    }
    
    if (max_extent > 1000.0) {  // Arbitrary threshold - adjust based on model scale
        std::cerr << "  ✗ FAIL: Mesh extent too large (" << max_extent << ")" << std::endl;
        return false;
    }
    
    std::cout << "  ✓ PASS: Vertex count and bounds are reasonable" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " data/model" << std::endl;
        return 1;
    }
    
    std::string model_dir = argv[1];
    
    std::cout << "=== PCA Coefficient Evaluation Test ===" << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << std::endl;
    
    // Load model
    MorphableModel model;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "Failed to load model from: " << model_dir << std::endl;
        return 1;
    }
    
    // Print model statistics
    model.printStats();
    std::cout << std::endl;
    
    // Run tests
    std::cout << "--- Running Tests ---" << std::endl;
    std::cout << std::endl;
    
    bool test1 = testMeanShapeReconstruction(model);
    std::cout << std::endl;
    
    bool test2 = testSmoothCoefficientVariation(model);
    std::cout << std::endl;
    
    bool test3 = testStddevConsistency(model);
    std::cout << std::endl;
    
    bool test4 = testMeshBounds(model);
    std::cout << std::endl;
    
    // Summary
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Test 1 (Mean shape): " << (test1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 2 (Smooth variation): " << (test2 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 3 (Stddev consistency): " << (test3 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Test 4 (Mesh bounds): " << (test4 ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;
    
    if (test1 && test2 && test3 && test4) {
        std::cout << "✓ All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << "✗ Some tests FAILED!" << std::endl;
        return 1;
    }
}

