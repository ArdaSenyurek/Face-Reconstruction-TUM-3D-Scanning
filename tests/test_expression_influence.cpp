/**
 * Expression Influence Sanity Test
 *
 * Verifies that expression basis coefficients (delta) actually change the mesh.
 * If this test fails, the expression basis is not applied correctly.
 *
 * Usage:
 *   build/bin/test_expression_influence --model-dir <path>
 */

#include "model/MorphableModel.h"
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>

using namespace face_reconstruction;

int main(int argc, char* argv[]) {
    std::string model_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        }
    }

    if (model_dir.empty()) {
        std::cerr << "Usage: test_expression_influence --model-dir <path>" << std::endl;
        return 1;
    }

    MorphableModel model;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "FAIL: Could not load model from " << model_dir << std::endl;
        return 1;
    }

    if (!model.isValid()) {
        std::cerr << "FAIL: Model is not valid" << std::endl;
        return 1;
    }

    std::cout << "Model loaded: " << model.num_vertices << " vertices, "
              << model.num_expression_components << " expression components" << std::endl;

    if (model.num_expression_components == 0) {
        std::cerr << "FAIL: Model has 0 expression components" << std::endl;
        return 1;
    }

    // Reconstruct with alpha=0, delta=0 (mean shape)
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(model.num_identity_components);
    Eigen::VectorXd delta_zero = Eigen::VectorXd::Zero(model.num_expression_components);

    Eigen::MatrixXd vertices_A = model.reconstructFace(alpha, delta_zero);
    if (vertices_A.rows() == 0) {
        std::cerr << "FAIL: reconstructFace returned empty for mean shape" << std::endl;
        return 1;
    }

    int num_tested = std::min(3, model.num_expression_components);
    int pass_count = 0;
    double displacement_threshold = 1e-4;

    for (int j = 0; j < num_tested; ++j) {
        Eigen::VectorXd delta_perturbed = Eigen::VectorXd::Zero(model.num_expression_components);

        double sigma = 1.0;
        if (j < model.expression_stddev.size() && model.expression_stddev(j) > 1e-10) {
            sigma = model.expression_stddev(j);
        }
        delta_perturbed(j) = 3.0 * sigma;

        Eigen::MatrixXd vertices_B = model.reconstructFace(alpha, delta_perturbed);
        if (vertices_B.rows() != vertices_A.rows()) {
            std::cerr << "FAIL: Vertex count mismatch for delta[" << j << "]" << std::endl;
            return 1;
        }

        double total_disp = 0.0;
        double max_disp = 0.0;
        for (int v = 0; v < vertices_A.rows(); ++v) {
            double d = (vertices_B.row(v) - vertices_A.row(v)).norm();
            total_disp += d;
            if (d > max_disp) max_disp = d;
        }
        double mean_disp = total_disp / vertices_A.rows();

        std::cout << "  delta[" << j << "] = 3*sigma (" << 3.0 * sigma << "): "
                  << "mean_disp=" << mean_disp << ", max_disp=" << max_disp << std::endl;

        if (mean_disp > displacement_threshold) {
            std::cout << "    PASS (mean displacement > " << displacement_threshold << ")" << std::endl;
            pass_count++;
        } else {
            std::cerr << "    FAIL: expression component " << j
                      << " does not change mesh (mean_disp=" << mean_disp << ")" << std::endl;
        }
    }

    std::cout << "\n=== Expression Influence Test ===" << std::endl;
    std::cout << "Tested " << num_tested << " components, " << pass_count << " passed" << std::endl;

    if (pass_count == num_tested) {
        std::cout << "RESULT: PASS" << std::endl;
        return 0;
    } else {
        std::cerr << "RESULT: FAIL (" << (num_tested - pass_count) << " components had no effect)" << std::endl;
        return 1;
    }
}
