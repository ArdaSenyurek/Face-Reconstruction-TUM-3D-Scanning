/**
 * Coefficient Clipping Test
 *
 * Verifies that per-coefficient sigma clipping works correctly:
 * coefficients are clamped to +/- 3*sigma after applyUpdate().
 *
 * Usage:
 *   build/bin/test_coefficient_clipping [--model-dir <path>]
 *
 * If --model-dir is provided, uses real stddev. Otherwise uses synthetic values.
 */

#include "optimization/Parameters.h"
#include "model/MorphableModel.h"
#include <iostream>
#include <string>
#include <cmath>

using namespace face_reconstruction;

static constexpr double CLIP_SIGMA = 3.0;

bool checkClipping(const Eigen::VectorXd& coeffs,
                   const Eigen::VectorXd& stddev,
                   const std::string& name) {
    bool ok = true;
    for (int i = 0; i < coeffs.size() && i < stddev.size(); ++i) {
        double limit = CLIP_SIGMA * stddev(i);
        if (limit <= 0) continue;
        if (std::abs(coeffs(i)) > limit + 1e-12) {
            std::cerr << "  FAIL: " << name << "[" << i << "] = " << coeffs(i)
                      << " exceeds limit " << limit << std::endl;
            ok = false;
        }
    }
    return ok;
}

int main(int argc, char* argv[]) {
    std::string model_dir;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        }
    }

    int num_id = 10;
    int num_expr = 10;
    Eigen::VectorXd id_stddev, expr_stddev;

    if (!model_dir.empty()) {
        MorphableModel model;
        if (model.loadFromFiles(model_dir) && model.isValid()) {
            num_id = model.num_identity_components;
            num_expr = model.num_expression_components;
            id_stddev = model.identity_stddev;
            expr_stddev = model.expression_stddev;
            std::cout << "Using model stddev (id=" << num_id << ", expr=" << num_expr << ")" << std::endl;
        } else {
            std::cerr << "Warning: Could not load model, using synthetic stddev" << std::endl;
        }
    }

    if (id_stddev.size() == 0) {
        id_stddev = Eigen::VectorXd::Constant(num_id, 100.0);
        expr_stddev = Eigen::VectorXd::Constant(num_expr, 50.0);
        std::cout << "Using synthetic stddev (id sigma=100, expr sigma=50)" << std::endl;
    }

    // Test 1: Large coefficients should be clipped
    std::cout << "\n--- Test 1: Clipping large coefficients ---" << std::endl;
    {
        OptimizationParams params(num_id, num_expr);
        params.identity_stddev = id_stddev;
        params.expression_stddev = expr_stddev;

        // Set coefficients to 10 * sigma (way over 3 * sigma limit)
        for (int i = 0; i < num_id && i < id_stddev.size(); ++i)
            params.alpha(i) = 10.0 * id_stddev(i);
        for (int i = 0; i < num_expr && i < expr_stddev.size(); ++i)
            params.delta(i) = -10.0 * expr_stddev(i);

        // Apply zero update (clipping still fires)
        Eigen::VectorXd zero_delta = Eigen::VectorXd::Zero(params.numParameters());
        params.applyUpdate(zero_delta);

        bool alpha_ok = checkClipping(params.alpha, id_stddev, "alpha");
        bool delta_ok = checkClipping(params.delta, expr_stddev, "delta");

        if (alpha_ok && delta_ok) {
            std::cout << "  PASS: All coefficients clipped to +/- 3*sigma" << std::endl;
        } else {
            std::cerr << "  FAIL: Some coefficients exceed limits" << std::endl;
            return 1;
        }
    }

    // Test 2: Small coefficients should NOT be clipped
    std::cout << "\n--- Test 2: Small coefficients unchanged ---" << std::endl;
    {
        OptimizationParams params(num_id, num_expr);
        params.identity_stddev = id_stddev;
        params.expression_stddev = expr_stddev;

        for (int i = 0; i < num_id && i < id_stddev.size(); ++i)
            params.alpha(i) = 1.0 * id_stddev(i);
        for (int i = 0; i < num_expr && i < expr_stddev.size(); ++i)
            params.delta(i) = -0.5 * expr_stddev(i);

        Eigen::VectorXd alpha_before = params.alpha;
        Eigen::VectorXd delta_before = params.delta;

        Eigen::VectorXd zero_delta = Eigen::VectorXd::Zero(params.numParameters());
        params.applyUpdate(zero_delta);

        bool unchanged = true;
        for (int i = 0; i < num_id; ++i) {
            if (std::abs(params.alpha(i) - alpha_before(i)) > 1e-12) {
                std::cerr << "  FAIL: alpha[" << i << "] changed from " << alpha_before(i)
                          << " to " << params.alpha(i) << std::endl;
                unchanged = false;
            }
        }
        for (int i = 0; i < num_expr; ++i) {
            if (std::abs(params.delta(i) - delta_before(i)) > 1e-12) {
                std::cerr << "  FAIL: delta[" << i << "] changed from " << delta_before(i)
                          << " to " << params.delta(i) << std::endl;
                unchanged = false;
            }
        }

        if (unchanged) {
            std::cout << "  PASS: Small coefficients not affected by clipping" << std::endl;
        } else {
            std::cerr << "  FAIL: Clipping modified small coefficients" << std::endl;
            return 1;
        }
    }

    // Test 3: Exact boundary
    std::cout << "\n--- Test 3: Exact boundary (3*sigma) ---" << std::endl;
    {
        OptimizationParams params(num_id, num_expr);
        params.identity_stddev = id_stddev;
        params.expression_stddev = expr_stddev;

        for (int i = 0; i < num_id && i < id_stddev.size(); ++i)
            params.alpha(i) = CLIP_SIGMA * id_stddev(i);

        Eigen::VectorXd zero_delta = Eigen::VectorXd::Zero(params.numParameters());
        params.applyUpdate(zero_delta);

        bool ok = true;
        for (int i = 0; i < num_id && i < id_stddev.size(); ++i) {
            double expected = CLIP_SIGMA * id_stddev(i);
            if (std::abs(params.alpha(i) - expected) > 1e-12) {
                std::cerr << "  FAIL: alpha[" << i << "] = " << params.alpha(i)
                          << " != expected " << expected << std::endl;
                ok = false;
            }
        }
        if (ok) {
            std::cout << "  PASS: Coefficients at boundary remain unchanged" << std::endl;
        } else {
            return 1;
        }
    }

    std::cout << "\n=== Coefficient Clipping Test ===" << std::endl;
    std::cout << "RESULT: PASS" << std::endl;
    return 0;
}
