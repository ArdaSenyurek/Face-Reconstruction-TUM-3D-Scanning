/**
 * Quick test to export mean shape for comparison
 */

#include "model/MorphableModel.h"
#include <iostream>

using namespace face_reconstruction;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <output_ply>" << std::endl;
        return 1;
    }
    
    std::string model_dir = argv[1];
    std::string output_ply = argv[2];
    
    MorphableModel model;
    if (!model.loadFromFiles(model_dir)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    Eigen::MatrixXd mean_shape = model.getMeanShapeMatrix();
    
    std::cout << "Mean shape: " << mean_shape.rows() << " vertices" << std::endl;
    Eigen::Vector3d min_vals = mean_shape.colwise().minCoeff();
    Eigen::Vector3d max_vals = mean_shape.colwise().maxCoeff();
    std::cout << "Bounds: X:[" << min_vals.x() << ", " << max_vals.x() << "], "
              << "Y:[" << min_vals.y() << ", " << max_vals.y() << "], "
              << "Z:[" << min_vals.z() << ", " << max_vals.z() << "]" << std::endl;
    
    if (model.saveMeshPLY(mean_shape, output_ply)) {
        std::cout << "Saved mean shape to: " << output_ply << std::endl;
    } else {
        std::cerr << "Failed to save" << std::endl;
        return 1;
    }
    
    return 0;
}

