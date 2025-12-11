#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace face_reconstruction {

/**
 * PCA-based 3D Morphable Face Model
 * 
 * Face vertices = mean_shape + U_identity * alpha + U_expression * delta
 * where:
 *   - mean_shape: Mean 3D face shape (3N x 1, N = number of vertices)
 *   - U_identity: Identity basis vectors (3N x num_identity_components)
 *   - U_expression: Expression basis vectors (3N x num_expression_components)
 *   - alpha: Identity coefficients
 *   - delta: Expression coefficients
 */
struct MorphableModel {
    // Mean shape: flattened [x1, y1, z1, x2, y2, z2, ...]
    Eigen::VectorXd mean_shape;
    
    // Identity basis (shape PCA components)
    Eigen::MatrixXd identity_basis;  // Each column is a basis vector
    
    // Expression basis (expression PCA components)
    Eigen::MatrixXd expression_basis;
    
    // Standard deviations for identity and expression components
    Eigen::VectorXd identity_stddev;
    Eigen::VectorXd expression_stddev;
    
    // Number of vertices
    int num_vertices = 0;
    
    // Number of identity and expression components
    int num_identity_components = 0;
    int num_expression_components = 0;
    
    // Face connectivity (triangles) - F x 3 matrix, each row is [v1, v2, v3]
    Eigen::MatrixXi faces;
    
    MorphableModel() = default;
    
    /**
     * Load PCA model from files
     * Expected files:
     *   - mean_shape.bin or mean_shape.txt
     *   - identity_basis.bin or identity_basis.txt
     *   - expression_basis.bin or expression_basis.txt
     *   - identity_stddev.bin or identity_stddev.txt
     *   - expression_stddev.bin or expression_stddev.txt
     *   - faces.bin or faces.txt (optional - face connectivity)
     */
    bool loadFromFiles(const std::string& base_path);
    
    /**
     * Reconstruct face mesh from coefficients
     * @param identity_coeffs Identity coefficients (alpha)
     * @param expression_coeffs Expression coefficients (delta)
     * @return Reconstructed vertices (N x 3 matrix)
     */
    Eigen::MatrixXd reconstructFace(const Eigen::VectorXd& identity_coeffs,
                                     const Eigen::VectorXd& expression_coeffs) const;
    
    /**
     * Get mean shape as N x 3 matrix
     */
    Eigen::MatrixXd getMeanShapeMatrix() const;
    
    /**
     * Check if model is valid (loaded and has correct dimensions)
     */
    bool isValid() const;
    
    /**
     * Print model statistics
     */
    void printStats() const;
    
    /**
     * Export reconstructed mesh to PLY file
     * @param vertices N x 3 matrix of vertices
     * @param filepath Output PLY file path
     * @param faces Optional face indices (F x 3). If empty, uses model.faces if available.
     * @return true if successful
     */
    bool saveMeshPLY(const Eigen::MatrixXd& vertices, 
                     const std::string& filepath,
                     const Eigen::MatrixXi& faces = Eigen::MatrixXi()) const;
    
    /**
     * Export reconstructed mesh to OBJ file
     * @param vertices N x 3 matrix of vertices
     * @param filepath Output OBJ file path
     * @param faces Optional face indices (F x 3). If empty, uses model.faces if available.
     * @return true if successful
     */
    bool saveMeshOBJ(const Eigen::MatrixXd& vertices,
                     const std::string& filepath,
                     const Eigen::MatrixXi& faces = Eigen::MatrixXi()) const;
};

} // namespace face_reconstruction
