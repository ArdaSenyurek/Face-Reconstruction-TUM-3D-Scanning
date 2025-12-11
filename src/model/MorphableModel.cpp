#include "model/MorphableModel.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstdint>

namespace face_reconstruction {

// Helper function to load binary file (raw double array)
bool loadBinaryVector(const std::string& filepath, Eigen::VectorXd& vec, size_t expected_size = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate number of elements (assuming double precision)
    size_t num_elements = file_size / sizeof(double);
    if (expected_size > 0 && num_elements != expected_size) {
        std::cerr << "Warning: Expected " << expected_size << " elements, found " << num_elements << std::endl;
    }
    
    vec.resize(num_elements);
    file.read(reinterpret_cast<char*>(vec.data()), file_size);
    
    return !file.fail();
}

// Helper function to load binary matrix file
bool loadBinaryMatrix(const std::string& filepath, Eigen::MatrixXd& mat, 
                      size_t expected_rows = 0, size_t expected_cols = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Get file size first
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Try to read header (optional: rows, cols as int32)
    int32_t header_rows = 0, header_cols = 0;
    std::streampos start_pos = file.tellg();
    file.read(reinterpret_cast<char*>(&header_rows), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&header_cols), sizeof(int32_t));
    
    int32_t rows = header_rows;
    int32_t cols = header_cols;
    
    // Check if header values are reasonable
    bool has_header = (rows > 0 && cols > 0 && rows < 1000000 && cols < 1000000);
    
    if (!has_header) {
        // No header, go back to start
        file.seekg(start_pos);
        size_t num_elements = file_size / sizeof(double);
        
        // Try to infer dimensions from expected values
        if (expected_rows > 0 && expected_cols > 0) {
            rows = expected_rows;
            cols = expected_cols;
        } else if (expected_rows > 0) {
            rows = expected_rows;
            cols = num_elements / expected_rows;
            if (rows * cols * sizeof(double) != file_size) {
                return false;  // Size doesn't match
            }
        } else {
            return false;  // Can't load matrix without dimensions
        }
    }
    
    mat.resize(rows, cols);
    file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
    
    return !file.fail() && static_cast<size_t>(file.tellg()) == file_size;
}

// Helper function to load text file (space-separated values)
bool loadTextVector(const std::string& filepath, Eigen::VectorXd& vec) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    std::vector<double> values;
    double val;
    while (file >> val) {
        values.push_back(val);
    }
    
    if (values.empty()) {
        return false;
    }
    
    vec.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        vec(i) = values[i];
    }
    
    return true;
}

// Helper function to load text matrix (one row per line, space-separated)
bool loadTextMatrix(const std::string& filepath, Eigen::MatrixXd& mat) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    std::vector<std::vector<double>> rows;
    std::string line;
    size_t num_cols = 0;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        
        while (iss >> val) {
            row.push_back(val);
        }
        
        if (!row.empty()) {
            if (num_cols == 0) {
                num_cols = row.size();
            } else if (row.size() != num_cols) {
                std::cerr << "Warning: Inconsistent row sizes in matrix file" << std::endl;
            }
            rows.push_back(row);
        }
    }
    
    if (rows.empty() || num_cols == 0) {
        return false;
    }
    
    mat.resize(rows.size(), num_cols);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            mat(i, j) = rows[i][j];
        }
    }
    
    return true;
}

bool MorphableModel::loadFromFiles(const std::string& base_path) {
    // Ensure base_path ends with '/'
    std::string path = base_path;
    if (!path.empty() && path.back() != '/') {
        path += "/";
    }
    
    // Try to load mean shape (required)
    // Try binary first, then text
    if (!loadBinaryVector(path + "mean_shape.bin", mean_shape)) {
        if (!loadTextVector(path + "mean_shape.txt", mean_shape)) {
            std::cerr << "Error: Could not load mean_shape from " << path << std::endl;
            return false;
        }
    }
    
    // Calculate number of vertices from mean_shape dimension
    if (mean_shape.size() % 3 != 0) {
        std::cerr << "Error: Mean shape dimension must be divisible by 3" << std::endl;
        return false;
    }
    num_vertices = mean_shape.size() / 3;
    int dim = mean_shape.size();
    
    // Load identity basis
    if (!loadBinaryMatrix(path + "identity_basis.bin", identity_basis, dim, 0)) {
        if (!loadTextMatrix(path + "identity_basis.txt", identity_basis)) {
            std::cerr << "Warning: Could not load identity_basis, setting to empty" << std::endl;
            identity_basis = Eigen::MatrixXd::Zero(dim, 0);
        }
    }
    num_identity_components = identity_basis.cols();
    
    // Load expression basis
    if (!loadBinaryMatrix(path + "expression_basis.bin", expression_basis, dim, 0)) {
        if (!loadTextMatrix(path + "expression_basis.txt", expression_basis)) {
            std::cerr << "Warning: Could not load expression_basis, setting to empty" << std::endl;
            expression_basis = Eigen::MatrixXd::Zero(dim, 0);
        }
    }
    num_expression_components = expression_basis.cols();
    
    // Verify basis dimensions
    if (identity_basis.rows() != dim) {
        std::cerr << "Error: Identity basis rows (" << identity_basis.rows() 
                  << ") must match mean_shape size (" << dim << ")" << std::endl;
        return false;
    }
    if (expression_basis.rows() != dim) {
        std::cerr << "Error: Expression basis rows (" << expression_basis.rows() 
                  << ") must match mean_shape size (" << dim << ")" << std::endl;
        return false;
    }
    
    // Load standard deviations
    if (num_identity_components > 0) {
        if (!loadBinaryVector(path + "identity_stddev.bin", identity_stddev, num_identity_components)) {
            if (!loadTextVector(path + "identity_stddev.txt", identity_stddev)) {
                std::cerr << "Warning: Could not load identity_stddev, setting to ones" << std::endl;
                identity_stddev = Eigen::VectorXd::Ones(num_identity_components);
            }
        }
        if (identity_stddev.size() != num_identity_components) {
            std::cerr << "Warning: identity_stddev size mismatch, resizing" << std::endl;
            identity_stddev = Eigen::VectorXd::Ones(num_identity_components);
        }
    }
    
    if (num_expression_components > 0) {
        if (!loadBinaryVector(path + "expression_stddev.bin", expression_stddev, num_expression_components)) {
            if (!loadTextVector(path + "expression_stddev.txt", expression_stddev)) {
                std::cerr << "Warning: Could not load expression_stddev, setting to ones" << std::endl;
                expression_stddev = Eigen::VectorXd::Ones(num_expression_components);
            }
        }
        if (expression_stddev.size() != num_expression_components) {
            std::cerr << "Warning: expression_stddev size mismatch, resizing" << std::endl;
            expression_stddev = Eigen::VectorXd::Ones(num_expression_components);
        }
    }
    
    // Load face connectivity (optional)
    // Faces are stored as integers, so we need a special loader
    std::ifstream faces_file_bin(path + "faces.bin", std::ios::binary);
    if (faces_file_bin.is_open()) {
        int32_t rows = 0, cols = 0;
        faces_file_bin.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
        faces_file_bin.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
        
        if (rows > 0 && cols == 3 && rows < 10000000) {
            // Read integer data
            std::vector<int32_t> face_data(rows * cols);
            faces_file_bin.read(reinterpret_cast<char*>(face_data.data()), rows * cols * sizeof(int32_t));
            if (!faces_file_bin.fail()) {
                // Copy to Eigen matrix
                faces.resize(rows, cols);
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        faces(i, j) = face_data[i * cols + j];
                    }
                }
                std::cout << "Loaded " << faces.rows() << " faces from binary file" << std::endl;
            } else {
                faces.resize(0, 0);
            }
        }
        faces_file_bin.close();
    }
    
    // If binary load failed, try text
    if (faces.rows() == 0) {
        std::ifstream faces_file_txt(path + "faces.txt");
        if (faces_file_txt.is_open()) {
            std::vector<std::vector<int>> face_rows;
            std::string line;
            while (std::getline(faces_file_txt, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                std::vector<int> face;
                double val;  // Read as double first (text file may have decimals)
                while (iss >> val) {
                    face.push_back(static_cast<int>(val));  // Convert to int
                }
                if (face.size() == 3) {
                    face_rows.push_back(face);
                }
            }
            if (!face_rows.empty()) {
                faces.resize(face_rows.size(), 3);
                for (size_t i = 0; i < face_rows.size(); ++i) {
                    faces(i, 0) = face_rows[i][0];
                    faces(i, 1) = face_rows[i][1];
                    faces(i, 2) = face_rows[i][2];
                }
                std::cout << "Loaded " << faces.rows() << " faces from text file" << std::endl;
            }
            faces_file_txt.close();
        }
    }
    
    if (faces.rows() == 0) {
        std::cout << "Note: No face connectivity file found. Mesh will have vertices only." << std::endl;
    }
    
    std::cout << "Successfully loaded PCA model from " << path << std::endl;
    return true;
}

Eigen::MatrixXd MorphableModel::reconstructFace(const Eigen::VectorXd& identity_coeffs,
                                                  const Eigen::VectorXd& expression_coeffs) const {
    if (!isValid()) {
        throw std::runtime_error("Model is not valid");
    }
    
    if (identity_coeffs.size() != num_identity_components) {
        throw std::runtime_error("Identity coefficients size mismatch");
    }
    
    if (expression_coeffs.size() != num_expression_components) {
        throw std::runtime_error("Expression coefficients size mismatch");
    }
    
    // Reconstruct: mean + U_identity * alpha + U_expression * delta
    Eigen::VectorXd vertices = mean_shape;
    
    if (num_identity_components > 0) {
        vertices += identity_basis * identity_coeffs;
    }
    
    if (num_expression_components > 0) {
        vertices += expression_basis * expression_coeffs;
    }
    
    // Reshape from [x1, y1, z1, x2, y2, z2, ...] to N x 3
    Eigen::MatrixXd vertices_matrix(num_vertices, 3);
    for (int i = 0; i < num_vertices; ++i) {
        vertices_matrix(i, 0) = vertices(3 * i);
        vertices_matrix(i, 1) = vertices(3 * i + 1);
        vertices_matrix(i, 2) = vertices(3 * i + 2);
    }
    
    return vertices_matrix;
}

Eigen::MatrixXd MorphableModel::getMeanShapeMatrix() const {
    if (!isValid()) {
        return Eigen::MatrixXd();
    }
    
    Eigen::MatrixXd mean_matrix(num_vertices, 3);
    for (int i = 0; i < num_vertices; ++i) {
        mean_matrix(i, 0) = mean_shape(3 * i);
        mean_matrix(i, 1) = mean_shape(3 * i + 1);
        mean_matrix(i, 2) = mean_shape(3 * i + 2);
    }
    
    return mean_matrix;
}

bool MorphableModel::isValid() const {
    if (num_vertices <= 0) return false;
    if (mean_shape.size() != 3 * num_vertices) return false;
    if (identity_basis.rows() != 3 * num_vertices) return false;
    if (expression_basis.rows() != 3 * num_vertices) return false;
    if (identity_stddev.size() != num_identity_components) return false;
    if (expression_stddev.size() != num_expression_components) return false;
    return true;
}

void MorphableModel::printStats() const {
    std::cout << "=== Morphable Model Statistics ===" << std::endl;
    std::cout << "Number of vertices: " << num_vertices << std::endl;
    std::cout << "Identity components: " << num_identity_components << std::endl;
    std::cout << "Expression components: " << num_expression_components << std::endl;
    std::cout << "Mean shape dimension: " << mean_shape.size() << std::endl;
    std::cout << "Valid: " << (isValid() ? "Yes" : "No") << std::endl;
    std::cout << "===================================" << std::endl;
}

bool MorphableModel::saveMeshPLY(const Eigen::MatrixXd& vertices, 
                                  const std::string& filepath,
                                  const Eigen::MatrixXi& faces_in) const {
    // Use provided faces or model's faces
    const Eigen::MatrixXi& faces_to_use = (faces_in.rows() > 0) ? faces_in : faces;
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << vertices.rows() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    
    if (faces_to_use.rows() > 0) {
        file << "element face " << faces_to_use.rows() << "\n";
        file << "property list uchar int vertex_indices\n";
    }
    
    file << "end_header\n";
    
    // Write vertices
    for (int i = 0; i < vertices.rows(); ++i) {
        file << std::fixed << std::setprecision(6)
             << vertices(i, 0) << " " 
             << vertices(i, 1) << " " 
             << vertices(i, 2) << "\n";
    }
    
    // Write faces
    if (faces_to_use.rows() > 0) {
        for (int i = 0; i < faces_to_use.rows(); ++i) {
            file << "3 " << faces_to_use(i, 0) << " " 
                 << faces_to_use(i, 1) << " " 
                 << faces_to_use(i, 2) << "\n";
        }
    }
    
    file.close();
    return true;
}

bool MorphableModel::saveMeshOBJ(const Eigen::MatrixXd& vertices,
                                  const std::string& filepath,
                                  const Eigen::MatrixXi& faces_in) const {
    // Use provided faces or model's faces
    const Eigen::MatrixXi& faces_to_use = (faces_in.rows() > 0) ? faces_in : faces;
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write vertices
    for (int i = 0; i < vertices.rows(); ++i) {
        file << std::fixed << std::setprecision(6)
             << "v " << vertices(i, 0) << " " 
             << vertices(i, 1) << " " 
             << vertices(i, 2) << "\n";
    }
    
    // Write faces (OBJ uses 1-based indexing)
    if (faces_to_use.rows() > 0) {
        for (int i = 0; i < faces_to_use.rows(); ++i) {
            file << "f " << (faces_to_use(i, 0) + 1) << " " 
                 << (faces_to_use(i, 1) + 1) << " " 
                 << (faces_to_use(i, 2) + 1) << "\n";
        }
    }
    
    file.close();
    return true;
}


} // namespace face_reconstruction
