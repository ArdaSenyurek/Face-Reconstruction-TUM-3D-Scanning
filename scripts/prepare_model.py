#!/usr/bin/env python3
"""
Script to prepare PCA morphable face model files for the face reconstruction project.

This script can:
1. Convert models from NumPy/PyTorch/TensorFlow formats
2. Generate dummy/test model data
3. Convert between binary and text formats
"""

import numpy as np
import os
import sys
import argparse


def save_binary_vector(filename, vector):
    """Save vector as binary file (raw double array)"""
    vector = np.asarray(vector, dtype=np.float64)
    vector.tofile(filename)
    print(f"Saved binary vector: {filename} ({vector.size} elements)")


def save_binary_matrix(filename, matrix, with_header=True):
    """Save matrix as binary file (with optional header)"""
    matrix = np.asarray(matrix, dtype=np.float64)
    
    with open(filename, 'wb') as f:
        if with_header:
            # Write header: rows and cols as int32
            rows, cols = matrix.shape
            f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        # Write matrix data
        f.write(matrix.tobytes())
    
    if with_header:
        print(f"Saved binary matrix with header: {filename} ({matrix.shape[0]}x{matrix.shape[1]})")
    else:
        print(f"Saved binary matrix (no header): {filename} ({matrix.shape[0]}x{matrix.shape[1]})")


def save_text_vector(filename, vector):
    """Save vector as text file"""
    vector = np.asarray(vector)
    np.savetxt(filename, vector, fmt='%.10f')
    print(f"Saved text vector: {filename} ({vector.size} elements)")


def save_text_matrix(filename, matrix):
    """Save matrix as text file"""
    matrix = np.asarray(matrix)
    np.savetxt(filename, matrix, fmt='%.10f')
    print(f"Saved text matrix: {filename} ({matrix.shape[0]}x{matrix.shape[1]})")


def generate_dummy_model(output_dir, num_vertices=53490, num_identity=199, num_expression=100):
    """
    Generate a dummy/test PCA model.
    
    Parameters:
    - num_vertices: Number of vertices in the model (Basel Face Model uses ~53490)
    - num_identity: Number of identity components
    - num_expression: Number of expression components
    """
    print(f"\nGenerating dummy model:")
    print(f"  Vertices: {num_vertices}")
    print(f"  Identity components: {num_identity}")
    print(f"  Expression components: {num_expression}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dim = 3 * num_vertices
    
    # Generate mean shape (simple face-like shape)
    mean_shape = np.zeros(dim)
    for i in range(num_vertices):
        # Create a simple ellipsoid-like shape
        angle = 2 * np.pi * i / num_vertices
        radius = 0.1 + 0.05 * np.cos(angle)
        mean_shape[3*i] = radius * np.cos(angle)  # x
        mean_shape[3*i+1] = radius * np.sin(angle)  # y
        mean_shape[3*i+2] = 0.0 + 0.02 * np.sin(2 * angle)  # z
    
    # Generate identity basis (random but normalized)
    identity_basis = np.random.randn(dim, num_identity).astype(np.float64)
    # Normalize each column
    identity_basis = identity_basis / np.linalg.norm(identity_basis, axis=0)
    # Scale by small values
    identity_basis *= 0.01
    
    # Generate expression basis
    expression_basis = np.random.randn(dim, num_expression).astype(np.float64)
    expression_basis = expression_basis / np.linalg.norm(expression_basis, axis=0)
    expression_basis *= 0.005
    
    # Generate standard deviations
    identity_stddev = np.random.rand(num_identity).astype(np.float64) * 0.1 + 0.05
    expression_stddev = np.random.rand(num_expression).astype(np.float64) * 0.05 + 0.02
    
    # Generate simple face connectivity (triangulation)
    # This is a simplified approach - creates a basic triangulation
    # For a proper mesh, you'd want Delaunay triangulation or structured grid
    print("\nGenerating face connectivity...")
    faces = generate_simple_faces(num_vertices)
    print(f"  Generated {len(faces)} faces")
    
    # Save binary files
    save_binary_vector(os.path.join(output_dir, "mean_shape.bin"), mean_shape)
    save_binary_matrix(os.path.join(output_dir, "identity_basis.bin"), identity_basis, with_header=True)
    save_binary_matrix(os.path.join(output_dir, "expression_basis.bin"), expression_basis, with_header=True)
    save_binary_vector(os.path.join(output_dir, "identity_stddev.bin"), identity_stddev)
    save_binary_vector(os.path.join(output_dir, "expression_stddev.bin"), expression_stddev)
    
    # Save faces (as integer matrix)
    faces_array = np.array(faces, dtype=np.int32)
    # Save binary with integer header and data
    with open(os.path.join(output_dir, "faces.bin"), 'wb') as f:
        rows, cols = faces_array.shape
        f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        f.write(faces_array.tobytes())
    print(f"Saved binary faces: {output_dir}/faces.bin ({faces_array.shape[0]}x{faces_array.shape[1]})")
    save_text_matrix(os.path.join(output_dir, "faces.txt"), faces_array)
    
    print(f"\nDummy model saved to: {output_dir}")
    return True


def generate_simple_faces(num_vertices):
    """
    Generate a simple face connectivity for testing.
    Creates a basic triangulation pattern.
    
    Note: This is a simplified approach. For real models, use proper
    triangulation or load face connectivity from the model data.
    """
    faces = []
    
    # Try to infer a grid-like structure
    # Estimate grid dimensions (assuming roughly square layout)
    grid_size = int(np.sqrt(num_vertices))
    if grid_size * grid_size != num_vertices:
        # Not a perfect square, try rectangle
        grid_size = int(np.sqrt(num_vertices * 2))
    
    # If we can create a grid, triangulate it
    if grid_size * grid_size <= num_vertices:
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                idx = i * grid_size + j
                if idx + grid_size + 1 < num_vertices:
                    # Create two triangles per quad
                    faces.append([idx, idx + 1, idx + grid_size])
                    faces.append([idx + 1, idx + grid_size + 1, idx + grid_size])
    
    # If no faces generated, create a minimal connectivity
    # (just connect first few vertices to form a simple shape)
    if len(faces) == 0 and num_vertices >= 3:
        # Create a simple fan triangulation
        for i in range(1, min(num_vertices - 1, 100)):  # Limit to avoid too many faces
            faces.append([0, i, i + 1])
    
    return faces


def convert_from_numpy(input_dir, output_dir, format='binary'):
    """
    Convert model from NumPy .npy files.
    
    Expected input files:
    - mean_shape.npy
    - identity_basis.npy
    - expression_basis.npy
    - identity_stddev.npy
    - expression_stddev.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_convert = [
        ('mean_shape.npy', 'mean_shape', True),
        ('identity_basis.npy', 'identity_basis', False),
        ('expression_basis.npy', 'expression_basis', False),
        ('identity_stddev.npy', 'identity_stddev', True),
        ('expression_stddev.npy', 'expression_stddev', True),
    ]
    
    for input_file, output_name, is_vector in files_to_convert:
        input_path = os.path.join(input_dir, input_file)
        if not os.path.exists(input_path):
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        data = np.load(input_path)
        
        if format == 'binary':
            if is_vector:
                save_binary_vector(os.path.join(output_dir, f"{output_name}.bin"), data)
            else:
                save_binary_matrix(os.path.join(output_dir, f"{output_name}.bin"), data, with_header=True)
        else:  # text
            if is_vector:
                save_text_vector(os.path.join(output_dir, f"{output_name}.txt"), data)
            else:
                save_text_matrix(os.path.join(output_dir, f"{output_name}.txt"), data)
    
    print(f"\nConversion complete. Files saved to: {output_dir}")


def convert_between_formats(input_dir, output_dir, from_format='binary', to_format='text'):
    """Convert model files between binary and text formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    vector_files = ['mean_shape', 'identity_stddev', 'expression_stddev']
    matrix_files = ['identity_basis', 'expression_basis']
    
    for filename in vector_files:
        input_path = os.path.join(input_dir, f"{filename}.{from_format}")
        output_path = os.path.join(output_dir, f"{filename}.{to_format}")
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        if from_format == 'binary':
            data = np.fromfile(input_path, dtype=np.float64)
        else:  # text
            data = np.loadtxt(input_path)
        
        if to_format == 'binary':
            save_binary_vector(output_path, data)
        else:  # text
            save_text_vector(output_path, data)
    
    for filename in matrix_files:
        input_path = os.path.join(input_dir, f"{filename}.{from_format}")
        output_path = os.path.join(output_dir, f"{filename}.{to_format}")
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        if from_format == 'bin':
            # Try reading with header first
            with open(input_path, 'rb') as f:
                header = np.frombuffer(f.read(8), dtype=np.int32)
                if header[0] > 0 and header[1] > 0 and header[0] < 1000000 and header[1] < 1000000:
                    rows, cols = header[0], header[1]
                    data = np.frombuffer(f.read(), dtype=np.float64).reshape(rows, cols)
                else:
                    # No header, need dimensions
                    print(f"Error: Cannot determine dimensions for {filename}.bin")
                    print("  Please specify dimensions manually or use --from-numpy")
                    continue
        else:  # text
            data = np.loadtxt(input_path)
        
        if to_format == 'bin':
            save_binary_matrix(output_path, data, with_header=True)
        else:  # text
            save_text_matrix(output_path, data)
    
    print(f"\nConversion complete. Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare PCA morphable face model files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dummy/test model
  python prepare_model.py --generate-dummy --output data/model
  
  # Convert from NumPy files
  python prepare_model.py --from-numpy input_dir --output data/model --format binary
  
  # Convert between formats
  python prepare_model.py --convert --input input_dir --output output_dir --from-format bin --to-format txt
        """
    )
    
    parser.add_argument('--generate-dummy', action='store_true',
                       help='Generate dummy/test model data')
    parser.add_argument('--from-numpy', type=str,
                       help='Convert from NumPy .npy files in this directory')
    parser.add_argument('--convert', action='store_true',
                       help='Convert between binary/text formats')
    parser.add_argument('--input', type=str,
                       help='Input directory for conversion')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--format', choices=['binary', 'text'], default='binary',
                       help='Output format (default: binary)')
    parser.add_argument('--from-format', choices=['bin', 'txt'], default='bin',
                       help='Input format for conversion (default: bin)')
    parser.add_argument('--to-format', choices=['bin', 'txt'], default='txt',
                       help='Output format for conversion (default: txt)')
    
    # Model parameters for dummy generation
    parser.add_argument('--num-vertices', type=int, default=53490,
                       help='Number of vertices for dummy model (default: 53490)')
    parser.add_argument('--num-identity', type=int, default=199,
                       help='Number of identity components (default: 199)')
    parser.add_argument('--num-expression', type=int, default=100,
                       help='Number of expression components (default: 100)')
    
    args = parser.parse_args()
    
    if args.generate_dummy:
        generate_dummy_model(
            args.output,
            args.num_vertices,
            args.num_identity,
            args.num_expression
        )
    elif args.from_numpy:
        convert_from_numpy(args.from_numpy, args.output, args.format)
    elif args.convert:
        if not args.input:
            parser.error("--convert requires --input")
        convert_between_formats(args.input, args.output, args.from_format, args.to_format)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
