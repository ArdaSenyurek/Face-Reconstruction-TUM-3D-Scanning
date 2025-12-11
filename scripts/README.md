# Model Preparation Scripts

This directory contains scripts to help prepare PCA morphable face model files.

## Quick Start

### 1. Generate Dummy/Test Model

The easiest way to get started is to generate a dummy model for testing:

```bash
cd scripts
python3 prepare_model.py --generate-dummy --output ../data/model
```

This creates a test model in `data/model/` directory with:
- `mean_shape.bin`
- `identity_basis.bin`
- `expression_basis.bin`
- `identity_stddev.bin`
- `expression_stddev.bin`

### 2. Convert from NumPy Files

If you have model files in NumPy format:

```bash
python3 prepare_model.py --from-numpy /path/to/numpy/files --output ../data/model --format binary
```

Expected input files:
- `mean_shape.npy`
- `identity_basis.npy`
- `expression_basis.npy`
- `identity_stddev.npy`
- `expression_stddev.npy`

### 3. Convert Between Formats

Convert from binary to text or vice versa:

```bash
# Binary to text
python3 prepare_model.py --convert --input ../data/model --output ../data/model_text \
    --from-format bin --to-format txt

# Text to binary
python3 prepare_model.py --convert --input ../data/model_text --output ../data/model \
    --from-format txt --to-format bin
```

## Using Existing Models

### Basel Face Model (BFM)

If you have the Basel Face Model, you can convert it:

```python
# Example conversion script for BFM
import numpy as np
import os

def convert_bfm(bfm_path, output_dir):
    # Load BFM (example - adjust based on your BFM format)
    # BFM typically comes as .mat files or .h5 files
    
    # If .mat file (MATLAB format):
    from scipy.io import loadmat
    bfm = loadmat(os.path.join(bfm_path, '01_MorphableModel.mat'))
    
    # Extract data (adjust keys based on your BFM version)
    mean_shape = bfm['shapeMU'].flatten()  # Mean shape
    identity_basis = bfm['shapePC']  # Identity PCA basis
    identity_stddev = bfm['shapeEV'].flatten()  # Standard deviations
    
    # Expression model (if available)
    # expression_basis = bfm['expressionPC']
    # expression_stddev = bfm['expressionEV'].flatten()
    
    # Save using the prepare_model.py script or directly:
    os.makedirs(output_dir, exist_ok=True)
    mean_shape.astype(np.float64).tofile(os.path.join(output_dir, 'mean_shape.bin'))
    
    # Save matrix with header
    rows, cols = identity_basis.shape
    with open(os.path.join(output_dir, 'identity_basis.bin'), 'wb') as f:
        f.write(np.array([rows, cols], dtype=np.int32).tobytes())
        f.write(identity_basis.astype(np.float64).tobytes())
    
    identity_stddev.astype(np.float64).tofile(os.path.join(output_dir, 'identity_stddev.bin'))

convert_bfm('/path/to/bfm', '../data/model')
```

### Other Models

For other model formats (PyTorch, TensorFlow, etc.), you'll need to:

1. Load the model in Python
2. Extract the required arrays (mean_shape, basis vectors, stddevs)
3. Convert to NumPy arrays
4. Save using the prepare script

Example for PyTorch:

```python
import torch
import numpy as np

# Load PyTorch model
model = torch.load('model.pth', map_location='cpu')

# Extract tensors
mean_shape = model['mean_shape'].numpy().flatten()
identity_basis = model['identity_basis'].numpy()
identity_stddev = model['identity_stddev'].numpy()

# Save as .npy files first, then convert
np.save('mean_shape.npy', mean_shape)
np.save('identity_basis.npy', identity_basis)
np.save('identity_stddev.npy', identity_stddev)

# Then use prepare_model.py to convert to binary/text
```

## Model Requirements

Your model files should have the following structure:

### Mean Shape
- **Type**: Vector
- **Size**: `3*N` where `N` = number of vertices
- **Format**: `[x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]`

### Identity Basis
- **Type**: Matrix
- **Size**: `(3*N) × M_id` where `M_id` = number of identity components
- **Format**: Each column is a PCA basis vector

### Expression Basis
- **Type**: Matrix
- **Size**: `(3*N) × M_exp` where `M_exp` = number of expression components
- **Format**: Each column is a PCA basis vector

### Standard Deviations
- **Type**: Vector
- **Size**: Matching number of components
- **Format**: Standard deviations for each component

## Verifying Your Model

After preparing your model, verify it works:

```bash
# Test loading
cd ../build
./bin/test_real_data --model-dir ../data/model --output-mesh test.ply
```

If you get errors, check:
1. File paths are correct
2. File formats match (binary vs text)
3. Dimensions are consistent
4. Data types are correct (should be float64/double)
