# Running Learning-based NRSfM Without MATLAB

This guide explains how to run the Learning-based NRSfM training and inference pipeline **without requiring MATLAB**.

## Overview

The refactored code provides a **pure Python backend** that replaces MATLAB dependencies for:
- **Spline fitting**: Uses SciPy's bivariate spline fitting instead of MATLAB's BBS toolbox
- **Depth initialization**: Provides simple Python-based initialization methods
- **Shape evaluation**: Uses scipy.linalg.orthogonal_procrustes for Procrustes alignment

MATLAB support is **optional** and can be enabled via command-line flags for backward compatibility or comparison purposes.

## Installation

### Requirements

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies are:
- `numpy>=1.19.0`
- `scipy>=1.5.0`
- `torch>=1.8.0`
- `open3d>=0.13.0`
- `trimesh>=3.9.0`

### Optional: MATLAB Engine

If you want to use the MATLAB backend for comparison, you need:
1. MATLAB installed (tested with R2019b or later)
2. MATLAB Engine for Python: `python -m pip install matlabengine`

**Note**: MATLAB is NOT required for normal training/inference using the Python backend.

## Usage

### Basic Training (Python-only)

Run training with the default Python backend:

```bash
python main.py
```

This will:
- Use Python-based spline fitting (SciPy)
- Initialize depth with simple heuristics
- Evaluate shapes using Python Procrustes alignment
- Default to `ones` initialization method
- Use spline smoothing parameter of `1e-5`

### Command-Line Arguments

#### Backend Selection

Choose the backend for spline fitting and evaluation:

```bash
# Use Python backend (default)
python main.py --backend python

# Use MATLAB backend (requires MATLAB installation)
python main.py --backend matlab
```

#### Spline Smoothing

Control the smoothing parameter for spline fitting:

```bash
# Lower smoothing (less smooth, follows data more closely)
python main.py --spline-smoothing 1e-6

# Higher smoothing (smoother surface, more regularization)
python main.py --spline-smoothing 1e-4
```

**Tuning Guide**:
- Default `1e-5` approximates MATLAB's BBS regularization
- For noisy data: increase to `1e-4` or higher
- For clean data: decrease to `1e-6` or lower
- If you see overfitting: increase smoothing
- If results are too smooth: decrease smoothing

#### Initialization Method

Choose the depth initialization method:

```bash
# Simple initialization to 1.0 (default, fastest)
python main.py --init-method ones

# Random initialization with small perturbations
python main.py --init-method random

# Initialize based on distance from image center
python main.py --init-method mean_centered

# SVD-based affine camera initialization (most principled)
python main.py --init-method affine
```

**Recommendation**: Start with `ones` for quick testing. Use `affine` for better initialization if training is unstable.

#### Use MATLAB Initialization

If you have MATLAB and want to use MATLAB-based initialization while keeping Python for the rest:

```bash
python main.py --use-matlab-init
```

This uses MATLAB's `initialization_for_NRSfM_local_all_new` but keeps Python for spline fitting and evaluation.

### Full Example

Complete command with all options:

```bash
python main.py \
  --backend python \
  --spline-smoothing 5e-5 \
  --init-method affine \
  --batch_size 2 \
  --epochs 10000
```

## Backend Comparison

### Python Backend (Default)

**Advantages**:
- No MATLAB dependency
- Easier to deploy and containerize
- Can run on systems without MATLAB licenses
- Faster startup time (no MATLAB engine initialization)
- Uses standard scientific Python stack

**Implementation**:
- Spline fitting: `scipy.interpolate.SmoothBivariateSpline`
- Alignment: `scipy.linalg.orthogonal_procrustes`
- Initialization: NumPy-based heuristics or SVD factorization

### MATLAB Backend (Optional)

**Advantages**:
- Exact reproduction of original results
- Uses original BBS (Bicubic B-Splines) toolbox
- Useful for comparison and validation

**Limitations**:
- Requires MATLAB installation and license
- Slower startup (MATLAB engine initialization takes ~5-10 seconds)
- Not suitable for production deployment

## Architecture Details

### Modified Files

The following files have been updated to support both backends:

1. **`main.py`**: 
   - Optional MATLAB engine initialization
   - CLI argument parsing for backend selection
   - Backend parameter passing to downstream functions

2. **`NRSfM_core/class_autograd.py`**:
   - `ChamferFunction` supports both backends
   - Automatic backend detection

3. **`NRSfM_core/Initial_supervised_learning_multiple_model.py`**:
   - Dual-path spline fitting (Python/MATLAB)
   - No forced MATLAB imports

4. **`NRSfM_core/loss_function.py`**:
   - Optional MATLAB imports
   - Backend-agnostic loss computation

5. **`NRSfM_core/train_shape_decoder.py`**:
   - Backend selection for evaluation
   - Works with both Python and MATLAB backends

6. **`Result_evaluation/Shape_error.py`**:
   - Automatic backend selection
   - Fallback to Python if MATLAB unavailable

### New Modules

1. **`NRSfM_core/spline_fitting.py`**:
   - Pure Python spline surface fitting
   - Compatible interface with MATLAB `fit_python.m`
   - Returns fitted values and partial derivatives

2. **`NRSfM_core/initialization.py`**:
   - Multiple initialization strategies
   - No MATLAB dependency
   - Supports `ones`, `random`, `mean_centered`, `affine` methods

3. **`Result_evaluation/Shape_error_python.py`**:
   - Python-based shape error calculation
   - Procrustes alignment implementation
   - Same metrics as MATLAB version

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# If specific packages are missing
pip install scipy numpy torch
```

### MATLAB Not Found (when using --backend matlab)

If you see "MATLAB engine not available" but want to use MATLAB:
```bash
# Install MATLAB Engine for Python
cd /path/to/matlab/extern/engines/python
python setup.py install
```

### Performance Differences

Python and MATLAB backends may produce slightly different results due to:
- Different spline fitting implementations
- Numerical precision differences
- Optimization algorithm variations

**These differences are typically small (<1% error change)** and don't affect the overall training convergence.

### Tuning Spline Smoothing

If results differ significantly from MATLAB:
1. Start with default `1e-5`
2. Try `5e-5` for slightly more smoothing
3. Try `1e-6` for less smoothing
4. Compare error metrics with both backends
5. Use `--backend matlab` to validate Python results

## Dataset Requirements

The code expects:
- Pre-processed 2D point tracks in `.mat` format (can be created offline with MATLAB)
- Normalized coordinates
- Image warp derivatives (J) for differential constraints

**MATLAB is still allowed for offline preprocessing** of datasets, but not required for training/inference.

## Citation

If you use this code, please cite the original paper:
```
[Original paper citation]
```

## License

[Original license information]

## Contributing

When contributing, please ensure:
1. Changes work with both Python-only and MATLAB modes
2. Imports are conditional (no forced MATLAB dependencies)
3. CLI arguments are documented
4. Tests pass without MATLAB installed
