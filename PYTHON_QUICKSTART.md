# Quick Start Guide - Python-Only Mode

## Overview

This repository now supports **MATLAB-free operation** for training and inference. MATLAB is completely optional and only needed for visualization features.

## Installation

### Python-Only Mode (Recommended)

```bash
# Install Python dependencies
pip install -r requirements.txt

# That's it! You can now run the code without MATLAB.
```

### With MATLAB (Optional, for visualization)

If you want to use MATLAB visualization features:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install MATLAB Engine for Python
cd "$(matlab -batch 'disp(matlabroot)')/extern/engines/python"
python setup.py install
```

## Running the Code

### Single Dataset Training (Python-Only)

```bash
python main.py --epochs 10 --batch_size 2
```

The code will automatically:
- Detect that MATLAB is not available
- Use Python implementations for all operations
- Print status messages about MATLAB availability

### Expected Output

```
MATLAB engine is NOT available. Running in Python-only mode.
Using Python initialization...
```

## Key Features

✅ **No MATLAB Required** - Complete training/inference pipeline in Python  
✅ **Automatic Fallback** - Uses MATLAB if available, Python otherwise  
✅ **Drop-in Replacement** - Same API, different backend  
✅ **Tested** - All core functions verified to work without MATLAB

## What's Changed?

### New Python Implementations

- **B-spline fitting** (`NRSfM_core/spline_fitting.py`)
  - Replaces MATLAB `fit_python.m`
  - Uses SciPy's bicubic B-spline with bending regularization
  
- **Initialization** (`NRSfM_core/initialization.py`)
  - Replaces MATLAB `initialization_for_NRSfM_local_all_new.m`
  - Least-squares depth estimation

- **Shape error** (Python mode in `Result_evaluation/Shape_error.py`)
  - Procrustes alignment for error calculation
  - No MATLAB needed for evaluation

### Modified Files

All files that previously required MATLAB now have optional MATLAB support:
- `main.py` - Optional MATLAB engine
- `class_autograd.py` - Python spline fitting
- `Initial_supervised_learning_multiple_model.py` - Python backend
- `loss_function.py` - Optional MATLAB import
- `train_shape_decoder.py` - Works with or without MATLAB
- And more...

## Examples

See `example_python_only.py` for demonstrations:

```bash
python example_python_only.py
```

This shows:
1. B-spline surface fitting
2. Depth initialization
3. Shape error calculation
4. Complete NRSfM workflow

## Documentation

For detailed information, see:
- **[MATLAB_REMOVAL_GUIDE.md](MATLAB_REMOVAL_GUIDE.md)** - Complete documentation
  - File-by-file changes
  - Parameter tuning guide
  - Troubleshooting
  - API reference

## Backward Compatibility

All existing MATLAB-based workflows still work:
- If MATLAB is installed, it will be used automatically
- MATLAB visualization functions work as before
- No changes needed to existing scripts

## Performance Notes

Python implementations have been optimized for:
- **Speed**: Comparable to MATLAB for most operations
- **Accuracy**: Validated against MATLAB outputs
- **Memory**: Efficient numpy/scipy implementations

For very large point clouds (>1000 points), MATLAB BBS may be slightly faster.

## Troubleshooting

### "No module named 'matlab.engine'"

This is **normal and expected** in Python-only mode. The code handles this automatically.

### Poor spline fitting results

Try adjusting parameters in `spline_fitting.py`:
```python
# For smoother surfaces
fit_python(..., smoothing=1e-4, grid_size=40)

# For more detail
fit_python(..., smoothing=1e-6, grid_size=60)
```

See [MATLAB_REMOVAL_GUIDE.md](MATLAB_REMOVAL_GUIDE.md) for detailed tuning instructions.

## Testing

Run tests to verify installation:

```bash
# Test imports
python -c "from NRSfM_core.spline_fitting import fit_python; print('OK')"

# Run example script
python example_python_only.py
```

## Support

For issues related to:
- **Python implementations**: See [MATLAB_REMOVAL_GUIDE.md](MATLAB_REMOVAL_GUIDE.md)
- **Original code**: See main repository documentation
- **MATLAB integration**: Check MATLAB Engine API documentation

## Citation

If you use this code, please cite the original paper [add citation info].

Note: Python implementations added to support MATLAB-free operation.
