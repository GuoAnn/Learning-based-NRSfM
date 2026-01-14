import numpy as np
import scipy
import torch

# Optional MATLAB support for backward compatibility
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None

# Import Python backend
from Result_evaluation.Shape_error_python import (
    shape_error as shape_error_python,
    shape_error_image as shape_error_image_python,
    shape_error_save as shape_error_save_python,
    view_shape as view_shape_python,
    view_shape_image as view_shape_image_python
)


def shape_error(Estimation_all, Groundtruth_all, m_or_backend='python'):
    """
    Compute shape error with support for both Python and MATLAB backends.
    
    Args:
        Estimation_all: (F, 3, N) array - estimated 3D points
        Groundtruth_all: (F, 3, N) array - ground truth 3D points
        m_or_backend: Either 'python' string, MATLAB engine object, or None (defaults to Python)
    """
    # Determine backend
    use_matlab = MATLAB_AVAILABLE and hasattr(m_or_backend, 'draw_image_sparse')
    
    if use_matlab:
        # Original MATLAB path
        accuracy = np.zeros(shape=(1, Estimation_all.shape[0]), dtype=np.float32)
        for i in range(Estimation_all.shape[0]):
            Groundtruth = Groundtruth_all[i, :, :]
            Estimation = Estimation_all[i, :, :]
            accuracy[0, i] = view_shape(Estimation, Groundtruth, m_or_backend)
        return np.mean(accuracy)
    else:
        # Python backend
        return shape_error_python(Estimation_all, Groundtruth_all)


def view_shape(Shape_A, Shape_B, m_or_backend='python'):
    """
    Compute error between two shapes with backend support.
    
    Args:
        Shape_A: (3, N) array - estimated shape
        Shape_B: (3, N) array - ground truth shape
        m_or_backend: Either 'python', MATLAB engine, or None
    """
    use_matlab = MATLAB_AVAILABLE and hasattr(m_or_backend, 'draw_image_sparse')
    
    if use_matlab:
        Shape_A_matlab = matlab.double(Shape_A.tolist())
        Shape_B_matlab = matlab.double(Shape_B.tolist())
        error_result = m_or_backend.draw_image_sparse(Shape_A_matlab, Shape_B_matlab, nargout=3)
        error_np = np.array(error_result[0])
        return error_np
    else:
        return view_shape_python(Shape_A, Shape_B)


def shape_error_image(Estimation_all, Groundtruth_all, m_or_backend='python'):
    """
    Compute shape error with image visualization (backend-aware).
    
    Args:
        Estimation_all: (F, 3, N) array
        Groundtruth_all: (F, 3, N) array
        m_or_backend: Backend selector
    """
    use_matlab = MATLAB_AVAILABLE and hasattr(m_or_backend, 'draw_image_sparse_with_image')
    
    if use_matlab:
        accuracy = np.zeros(shape=(1, Estimation_all.shape[0]), dtype=np.float32)
        for i in range(Estimation_all.shape[0]):
            Groundtruth = Groundtruth_all[i, :, :]
            Estimation = Estimation_all[i, :, :]
            accuracy[0, i] = view_shape_image(Estimation, Groundtruth, m_or_backend)
        return np.mean(accuracy)
    else:
        return shape_error_image_python(Estimation_all, Groundtruth_all)


def view_shape_image(Shape_A, Shape_B, m_or_backend='python'):
    """
    Compute error with image visualization (backend-aware).
    
    Args:
        Shape_A: (3, N) array
        Shape_B: (3, N) array
        m_or_backend: Backend selector
    """
    use_matlab = MATLAB_AVAILABLE and hasattr(m_or_backend, 'draw_image_sparse_with_image')
    
    if use_matlab:
        Shape_A_matlab = matlab.double(Shape_A.tolist())
        Shape_B_matlab = matlab.double(Shape_B.tolist())
        error_result = m_or_backend.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
        error_np = np.array(error_result[0])
        return error_np
    else:
        return view_shape_image_python(Shape_A, Shape_B)


def shape_error_save(Estimation_all, Groundtruth_all, m_or_backend='python'):
    """
    Compute per-frame errors and return for saving (backend-aware).
    
    Args:
        Estimation_all: (F, 3, N) array
        Groundtruth_all: (F, 3, N) array
        m_or_backend: Backend selector
    """
    use_matlab = MATLAB_AVAILABLE and hasattr(m_or_backend, 'draw_image_sparse_with_image')
    
    if use_matlab:
        accuracy = np.zeros(shape=(1, Estimation_all.shape[0]), dtype=np.float32)
        for i in range(Estimation_all.shape[0]):
            Groundtruth = Groundtruth_all[i, :, :]
            Estimation = Estimation_all[i, :, :]
            Shape_A_matlab = matlab.double(Groundtruth.tolist())
            Shape_B_matlab = matlab.double(Estimation.tolist())
            error_result = m_or_backend.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
            accuracy[0, i] = np.array(error_result[0])
        accuracy_tensor = torch.tensor(accuracy)
        Estimation_all_tensor = torch.tensor(Estimation_all)
        return accuracy_tensor, Estimation_all_tensor
    else:
        return shape_error_save_python(Estimation_all, Groundtruth_all)

