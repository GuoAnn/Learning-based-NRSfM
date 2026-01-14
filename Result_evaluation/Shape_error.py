import numpy as np
import scipy
# Make MATLAB import optional
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None
import torch


def procrustes_alignment(X, Y):
    """
    Procrustes alignment to find optimal rotation, translation, and scale.
    
    Args:
        X: np.ndarray of shape (3, N) - source points
        Y: np.ndarray of shape (3, N) - target points (ground truth)
    
    Returns:
        tuple: (aligned_X, scale, rotation, translation, error)
    """
    # Center the point clouds
    X_mean = np.mean(X, axis=1, keepdims=True)
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # Compute optimal scale
    scale = np.sqrt(np.sum(Y_centered ** 2) / np.sum(X_centered ** 2))
    
    # Scale X
    X_scaled = X_centered * scale
    
    # Compute optimal rotation using SVD
    H = X_scaled @ Y_centered.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply transformation
    X_aligned = R @ X_scaled + Y_mean
    
    # Compute error
    error = np.sqrt(np.mean(np.sum((X_aligned - Y) ** 2, axis=0)))
    
    return X_aligned, scale, R, Y_mean, error


def calculate_shape_error_python(estimation, groundtruth):
    """
    Calculate shape error using Procrustes alignment (Python implementation).
    
    Args:
        estimation: np.ndarray of shape (3, N) - estimated shape
        groundtruth: np.ndarray of shape (3, N) - ground truth shape
    
    Returns:
        float - normalized shape error
    """
    try:
        # Perform Procrustes alignment
        aligned, scale, R, t, error = procrustes_alignment(estimation, groundtruth)
        
        # Normalize error by ground truth scale
        gt_scale = np.sqrt(np.mean(np.sum((groundtruth - np.mean(groundtruth, axis=1, keepdims=True)) ** 2)))
        normalized_error = error / (gt_scale + 1e-8)
        
        return normalized_error
    except Exception as e:
        print(f"Warning: Error calculation failed: {e}")
        # Fallback: simple RMSE
        return np.sqrt(np.mean((estimation - groundtruth) ** 2))


def shape_error(Estimation_all, Groundtruth_all, m=None):
    """
    Calculate shape error between estimation and ground truth.
    
    Args:
        Estimation_all: np.ndarray - estimated shapes
        Groundtruth_all: np.ndarray - ground truth shapes
        m: matlab.engine or None - MATLAB engine (optional, uses Python if None)
    
    Returns:
        float - mean error across all frames
    """
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        accuracy[0,i] = view_shape(Estimation, Groundtruth, m)

    return np.mean(accuracy)


def view_shape(Shape_A, Shape_B, m=None):
    """
    Calculate error between two shapes.
    
    Args:
        Shape_A: np.ndarray - first shape
        Shape_B: np.ndarray - second shape (ground truth)
        m: matlab.engine or None - MATLAB engine (optional)
    
    Returns:
        float - shape error
    """
    if m is not None and MATLAB_AVAILABLE:
        # Use MATLAB for error calculation if available
        Shape_A_matlab = matlab.double(Shape_A.tolist())
        Shape_B_matlab = matlab.double(Shape_B.tolist())
        #error_result = m.draw_image_dense(Shape_A_matlab,Shape_B_matlab)
        error_result = m.draw_image_sparse(Shape_A_matlab, Shape_B_matlab, nargout=3)
        error_np = np.array(error_result[0])
    else:
        # Python-only error calculation (Procrustes alignment + RMSE)
        error_np = calculate_shape_error_python(Shape_A, Shape_B)
    
    return error_np


def shape_error_image(Estimation_all, Groundtruth_all, m=None):
    """
    Calculate shape error with image visualization (MATLAB only).
    
    Args:
        Estimation_all: np.ndarray - estimated shapes
        Groundtruth_all: np.ndarray - ground truth shapes
        m: matlab.engine or None - MATLAB engine (required for visualization)
    
    Returns:
        float - mean error across all frames
    """
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        accuracy[0,i] = view_shape_image(Estimation, Groundtruth, m)

    return np.mean(accuracy)


def view_shape_image(Shape_A, Shape_B, m=None):
    """
    Calculate error and visualize shapes (MATLAB only).
    
    Args:
        Shape_A: np.ndarray - first shape
        Shape_B: np.ndarray - second shape (ground truth)
        m: matlab.engine or None - MATLAB engine (optional, uses Python if None)
    
    Returns:
        float - shape error
    """
    if m is not None and MATLAB_AVAILABLE:
        Shape_A_matlab = matlab.double(Shape_A.tolist())
        Shape_B_matlab = matlab.double(Shape_B.tolist())
        #error_result = m.draw_image_dense(Shape_A_matlab,Shape_B_matlab)
        error_result = m.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
        error_np = np.array(error_result[0])
        P2 = np.array(error_result[1])
        scale = np.array(error_result[2])
    else:
        # Python-only error calculation
        error_np = calculate_shape_error_python(Shape_A, Shape_B)

    return error_np

def shape_error_save(Estimation_all, Groundtruth_all, m=None):
    """
    Calculate shape error and save results.
    
    Args:
        Estimation_all: np.ndarray - estimated shapes
        Groundtruth_all: np.ndarray - ground truth shapes
        m: matlab.engine or None - MATLAB engine (optional)
    
    Returns:
        tuple: (accuracy_tensor, Estimation_all_tensor)
    """
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        if m is not None and MATLAB_AVAILABLE:
            Shape_A_matlab = matlab.double(Groundtruth.tolist())
            Shape_B_matlab = matlab.double(Estimation.tolist())
            error_result = m.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
            accuracy[0,i] = np.array(error_result[0])
        else:
            accuracy[0,i] = calculate_shape_error_python(Estimation, Groundtruth)
    accuracy_tensor = torch.tensor(accuracy)
    Estimation_all_tensor = torch.tensor(Estimation_all)

    return accuracy_tensor, Estimation_all_tensor

