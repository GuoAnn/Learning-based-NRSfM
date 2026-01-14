"""
Pure Python implementation of shape error evaluation.

This module provides shape error calculation without MATLAB, replacing the
functionality in Result_evaluation/Shape_error.py that relies on MATLAB's
draw_image_sparse.m and absor.m (absolute orientation/Procrustes alignment).
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes


def align_shapes_procrustes(source, target, scale=True):
    """
    Align source shape to target using Procrustes analysis.
    
    This replaces MATLAB's absor.m function.
    
    Args:
        source: (3, N) array - source 3D points
        target: (3, N) array - target 3D points (ground truth)
        scale: Whether to allow scaling (default True)
        
    Returns:
        aligned: (3, N) array - aligned source points
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale factor (1.0 if scale=False)
    """
    # Center both point sets
    source_mean = np.mean(source, axis=1, keepdims=True)
    target_mean = np.mean(target, axis=1, keepdims=True)
    
    source_centered = source - source_mean
    target_centered = target - target_mean
    
    # Compute optimal rotation using orthogonal Procrustes
    R, _ = orthogonal_procrustes(source_centered.T, target_centered.T)
    R = R.T  # Transpose to get correct orientation
    
    # Apply rotation
    source_rotated = R @ source_centered
    
    # Compute optimal scale if requested
    if scale:
        numerator = np.sum(source_rotated * target_centered)
        denominator = np.sum(source_centered * source_centered)
        s = numerator / denominator if denominator > 0 else 1.0
    else:
        s = 1.0
    
    # Apply scale and translation
    aligned = s * source_rotated + target_mean
    t = target_mean.flatten() - s * (R @ source_mean).flatten()
    
    return aligned, R, t, s


def compute_shape_error(estimation, ground_truth, scale=True):
    """
    Compute shape error between estimation and ground truth.
    
    This replaces MATLAB's draw_image_sparse.m function.
    
    Args:
        estimation: (3, N) array - estimated 3D points
        ground_truth: (3, N) array - ground truth 3D points
        scale: Whether to allow scaling in alignment (default True)
        
    Returns:
        error: Normalized mean shape error
        aligned: (3, N) array - aligned estimation
        scale_factor: Normalization scale based on ground truth extent
    """
    # Align estimation to ground truth
    aligned, R, t, s = align_shapes_procrustes(estimation, ground_truth, scale=scale)
    
    # Compute per-point errors
    diff = ground_truth - aligned
    point_errors = np.sqrt(np.sum(diff ** 2, axis=0))
    
    # Compute scale factor (maximum extent of ground truth)
    gt_min = np.min(ground_truth, axis=1)
    gt_max = np.max(ground_truth, axis=1)
    scale_factor = np.max(gt_max - gt_min)
    
    # Normalize error by scale
    normalized_error = np.mean(point_errors) / scale_factor if scale_factor > 0 else 0.0
    
    return normalized_error, aligned, scale_factor


def shape_error(estimation_all, groundtruth_all):
    """
    Compute average shape error across all frames.
    
    Replaces the MATLAB-based shape_error function in Shape_error.py.
    
    Args:
        estimation_all: (F, 3, N) array - estimated 3D points for F frames
        groundtruth_all: (F, 3, N) array - ground truth 3D points for F frames
        
    Returns:
        mean_error: Average normalized shape error across all frames
    """
    num_frames = estimation_all.shape[0]
    errors = np.zeros(num_frames)
    
    for i in range(num_frames):
        estimation = estimation_all[i, :, :]
        ground_truth = groundtruth_all[i, :, :]
        
        errors[i], _, _ = compute_shape_error(estimation, ground_truth, scale=True)
    
    return np.mean(errors)


def shape_error_image(estimation_all, groundtruth_all):
    """
    Compute shape error with optional visualization.
    
    This is a placeholder that matches the interface of the MATLAB version
    (draw_image_sparse_with_image.m), but without actual image generation.
    
    Args:
        estimation_all: (F, 3, N) array - estimated 3D points
        groundtruth_all: (F, 3, N) array - ground truth 3D points
        
    Returns:
        mean_error: Average normalized shape error across all frames
    """
    # For now, just compute error without visualization
    # Could be extended to save plots using matplotlib
    return shape_error(estimation_all, groundtruth_all)


def shape_error_save(estimation_all, groundtruth_all):
    """
    Compute per-frame errors and return for saving.
    
    Args:
        estimation_all: (F, 3, N) array - estimated 3D points
        groundtruth_all: (F, 3, N) array - ground truth 3D points
        
    Returns:
        accuracy_tensor: (1, F) array of per-frame errors
        estimation_tensor: (F, 3, N) array of estimations (for reference)
    """
    import torch
    
    num_frames = estimation_all.shape[0]
    errors = np.zeros((1, num_frames))
    
    for i in range(num_frames):
        estimation = estimation_all[i, :, :]
        ground_truth = groundtruth_all[i, :, :]
        
        errors[0, i], _, _ = compute_shape_error(estimation, ground_truth, scale=True)
    
    accuracy_tensor = torch.tensor(errors)
    estimation_tensor = torch.tensor(estimation_all)
    
    return accuracy_tensor, estimation_tensor


def view_shape(shape_a, shape_b):
    """
    Compute error between two shapes (replaces MATLAB view_shape).
    
    Args:
        shape_a: (3, N) array - first shape (estimation)
        shape_b: (3, N) array - second shape (ground truth)
        
    Returns:
        error: Normalized shape error
    """
    error, _, _ = compute_shape_error(shape_a, shape_b, scale=True)
    return error


def view_shape_image(shape_a, shape_b):
    """
    Compute error with image visualization (placeholder).
    
    Args:
        shape_a: (3, N) array - first shape (estimation)
        shape_b: (3, N) array - second shape (ground truth)
        
    Returns:
        error: Normalized shape error
    """
    # Same as view_shape, but could add visualization later
    return view_shape(shape_a, shape_b)
