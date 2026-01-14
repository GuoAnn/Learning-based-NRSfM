"""
Pure Python initialization for NRSfM without MATLAB dependency.

This module provides a simplified initialization that replaces the MATLAB
initialization_for_NRSfM_local_all_new.m function. The MATLAB version uses
LLS (Local Linear Structure) and depth recovery algorithms.

For training purposes, a simple initialization is often sufficient, as the
network learns to refine the initial estimates during training.
"""

import numpy as np


def initialize_depth_simple(normalized_points, num_frames, method='ones'):
    """
    Simple depth initialization without MATLAB.
    
    Args:
        normalized_points: (2F, N) array of normalized 2D points, where F is frames
                          Organized as [u0, v0, u1, v1, ..., uF-1, vF-1]
        num_frames: Number of frames
        method: Initialization method:
                - 'ones': Initialize all depths to 1.0 (perspective camera assumption)
                - 'random': Small random perturbations around 1.0
                - 'mean_centered': Initialize based on distance from centroid
                
    Returns:
        depth: (F, N) array of initial depth estimates
    """
    num_points = normalized_points.shape[1]
    
    if method == 'ones':
        # Simple initialization: assume depth = 1 for all points
        depth = np.ones((num_frames, num_points), dtype=np.float32)
        
    elif method == 'random':
        # Small random perturbations around 1.0
        # This can help with training diversity
        depth = 0.8 + 0.4 * np.random.rand(num_frames, num_points).astype(np.float32)
        
    elif method == 'mean_centered':
        # Initialize based on distance from mean (center of mass)
        depth = np.ones((num_frames, num_points), dtype=np.float32)
        
        for frame_idx in range(num_frames):
            u = normalized_points[frame_idx * 2, :]
            v = normalized_points[frame_idx * 2 + 1, :]
            
            # Compute center
            u_mean = np.mean(u)
            v_mean = np.mean(v)
            
            # Distance from center (in normalized coordinates)
            dist = np.sqrt((u - u_mean)**2 + (v - v_mean)**2)
            
            # Simple heuristic: points farther from center might be slightly closer
            # This is a very rough approximation
            max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
            depth[frame_idx, :] = 1.0 - 0.2 * (dist / max_dist)
            
            # Ensure positive depths
            depth[frame_idx, :] = np.maximum(depth[frame_idx, :], 0.5)
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return depth


def initialize_from_affine(normalized_points, num_frames):
    """
    Affine camera initialization (orthographic approximation).
    
    This provides a more principled initialization by assuming an affine camera
    model, which is a good approximation for distant objects.
    
    Args:
        normalized_points: (2F, N) array of normalized 2D points
        num_frames: Number of frames
        
    Returns:
        depth: (F, N) array of initial depth estimates
    """
    num_points = normalized_points.shape[1]
    depth = np.ones((num_frames, num_points), dtype=np.float32)
    
    # For affine camera, we can use a simple SVD-based factorization
    # This is a simplified version - full implementation would use
    # proper structure-from-motion techniques
    
    # Stack all 2D points into measurement matrix
    W = normalized_points  # (2F, N)
    
    # Center the data
    W_centered = W - np.mean(W, axis=1, keepdims=True)
    
    # SVD factorization: W = R * S
    # where R is rotation (2F x 3) and S is shape (3 x N)
    try:
        U, s, Vt = np.linalg.svd(W_centered, full_matrices=False)
        
        # Take first 3 components (rank-3 approximation)
        k = min(3, len(s))
        R = U[:, :k] @ np.diag(np.sqrt(s[:k]))
        S = np.diag(np.sqrt(s[:k])) @ Vt[:k, :]
        
        # Extract depth from the third row of shape matrix
        if k >= 3:
            z_init = S[2, :]
            # Normalize and ensure positive
            z_init = np.abs(z_init)
            z_init = z_init / np.mean(z_init) if np.mean(z_init) > 0 else np.ones_like(z_init)
            
            # Replicate for all frames
            for frame_idx in range(num_frames):
                depth[frame_idx, :] = z_init
        
    except Exception as e:
        print(f"Warning: SVD initialization failed, using ones: {e}")
        # Fallback to ones
        depth = np.ones((num_frames, num_points), dtype=np.float32)
    
    return depth


def initialization_for_NRSfM_local_all_new(file_path, method='ones'):
    """
    Main initialization function that replaces MATLAB version.
    
    This function mimics the interface of the MATLAB function:
    initialization_for_NRSfM_local_all_new(file_id, nargout=1)
    
    Args:
        file_path: Path to .mat file (loaded separately by caller)
                   OR numpy array of normalized points if already loaded
        method: Initialization method ('ones', 'random', 'mean_centered', 'affine')
        
    Returns:
        depth: (F, N) array of initial depth values
        
    Note:
        The caller should load the .mat file and extract normalized_points
        before calling this function, or pass the array directly.
    """
    # If file_path is a string, we expect the caller to have loaded it
    # If it's an array, use it directly
    if isinstance(file_path, str):
        # In the actual workflow, the .mat file is loaded by load_mat_dataset
        # This function receives the path but the data should be provided
        raise ValueError(
            "Please load the .mat file externally and pass normalized_points array. "
            "This function no longer loads MATLAB files directly."
        )
    else:
        # Assume it's a numpy array of normalized points
        normalized_points = file_path
    
    # Infer number of frames from shape
    # normalized_points should be (2F, N) where F is number of frames
    assert normalized_points.shape[0] % 2 == 0, "Invalid shape for normalized_points"
    num_frames = normalized_points.shape[0] // 2
    
    # Perform initialization
    if method == 'affine':
        depth = initialize_from_affine(normalized_points, num_frames)
    else:
        depth = initialize_depth_simple(normalized_points, num_frames, method=method)
    
    return depth


def initialization_wrapper(normalized_points, method='ones'):
    """
    Simplified wrapper for initialization.
    
    Args:
        normalized_points: (2F, N) array of normalized 2D points
        method: Initialization method
        
    Returns:
        depth: (F, N) array of initial depth estimates
    """
    num_frames = normalized_points.shape[0] // 2
    
    if method == 'affine':
        return initialize_from_affine(normalized_points, num_frames)
    else:
        return initialize_depth_simple(normalized_points, num_frames, method=method)
