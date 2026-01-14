"""
Python implementation of bicubic B-spline surface fitting with bending regularization.
Replaces MATLAB fit_python.m functionality using SciPy.

This module provides B-spline fitting for 3D point clouds with derivatives,
supporting the NRSfM pipeline without MATLAB dependency.
"""

import numpy as np
from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
import warnings


def fit_python(image_2d, point_3d, points_evaluation_2d, smoothing=1e-5, grid_size=50):
    """
    Fit a bicubic B-spline surface to 3D points and evaluate at query points.
    
    This function mimics the MATLAB fit_python.m behavior:
    - Fits a bicubic B-spline surface with bending regularization
    - Evaluates the surface and its derivatives at specified points
    
    Args:
        image_2d: np.ndarray of shape (2, N) - 2D coordinates (u, v) of input points
        point_3d: np.ndarray of shape (3, N) - 3D coordinates (x, y, z) of input points
        points_evaluation_2d: np.ndarray of shape (2, M) - 2D coordinates where to evaluate
        smoothing: float - regularization parameter (similar to MATLAB 'er' parameter)
        grid_size: int - number of grid points (similar to MATLAB 'nC' parameter)
    
    Returns:
        tuple: (quv, dqu, dqv, ddqu, ddqv, ddquv)
            - quv: np.ndarray of shape (3, M) - evaluated 3D positions
            - dqu: np.ndarray of shape (3, M) - partial derivatives w.r.t. u
            - dqv: np.ndarray of shape (3, M) - partial derivatives w.r.t. v
            - ddqu: np.ndarray of shape (3, M) or empty - second derivatives w.r.t. u
            - ddqv: np.ndarray of shape (3, M) or empty - second derivatives w.r.t. v
            - ddquv: np.ndarray of shape (3, M) or empty - mixed second derivatives
    """
    
    # Handle empty or invalid input
    if image_2d.shape[1] == 0 or point_3d.shape[1] == 0:
        M = points_evaluation_2d.shape[1]
        empty_result = np.zeros((3, M))
        return empty_result, empty_result, empty_result, np.array([]), np.array([]), np.array([])
    
    # Find non-zero points (handle missing features)
    idx = np.where(image_2d[0, :] != 0)[0]
    if len(idx) == 0:
        M = points_evaluation_2d.shape[1]
        empty_result = np.zeros((3, M))
        return empty_result, empty_result, empty_result, np.array([]), np.array([]), np.array([])
    
    # Extract valid points
    u_data = image_2d[0, idx]
    v_data = image_2d[1, idx]
    x_data = point_3d[0, idx]
    y_data = point_3d[1, idx]
    z_data = point_3d[2, idx]
    
    # Evaluation points
    u_eval = points_evaluation_2d[0, :]
    v_eval = points_evaluation_2d[1, :]
    
    # Fit B-spline for each coordinate (x, y, z)
    # Using SmoothBivariateSpline with appropriate smoothing
    # The smoothing parameter 's' controls the trade-off between fitting and smoothness
    # We scale it based on the number of points and the smoothing parameter
    s_param = smoothing * len(idx)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fit splines for each coordinate with bending regularization
            # kx=3, ky=3 gives cubic splines (bicubic)
            spline_x = SmoothBivariateSpline(u_data, v_data, x_data, kx=3, ky=3, s=s_param)
            spline_y = SmoothBivariateSpline(u_data, v_data, y_data, kx=3, ky=3, s=s_param)
            spline_z = SmoothBivariateSpline(u_data, v_data, z_data, kx=3, ky=3, s=s_param)
            
            # Evaluate at query points
            # For each evaluation point, we need to evaluate individually
            M = len(u_eval)
            quv = np.zeros((3, M))
            dqu = np.zeros((3, M))
            dqv = np.zeros((3, M))
            
            for i in range(M):
                # Evaluate function values
                quv[0, i] = spline_x(u_eval[i], v_eval[i], grid=False)
                quv[1, i] = spline_y(u_eval[i], v_eval[i], grid=False)
                quv[2, i] = spline_z(u_eval[i], v_eval[i], grid=False)
                
                # Evaluate derivatives w.r.t. u (dx=1, dy=0)
                dqu[0, i] = spline_x(u_eval[i], v_eval[i], dx=1, dy=0, grid=False)
                dqu[1, i] = spline_y(u_eval[i], v_eval[i], dx=1, dy=0, grid=False)
                dqu[2, i] = spline_z(u_eval[i], v_eval[i], dx=1, dy=0, grid=False)
                
                # Evaluate derivatives w.r.t. v (dx=0, dy=1)
                dqv[0, i] = spline_x(u_eval[i], v_eval[i], dx=0, dy=1, grid=False)
                dqv[1, i] = spline_y(u_eval[i], v_eval[i], dx=0, dy=1, grid=False)
                dqv[2, i] = spline_z(u_eval[i], v_eval[i], dx=0, dy=1, grid=False)
            
            # Second derivatives (empty for now, as in MATLAB version)
            ddqu = np.array([])
            ddqv = np.array([])
            ddquv = np.array([])
            
    except Exception as e:
        print(f"Warning: Spline fitting failed: {e}")
        # Return zero results on failure
        M = len(u_eval)
        quv = np.zeros((3, M))
        dqu = np.zeros((3, M))
        dqv = np.zeros((3, M))
        ddqu = np.array([])
        ddqv = np.array([])
        ddquv = np.array([])
    
    return quv, dqu, dqv, ddqu, ddqv, ddquv


def fit_python_batch(image_2d_list, point_3d_list, points_evaluation_2d_list, 
                     smoothing=1e-5, grid_size=50):
    """
    Batch version of fit_python for processing multiple frames efficiently.
    
    Args:
        image_2d_list: list of np.ndarray - list of 2D input coordinates
        point_3d_list: list of np.ndarray - list of 3D input coordinates
        points_evaluation_2d_list: list of np.ndarray - list of evaluation coordinates
        smoothing: float - regularization parameter
        grid_size: int - grid size parameter
    
    Returns:
        list of tuples - results for each frame
    """
    results = []
    for img_2d, pt_3d, eval_2d in zip(image_2d_list, point_3d_list, points_evaluation_2d_list):
        result = fit_python(img_2d, pt_3d, eval_2d, smoothing, grid_size)
        results.append(result)
    return results


def estimate_optimal_smoothing(image_2d, point_3d, test_smoothings=None):
    """
    Estimate optimal smoothing parameter using cross-validation.
    
    Args:
        image_2d: np.ndarray of shape (2, N) - 2D coordinates
        point_3d: np.ndarray of shape (3, N) - 3D coordinates
        test_smoothings: list of floats - smoothing values to test
    
    Returns:
        float - estimated optimal smoothing parameter
    """
    if test_smoothings is None:
        test_smoothings = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    idx = np.where(image_2d[0, :] != 0)[0]
    if len(idx) < 10:
        return 1e-5  # Default
    
    u_data = image_2d[0, idx]
    v_data = image_2d[1, idx]
    z_data = point_3d[2, idx]  # Use z-coordinate for testing
    
    best_smoothing = 1e-5
    best_score = float('inf')
    
    # Split data for cross-validation
    n_train = int(0.8 * len(idx))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    
    for s in test_smoothings:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spline = SmoothBivariateSpline(
                    u_data[train_idx], v_data[train_idx], z_data[train_idx],
                    kx=3, ky=3, s=s * n_train
                )
                
                # Evaluate on test set
                pred = np.array([spline(u_data[i], v_data[i], grid=False) 
                                for i in test_idx])
                error = np.mean((pred - z_data[test_idx]) ** 2)
                
                if error < best_score:
                    best_score = error
                    best_smoothing = s
        except:
            continue
    
    return best_smoothing
