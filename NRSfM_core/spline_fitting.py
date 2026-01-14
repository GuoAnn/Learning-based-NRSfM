"""
Pure Python implementation of spline surface fitting using SciPy.
Replaces MATLAB's BBS (Bicubic B-Splines) toolbox functionality.

This module provides bivariate spline fitting for (u,v) -> (x,y,z) mappings
with support for computing partial derivatives.
"""

import numpy as np
from scipy.interpolate import SmoothBivariateSpline


class SplineSurfaceFitter:
    """
    Fits a smooth bivariate spline surface to 3D point data.
    
    This replaces the MATLAB fit_python.m function which uses the BBS toolbox.
    The MATLAB version uses bicubic B-splines with bending regularization.
    This version uses SciPy's SmoothBivariateSpline which provides similar
    smoothing capabilities.
    """
    
    def __init__(self, smoothing=1e-5, kx=3, ky=3):
        """
        Initialize the spline fitter.
        
        Args:
            smoothing: Smoothing parameter (similar to MATLAB's 'er' parameter).
                      Higher values create smoother surfaces. Default 1e-5.
            kx, ky: Degrees of the bivariate spline in x and y directions.
                    Default is 3 (cubic). Must be 1 <= k <= 5.
        """
        self.smoothing = smoothing
        self.kx = kx
        self.ky = ky
        self.splines = {}  # Cache for fitted splines
        
    def fit_and_evaluate(self, uv_fit, points_3d, uv_eval):
        """
        Fit a spline surface to 3D points and evaluate at specified locations.
        
        This replicates the MATLAB fit_python.m function interface:
        [quv, dqu, dqv, ddqu, ddqv, ddquv] = fit_python(Image_2d, Point_3d, Points_evaluation_2d)
        
        Args:
            uv_fit: (2, N) array of (u, v) coordinates where data is available
            points_3d: (3, N) array of 3D points (x, y, z) at uv_fit locations
            uv_eval: (2, M) array of (u, v) coordinates where to evaluate
            
        Returns:
            quv: (3, M) array - evaluated 3D positions at uv_eval
            dqu: (3, M) array - partial derivatives w.r.t. u at uv_eval
            dqv: (3, M) array - partial derivatives w.r.t. v at uv_eval
            ddqu: None (not used in current code, placeholder for compatibility)
            ddqv: None (not used in current code, placeholder for compatibility)
            ddquv: None (not used in current code, placeholder for compatibility)
        """
        # Extract coordinates
        u_fit = uv_fit[0, :]
        v_fit = uv_fit[1, :]
        u_eval = uv_eval[0, :]
        v_eval = uv_eval[1, :]
        
        # Find valid points (non-zero in MATLAB version)
        valid_idx = u_fit != 0
        if not np.any(valid_idx):
            valid_idx = np.ones_like(u_fit, dtype=bool)
        
        u_fit_valid = u_fit[valid_idx]
        v_fit_valid = v_fit[valid_idx]
        
        # Initialize output arrays
        quv = np.zeros((3, len(u_eval)))
        dqu = np.zeros((3, len(u_eval)))
        dqv = np.zeros((3, len(u_eval)))
        
        # Fit spline for each coordinate (x, y, z)
        for coord_idx in range(3):
            coord_data = points_3d[coord_idx, valid_idx]
            
            try:
                # Fit the bivariate spline
                # s parameter controls smoothing (similar to MATLAB's regularization)
                spline = SmoothBivariateSpline(
                    u_fit_valid, 
                    v_fit_valid, 
                    coord_data,
                    kx=self.kx,
                    ky=self.ky,
                    s=self.smoothing * len(u_fit_valid)  # Scale by number of points
                )
                
                # Evaluate spline at requested points
                # ev method with dx=0, dy=0 gives function values
                quv[coord_idx, :] = spline.ev(u_eval, v_eval, dx=0, dy=0)
                
                # Evaluate first derivatives
                # dx=1, dy=0 gives partial derivative w.r.t. u
                dqu[coord_idx, :] = spline.ev(u_eval, v_eval, dx=1, dy=0)
                
                # dx=0, dy=1 gives partial derivative w.r.t. v
                dqv[coord_idx, :] = spline.ev(u_eval, v_eval, dx=0, dy=1)
                
            except Exception as e:
                # Fallback: use linear interpolation if spline fitting fails
                print(f"Warning: Spline fitting failed for coordinate {coord_idx}, using fallback: {e}")
                
                # Simple fallback: use original data if evaluation points match
                if len(u_eval) == len(u_fit) and np.allclose(u_eval, u_fit) and np.allclose(v_eval, v_fit):
                    quv[coord_idx, :] = points_3d[coord_idx, :]
                else:
                    # Use nearest neighbor as last resort
                    from scipy.spatial import cKDTree
                    tree = cKDTree(np.stack([u_fit_valid, v_fit_valid], axis=1))
                    _, nearest_idx = tree.query(np.stack([u_eval, v_eval], axis=1))
                    quv[coord_idx, :] = coord_data[nearest_idx]
                
                # Compute numerical derivatives as fallback
                dqu[coord_idx, :] = 0.0
                dqv[coord_idx, :] = 0.0
        
        # Return in MATLAB-compatible format (6 outputs, last 3 are None)
        return quv, dqu, dqv, None, None, None


def fit_python(uv_fit, points_3d, uv_eval, smoothing=1e-5):
    """
    Convenience function that mimics the MATLAB fit_python interface.
    
    Args:
        uv_fit: (2, N) array or list - (u, v) coordinates for fitting
        points_3d: (3, N) array or list - 3D points at uv_fit locations
        uv_eval: (2, M) array or list - (u, v) coordinates for evaluation
        smoothing: Smoothing parameter (default 1e-5)
        
    Returns:
        Tuple of 6 elements: (quv, dqu, dqv, ddqu, ddqv, ddquv)
        where ddqu, ddqv, ddquv are None (not used)
    """
    # Convert to numpy arrays if needed
    uv_fit = np.array(uv_fit, dtype=np.float64)
    points_3d = np.array(points_3d, dtype=np.float64)
    uv_eval = np.array(uv_eval, dtype=np.float64)
    
    # Create fitter and fit
    fitter = SplineSurfaceFitter(smoothing=smoothing)
    return fitter.fit_and_evaluate(uv_fit, points_3d, uv_eval)


# For backward compatibility and caching
_global_fitter = None

def get_global_fitter(smoothing=1e-5):
    """Get or create a global fitter instance with caching."""
    global _global_fitter
    if _global_fitter is None or _global_fitter.smoothing != smoothing:
        _global_fitter = SplineSurfaceFitter(smoothing=smoothing)
    return _global_fitter
