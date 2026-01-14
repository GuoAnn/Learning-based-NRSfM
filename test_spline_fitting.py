"""
Quick test to validate the Python spline fitting module.
Run with: python test_spline_fitting.py
"""

import numpy as np
from NRSfM_core.spline_fitting import fit_python, SplineSurfaceFitter

def test_basic_spline_fitting():
    """Test basic spline fitting functionality."""
    print("Testing Python spline fitting...")
    
    # Create synthetic data: a simple surface z = x^2 + y^2
    n_points = 50
    u = np.random.uniform(-1, 1, n_points)
    v = np.random.uniform(-1, 1, n_points)
    
    # Create 3D points (x, y, z) where z = u^2 + v^2 (paraboloid)
    x = u
    y = v
    z = u**2 + v**2
    
    # Stack into (3, N) format
    uv_fit = np.stack([u, v], axis=0)  # (2, N)
    points_3d = np.stack([x, y, z], axis=0)  # (3, N)
    
    # Evaluate at the same points
    uv_eval = uv_fit.copy()
    
    # Fit spline
    print(f"Fitting spline with {n_points} points...")
    quv, dqu, dqv, _, _, _ = fit_python(uv_fit, points_3d, uv_eval, smoothing=1e-5)
    
    # Check shapes
    assert quv.shape == (3, n_points), f"Expected shape (3, {n_points}), got {quv.shape}"
    assert dqu.shape == (3, n_points), f"Expected shape (3, {n_points}), got {dqu.shape}"
    assert dqv.shape == (3, n_points), f"Expected shape (3, {n_points}), got {dqv.shape}"
    
    # Check that fitted values are close to original (should interpolate well)
    error = np.mean(np.abs(quv - points_3d))
    print(f"Mean absolute error in fitted values: {error:.6f}")
    
    # For derivatives, we expect dz/du ≈ 2u and dz/dv ≈ 2v
    # Check if derivatives have reasonable magnitudes
    dz_du_expected = 2 * u
    dz_dv_expected = 2 * v
    dz_du_fitted = dqu[2, :]
    dz_dv_fitted = dqv[2, :]
    
    deriv_error_u = np.mean(np.abs(dz_du_fitted - dz_du_expected))
    deriv_error_v = np.mean(np.abs(dz_dv_fitted - dz_dv_expected))
    
    print(f"Mean error in dz/du: {deriv_error_u:.6f}")
    print(f"Mean error in dz/dv: {deriv_error_v:.6f}")
    
    # Check that fitting was successful (error should be small)
    if error < 0.1:
        print("✓ Spline fitting test PASSED!")
        return True
    else:
        print("✗ Spline fitting test FAILED - error too large")
        return False

def test_initialization():
    """Test depth initialization methods."""
    print("\nTesting initialization methods...")
    
    from NRSfM_core.initialization import initialization_wrapper
    
    # Create synthetic normalized points
    num_frames = 4
    num_points = 30
    normalized_points = np.random.uniform(-0.5, 0.5, (num_frames * 2, num_points))
    
    methods = ['ones', 'random', 'mean_centered', 'affine']
    
    for method in methods:
        print(f"Testing {method} initialization...")
        depth = initialization_wrapper(normalized_points, method=method)
        
        # Check shape
        assert depth.shape == (num_frames, num_points), \
            f"Expected shape ({num_frames}, {num_points}), got {depth.shape}"
        
        # Check that depths are positive
        assert np.all(depth > 0), f"{method}: Some depths are not positive"
        
        # Check reasonable range
        assert np.all(depth < 10), f"{method}: Some depths are too large"
        
        print(f"  ✓ {method}: mean depth = {np.mean(depth):.4f}, std = {np.std(depth):.4f}")
    
    print("✓ All initialization methods PASSED!")
    return True

def test_shape_error():
    """Test shape error calculation."""
    print("\nTesting shape error calculation...")
    
    from Result_evaluation.Shape_error_python import shape_error, compute_shape_error
    
    # Create synthetic shapes
    num_frames = 3
    num_points = 20
    
    # Ground truth: random 3D points
    gt = np.random.randn(num_frames, 3, num_points)
    
    # Estimation: ground truth with small noise
    estimation = gt + 0.1 * np.random.randn(num_frames, 3, num_points)
    
    # Compute error
    error = shape_error(estimation, gt)
    
    print(f"Shape error with small noise: {error:.6f}")
    
    # Error should be small
    if error < 1.0:
        print("✓ Shape error calculation test PASSED!")
        return True
    else:
        print("✗ Shape error calculation test FAILED")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Running Python Backend Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Test spline fitting
    all_passed &= test_basic_spline_fitting()
    
    # Test initialization
    all_passed &= test_initialization()
    
    # Test shape error
    all_passed &= test_shape_error()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 60)
