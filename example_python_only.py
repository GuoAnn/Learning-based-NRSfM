#!/usr/bin/env python3
"""
Example script demonstrating MATLAB-free usage of the NRSfM pipeline.

This script shows how to:
1. Initialize depth without MATLAB
2. Fit B-spline surfaces to 3D points
3. Calculate shape errors
All using pure Python implementations.
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from NRSfM_core.spline_fitting import fit_python
from NRSfM_core.initialization import initialization_simple, initialization_from_observations
from Result_evaluation.Shape_error import calculate_shape_error_python


def example_spline_fitting():
    """Example: Fit B-spline surface to synthetic 3D points."""
    print("\n" + "="*60)
    print("Example 1: B-Spline Surface Fitting")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    num_points = 100
    
    # Normalized image coordinates (u, v)
    u = np.random.uniform(-0.5, 0.5, num_points)
    v = np.random.uniform(-0.5, 0.5, num_points)
    
    # 3D points (with varying depth)
    depth = 1.0 + 0.2 * (u**2 + v**2)  # Parabolic surface
    x = u * depth
    y = v * depth
    z = depth
    
    image_2d = np.vstack([u, v])
    point_3d = np.vstack([x, y, z])
    
    # Fit B-spline surface
    print(f"\nFitting B-spline to {num_points} points...")
    quv, dqu, dqv, _, _, _ = fit_python(
        image_2d, 
        point_3d, 
        image_2d,
        smoothing=1e-5,  # Adjust for your data
        grid_size=50
    )
    
    # Evaluate fit quality
    error = np.mean(np.abs(quv - point_3d))
    print(f"✓ Fitting successful!")
    print(f"  Mean absolute error: {error:.6f}")
    print(f"  Fitted surface shape: {quv.shape}")
    print(f"  Derivative dqu shape: {dqu.shape}")
    print(f"  Derivative dqv shape: {dqv.shape}")
    
    # Calculate partial derivatives for NRSfM
    dz_du = -dqu[2, :] / point_3d[2, :]
    dz_dv = -dqv[2, :] / point_3d[2, :]
    print(f"\n  Partial derivatives computed:")
    print(f"  dz/du range: [{dz_du.min():.3f}, {dz_du.max():.3f}]")
    print(f"  dz/dv range: [{dz_dv.min():.3f}, {dz_dv.max():.3f}]")


def example_initialization():
    """Example: Initialize depth for NRSfM."""
    print("\n" + "="*60)
    print("Example 2: Depth Initialization")
    print("="*60)
    
    # Synthetic data
    np.random.seed(42)
    num_frames = 10
    num_points = 150
    
    # Normalized 2D observations
    normalized_points = np.random.randn(num_frames, 2, num_points) * 0.1
    
    print(f"\nInitializing depth for {num_frames} frames, {num_points} points...")
    
    # Method 1: Uniform initialization
    depth_uniform = initialization_simple(normalized_points, initial_depth=1.0)
    print(f"✓ Uniform initialization:")
    print(f"  Depth shape: {depth_uniform.shape}")
    print(f"  Mean depth: {np.mean(depth_uniform):.6f}")
    print(f"  Std depth: {np.std(depth_uniform):.6f}")
    
    # Method 2: Magnitude-based initialization
    depth_magnitude = initialization_from_observations(
        normalized_points, 
        method='magnitude'
    )
    print(f"\n✓ Magnitude-based initialization:")
    print(f"  Depth shape: {depth_magnitude.shape}")
    print(f"  Mean depth: {np.mean(depth_magnitude):.6f}")
    print(f"  Std depth: {np.std(depth_magnitude):.6f}")
    
    # Method 3: Random initialization
    depth_random = initialization_from_observations(
        normalized_points, 
        method='random'
    )
    print(f"\n✓ Random initialization:")
    print(f"  Depth shape: {depth_random.shape}")
    print(f"  Mean depth: {np.mean(depth_random):.6f}")
    print(f"  Std depth: {np.std(depth_random):.6f}")


def example_shape_error():
    """Example: Calculate shape error without MATLAB."""
    print("\n" + "="*60)
    print("Example 3: Shape Error Calculation")
    print("="*60)
    
    # Create ground truth shape
    np.random.seed(42)
    num_points = 200
    
    # Ground truth: sphere-like shape
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = 10.0
    
    x_gt = r * np.sin(phi) * np.cos(theta)
    y_gt = r * np.sin(phi) * np.sin(theta)
    z_gt = r * np.cos(phi)
    groundtruth = np.vstack([x_gt, y_gt, z_gt])
    
    # Estimated shape: ground truth + noise + rotation
    noise = np.random.randn(3, num_points) * 0.5
    
    # Apply small rotation
    angle = 0.1
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    estimation = R @ groundtruth + noise + np.array([[1], [2], [3]])  # Add translation
    
    print(f"\nCalculating error between estimated and ground truth shapes...")
    print(f"  Number of points: {num_points}")
    
    # Calculate error using Procrustes alignment
    error = calculate_shape_error_python(estimation, groundtruth)
    
    print(f"✓ Shape error calculated:")
    print(f"  Normalized RMSE: {error:.6f}")
    print(f"\n  Note: Error includes optimal rotation, translation, and scale alignment")


def example_full_workflow():
    """Example: Complete workflow without MATLAB."""
    print("\n" + "="*60)
    print("Example 4: Complete NRSfM Workflow (Python-only)")
    print("="*60)
    
    np.random.seed(42)
    
    # Step 1: Setup
    num_frames = 5
    num_points = 80
    print(f"\n[Step 1] Setting up data:")
    print(f"  Frames: {num_frames}")
    print(f"  Points per frame: {num_points}")
    
    # Step 2: Generate synthetic observations
    normalized_points = np.random.randn(num_frames, 2, num_points) * 0.1
    print(f"\n[Step 2] Generated 2D observations")
    print(f"  Shape: {normalized_points.shape}")
    
    # Step 3: Initialize depth
    print(f"\n[Step 3] Initializing depth...")
    initial_depth = initialization_simple(normalized_points, initial_depth=1.0)
    print(f"  ✓ Initial depth shape: {initial_depth.shape}")
    
    # Step 4: Fit B-splines for one frame
    print(f"\n[Step 4] Fitting B-spline surface for frame 0...")
    frame_idx = 0
    u = normalized_points[frame_idx, 0, :]
    v = normalized_points[frame_idx, 1, :]
    depth = initial_depth[frame_idx, :]
    
    x = u * depth
    y = v * depth
    z = depth
    
    image_2d = np.vstack([u, v])
    point_3d = np.vstack([x, y, z])
    
    quv, dqu, dqv, _, _, _ = fit_python(image_2d, point_3d, image_2d)
    print(f"  ✓ B-spline fitted")
    
    # Step 5: Compute derivatives
    print(f"\n[Step 5] Computing partial derivatives...")
    dz_du = -dqu[2, :] / point_3d[2, :]
    dz_dv = -dqv[2, :] / point_3d[2, :]
    print(f"  ✓ Derivatives computed")
    print(f"    dz/du mean: {np.mean(dz_du):.6f}")
    print(f"    dz/dv mean: {np.mean(dz_dv):.6f}")
    
    # Step 6: Reconstruct 3D shape
    print(f"\n[Step 6] Reconstructing 3D shape...")
    reconstruction = quv
    print(f"  ✓ Reconstruction shape: {reconstruction.shape}")
    
    # Step 7: Evaluate (if we had ground truth)
    print(f"\n[Step 7] Evaluation:")
    error = np.mean(np.abs(reconstruction - point_3d))
    print(f"  ✓ Reconstruction error: {error:.6f}")
    
    print(f"\n{'='*60}")
    print("Workflow completed successfully! ✓")
    print("All steps executed without MATLAB.")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MATLAB-FREE NRSfM EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates core functionality without MATLAB:")
    print("  1. B-spline surface fitting")
    print("  2. Depth initialization")
    print("  3. Shape error calculation")
    print("  4. Complete workflow")
    
    try:
        example_spline_fitting()
        example_initialization()
        example_shape_error()
        example_full_workflow()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nYou can now use these functions in main.py without MATLAB.")
        print("For more details, see MATLAB_REMOVAL_GUIDE.md")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
