#!/usr/bin/env python
"""
System check script to verify Python-only mode is working.
Run this to ensure all dependencies are installed correctly.
"""

import sys

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking Python dependencies...\n")
    
    required_modules = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('torch', 'PyTorch'),
        ('trimesh', 'Trimesh'),
        ('open3d', 'Open3D'),
    ]
    
    all_good = True
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:15} - OK")
        except ImportError as e:
            print(f"✗ {display_name:15} - MISSING")
            all_good = False
    
    return all_good

def check_custom_modules():
    """Check if custom modules can be imported."""
    print("\nChecking custom modules...\n")
    
    custom_modules = [
        ('NRSfM_core.spline_fitting', 'Spline fitting'),
        ('NRSfM_core.initialization', 'Initialization'),
        ('Result_evaluation.Shape_error_python', 'Shape error (Python)'),
        ('NRSfM_core.class_autograd', 'Class autograd'),
        ('main', 'Main module'),
    ]
    
    all_good = True
    for module_name, display_name in custom_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:25} - OK")
        except ImportError as e:
            print(f"✗ {display_name:25} - ERROR: {e}")
            all_good = False
    
    return all_good

def check_matlab():
    """Check MATLAB availability (optional)."""
    print("\nChecking MATLAB (optional)...\n")
    
    try:
        import matlab.engine
        print("✓ MATLAB Engine - Available (optional)")
        print("  You can use --backend matlab if desired")
        return True
    except ImportError:
        print("○ MATLAB Engine - Not available (this is OK)")
        print("  Using Python-only mode")
        return False

def run_quick_test():
    """Run a quick functional test."""
    print("\nRunning quick functional test...\n")
    
    try:
        import numpy as np
        from NRSfM_core.spline_fitting import fit_python
        
        # Simple test
        u = np.array([0.0, 0.5, 1.0])
        v = np.array([0.0, 0.5, 1.0])
        uv = np.stack([u, v], axis=0)
        points_3d = np.stack([u, v, u*v], axis=0)
        
        quv, dqu, dqv, _, _, _ = fit_python(uv, points_3d, uv)
        
        print("✓ Spline fitting functional test - PASSED")
        return True
    except Exception as e:
        print(f"✗ Spline fitting functional test - FAILED: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 70)
    print("Learning-based NRSfM - System Check (Python-only mode)")
    print("=" * 70)
    
    checks_passed = []
    
    # Check imports
    checks_passed.append(check_imports())
    
    # Check custom modules
    checks_passed.append(check_custom_modules())
    
    # Check MATLAB (optional, doesn't affect pass/fail)
    check_matlab()
    
    # Run quick test
    checks_passed.append(run_quick_test())
    
    # Summary
    print("\n" + "=" * 70)
    if all(checks_passed):
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nYour system is ready to run Learning-based NRSfM without MATLAB!")
        print("\nQuick start:")
        print("  python main.py")
        print("\nFor more options:")
        print("  python main.py --help")
        print("  See PYTHON_BACKEND_README.md for complete documentation")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
