#!/usr/bin/env python3
"""
Test script to verify depth map storage and warping functionality.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.utils.eval_utils import warp_depth_map, save_depth_map

def test_depth_warping():
    """Test depth map warping functionality."""
    print("🧪 Testing depth map warping...")
    
    # Create a simple depth map
    H, W = 100, 100
    depth_map = np.ones((H, W)) * 5.0  # 5m depth everywhere
    
    # Create a simple transformation (translation)
    transform = np.eye(4)
    transform[0, 3] = 1.0  # 1m translation in X
    
    # Create intrinsic matrix
    intrinsic = np.array([
        [500, 0, W/2],
        [0, 500, H/2],
        [0, 0, 1]
    ])
    
    # Test warping
    try:
        warped_depth = warp_depth_map(depth_map, transform, intrinsic)
        print(f"✅ Depth warping successful. Shape: {warped_depth.shape}")
        print(f"   Original depth range: {np.nanmin(depth_map):.2f} - {np.nanmax(depth_map):.2f}")
        print(f"   Warped depth range: {np.nanmin(warped_depth):.2f} - {np.nanmax(warped_depth):.2f}")
        return True
    except Exception as e:
        print(f"❌ Depth warping failed: {e}")
        return False

def test_depth_saving():
    """Test depth map saving functionality."""
    print("🧪 Testing depth map saving...")
    
    # Create a test depth map
    H, W = 50, 50
    depth_map = np.random.rand(H, W) * 10.0  # Random depths 0-10m
    
    # Create test output directory
    test_dir = Path("/tmp/depth_test")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Test saving
        save_depth_map(depth_map, test_dir, 0, "test_depth.png")
        
        # Check if file was created
        expected_file = test_dir / "000000_test_depth.png"
        if expected_file.exists():
            print(f"✅ Depth map saved successfully: {expected_file}")
            print(f"   File size: {expected_file.stat().st_size} bytes")
            return True
        else:
            print(f"❌ Depth map file not found: {expected_file}")
            return False
    except Exception as e:
        print(f"❌ Depth map saving failed: {e}")
        return False
    finally:
        # Clean up
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

def main():
    """Run all tests."""
    print("🚀 Testing depth map implementation...")
    
    tests_passed = 0
    total_tests = 2
    
    # Test depth warping
    if test_depth_warping():
        tests_passed += 1
    
    # Test depth saving
    if test_depth_saving():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Depth map implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
