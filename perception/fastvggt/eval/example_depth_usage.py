#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced windowed inference with depth maps.
"""

import argparse
from pathlib import Path
import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def main():
    """Example usage of windowed inference with depth maps."""
    parser = argparse.ArgumentParser(
        description="Example: Run windowed inference with depth map storage"
    )
    
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to directory containing sequence images"
    )
    
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Window size (default: 20)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=5,
        help="Overlap size (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Running windowed inference with depth map storage...")
    print(f"📁 Data path: {args.data_path}")
    print(f"📁 Output path: {args.output_path}")
    print(f"⚙️  Window size: {args.window_size}, Overlap: {args.overlap}")
    
    # Import and run windowed inference
    from windowed_inference import main as run_windowed_inference
    
    # Set up arguments for windowed inference
    sys.argv = [
        "windowed_inference.py",
        "--data_path", str(args.data_path),
        "--output_path", str(args.output_path),
        "--window_size", str(args.window_size),
        "--overlap", str(args.overlap),
        "--smoothing_factor", "0.1",
        "--alignment_method", "icp"
    ]
    
    try:
        run_windowed_inference()
        print("\n✅ Windowed inference completed successfully!")
        print(f"📁 Results saved to: {args.output_path}")
        print(f"📁 Depth maps saved to: {args.output_path}/depth_maps/")
        print("\n📋 Output structure:")
        print(f"  - final_trajectory.txt: Aligned camera poses")
        print(f"  - before_trajectory.txt: Original poses (for comparison)")
        print(f"  - depth_maps/: Directory containing depth maps with 'depth.png' suffix")
        print(f"  - trajectory_comparison.png: Visualization of trajectory alignment")
        print(f"  - processing_metadata.json: Processing parameters and statistics")
        
    except Exception as e:
        print(f"❌ Error running windowed inference: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
