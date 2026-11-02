#!/usr/bin/env python3
"""
Example script to run windowed inference on horse motion sequences.
This script demonstrates how to use the windowed inference system.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_windowed_inference(data_dir: str, out_dir: str, 
                          window_size: int = 20, overlap: int = 5,
                          smoothing_factor: float = 0.1,
                          alignment_method: str = 'icp'):
    """
    Run windowed inference on a sequence directory.
    
    Args:
        data_dir: Path to directory containing images
        out_dir: Path for output results
        window_size: Number of frames per window
        overlap: Number of overlapping frames
        smoothing_factor: Trajectory smoothing factor (0.0-1.0)
        alignment_method: Window alignment method
    """
    
    cmd = [
        sys.executable, "windowed_inference.py",
        "--data_dir", data_dir,
        "--out_dir", out_dir,
        "--window_size", str(window_size),
        "--overlap", str(overlap),
        "--smoothing_factor", str(smoothing_factor),
        "--alignment_method", alignment_method
    ]
    
    print(f"🚀 Running windowed inference...")
    print(f"📁 Data: {data_dir}")
    print(f"📁 Output: {out_dir}")
    print(f"⚙️  Window: {window_size}, Overlap: {overlap}, Smoothing: {smoothing_factor}")
    print(f"🔧 Alignment: {alignment_method}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Windowed inference completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
        else:
            print("❌ Windowed inference failed!")
            print("\n📊 Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running windowed inference: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run windowed inference on horse motion sequences"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing sequence images"
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Number of frames per window (default: 20)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=5,
        help="Number of overlapping frames (default: 5)"
    )
    
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.1,
        help="Trajectory smoothing factor 0.0-1.0 (default: 0.1)"
    )
    
    parser.add_argument(
        "--alignment_method",
        type=str,
        choices=['icp', 'translation_only', 'rotation_only'],
        default='icp',
        help="Window alignment method (default: icp)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data path does not exist: {data_dir}")
        return
    
    # Run windowed inference
    run_windowed_inference(
        args.data_dir,
        args.out_dir,
        args.window_size,
        args.overlap,
        args.smoothing_factor,
        args.alignment_method
    )


if __name__ == "__main__":
    main()
