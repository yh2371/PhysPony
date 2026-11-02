#!/usr/bin/env python3
"""
Example script demonstrating how to use the windowed inference system.
This shows the key concepts and provides a simple interface for testing.
"""

import sys
from pathlib import Path

# Add the eval directory to path for imports
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

from windowed_inference import (
    WindowedInferenceProcessor, 
    TrajectoryStitcher, 
    TrajectoryVisualizer
)


def example_single_sequence():
    """Example: Process a single sequence with windowed inference."""
    
    print("=== Example: Single Sequence Processing ===\n")
    
    # This is a conceptual example - you would need actual model and data
    print("Conceptual usage:")
    print("""
    # Load your VGGT model
    model = VGGT(merging=0, vis_attn_map=False)
    # ... load checkpoint ...
    
    # Create processor
    processor = WindowedInferenceProcessor(
        model, 
        window_size=20, 
        overlap=5
    )
    
    # Process sequence
    results = processor.process_sequence_windows(image_paths)
    
    # Stitch and smooth
    stitcher = TrajectoryStitcher(smoothing_factor=0.1)
    aligned_poses = stitcher.align_windows(
        results['window_results'], 
        results['window_metadata']
    )
    smoothed_poses = stitcher.smooth_trajectory(aligned_poses)
    
    # Visualize
    visualizer = TrajectoryVisualizer()
    visualizer.visualize_trajectories(
        before_poses, 
        smoothed_poses, 
        output_path, 
        "Windowed Inference Results"
    )
    """)


def example_parameter_comparison():
    """Example: Compare different parameter settings."""
    
    print("=== Example: Parameter Comparison ===\n")
    
    print("To compare different parameters, you can:")
    print("""
    1. Run with different window sizes:
       python run_windowed_inference.py --window_size 10 --overlap 3
       python run_windowed_inference.py --window_size 20 --overlap 5
       python run_windowed_inference.py --window_size 30 --overlap 7
    
    2. Compare smoothing factors:
       python run_windowed_inference.py --smoothing_factor 0.0  # No smoothing
       python run_windowed_inference.py --smoothing_factor 0.1  # Light smoothing
       python run_windowed_inference.py --smoothing_factor 0.3  # Heavy smoothing
    
    3. Try different alignment methods:
       python run_windowed_inference.py --alignment_method icp
       python run_windowed_inference.py --alignment_method translation_only
       python run_windowed_inference.py --alignment_method rotation_only
    """)


def example_batch_processing():
    """Example: Batch processing multiple sequences."""
    
    print("=== Example: Batch Processing ===\n")
    
    print("For batch processing multiple sequences:")
    print("""
    # Process all sequences with multiple parameter combinations
    python batch_windowed_inference.py \\
        --data_dir /path/to/sequences \\
        --out_dir /path/to/batch_results \\
        --window_sizes 10 20 \\
        --overlaps 3 5 \\
        --smoothing_factors 0.0 0.1 0.2 \\
        --alignment_methods icp translation_only
    
    This will:
    - Find all sequence directories
    - Test all parameter combinations
    - Generate comparison results
    - Identify best parameters per sequence
    """)


def example_interpretation():
    """Example: How to interpret results."""
    
    print("=== Example: Interpreting Results ===\n")
    
    print("After processing, you'll get several outputs:")
    print("""
    1. final_trajectory.txt - Final camera poses (use this for your application)
    2. before_trajectory.txt - Original poses (for comparison)
    3. trajectory_comparison.png - Visual comparison plots
    4. trajectory_stats.json - Quantitative metrics
    
    Key metrics to look for:
    - position_std: Lower is better (less noisy trajectory)
    - total_distance: Should be reasonable for your sequence
    - num_frames: Should match your input sequence length
    
    Visual inspection:
    - Check trajectory_comparison.png for smoothness
    - Look for discontinuities or jumps
    - Verify the trajectory makes sense for your scene
    """)


def example_integration():
    """Example: How to integrate with existing code."""
    
    print("=== Example: Integration with Existing Code ===\n")
    
    print("To integrate windowed inference with your existing pipeline:")
    print("""
    # Option 1: Replace eval_custom.py calls
    # Instead of running inference on full sequence:
    
    # OLD:
    # poses = infer_vggt_and_reconstruct(model, full_sequence)
    
    # NEW:
    processor = WindowedInferenceProcessor(model, window_size=20, overlap=5)
    results = processor.process_sequence_windows(image_paths)
    
    stitcher = TrajectoryStitcher(smoothing_factor=0.1)
    poses = stitcher.smooth_trajectory(
        stitcher.align_windows(results['window_results'], results['window_metadata'])
    )
    
    # Option 2: Use as preprocessing step
    # Run windowed inference first, then use results in your pipeline
    
    # Option 3: Compare with original
    # Run both methods and compare results
    """)


def main():
    """Run all examples."""
    
    print("🔍 Windowed Inference Examples\n")
    print("This script demonstrates key concepts and usage patterns.\n")
    
    example_single_sequence()
    print()
    
    example_parameter_comparison()
    print()
    
    example_batch_processing()
    print()
    
    example_interpretation()
    print()
    
    example_integration()
    print()
    
    print("=== Quick Start Commands ===\n")
    print("1. Process single sequence:")
    print("   python run_windowed_inference.py --data_dir /path/to/images --out_dir /path/to/output")
    print()
    print("2. Batch process with parameter search:")
    print("   python batch_windowed_inference.py --data_dir /path/to/sequences --out_dir /path/to/results")
    print()
    print("3. Custom parameters:")
    print("   python windowed_inference.py --data_dir /path/to/images --out_dir /path/to/output --window_size 15 --overlap 5 --smoothing_factor 0.2")


if __name__ == "__main__":
    main()
