# Windowed Inference for FastVGGT

This system processes long sequences by running inference on overlapping windows and then stitching the results together with smoothing. This approach helps reduce noise in long sequences that tend to accumulate drift and artifacts.

## Overview

The windowed inference system consists of three main components:

1. **WindowedInferenceProcessor**: Processes sequences in overlapping windows
2. **TrajectoryStitcher**: Aligns and smooths windowed results
3. **TrajectoryVisualizer**: Creates before/after comparison visualizations

## Files

- `windowed_inference.py`: Main processing script
- `run_windowed_inference.py`: Simplified usage script
- `batch_windowed_inference.py`: Batch processing for multiple sequences
- `README_windowed_inference.md`: This documentation

## Usage

### Basic Usage

```bash
# Process a single sequence with default parameters
python run_windowed_inference.py \
    --data_dir /path/to/sequence/images \
    --out_dir /path/to/output

# With custom parameters
python run_windowed_inference.py \
    --data_dir /path/to/sequence/images \
    --out_dir /path/to/output \
    --window_size 20 \
    --overlap 5 \
    --smoothing_factor 0.1 \
    --alignment_method icp
```

### Advanced Usage

```bash
# Direct usage of main script
python windowed_inference.py \
    --data_dir /path/to/sequence/images \
    --out_dir /path/to/output \
    --window_size 20 \
    --overlap 5 \
    --smoothing_factor 0.1 \
    --alignment_method icp \
    --depth_conf_thresh 0.0
```

### Batch Processing

```bash
# Process multiple sequences with different parameter combinations
python batch_windowed_inference.py \
    --data_dir /path/to/all/sequences \
    --out_dir /path/to/batch/output \
    --window_sizes 10 20 \
    --overlaps 3 5 \
    --smoothing_factors 0.0 0.1 0.2 \
    --alignment_methods icp translation_only \
    --max_sequences 5
```

## Parameters

### Window Parameters
- `--window_size`: Number of frames per window (default: 20)
- `--overlap`: Number of overlapping frames between windows (default: 5)

### Smoothing Parameters
- `--smoothing_factor`: Trajectory smoothing factor (0.0 = no smoothing, 1.0 = heavy smoothing)

### Alignment Parameters
- `--alignment_method`: Method for aligning windows
  - `icp`: Full 6DOF alignment (translation + rotation)
  - `translation_only`: Only align translations
  - `rotation_only`: Only align rotations

### Model Parameters
- `--ckpt_path`: Path to VGGT model checkpoint
- `--depth_conf_thresh`: Depth confidence threshold for filtering

## Output

The system generates several output files:

### Main Results
- `final_trajectory.txt`: Final stitched and smoothed camera poses
- `before_trajectory.txt`: Original poses before processing (for comparison)

### Visualizations
- `trajectory_comparison.png`: Before/after trajectory comparison plots
- `trajectory_stats.json`: Trajectory statistics and metrics

### Metadata
- `processing_metadata.json`: Processing parameters and statistics

## How It Works

### 1. Windowed Processing
- Divides long sequences into overlapping windows
- Runs FastVGGT inference on each window independently
- Handles edge cases (insufficient frames, failed inference)

### 2. Window Alignment
- Aligns subsequent windows using overlap regions
- Supports different alignment strategies:
  - **ICP-style**: Full 6DOF alignment
  - **Translation-only**: Only align camera positions
  - **Rotation-only**: Only align camera orientations

### 3. Trajectory Smoothing
- Applies temporal smoothing to reduce noise
- Uses moving averages for positions
- Uses SLERP (Spherical Linear Interpolation) for rotations

### 4. Visualization
- Creates comprehensive before/after comparisons
- Shows 2D projections (XZ, XY planes)
- Shows 3D trajectory plots
- Provides statistical analysis

## Example Results

After processing, you'll see output like:

```
🚀 Starting windowed inference processing
📁 Data path: /path/to/sequence
📁 Output path: /path/to/output
⚙️  Window size: 20, Overlap: 5
🌊 Smoothing factor: 0.1

🔄 Processing 150 frames in windows of 20 with 5 overlap
  Window 1/8: frames 0-19 (20 frames)
    ✅ Window 0 completed: 20 poses
  Window 2/8: frames 15-34 (20 frames)
    ✅ Window 1 completed: 20 poses
  ...

🔗 Aligning 8 windows...
  Window 0: Reference (no alignment)
  Window 1: Aligned 20 poses
  ...

🌊 Smoothing trajectory with factor 0.1...
✅ Final trajectory has 150 poses

📊 Creating trajectory visualization...
📊 Trajectory visualization saved: /path/to/output/trajectory_comparison.png
📊 Trajectory statistics saved: /path/to/output/trajectory_stats.json

🎉 Processing complete!
📁 Results saved to: /path/to/output
📊 Final trajectory: 150 poses
```

## Tips for Best Results

### Window Size Selection
- **Smaller windows (10-15 frames)**: Better for sequences with rapid motion changes
- **Larger windows (20-30 frames)**: Better for sequences with smooth motion
- **Avoid windows too small**: Need at least 3 frames for stable inference

### Overlap Selection
- **Higher overlap (30-50% of window size)**: Better alignment but more computation
- **Lower overlap (10-25% of window size)**: Faster processing but may have alignment issues

### Smoothing Factor
- **0.0**: No smoothing (preserves all details but may be noisy)
- **0.1-0.2**: Light smoothing (recommended for most cases)
- **0.3+**: Heavy smoothing (may lose important motion details)

### Alignment Method
- **icp**: Best for general cases with both translation and rotation
- **translation_only**: Good for sequences where camera mainly translates
- **rotation_only**: Good for sequences where camera mainly rotates

## Troubleshooting

### Common Issues

1. **"Not enough valid images"**: Ensure sequence has at least 3 valid images
2. **"Window failed with error"**: Check image quality and format
3. **"No valid windows processed"**: Reduce window size or check image paths
4. **Poor alignment**: Increase overlap or try different alignment method

### Performance Tips

1. **GPU Memory**: Large windows may require more GPU memory
2. **Processing Time**: Larger windows are faster overall but use more memory
3. **Batch Processing**: Use batch script for parameter optimization

## Integration with Existing Code

The windowed inference system is designed to work with the existing FastVGGT evaluation pipeline:

```python
from windowed_inference import WindowedInferenceProcessor, TrajectoryStitcher

# Use with existing model
processor = WindowedInferenceProcessor(model, window_size=20, overlap=5)
results = processor.process_sequence_windows(image_paths)

# Stitch results
stitcher = TrajectoryStitcher(smoothing_factor=0.1)
aligned_poses = stitcher.align_windows(results['window_results'], results['window_metadata'])
smoothed_poses = stitcher.smooth_trajectory(aligned_poses)
```

This system provides a robust solution for processing long sequences while maintaining trajectory quality and reducing noise accumulation.
