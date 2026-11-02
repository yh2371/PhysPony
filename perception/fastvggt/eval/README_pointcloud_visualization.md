# RGB Point Cloud Visualization with Camera Trajectory

This script visualizes RGB point clouds created from metric depth maps along with the scaled camera trajectory.

## Overview

The `visualize_rgb_pointcloud_with_trajectory.py` script:
1. Loads metric depth maps from `*_metric_depth.npy` files
2. Loads RGB images from `*_rgb.jpg` files
3. Loads camera poses from `*_scaled_rotation.npy` files (4x4 transformation matrices)
4. Converts depth maps to 3D point clouds using camera intrinsics
5. Colors the point clouds with RGB values from images
6. Visualizes the combined point cloud with the camera trajectory

## Usage

### Basic Usage

```bash
python visualize_rgb_pointcloud_with_trajectory.py \
    --sequence_dir $PHYSPONY_DATA/horse_combined/train/0000001 \
    --output_dir ./output_visualization
```

### Advanced Options

```bash
python visualize_rgb_pointcloud_with_trajectory.py \
    --sequence_dir /path/to/sequence \
    --output_dir ./output_visualization \
    --depth_threshold 50.0 \
    --subsample 2 \
    --max_points_per_frame 50000 \
    --focal_length 800.0 \
    --intrinsic_file /path/to/intrinsics.npy
```

## Parameters

- `--sequence_dir`: Directory containing sequence data (required)
- `--output_dir`: Output directory for visualizations (required)
- `--intrinsic_file`: Optional path to camera intrinsic matrix file (npy format)
- `--focal_length`: Optional camera focal length in pixels (for intrinsic estimation)
- `--depth_threshold`: Maximum depth value to consider in meters (default: 50.0)
- `--subsample`: Subsample factor for point clouds (default: 2, use every 2nd pixel)
- `--max_points_per_frame`: Maximum points to visualize per frame (default: 50000)

## Output

The script generates:

1. **pointcloud_with_trajectory.png**: Matplotlib visualization showing:
   - RGB point cloud
   - Camera trajectory (red line)
   - Camera positions (red dots)
   - Camera coordinate frames (RGB axes)

2. **pointcloud.ply**: Point cloud file that can be opened in:
   - MeshLab
   - CloudCompare
   - Open3D viewer
   - Other 3D visualization software

## Data Format

### Expected File Structure

The sequence directory should contain files with the following naming pattern:
```
sequence_dir/
├── 0010192_rgb.jpg
├── 0010192_metric_depth.npy
├── 0010192_scaled_rotation.npy
├── 0010193_rgb.jpg
├── 0010193_metric_depth.npy
├── 0010193_scaled_rotation.npy
└── ...
```

### File Formats

- **RGB images**: `*_rgb.jpg` - Standard JPEG images
- **Depth maps**: `*_metric_depth.npy` - Numpy arrays with depth values in meters
- **Camera poses**: `*_scaled_rotation.npy` - Numpy arrays with 4x4 camera-to-world transformation matrices
  - Can also be 3x3 rotation matrices (will be converted to 4x4 with zero translation)

## Camera Intrinsics

If not provided, the script estimates camera intrinsics from image dimensions assuming a 60-degree field of view. You can:

1. Provide a focal length: `--focal_length 800.0`
2. Provide an intrinsic matrix file: `--intrinsic_file intrinsics.npy`

The intrinsic matrix should be a 3x3 numpy array:
```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

## Tips

1. **Performance**: Use `--subsample` to reduce point cloud density for faster processing
2. **Memory**: Use `--max_points_per_frame` to limit points per frame if running out of memory
3. **Depth Range**: Adjust `--depth_threshold` based on your scene scale
4. **Large Sequences**: For very long sequences, consider processing a subset of frames

## Troubleshooting

### "No depth maps found"
- Check that `*_metric_depth.npy` files exist in the sequence directory
- Verify the directory path is correct
- Ensure RGB files exist (script uses RGB files as reference)

### "Pose file not found"
- Check that `*_scaled_rotation.npy` files exist for each frame
- Verify the file naming matches the pattern

### Point cloud appears distorted
- Check camera intrinsics are correct
- Verify depth maps are in meters (not millimeters or other units)
- Ensure camera poses are camera-to-world transformation matrices

### Out of memory
- Reduce `--max_points_per_frame`
- Increase `--subsample` factor
- Process a subset of frames
