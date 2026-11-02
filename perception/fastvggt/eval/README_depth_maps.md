# Depth Map Storage and Warping in Windowed Inference

This document describes the enhanced windowed inference functionality that now stores depth maps estimated from VGGT and applies warping to maintain consistency after trajectory alignment.

## Overview

The windowed inference system has been enhanced to:

1. **Store depth maps**: Save depth maps estimated from VGGT with the suffix `depth.png`
2. **Apply depth warping**: Warp depth maps to maintain consistency after trajectory alignment
3. **Scale estimation**: Provide depth maps for later scale estimation

## Key Features

### Depth Map Storage
- Depth maps are saved as PNG files with 16-bit precision
- Files are named with the pattern: `{frame_idx:06d}_depth.png`
- Depth values are scaled to the full uint16 range (0-65535)
- NaN values are handled by setting them to 0

### Depth Map Warping
- Depth maps are warped using the same transformation applied to camera poses
- Bilinear interpolation is used for smooth depth map warping
- Warping maintains depth consistency across window boundaries

### File Structure
```
output_directory/
├── final_trajectory.txt          # Aligned camera poses
├── before_trajectory.txt         # Original poses (for comparison)
├── depth_maps/                   # Directory containing depth maps
│   ├── 000000_depth.png         # Depth map for frame 0
│   ├── 000001_depth.png         # Depth map for frame 1
│   └── ...
├── trajectory_comparison.png     # Visualization
└── processing_metadata.json      # Processing parameters
```

## Usage

### Basic Usage
```bash
python windowed_inference.py \
    --data_dir /path/to/images \
    --out_dir /path/to/output \
    --window_size 20 \
    --overlap 5 \
    --smoothing_factor 0.1 \
    --alignment_method icp
```

### Batch Processing
```bash
python batch_windowed_inference.py \
    --data_dir /path/to/sequences \
    --out_dir /path/to/output \
    --window_sizes 10 20 \
    --overlaps 3 5 \
    --smoothing_factors 0.0 0.1 0.2
```

## Implementation Details

### New Functions Added

#### `warp_depth_map(depth_map, transform, intrinsic, output_shape=None)`
- Warps a depth map according to a 4x4 transformation matrix
- Uses bilinear interpolation for smooth results
- Handles coordinate transformations properly

#### `save_depth_map(depth_map, output_path, frame_idx, suffix="depth.png")`
- Saves depth map as 16-bit PNG
- Scales depth values to full uint16 range
- Handles NaN values appropriately

### Modified Functions

#### `infer_vggt_and_reconstruct()`
- Now returns depth maps as an additional return value
- Depth maps are filtered by confidence threshold
- Returns: `(extrinsic, intrinsic, world_points, colors, poses, time, depth_maps)`

#### `TrajectoryStitcher.align_windows()`
- Now handles depth map warping during alignment
- Returns: `(aligned_poses, aligned_depth_maps)`

#### `TrajectoryStitcher._align_window_to_reference()`
- Applies depth warping using the computed transformation
- Maintains depth consistency across windows

## Depth Map Format

### Storage Format
- **File format**: PNG (16-bit)
- **Naming**: `{frame_idx:06d}_depth.png`
- **Scaling**: Depth values scaled to 0-65535 range
- **Units**: Depth values in meters (original VGGT output)

### Loading Depth Maps
```python
import cv2
import numpy as np

# Load depth map
depth_image = cv2.imread("000000_depth.png", cv2.IMREAD_UNCHANGED)

# Convert back to depth values (assuming original range was 0-10m)
depth_min, depth_max = 0.0, 10.0
depth_values = depth_image.astype(np.float32) / 65535.0 * (depth_max - depth_min) + depth_min
```

## Scale Estimation

The stored depth maps can be used for scale estimation by:

1. Loading the depth maps from the `depth_maps/` directory
2. Converting them back to metric depth values
3. Using the depth information to estimate scene scale
4. Applying scale correction to the trajectory

### Example Scale Estimation
```python
import numpy as np
from pathlib import Path

def estimate_scale_from_depth_maps(depth_dir, reference_depth=5.0):
    """Estimate scale from depth maps."""
    depth_files = sorted(depth_dir.glob("*_depth.png"))
    
    if not depth_files:
        return 1.0
    
    # Load first depth map
    depth_map = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    depth_values = depth_map.astype(np.float32) / 65535.0 * 10.0  # Assuming 0-10m range
    
    # Estimate scale based on median depth
    valid_depths = depth_values[depth_values > 0]
    if len(valid_depths) > 0:
        median_depth = np.median(valid_depths)
        scale_factor = reference_depth / median_depth
        return scale_factor
    
    return 1.0
```

## Performance Considerations

- Depth map warping adds computational overhead
- Memory usage increases due to depth map storage
- PNG compression helps reduce storage requirements
- Warping is only applied during window alignment

## Troubleshooting

### Common Issues

1. **Missing depth maps**: Ensure VGGT model is properly loaded and depth confidence threshold is appropriate
2. **Warping artifacts**: Check that intrinsic matrix is correct
3. **Storage issues**: Ensure output directory has write permissions

### Debug Information
- Check `processing_metadata.json` for processing parameters
- Verify depth map files are created in `depth_maps/` directory
- Use trajectory visualization to check alignment quality

## Future Enhancements

- Depth map compression options
- Multi-scale depth processing
- Depth-based trajectory optimization
- Real-time depth visualization
