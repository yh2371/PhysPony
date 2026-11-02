#!/usr/bin/env python3
"""
Visualize RGB point clouds with scaled camera trajectory.

This script loads:
1. Metric depth maps from *_metric_depth.npy files
2. RGB images from *_rgb.jpg files  
3. Camera poses from scaled_rotation.npy files (4x4 transformation matrices)
4. Creates colored 3D point clouds and visualizes them with the camera trajectory
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import glob
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval.utils import depthmap_to_absolute_camera_coordinates


def load_sequence_data(sequence_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load depth maps, RGB images, and camera poses from a sequence directory.
    
    Args:
        sequence_dir: Directory containing sequence data
        
    Returns:
        Tuple of (depth_maps, rgb_images, camera_poses, frame_ids)
    """
    depth_maps = []
    rgb_images = []
    camera_poses = []
    frame_ids = []
    
    # Find all RGB files as reference
    rgb_files = sorted(glob.glob(str(sequence_dir / "*_rgb.jpg")))
    
    if len(rgb_files) == 0:
        print(f"Warning: No RGB files found in {sequence_dir}")
        return [], [], [], []
    
    for rgb_file in rgb_files:
        # Extract frame ID from filename (e.g., "0010192_rgb.jpg" -> "0010192")
        frame_id = Path(rgb_file).stem.replace("_rgb", "")
        
        # Load RGB image
        rgb_image = cv2.imread(str(rgb_file))
        if rgb_image is None:
            print(f"Warning: Could not load RGB image {rgb_file}, skipping")
            continue
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding depth map
        depth_file = sequence_dir / f"{frame_id}_metric_depth.npy"
        if not depth_file.exists():
            print(f"Warning: Depth file not found for {frame_id}, skipping")
            continue
        
        depth_map = np.load(depth_file)
        
        # Load camera pose (scaled_rotation.npy)
        pose_file = sequence_dir / f"{frame_id}_scaled_rotation.npy"
        if not pose_file.exists():
            print(f"Warning: Pose file not found for {frame_id}, skipping")
            continue
        
        camera_pose = np.load(pose_file).astype(np.float32)
        
        # Ensure pose is 4x4
        if camera_pose.shape == (3, 3):
            # If it's a 3x3 rotation matrix, convert to 4x4
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :3] = camera_pose
            camera_pose = pose_4x4
        elif camera_pose.shape != (4, 4):
            print(f"Warning: Unexpected pose shape {camera_pose.shape} for {frame_id}, skipping")
            continue
        
        # Resize RGB to match depth if needed
        print(rgb_image.shape, "????", depth_map.shape)
        depth_maps.append(depth_map)
        rgb_images.append(rgb_image)
        camera_poses.append(camera_pose)
        frame_ids.append(frame_id)
    
    return depth_maps, rgb_images, camera_poses, frame_ids


def estimate_intrinsics_from_image(image: np.ndarray, default_fov: float = 60.0) -> np.ndarray:
    """
    Estimate camera intrinsics from image dimensions.
    
    Args:
        image: RGB image array
        default_fov: Default field of view in degrees
        
    Returns:
        3x3 camera intrinsic matrix
    """
    h, w = image.shape[:2]
    fov_rad = np.radians(default_fov)
    focal_length = w / (2.0 * np.tan(fov_rad / 2.0))
    
    K = np.array([
        [focal_length, 0.0, w / 2.0],
        [0.0, focal_length, h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    K = np.array([[928., 0., 259.],
                  [0., 956., 259.],
                  [0., 0., 1.]], dtype=np.float32)
    
    return K


def depth_to_pointcloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    camera_pose: np.ndarray,
    intrinsic: np.ndarray,
    depth_threshold: float = 50.0,
    subsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to 3D point cloud with RGB colors.
    
    Args:
        depth_map: Depth map (H, W)
        rgb_image: RGB image (H, W, 3)
        camera_pose: 4x4 camera-to-world transformation matrix
        intrinsic: 3x3 camera intrinsic matrix
        depth_threshold: Maximum depth value to consider (in meters)
        subsample: Subsample factor (1 = use all pixels, 2 = every 2nd pixel, etc.)
        
    Returns:
        Tuple of (points, colors) where points are (N, 3) and colors are (N, 3)
    """
    # Convert depth map to world coordinates
    X_world, valid_mask = depthmap_to_absolute_camera_coordinates(
        depth_map, intrinsic, camera_pose
    )
    intrinsic = np.array([[928., 0., 259.],
                  [0., 956., 259.],
                  [0., 0., 1.]], dtype=np.float32)
    
    # Apply depth threshold
    valid_mask = valid_mask & (depth_map < depth_threshold) & (depth_map > 0)

    
    # Subsample
    if subsample > 1:
        valid_mask = valid_mask[::subsample, ::subsample]
        X_world = X_world[::subsample, ::subsample]
        rgb_image = rgb_image[::subsample, ::subsample]

    
    # Extract valid points and colors
    valid_points = X_world[valid_mask]
    valid_colors = rgb_image[valid_mask]
    
    return valid_points, valid_colors


def visualize_pointcloud_with_trajectory(
    point_clouds: List[np.ndarray],
    colors: List[np.ndarray],
    camera_poses: List[np.ndarray],
    output_path: Path,
    max_points_per_frame: int = 50000,
    trajectory_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    camera_size: float = 0.1
):
    """
    Visualize RGB point clouds with camera trajectory.
    
    Args:
        point_clouds: List of point clouds, each (N, 3)
        colors: List of colors, each (N, 3) with values in [0, 255]
        camera_poses: List of camera poses (4x4 matrices)
        output_path: Output directory for visualizations
        max_points_per_frame: Maximum points to visualize per frame
        trajectory_color: RGB color for trajectory (values in [0, 1])
        camera_size: Size of camera markers
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine all point clouds
    print("Combining point clouds...")
    all_points = []
    all_colors = []
    
    for i, (pc, col) in enumerate(zip(point_clouds, colors)):
        if len(pc) == 0:
            continue
        
        # Subsample if too many points
        if len(pc) > max_points_per_frame:
            indices = np.random.choice(len(pc), max_points_per_frame, replace=False)
            pc = pc[indices]
            col = col[indices]
        
        all_points.append(pc)
        all_colors.append(col)
    
    if len(all_points) == 0:
        print("Error: No valid points to visualize")
        return
    
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors) / 255.0  # Normalize to [0, 1]
    
    print(f"Total points: {len(combined_points)}")
    
    # Extract camera positions from poses
    camera_positions = np.array([pose[:3, 3] for pose in camera_poses])
    
    # Create visualization using matplotlib
    print("Creating matplotlib visualization...")
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud (subsampled for performance)
    if len(combined_points) > 500000:
        print("Subsampling point cloud for visualization...")
        indices = np.random.choice(len(combined_points), 500000, replace=False)
        plot_points = combined_points[indices]
        plot_colors = combined_colors[indices]
    else:
        plot_points = combined_points
        plot_colors = combined_colors
    
    # Scatter plot with colors
    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        c=plot_colors,
        s=0.1,
        alpha=0.6
    )
    
    # Plot camera trajectory
    ax.plot(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        color=trajectory_color,
        linewidth=2,
        label='Camera Trajectory'
    )
    
    # Plot camera positions as larger points
    ax.scatter(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        color=trajectory_color,
        s=50,
        marker='o',
        label='Camera Positions'
    )
    
    # Draw camera orientations (coordinate frames)
    for i in range(0, len(camera_poses), max(1, len(camera_poses) // 20)):  # Draw every Nth camera
        pose = camera_poses[i]
        pos = pose[:3, 3]
        # Draw camera coordinate axes
        axis_length = camera_size
        x_axis = pose[:3, 0] * axis_length
        y_axis = pose[:3, 1] * axis_length
        z_axis = pose[:3, 2] * axis_length
        
        ax.plot([pos[0], pos[0] + x_axis[0]], [pos[1], pos[1] + x_axis[1]], [pos[2], pos[2] + x_axis[2]], 'r-', linewidth=1)
        ax.plot([pos[0], pos[0] + y_axis[0]], [pos[1], pos[1] + y_axis[1]], [pos[2], pos[2] + y_axis[2]], 'g-', linewidth=1)
        ax.plot([pos[0], pos[0] + z_axis[0]], [pos[1], pos[1] + z_axis[1]], [pos[2], pos[2] + z_axis[2]], 'b-', linewidth=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('RGB Point Cloud with Camera Trajectory')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        plot_points[:, 0].max() - plot_points[:, 0].min(),
        plot_points[:, 1].max() - plot_points[:, 1].min(),
        plot_points[:, 2].max() - plot_points[:, 2].min()
    ]).max() / 2.0
    mid_x = (plot_points[:, 0].max() + plot_points[:, 0].min()) * 0.5
    mid_y = (plot_points[:, 1].max() + plot_points[:, 1].min()) * 0.5
    mid_z = (plot_points[:, 2].max() + plot_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path / "pointcloud_with_trajectory.png", dpi=150, bbox_inches='tight')
    print(f"Saved matplotlib visualization to {output_path / 'pointcloud_with_trajectory.png'}")
    plt.close()
    
    # Also create Open3D visualization (interactive)
    print("Creating Open3D visualization...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Create camera trajectory line
    camera_line = o3d.geometry.LineSet()
    camera_line.points = o3d.utility.Vector3dVector(camera_positions)
    camera_line.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(camera_positions)-1)])
    camera_line.colors = o3d.utility.Vector3dVector([trajectory_color for _ in range(len(camera_positions)-1)])
    
    # Create camera position spheres
    camera_spheres = []
    for pos in camera_positions[::max(1, len(camera_positions) // 50)]:  # Every Nth camera
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=camera_size)
        sphere.translate(pos)
        sphere.paint_uniform_color(trajectory_color)
        camera_spheres.append(sphere)
    
    # Save point cloud
    o3d.io.write_point_cloud(str(output_path / "pointcloud.ply"), pcd)
    print(f"Saved point cloud to {output_path / 'pointcloud.ply'}")
    
    # Visualize (optional, can be commented out for headless operation)
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.add_geometry(pcd)
        vis.add_geometry(camera_line)
        for sphere in camera_spheres:
            vis.add_geometry(sphere)
        
        # Set view point
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.7)
        
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Could not display Open3D visualization: {e}")
        print("Point cloud saved to PLY file for viewing in other software")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RGB point clouds with scaled camera trajectory"
    )
    
    parser.add_argument(
        "--sequence_dir",
        type=Path,
        required=True,
        help="Directory containing sequence data (with *_rgb.jpg, *_metric_depth.npy, *_scaled_rotation.npy files)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--intrinsic_file",
        type=Path,
        default=None,
        help="Optional: Path to camera intrinsic matrix file (npy format)"
    )
    
    parser.add_argument(
        "--focal_length",
        type=float,
        default=5000,
        help="Optional: Camera focal length in pixels (for intrinsic estimation)"
    )
    
    parser.add_argument(
        "--depth_threshold",
        type=float,
        default=50.0,
        help="Maximum depth value to consider (meters, default: 50.0)"
    )
    
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for point clouds (default: 2, use every 2nd pixel)"
    )
    
    parser.add_argument(
        "--max_points_per_frame",
        type=int,
        default=50000,
        help="Maximum points to visualize per frame (default: 50000)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Sequence directory: {args.sequence_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load sequence data
    print("🔄 Loading sequence data...")
    depth_maps, rgb_images, camera_poses, frame_ids = load_sequence_data(args.sequence_dir)
    print(rgb_images[0].shape, "what", rgb_images[1].shape)
    if len(depth_maps) == 0:
        print("❌ Error: No depth maps found")
        return
    
    print(f"✅ Loaded {len(depth_maps)} frames")
    
    # Load or estimate camera intrinsics
    if args.intrinsic_file and args.intrinsic_file.exists():
        intrinsic = np.load(args.intrinsic_file)
        print(f"✅ Loaded intrinsics from file: {intrinsic.shape}")
    else:
        # Estimate from first image
        print("🔍 Estimating camera intrinsics from image dimensions...")
        intrinsic = estimate_intrinsics_from_image(rgb_images[0], default_fov=3.0)
        if args.focal_length:
            intrinsic[0, 0] = args.focal_length
            intrinsic[1, 1] = args.focal_length
        print(f"✅ Estimated intrinsics:\n{intrinsic}")
    
    # Convert depth maps to point clouds
    print("🔄 Converting depth maps to point clouds...")
    point_clouds = []
    colors = []
    
    for i, (depth_map, rgb_image, camera_pose) in enumerate(zip(depth_maps, rgb_images, camera_poses)):
        if i % 10 == 0:
            print(f"  Processing frame {i+1}/{len(depth_maps)}")
        depth_map = depth_map[0,0]
        intrinsic = np.array([[928., 0., 259.],
                  [0., 956., 259.],
                  [0., 0., 1.]], dtype=np.float32)
        points, cols = depth_to_pointcloud(
            depth_map,
            rgb_image,
            camera_pose,
            intrinsic,
            depth_threshold=args.depth_threshold,
            subsample=args.subsample
        )
        point_clouds.append(points)
        colors.append(cols)
    
    print(f"✅ Created {len(point_clouds)} point clouds")
    
    # Visualize
    print("🎨 Creating visualizations...")
    visualize_pointcloud_with_trajectory(
        point_clouds,
        colors,
        camera_poses,
        args.output_dir,
        max_points_per_frame=args.max_points_per_frame
    )
    
    print("🎉 Done!")


if __name__ == "__main__":
    main()
