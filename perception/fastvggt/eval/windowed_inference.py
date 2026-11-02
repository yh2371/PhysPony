#!/usr/bin/env python3
"""
Windowed Inference for FastVGGT
Processes long sequences in overlapping windows and stitches them together with smoothing.
"""

import argparse
import numpy as np
import torch
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import json
from tqdm import tqdm

# Ensure project root is in sys.path for absolute imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    warp_depth_map,
    save_depth_map,
)


class WindowedInferenceProcessor:
    """Processes sequences in overlapping windows and stitches them together."""
    
    def __init__(self, model, dtype=torch.bfloat16, window_size=20, overlap=5):
        """
        Args:
            model: VGGT model
            dtype: Model dtype
            window_size: Number of frames per window
            overlap: Number of overlapping frames between windows
        """
        self.model = model
        self.dtype = dtype
        self.window_size = window_size
        self.overlap = overlap
        
    def process_sequence_windows(self, image_paths: List[Path], 
                                depth_conf_thresh: float = 0.0) -> Dict:
        """
        Process a sequence in overlapping windows.
        
        Args:
            image_paths: List of image paths
            depth_conf_thresh: Depth confidence threshold
            
        Returns:
            Dictionary containing window results and metadata
        """
        total_frames = len(image_paths)
        print(f"🔄 Processing {total_frames} frames in windows of {self.window_size} with {self.overlap} overlap")
        
        window_results = []
        window_metadata = []
        
        # Calculate window positions
        step_size = self.window_size - self.overlap
        num_windows = max(1, (total_frames - self.overlap + step_size - 1) // step_size)
        
        for window_idx in tqdm(range(num_windows), desc="Processing windows"):
            start_idx = window_idx * step_size
            end_idx = min(start_idx + self.window_size, total_frames)
            
            # Ensure minimum window size
            if end_idx - start_idx < 3:
                continue
                
            window_paths = image_paths[start_idx:end_idx]
            print(f"  Window {window_idx+1}/{num_windows}: frames {start_idx}-{end_idx-1} ({len(window_paths)} frames)")
            
            try:
                # Load images for this window
                images = load_images_rgb(window_paths)
                if len(images) < 3:
                    print(f"    ⚠️  Skipping window {window_idx} - not enough valid images")
                    continue
                    
                # Prepare VGG input
                images_array = np.stack(images)
                vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
                
                # Update model patch dimensions
                self.model.update_patch_dimensions(patch_width, patch_height)
                
                # Run inference
                (
                    extrinsic_np,
                    intrinsic_np,
                    all_world_points,
                    all_point_colors,
                    all_cam_to_world_mat,
                    inference_time_ms,
                    depth_maps,
                ) = infer_vggt_and_reconstruct(
                    self.model, vgg_input, self.dtype, depth_conf_thresh, window_paths
                )
                if all_cam_to_world_mat and len(all_cam_to_world_mat) > 0:
                    window_results.append({
                        'poses': all_cam_to_world_mat,
                        'world_points': all_world_points,
                        'point_colors': all_point_colors,
                        'extrinsic': extrinsic_np,
                        'intrinsic': intrinsic_np,
                        'inference_time_ms': inference_time_ms,
                        'depth_maps': depth_maps
                    })
                    
                    window_metadata.append({
                        'window_idx': window_idx,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'frame_count': len(all_cam_to_world_mat)
                    })
                    print(f"    ✅ Window {window_idx} completed: {len(all_cam_to_world_mat)} poses")
                else:
                    print(f"    ❌ Window {window_idx} failed - no valid poses")
                    
            except Exception as e:
                print(f"    ❌ Window {window_idx} failed with error: {e}")
                continue
        
        return {
            'window_results': window_results,
            'window_metadata': window_metadata,
            'total_windows': len(window_results)
        }


class TrajectoryStitcher:
    """Stitches windowed trajectories together with alignment and smoothing."""
    
    def __init__(self, smoothing_factor=0.1, alignment_method='icp'):
        """
        Args:
            smoothing_factor: Factor for trajectory smoothing (0.0 = no smoothing, 1.0 = heavy smoothing)
            alignment_method: Method for aligning windows ('icp', 'translation_only', 'rotation_only')
        """
        self.smoothing_factor = smoothing_factor
        self.alignment_method = alignment_method
        
    def align_windows(self, window_results: List[Dict], 
                     window_metadata: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align windows to create a coherent trajectory.
        
        Args:
            window_results: List of window inference results
            window_metadata: List of window metadata
            
        Returns:
            Tuple of (aligned poses, aligned depth maps)
        """
        if not window_results:
            return [], []
            
        print(f"🔗 Aligning {len(window_results)} windows...")
        
        aligned_poses = []
        aligned_depth_maps = []
        cumulative_transform = np.eye(4)  # Identity transformation
        
        for i, (window_result, metadata) in enumerate(zip(window_results, window_metadata)):
            poses = np.array(window_result['poses'])
            depth_maps = window_result.get('depth_maps', None)
            intrinsic = window_result.get('intrinsic', None)
            
            if i == 0:
                # First window - use as reference
                aligned_poses.extend(poses)
                if depth_maps is not None:
                    aligned_depth_maps.extend(depth_maps)
                print(f"  Window {i}: Reference (no alignment)")
            else:
                # Align subsequent windows
                aligned_poses_window, aligned_depth_window = self._align_window_to_reference(
                    poses, aligned_poses, metadata, depth_maps, intrinsic
                )
                aligned_poses.extend(aligned_poses_window)
                if aligned_depth_window is not None:
                    aligned_depth_maps.extend(aligned_depth_window)
                print(f"  Window {i}: Aligned {len(poses)} poses")
        
        return aligned_poses, aligned_depth_maps
    
    def _align_window_to_reference(self, window_poses: np.ndarray, 
                                  reference_poses: List[np.ndarray],
                                  metadata: Dict, depth_maps: np.ndarray = None,
                                  intrinsic: np.ndarray = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Align a window to the reference trajectory.
        
        Args:
            window_poses: Poses from current window
            reference_poses: Previously aligned poses
            metadata: Window metadata
            depth_maps: Depth maps for current window
            intrinsic: Camera intrinsic matrix
            
        Returns:
            Tuple of (aligned poses, aligned depth maps)
        """
        if not reference_poses or len(window_poses) == 0:
            return window_poses.tolist(), depth_maps.tolist() if depth_maps is not None else []
        
        # Get overlap region for alignment
        overlap_size = 5  # Use last 5 frames of reference for alignment
        if len(reference_poses) >= overlap_size:
            reference_overlap = reference_poses[-overlap_size:]
            window_overlap = window_poses[:overlap_size]
            
            # Compute alignment transformation (rotation + translation [+ scale
            # if alignment_method == 'icp_scale'])
            transform, scale = self._compute_alignment_transform(
                window_overlap, reference_overlap
            )

            # Apply transformation to all poses in window. Scale must only touch
            # position, not the rotation block, or the pose stops being a valid
            # camera-to-world matrix (rotation submatrix would no longer be
            # orthonormal).
            R_align = transform[:3, :3]
            t_align = transform[:3, 3]
            aligned_poses = []
            for pose in window_poses:
                aligned_pose = np.eye(4)
                aligned_pose[:3, :3] = R_align @ pose[:3, :3]
                aligned_pose[:3, 3] = scale * (R_align @ pose[:3, 3]) + t_align
                aligned_poses.append(aligned_pose)
            
            # Warp depth maps if available
            aligned_depth_maps = []
            if depth_maps is not None and intrinsic is not None:
                for i in range(len(depth_maps)):
                    depth_map = depth_maps[i]
                    print(f"Depth map shape: {depth_map.shape}")
                    warped_depth = warp_depth_map(depth_map, transform, intrinsic[i])
                    aligned_depth_maps.append(warped_depth)
            else:
                aligned_depth_maps = depth_maps.tolist() if depth_maps is not None else []
            
            return aligned_poses, aligned_depth_maps
        else:
            return window_poses.tolist(), depth_maps.tolist() if depth_maps is not None else []
    
    def _compute_alignment_transform(self, source_poses: np.ndarray,
                                   target_poses: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Compute transformation to align source poses to target poses.

        Args:
            source_poses: Source poses (N, 4, 4)
            target_poses: Target poses (list of 4x4 matrices)

        Returns:
            (transform, scale): transform is a (4,4) matrix holding rotation in
            [:3,:3] and translation in [:3,3]; scale is a separate multiplicative
            factor that must be applied to position only (see call site).
            'translation_only'/'rotation_only'/'icp' (default) always return
            scale=1.0, matching original behavior; 'icp_scale' is an
            experimental variant that also solves for scale (see its comment
            for why it's not recommended as-is).
        """
        if len(source_poses) == 0 or len(target_poses) == 0:
            return np.eye(4), 1.0

        # Convert target poses to numpy array
        target_poses = np.array(target_poses)

        # Extract translations
        source_trans = source_poses[:, :3, 3]  # (N, 3)
        target_trans = target_poses[:, :3, 3]  # (N, 3)

        if self.alignment_method == 'translation_only':
            # Only align translations
            translation_diff = np.mean(target_trans - source_trans, axis=0)
            transform = np.eye(4)
            transform[:3, 3] = translation_diff
            return transform, 1.0

        elif self.alignment_method == 'rotation_only':
            # Only align rotations
            source_rots = Rotation.from_matrix(source_poses[:, :3, :3])
            target_rots = Rotation.from_matrix(target_poses[:, :3, :3])

            # Compute average rotation difference
            rot_diff = (target_rots * source_rots.inv()).mean()
            transform = np.eye(4)
            transform[:3, :3] = rot_diff.as_matrix()
            return transform, 1.0

        elif self.alignment_method == 'icp_scale':
            # EXPERIMENTAL, not currently recommended: similarity (rotation +
            # translation + scale) via Umeyama's closed-form fit on the
            # overlap-region camera centers, to test whether FastVGGT's
            # per-window reconstructions have inconsistent internal scale.
            # Tried on 20201128_ID_2_0006 (window_size=20, overlap=5): the
            # per-window scale estimate swung from 0.17x to 7.0x (should be
            # ~1x if windows share scale), because 5 overlap points spanning
            # only a few consecutive frames give too little baseline/motion
            # to constrain a scale fit - the estimate is dominated by pose
            # noise, not signal. Compounding ~40 such noisy factors across a
            # sequence made recovered displacement *worse* than the plain
            # rigid fit (path-length recovery dropped from 10.5% to 1.5% of
            # true mocap distance on that sequence), so this is not wired up
            # as the default. Left here in case a future attempt increases
            # the overlap window enough to get a stable estimate, or adds
            # regularization (e.g. shrink scale toward 1.0, or require a
            # minimum overlap baseline before trusting a non-unit scale).
            mu_s = source_trans.mean(axis=0)
            mu_t = target_trans.mean(axis=0)
            Xs = source_trans - mu_s
            Xt = target_trans - mu_t

            var_s = np.mean(np.sum(Xs ** 2, axis=1))
            if var_s < 1e-12:
                translation_diff = np.mean(target_trans - source_trans, axis=0)
                source_rots = Rotation.from_matrix(source_poses[:, :3, :3])
                target_rots = Rotation.from_matrix(target_poses[:, :3, :3])
                rot_diff = (target_rots * source_rots.inv()).mean()
                transform = np.eye(4)
                transform[:3, :3] = rot_diff.as_matrix()
                transform[:3, 3] = translation_diff
                return transform, 1.0

            H = Xs.T @ Xt / len(Xs)
            U, D, Vt = np.linalg.svd(H)
            S = np.eye(3)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S[2, 2] = -1.0
            R = Vt.T @ S @ U.T
            scale = float(np.trace(np.diag(D) @ S) / var_s)
            t = mu_t - scale * (R @ mu_s)

            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t
            return transform, scale

        else:  # 'icp' or default
            # Rigid alignment (rotation + translation); metric scale is
            # recovered later by align_depth.py.
            translation_diff = np.mean(target_trans - source_trans, axis=0)

            source_rots = Rotation.from_matrix(source_poses[:, :3, :3])
            target_rots = Rotation.from_matrix(target_poses[:, :3, :3])
            rot_diff = (target_rots * source_rots.inv()).mean()

            transform = np.eye(4)
            transform[:3, :3] = rot_diff.as_matrix()
            transform[:3, 3] = translation_diff
            return transform, 1.0
    
    def smooth_trajectory(self, poses: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply smoothing to the trajectory.
        
        Args:
            poses: List of camera-to-world matrices
            
        Returns:
            Smoothed poses
        """
        if len(poses) < 3 or self.smoothing_factor <= 0.0:
            return poses
        
        print(f"🌊 Smoothing trajectory with factor {self.smoothing_factor}...")
        
        poses_array = np.array(poses)
        
        # Extract positions and rotations
        positions = poses_array[:, :3, 3]  # (N, 3)
        rotations = Rotation.from_matrix(poses_array[:, :3, :3])
        
        # Apply smoothing
        smoothed_positions = self._smooth_positions(positions)
        smoothed_rotations = self._smooth_rotations(rotations)
        
        # Reconstruct poses
        smoothed_poses = []
        for i in range(len(poses)):
            pose = np.eye(4)
            pose[:3, :3] = smoothed_rotations[i].as_matrix()
            pose[:3, 3] = smoothed_positions[i]
            smoothed_poses.append(pose)
        
        return smoothed_poses
    
    def _smooth_positions(self, positions: np.ndarray) -> np.ndarray:
        """Smooth position trajectory using moving average."""
        if self.smoothing_factor <= 0.0:
            return positions
        
        window_size = max(3, int(len(positions) * self.smoothing_factor))
        smoothed = np.zeros_like(positions)
        
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def _smooth_rotations(self, rotations: Rotation) -> Rotation:
        """Smooth rotation trajectory using SLERP."""
        if self.smoothing_factor <= 0.0:
            return rotations
        
        window_size = max(3, int(len(rotations) * self.smoothing_factor))
        smoothed_quats = []
        
        for i in range(len(rotations)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(rotations), i + window_size // 2 + 1)
            
            if end_idx - start_idx > 1:
                # Use SLERP for rotation smoothing
                quat_weights = np.ones(end_idx - start_idx)
                quat_weights /= np.sum(quat_weights)
                
                # Weighted average of quaternions
                quats = rotations[start_idx:end_idx].as_quat()
                avg_quat = np.average(quats, axis=0, weights=quat_weights)
                
                # Normalize
                avg_quat = avg_quat / np.linalg.norm(avg_quat)
                smoothed_quats.append(avg_quat)
            else:
                smoothed_quats.append(rotations[i].as_quat())
        
        return Rotation.from_quat(smoothed_quats)


class TrajectoryVisualizer:
    """Visualizes trajectories before and after processing."""
    
    @staticmethod
    def visualize_trajectories(before_poses: List[np.ndarray], 
                              after_poses: List[np.ndarray],
                              output_path: Path,
                              title: str = "Trajectory Comparison"):
        """
        Create comparison visualization of trajectories.
        
        Args:
            before_poses: Original poses
            after_poses: Processed poses
            output_path: Output directory
            title: Plot title
        """
        print(f"📊 Creating trajectory visualization...")
        
        # Extract positions
        before_pos = np.array([pose[:3, 3] for pose in before_poses])
        after_pos = np.array([pose[:3, 3] for pose in after_poses])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # 2D XZ projection
        axes[0, 0].plot(before_pos[:, 0], before_pos[:, 2], 'r-', linewidth=2, label='Before', alpha=0.7)
        axes[0, 0].plot(after_pos[:, 0], after_pos[:, 2], 'b-', linewidth=2, label='After')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Z (m)')
        axes[0, 0].set_title('XZ Plane Projection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # 2D XY projection
        axes[0, 1].plot(before_pos[:, 0], before_pos[:, 1], 'r-', linewidth=2, label='Before', alpha=0.7)
        axes[0, 1].plot(after_pos[:, 0], after_pos[:, 1], 'b-', linewidth=2, label='After')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Y (m)')
        axes[0, 1].set_title('XY Plane Projection')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        
        # Position over time
        axes[1, 0].plot(before_pos[:, 0], 'r-', linewidth=2, label='Before X', alpha=0.7)
        axes[1, 0].plot(after_pos[:, 0], 'b-', linewidth=2, label='After X')
        axes[1, 0].plot(before_pos[:, 1], 'r--', linewidth=2, label='Before Y', alpha=0.7)
        axes[1, 0].plot(after_pos[:, 1], 'b--', linewidth=2, label='After Y')
        axes[1, 0].plot(before_pos[:, 2], 'r:', linewidth=2, label='Before Z', alpha=0.7)
        axes[1, 0].plot(after_pos[:, 2], 'b:', linewidth=2, label='After Z')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].set_title('Position vs Frame')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3D trajectory
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax_3d.plot(before_pos[:, 0], before_pos[:, 1], before_pos[:, 2], 'r-', linewidth=2, label='Before', alpha=0.7)
        ax_3d.plot(after_pos[:, 0], after_pos[:, 1], after_pos[:, 2], 'b-', linewidth=2, label='After')
        ax_3d.scatter(before_pos[0, 0], before_pos[0, 1], before_pos[0, 2], color='green', s=100, label='Start')
        ax_3d.scatter(before_pos[-1, 0], before_pos[-1, 1], before_pos[-1, 2], color='red', s=100, label='End')
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Trajectory')
        ax_3d.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "trajectory_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"📊 Trajectory visualization saved: {plot_path}")
        
        # Save trajectory statistics
        stats = {
            'before': {
                'num_frames': len(before_poses),
                'total_distance': np.sum(np.linalg.norm(np.diff(before_pos, axis=0), axis=1)),
                'position_std': np.std(before_pos, axis=0).tolist(),
                'position_range': [np.min(before_pos, axis=0).tolist(), np.max(before_pos, axis=0).tolist()]
            },
            'after': {
                'num_frames': len(after_poses),
                'total_distance': np.sum(np.linalg.norm(np.diff(after_pos, axis=0), axis=1)),
                'position_std': np.std(after_pos, axis=0).tolist(),
                'position_range': [np.min(after_pos, axis=0).tolist(), np.max(after_pos, axis=0).tolist()]
            }
        }
        
        stats_path = output_path / "trajectory_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Trajectory statistics saved: {stats_path}")
        
        return stats


def main():
    """Main function for windowed inference processing."""
    parser = argparse.ArgumentParser(
        description="Process long sequences with windowed inference and stitching"
    )
    
    # Required arguments
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset path containing images"
    )
    
    parser.add_argument(
        "--out_dir", 
        type=Path,
        required=True,
        help="Output path for results"
    )
    
    # Model arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get(
            "FASTVGGT_CKPT",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))),
                "checkpoints", "fastvggt", "model_tracker_fixed_e20.pt"),
        ),
        help="Model checkpoint path (env FASTVGGT_CKPT or <repo>/checkpoints/fastvggt/...)"
    )
    
    # Processing parameters
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Number of frames per window"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=5,
        help="Number of overlapping frames between windows"
    )
    
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.1,
        help="Trajectory smoothing factor (0.0 = no smoothing, 1.0 = heavy smoothing)"
    )
    
    parser.add_argument(
        "--alignment_method",
        type=str,
        choices=['icp', 'translation_only', 'rotation_only'],
        default='icp',
        help="Window alignment method"
    )
    
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=0.0,
        help="Depth confidence threshold"
    )
    
    args = parser.parse_args()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting windowed inference processing")
    print(f"📁 Data path: {args.data_dir}")
    print(f"📁 Output path: {args.out_dir}")
    print(f"⚙️  Window size: {args.window_size}, Overlap: {args.overlap}")
    print(f"🌊 Smoothing factor: {args.smoothing_factor}")
    
    # Load model
    print(f"🔄 Loading model: {args.ckpt_path}")
    model = VGGT(merging=0, vis_attn_map=False)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    print(f"✅ Model loaded")
    
    # Load images
    print(f"🖼️  Loading images...")
    image_paths = get_sorted_image_paths(args.data_dir)
    image_paths = [path for path in image_paths if "rgb.jpg" in path.name]
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {args.data_dir}")
        return
    
    print(f"🖼️  Found {len(image_paths)} images")
    
    # Process with windowed inference
    processor = WindowedInferenceProcessor(
        model, 
        dtype=torch.bfloat16,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    window_results = processor.process_sequence_windows(
        image_paths, 
        args.depth_conf_thresh
    )
    
    if window_results['total_windows'] == 0:
        print(f"❌ No valid windows processed")
        return
    
    print(f"✅ Processed {window_results['total_windows']} windows")
    
    # Stitch trajectories together
    stitcher = TrajectoryStitcher(
        smoothing_factor=args.smoothing_factor,
        alignment_method=args.alignment_method
    )
    
    # Collect all poses before alignment (for comparison)
    before_poses = []
    for window_result in window_results['window_results']:
        before_poses.extend(window_result['poses'])
    
    # Align and smooth
    aligned_poses, aligned_depth_maps = stitcher.align_windows(
        window_results['window_results'],
        window_results['window_metadata']
    )
    
    smoothed_poses = stitcher.smooth_trajectory(aligned_poses)
    
    print(f"✅ Final trajectory has {len(smoothed_poses)} poses")
    
    # Save results
    print(f"💾 Saving results...")
    
    # Save poses
    print(args.out_dir)
    poses_output_path = args.out_dir / "final_trajectory.txt"
    print(poses_output_path)
    with open(poses_output_path, "w") as f:
        for i, pose in enumerate(smoothed_poses):
            f.write(f"# Frame {i}\n")
            for row in pose:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n")
    
    # Save before poses for comparison
    before_poses_output_path = args.out_dir / "before_trajectory.txt"
    with open(before_poses_output_path, "w") as f:
        for i, pose in enumerate(before_poses):
            f.write(f"# Frame {i}\n")
            for row in pose:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n")
    
    # Save depth maps
    if aligned_depth_maps and len(aligned_depth_maps) > 0:
        print(f"💾 Saving {len(aligned_depth_maps)} depth maps...")
        depth_output_dir = args.out_dir / "depth_maps"
        depth_output_dir.mkdir(exist_ok=True)
        print(depth_output_dir, os.path.exists(depth_output_dir))
        
        for i, depth_map in enumerate(aligned_depth_maps):
            save_depth_map(depth_map, depth_output_dir, i, "depth.png")
        
        print(f"📁 Depth maps saved to: {depth_output_dir}")
    
    # Visualize trajectories
    visualizer = TrajectoryVisualizer()
    stats = visualizer.visualize_trajectories(
        before_poses,
        smoothed_poses,
        args.out_dir,
        f"Windowed Inference Results (Windows: {window_results['total_windows']})"
    )
    
    # Save processing metadata
    metadata = {
        'window_size': args.window_size,
        'overlap': args.overlap,
        'smoothing_factor': args.smoothing_factor,
        'alignment_method': args.alignment_method,
        'total_windows': window_results['total_windows'],
        'total_frames': len(smoothed_poses),
        'processing_stats': stats
    }
    metadata_path = args.out_dir / "processing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"🎉 Processing complete!")
    print(f"📁 Results saved to: {args.out_dir}")
    print(f"📊 Final trajectory: {len(smoothed_poses)} poses")


if __name__ == "__main__":
    main()
