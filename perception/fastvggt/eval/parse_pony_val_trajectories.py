#!/usr/bin/env python3
"""
Parse PonyVal trajectory results and save rotation matrices per-frame.
Reads final_trajectory.txt files and matches them with actual image frame numbers.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import argparse
import glob
from tqdm import tqdm
import json


def load_trajectory_txt(fpath):
    """
    Load trajectory from text file format (4x4 matrices).
    
    Args:
        fpath: Path to trajectory text file
        
    Returns:
        np.ndarray: Camera poses as 4x4 transformation matrices (T, 4, 4)
        list: Frame indices from the file
    """
    poses = []
    frame_indices = []
    current_pose = []
    
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Frame '):
                # Extract frame index
                frame_idx = int(line.split('# Frame ')[1])
                frame_indices.append(frame_idx)
                continue
            elif line == '':
                if len(current_pose) == 4:
                    poses.append(np.array(current_pose, dtype=np.float32))
                    current_pose = []
            else:
                row = [float(x) for x in line.split()]
                if len(row) == 4:
                    current_pose.append(row)
    
    # Handle last pose if file doesn't end with empty line
    if len(current_pose) == 4:
        poses.append(np.array(current_pose, dtype=np.float32))
    
    if not poses:
        raise ValueError(f"No valid poses found in {fpath}")
    
    poses_array = np.stack(poses, axis=0)  # (T, 4, 4)
    return poses_array, frame_indices


def get_image_frame_numbers(image_dir):
    """
    Get the actual frame numbers from image filenames.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        list: Sorted list of frame numbers
    """
    image_files = glob.glob(os.path.join(image_dir, "*rgb.jpg"))
    frame_numbers = []
    
    for img_file in image_files:
        # Extract frame number from filename
        # Assuming format like "frame_00123_rgb.jpg" or "00123_rgb.jpg"
        filename = os.path.basename(img_file)
        frame_str = filename.split("_rgb.jpg")[0]
        frame_numbers.append(frame_str)

    return sorted(frame_numbers)


def extract_rotation_matrices(poses_array):
    """
    Pass through the 4x4 camera-to-world transforms unchanged (downstream
    consumers expect the full 4x4, not just the rotation submatrix).

    Args:
        poses_array: Camera poses as 4x4 transformation matrices (T, 4, 4)

    Returns:
        np.ndarray: Camera-to-world transforms (T, 4, 4)
    """
    # poses_array is already the full 4x4 camera-to-world transform; kept as-is
    # (not reduced to the top-left 3x3) since downstream consumers expect 4x4.
    return poses_array


def save_rotation_matrices(rotations, frame_indices, actual_frame_numbers, output_dir):
    """
    Save rotation matrices with correct frame numbering.
    
    Args:
        rotations: Camera rotation matrices (T, 3, 3)
        frame_indices: Frame indices from trajectory file
        actual_frame_numbers: Actual frame numbers from image files
        output_dir: Output directory to save .npy files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mapping from trajectory frame indices to actual frame numbers
    if len(frame_indices) != len(actual_frame_numbers):
        raise ValueError(f"Mismatch between trajectory frames ({len(frame_indices)}) and image frames ({len(actual_frame_numbers)})")


    # Save rotation matrix for each frame
    for i, (traj_frame_idx, actual_frame_num, rotation) in enumerate(zip(frame_indices, actual_frame_numbers, rotations)):
        # Save with actual frame number
        output_file = os.path.join(output_dir, f"{actual_frame_num}_rotation.npy")
        np.save(output_file, rotation)
        
        # Also save metadata for debugging
        if i == 0:  # Save metadata only once
            metadata = {
                "total_frames": len(rotations),
                "trajectory_frame_indices": frame_indices,
                "actual_frame_numbers": actual_frame_numbers,
                "rotation_matrix_shape": list(rotations.shape)
            }
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

def save_depth_matrices(depth_paths, frame_indices, actual_frame_numbers, output_dir):
    """
    Save depth matrices with correct frame numbering.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, (traj_frame_idx, actual_frame_num, depth_path) in enumerate(zip(frame_indices, actual_frame_numbers, depth_paths)):
        # Save with actual frame number
        depth = np.load(depth_path)
        output_file = os.path.join(output_dir, f"{actual_frame_num}_vggt_depth.npy")
        np.save(output_file, depth)

def process_sequence(sequence_dir, output_base_dir, image_base=None):
    """
    Process a single sequence directory.
    
    Args:
        sequence_dir: Path to sequence directory
        output_base_dir: Base output directory
        image_base: Root folder that contains per-sequence image subfolders
                    (the sequence_id is appended). Defaults to output_base_dir.
    """
    sequence_id = "_".join(sequence_dir.name.split("_")[:-4])
    
    # Look for final_trajectory.txt
    trajectory_file = sequence_dir / "final_trajectory.txt"
    if not trajectory_file.exists():
        print(f"Warning: No final_trajectory.txt found in {sequence_dir}")
        return False
    
    # Root folder holding per-sequence image subfolders. Configurable via
    # --image_dir (falls back to the output directory).
    if image_base is None:
        image_base = str(output_base_dir)
    image_dir = os.path.join(str(image_base), sequence_id)
    print(f"Processing sequence {sequence_id}, using image dir: {image_dir}")
    
    try:
        # Load trajectory
        poses_array, frame_indices = load_trajectory_txt(trajectory_file)
        depth_paths = glob.glob(os.path.join(sequence_dir, "depth_maps/*.npy"))
        print(f"Loaded {len(poses_array)} poses from trajectory")
        new_poses_array = []
        step_size = 5
        window_size = 20
        new_depth_paths = []
        for i in range(0,len(frame_indices),window_size):
            if i + window_size >= len(frame_indices):
                new_poses_array.append(poses_array[i:i+window_size])
                new_depth_paths.extend(depth_paths[i:i+window_size])
            else:
                new_poses_array.append(poses_array[i:i+window_size-step_size])
                new_depth_paths.extend(depth_paths[i:i+window_size-step_size])

        poses_array = np.concatenate(new_poses_array, axis=0)
        frame_indices = list(range(0,len(poses_array)))

        # Get actual frame numbers from images
        actual_frame_numbers = get_image_frame_numbers(str(image_dir))
        print(f"Found {len(actual_frame_numbers)} image frames")
        
        # Extract rotation matrices
        rotations = extract_rotation_matrices(poses_array)
        print(f"Extracted rotation matrices with shape: {rotations.shape}")
        
        # Create output directory
        try:
            output_dir = output_base_dir / f"{int(sequence_id):07d}"
        except:
            output_dir = output_base_dir / f"{sequence_id}"
        
        # Save rotation matrices
        save_rotation_matrices(rotations, frame_indices, actual_frame_numbers, output_dir)

        save_depth_matrices(new_depth_paths, frame_indices, actual_frame_numbers, output_dir)
        
        print(f"✅ Successfully processed sequence {sequence_id}")
        print(f"   Saved {len(rotations)} rotation matrices to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing sequence {sequence_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Parse PonyVal trajectory results and save rotation matrices"
    )
    
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing eval_pony_val results"
    )
    
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./fastvggt_out"),
        help="Output directory for rotation matrices"
    )
    
    parser.add_argument(
        "--sequence_ids",
        type=str,
        nargs='+',
        default=None,
        help="Specific sequence IDs to process (default: all)"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root folder holding per-sequence image subfolders (sequence_id is "
             "appended). Defaults to --out_dir.",
    )
    
    args = parser.parse_args()
    
    # Check input directory
    if not args.data_dir.exists():
        print(f"❌ Error: eval_pony_val directory does not exist: {args.data_dir}")
        return
    
    # Find sequence directories
    sequence_dirs = []
    for item in args.data_dir.iterdir():
        if item.is_dir():
            if args.sequence_ids is None or item.name in args.sequence_ids:
                sequence_dirs.append(item)

    if not sequence_dirs:
        print(f"❌ No sequence directories found in {args.data_dir}")
        return

    sequence_dirs = sorted(sequence_dirs, key=lambda x: x.name)

    print(f"🔍 Found {len(sequence_dirs)} sequences to process:")
    for seq_dir in sequence_dirs:
        print(f"  - {seq_dir.name}")
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sequence
    successful = 0
    failed = 0
    
    for seq_dir in tqdm(sequence_dirs, desc="Processing sequences"):
        if process_sequence(seq_dir, args.out_dir, image_base=args.image_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Processing Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
