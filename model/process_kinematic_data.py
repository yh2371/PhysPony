#!/usr/bin/env python3
"""
Standalone script to process kinematic data using physics pipeline for ground alignment.
This script processes pkl files and saves the ground-aligned data as npy files.
"""

import os
import sys
import re
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.mjx import pipeline as mjx_pipeline
from scipy.spatial.transform import Rotation as R
from phc_torch_utils import quat_to_exp_map
import torch.nn.functional as F

from scale_utils import add_trans_scale_args, resolve_trans_scale
from coord_utils import fix_camera_axes

# Add the model path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "dataset"))

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)

def load_pkl_data(pkl_path: Path, trans_scale: float = 1.0) -> Dict:
    """Load and process a single pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    M = data["rotation"]
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    if M.ndim == 3:
        M = M[0]
    assert M.shape == (4, 4), f"rotation in {pkl_path} must be 4x4, got {M.shape}"

    # Apply coordinate transformations
    M = fix_camera_axes(M)

    R_c2w = M[:3, :3]
    T_c2w = M[:3, 3]

    betas = torch.as_tensor(data["betas"], dtype=torch.float32).view(1, -1)
    pose6d = torch.as_tensor(data["pose"], dtype=torch.float32)
    if pose6d.shape[-1] == 6 or pose6d.shape[-1] == 36*6:
        pose6d = pose6d.view(36, -1)
        pose_R = rot6d_to_rotmat(pose6d).view(36, 3, 3).numpy()
    else:
        pose_R = pose6d.view(36, 3, 3).numpy()
    trans = trans_scale * torch.as_tensor(data["trans"], dtype=torch.float32).view(1, 3).numpy()

    pose_root = R_c2w @ pose_R[0:1]
    trans = (R_c2w @ trans.T).T + T_c2w[None, :]

    pose_root = R.from_matrix(pose_root).as_quat()
    poses = R.from_matrix(pose_R[1:]).as_quat()

    return {
        "poses_root": pose_root[0], 
        "betas": betas, 
        "poses": poses, 
        "trans": trans[0]
    }

def process_poses_with_physics_pipeline(ref_motion: Dict, sys, full: bool = True) -> Dict:
    """
    Process poses using physics pipeline to detect contact distances and align to ground.
    
    Args:
        ref_motion: Dictionary containing poses, poses_root, betas, trans
        sys: Physics system from mjcf.load
        full: Whether to use full pose representation
    
    Returns:
        Processed reference motion with ground-aligned translation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert poses to the format expected by the physics system
    poses = torch.tensor(ref_motion['poses'], dtype=torch.float32, device=device)
    poses_root = torch.tensor(ref_motion['poses_root'], dtype=torch.float32, device=device)
    betas = torch.tensor(ref_motion['betas'], dtype=torch.float32, device=device)
    trans = torch.tensor(ref_motion['trans'], dtype=torch.float32, device=device)
    
    T = poses.shape[0]
    
    # Convert quaternions to the format expected by the physics system
    root_rot = poses_root.clone()
    _root_rot = root_rot.clone()
    _root_rot_x = _root_rot[:, 0] 
    _root_rot_y = _root_rot[:, 1]
    _root_rot_z = _root_rot[:, 2]
    _root_rot_w = _root_rot[:, 3]
    root_rot[:, 0] = _root_rot_w
    root_rot[:, 1] = _root_rot_x
    root_rot[:, 2] = _root_rot_y
    root_rot[:, 3] = _root_rot_z 
    
    # Convert to exp map for physics system
    dof_pos = quat_to_exp_map(poses)
    N, _, _ = dof_pos.shape
    dof_pos = dof_pos.reshape(N, -1)
    
    # Create qpos for physics system
    root_transl = trans.clone()
    root_transl[:, 2] = 0 # Add some height offset
    
    if full:
        qpos_list = [root_transl, root_rot, dof_pos]
        root_exp_map = quat_to_exp_map(_root_rot)
        qpos_exp_map_list = [root_transl, root_exp_map, dof_pos]
    else:
        qpos_list = [root_transl, root_rot]
        root_exp_map = quat_to_exp_map(_root_rot)
        qpos_exp_map_list = [root_transl, root_exp_map]
        # Add selective body parts (simplified for now)
        qpos_list.append(dof_pos)
        qpos_exp_map_list.append(dof_pos)
    
    qpos_tensor = torch.cat(qpos_list, dim=1)
    qpos_exp_map_tensor = torch.cat(qpos_exp_map_list, dim=1)
    
    # Convert to numpy for JAX processing
    qpos_np = qpos_tensor.detach().cpu().numpy()
    qpos_exp_map_np = qpos_exp_map_tensor.detach().cpu().numpy()
    
    # Process through physics pipeline to detect contact distances
    min_contact_distances = []
    
    for t in range(T):
        # Initialize physics state for this frame
        q = jnp.array(qpos_np[t, :sys.nq])
        qd = jnp.zeros(sys.nv)  # Zero initial velocities
        
        # Initialize physics state
        state = mjx_pipeline.init(sys, q, qd)
        
        # Get contact information from the physics state
        # The contact information is in state.contact
        if hasattr(state, 'contact') and state.contact is not None:
            # Find minimum contact distance (closest point to ground)
            contact_distances = state.contact.dist  # Contact point positions
            min_contact_dist = jnp.min(contact_distances)
        else:
            # Fallback: use the minimum Z coordinate of the root
            min_contact_dist = q[2]  # Root Z position
        
        min_contact_distances.append(float(min_contact_dist))
    
    # Convert to numpy array
    min_contact_distances = np.array(min_contact_distances)
    
    # Adjust translation to set the minimum contact distance to 0
    trans_adjusted = trans.clone().cpu().detach().numpy()
    trans_adjusted[:, 2] = trans[:, 2].cpu().detach().numpy() - min_contact_distances
    
    # Update the reference motion with adjusted translation
    ref_motion_adjusted = ref_motion.copy()
    ref_motion_adjusted['trans'] = trans_adjusted#.cpu().numpy()
    
    return ref_motion_adjusted

def load_reference_motion(ref_motion, full=True):
    """Convert reference motion to the format expected by the physics system."""
    ref_mot_fps = 30 

    local_rotation = torch.tensor(ref_motion['poses'])
    root_transl = torch.tensor(ref_motion['trans']) 
    root_transl[:,2] += 1.1

    root_rot = torch.tensor(ref_motion['poses_root'])
    _root_rot = root_rot.clone()
    _root_rot_x = _root_rot[:, 0] 
    _root_rot_y = _root_rot[:, 1]
    _root_rot_z = _root_rot[:, 2]
    _root_rot_w = _root_rot[:, 3]
    root_rot[:, 0] = _root_rot_w
    root_rot[:, 1] = _root_rot_x
    root_rot[:, 2] = _root_rot_y
    root_rot[:, 3] = _root_rot_z 

    dof_pos = quat_to_exp_map(local_rotation)
    N, _, _ = dof_pos.shape
    dof_pos = dof_pos.reshape(N, -1)
    
    body_parts = [
        "spine1", "neck1", "neck2", "head", "frthigh", "frshin", "frankle", "frfoot", 
        "flthigh", "flshin", "flankle", "flfoot", "spine2", "tail1", "tail2", "brthigh", "brshin", 
        "brankle", "brfoot", "blthigh", "blshin", "blankle", "blfoot",
    ]

    if full:
        qpos_list = [root_transl, root_rot, dof_pos]
        root_exp_map = quat_to_exp_map(_root_rot)
        qpos_exp_map_list = [root_transl, root_exp_map, dof_pos]
    else:
        qpos_list = [root_transl, root_rot]
        root_exp_map = quat_to_exp_map(_root_rot)
        qpos_exp_map_list = [root_transl, root_exp_map]
        for i in range(len(body_parts)):
            part = body_parts[i]
            if part in ["head", "tail2"] or "ankle" in part or "shin" in part:
                qpos_list.append(dof_pos[:, i*3+1:i*3+2])
                qpos_exp_map_list.append(dof_pos[:, i*3+1:i*3+2])
            else:
                qpos_list.append(dof_pos[:, i*3:i*3+3])
                qpos_exp_map_list.append(dof_pos[:, i*3:i*3+3])
    
    qpos_tensor = torch.cat(qpos_list, dim=1)
    qpos_exp_map_tensor = torch.cat(qpos_exp_map_list, dim=1)

    _curr_qvel_tensor = (
        qpos_exp_map_tensor[1:] - qpos_exp_map_tensor[:-1]
    ) * ref_mot_fps
    
    _curr_qvel_tensor = torch.cat([_curr_qvel_tensor[:,:6],_curr_qvel_tensor[:,6:]], dim=1)
    qvel_tensor = torch.cat(
        [torch.zeros(_curr_qvel_tensor.size(1)).float()[None], _curr_qvel_tensor], dim=0
    )
    
    res = torch.cat([qpos_tensor.clone(), qvel_tensor.clone()], dim=-1).cpu().detach().numpy()
    return res

def process_sequence(seq_dir: Path, xml_path: str, filename_pattern: str = "0065000_{frame_num: 10d}_params.pkl",
                    max_frames: int = 30, use_physics_alignment: bool = True,
                    trans_scale: float = 1.0) -> np.ndarray:
    """Process a single sequence directory."""
    print(f"Processing sequence: {seq_dir}")
    
    # Load physics system
    sys = mjcf.load(xml_path)
    
    # Find all pkl files in the sequence
    frames = []
    for f in seq_dir.iterdir():
        name = f.name
        if name.endswith("_params.pkl") and name.startswith("0065000_"):
            try:
                num_str = name.split("_")[1]
                num = int(num_str)
                frames.append(num)
            except Exception:
                continue
        if len(frames) == max_frames:
            break
    frames.sort()
    
    if len(frames) == 0:
        print(f"No frames found in {seq_dir}")
        return None
    
    # Load all frame data
    hat_full_list = {'poses': [], 'poses_root': [], 'betas': [], 'trans': []}
    for fn in frames:
        pkl_path = seq_dir / filename_pattern.format(frame_num=fn)
        if not pkl_path.exists():
            print(f"Missing file: {pkl_path}")
            continue
        data = load_pkl_data(pkl_path, trans_scale=trans_scale)
        for key in hat_full_list.keys():
            hat_full_list[key].append(data[key])
    
    if len(hat_full_list['poses']) == 0:
        print(f"No valid data found in {seq_dir}")
        return None
    
    # Stack the data
    for key in hat_full_list.keys():
        hat_full_list[key] = np.stack(hat_full_list[key], axis=0)
    
    # Apply physics-based ground alignment if enabled
    if use_physics_alignment:
        #try:
            hat_full_list = process_poses_with_physics_pipeline(hat_full_list, sys)
            print(f"Applied physics-based ground alignment to {seq_dir}")
       # except Exception as e:
    
    # Convert to final format
    data = load_reference_motion(hat_full_list)
    return data


def group_params_by_sequence(input_dir: Path, filename_suffix: str = "_params.pkl") -> Dict[str, List[Path]]:
    """
    Group *_params.pkl files in a flat directory into sequences.

    Heuristic:
    - Consider all files ending with filename_suffix (default "_params.pkl").
    - For each file, strip the suffix and then split the stem into:
        stem = "<seq_key><frame_digits>"
      where <frame_digits> is the final integer run. We use a regex to find
      the last integer; everything before that is the seq_key.
    - Files that share the same seq_key form a sequence, ordered by frame id.
    """
    files = [p for p in input_dir.iterdir() if p.is_file() and p.name.endswith(filename_suffix)]
    if not files:
        return {}

    seq_map: Dict[str, List[Tuple[int, Path]]] = {}
    for p in files:
        stem = p.name[:-len(filename_suffix)]
        m = re.search(r"(.*?)(\d+)\s*$", stem)
        if m:
            seq_key = m.group(1).rstrip("_")
            frame_id = int(m.group(2))
        else:
            seq_key = stem
            frame_id = -1
        seq_map.setdefault(seq_key, []).append((frame_id, p))

    grouped: Dict[str, List[Path]] = {}
    for key, lst in seq_map.items():
        lst_sorted = sorted(lst, key=lambda t: (t[0] if t[0] >= 0 else 0, t[1].name))
        grouped[key] = [p for _, p in lst_sorted]

    return grouped

def main():
    parser = argparse.ArgumentParser(description='Process kinematic data with physics-based ground alignment')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing pkl files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed npy files')
    parser.add_argument('--xml_path', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            'sim', 'data', 'assets', 'horse', 'test_full_box2_full.xml'),
                       help='Path to physics XML file (override for a different skeleton)')
    parser.add_argument('--filename_pattern', type=str, default='0065000_{frame_num: 10d}_params.pkl',
                       help='Pattern for pkl filenames')
    parser.add_argument('--max_frames', type=int, default=30, help='Maximum number of frames per sequence')
    parser.add_argument('--no_physics_alignment', action='store_true',
                       help='Disable physics-based ground alignment')
    add_trans_scale_args(parser)

    args = parser.parse_args()
    trans_scale = resolve_trans_scale(args.trans_scale, args.median_scale_file)
    print(f"Using trans_scale = {trans_scale}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_physics_alignment = not args.no_physics_alignment
    
    print(f"Processing data from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Physics alignment: {'enabled' if use_physics_alignment else 'disabled'}")
    
    if not input_dir.exists():
        print(f"[Error] input_dir {input_dir} does not exist.")
        return
    
    # Discover sequences from flat directory
    seq_map = group_params_by_sequence(input_dir, filename_suffix="_params.pkl")
    if not seq_map:
        print(f"No *_params.pkl files found under {input_dir}")
        return
    
    print(f"Discovered {len(seq_map)} sequences.")
    
    # Load physics system once
    sys = mjcf.load(args.xml_path)
    
    for seq_key, file_list in sorted(seq_map.items()):
        print(f"\n=== Processing sequence '{seq_key}' with {len(file_list)} frames ===")
        # Respect max_frames if provided
        frame_paths = file_list[: args.max_frames] if args.max_frames > 0 else file_list
        
        hat_full_list = {"poses": [], "poses_root": [], "betas": [], "trans": []}
        for pkl_path in frame_paths:
            data = load_pkl_data(pkl_path, trans_scale=trans_scale)
            for k in hat_full_list.keys():
                hat_full_list[k].append(data[k])
        
        if len(hat_full_list["poses"]) == 0:
            print(f"No valid data for sequence '{seq_key}', skipping.")
            continue
        
        # Stack data
        for k in hat_full_list.keys():
            hat_full_list[k] = np.stack(hat_full_list[k], axis=0)
        
        # Optional physics-based ground alignment
        if use_physics_alignment:
            hat_full_list = process_poses_with_physics_pipeline(hat_full_list, sys)
            print(f"Applied physics-based ground alignment to '{seq_key}'")
        
        # Convert to final qpos|qvel format
        processed_data = load_reference_motion(hat_full_list)
        
        # Save per-sequence output
        out_file = output_dir / f"processed_kinematic_data_{seq_key}.npy"
        np.save(out_file, processed_data)
        print(f"Saved processed data for '{seq_key}' to: {out_file}")
        print(f"Data shape: {processed_data.shape}")

if __name__ == "__main__":
    main()
