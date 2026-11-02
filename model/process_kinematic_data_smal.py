#!/usr/bin/env python3
"""
Standalone script to process kinematic PKL data using the SMAL model to align
the mesh's lowest point to Z=0 for every frame, then export as a single NPY
array of concatenated [qpos | qvel] per frame compatible with the tracker.

Usage example:
  python process_kinematic_data_smal.py \
    --input_dir /path/to/sequence_dir \
    --output_dir /path/to/output_dir
  # --smal_model_path defaults to the bundled model/geometry/SMAL/*.pkl
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import pickle
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

from scale_utils import add_trans_scale_args, resolve_trans_scale
from coord_utils import fix_camera_axes

# SMAL import
SMAL_ROOT = os.environ.get(
    "PHYSPONY_GEOMETRY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "geometry"),
)
if SMAL_ROOT not in sys.path:
    sys.path.append(SMAL_ROOT)
from SMAL.smal_torch.smal_torch import SMAL


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix (Zhou et al.)."""
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def quat_to_exp_map(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (..., 4) to exponential map (..., 3).
    Expects XYZW quaternion order per provided datasets.
    """
    # Convert to rotation matrices, then to rotvec
    q_np = q.detach().cpu().numpy()
    rot = R.from_quat(q_np)
    rotvec = torch.from_numpy(rot.as_rotvec()).to(dtype=torch.float32)
    return rotvec


def load_pkl_data(pkl_path: Path, trans_scale: float = 1.0) -> Dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Camera/world transforms
    M = data["rotation"]
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    if M.ndim == 3:
        M = M[0]
    assert M.shape == (4, 4), f"rotation in {pkl_path} must be 4x4, got {M.shape}"

    M = fix_camera_axes(M)
    R_c2w = M[:3, :3]
    T_c2w = M[:3, 3]

    pose6d = torch.as_tensor(data["pose"], dtype=torch.float32)
    if pose6d.shape[-1] == 6 or pose6d.shape[-1] == 36 * 6:
        pose6d = pose6d.view(36, -1)
        pose_R = rot6d_to_rotmat(pose6d).view(36, 3, 3).numpy()
    else:
        pose_R = pose6d.view(36, 3, 3).numpy()

    trans = trans_scale * torch.as_tensor(data["trans"], dtype=torch.float32).view(1, 3).numpy()
    pose_root = R_c2w @ pose_R[0:1]
    trans = (R_c2w @ trans.T).T + T_c2w[None, :]

    pose_root_quat = R.from_matrix(pose_root).as_quat()  # xyzw
    poses_quat = R.from_matrix(pose_R[1:]).as_quat()     # xyzw for 35 joints

    return {
        "poses_root": pose_root_quat[0],
        "poses": poses_quat,
        "trans": trans[0],
    }


def convert_to_qpos_qvel(ref_motion: Dict, full: bool = True) -> np.ndarray:
    """Build [qpos|qvel] per-frame array following the dataset's convention."""
    ref_mot_fps = 30.0

    local_rotation = torch.tensor(ref_motion['poses'], dtype=torch.float32)   # (T, 35, 4)
    root_transl = torch.tensor(ref_motion['trans'], dtype=torch.float32)      # (T, 3)
    root_rot = torch.tensor(ref_motion['poses_root'], dtype=torch.float32)    # (T, 4 xyzw)

    # Reorder root quaternion from xyzw -> wxyz for the qpos packing used elsewhere
    _root_rot = root_rot.clone()
    _root_rot_x = _root_rot[:, 0]
    _root_rot_y = _root_rot[:, 1]
    _root_rot_z = _root_rot[:, 2]
    _root_rot_w = _root_rot[:, 3]
    root_rot[:, 0] = _root_rot_w
    root_rot[:, 1] = _root_rot_x
    root_rot[:, 2] = _root_rot_y
    root_rot[:, 3] = _root_rot_z

    # Convert body joint quats to exp-map
    local_rotation_flat = local_rotation.view(local_rotation.shape[0], -1, 4)
    dof_pos_list: List[torch.Tensor] = []
    for t in range(local_rotation_flat.shape[0]):
        rotvecs = quat_to_exp_map(local_rotation_flat[t])  # (35, 3)
        dof_pos_list.append(rotvecs)
    dof_pos = torch.stack(dof_pos_list, dim=0).reshape(local_rotation.shape[0], -1)

    if full:
        qpos_list = [root_transl, root_rot, dof_pos]
        root_exp_map = quat_to_exp_map(_root_rot)  # (T, 3)
        qpos_exp_map_list = [root_transl, root_exp_map, dof_pos]
    else:
        qpos_list = [root_transl, root_rot, dof_pos]
        root_exp_map = quat_to_exp_map(_root_rot)
        qpos_exp_map_list = [root_transl, root_exp_map, dof_pos]

    qpos_tensor = torch.cat(qpos_list, dim=1)
    qpos_exp_map_tensor = torch.cat(qpos_exp_map_list, dim=1)

    # Finite-difference velocities in exp-map space
    curr_qvel = (qpos_exp_map_tensor[1:] - qpos_exp_map_tensor[:-1]) * ref_mot_fps
    curr_qvel = torch.cat([curr_qvel[:, :6], curr_qvel[:, 6:]], dim=1)
    qvel_tensor = torch.cat([torch.zeros(curr_qvel.size(1)).float()[None], curr_qvel], dim=0)

    res = torch.cat([qpos_tensor.clone(), qvel_tensor.clone()], dim=-1).cpu().detach().numpy()
    return res


def process_sequence_with_smal(seq_dir: Path,
                               smal_model_path: str,
                               filename_prefix: str = "0065000_",
                               filename_suffix: str = "_params.pkl",
                               max_frames: int = 30,
                               align_per_frame: bool = True,
                               trans_scale: float = 1.0) -> np.ndarray:
    """Process one sequence directory using SMAL to ground-align per frame."""
    print(f"Processing (SMAL) sequence: {seq_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smal_model = SMAL(smal_model_path, device=device, use_smal_betas=False)

    # Discover frames
    frames: List[int] = []
    for f in seq_dir.iterdir():
        name = f.name
        if name.endswith(filename_suffix) and name.startswith(filename_prefix):
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

    # Load frame data
    ref = {"poses_root": [], "poses": [], "trans": []}
    for fn in frames:
        pkl_path = seq_dir / f"{filename_prefix}{fn: 10d}{filename_suffix}"
        if not pkl_path.exists():
            print(f"Missing file: {pkl_path}")
            continue
        d = load_pkl_data(pkl_path, trans_scale=trans_scale)
        ref["poses_root"].append(d["poses_root"])  # (4,)
        ref["poses"].append(d["poses"])            # (35, 4)
        ref["trans"].append(d["trans"])            # (3,)

    # Stack to arrays
    poses_root = np.stack(ref["poses_root"], axis=0)               # (T, 4)
    poses_body = np.stack(ref["poses"], axis=0)                    # (T, 35, 4)
    trans = np.stack(ref["trans"], axis=0)                         # (T, 3)

    T = poses_body.shape[0]

    # Convert quats (xyzw) to axis-angle for SMAL: theta (T, 108)
    theta = np.zeros((T, 36, 3), dtype=np.float32)
    root_rotvec = R.from_quat(poses_root).as_rotvec()               # (T, 3)
    theta[:, 0] = root_rotvec
    for t in range(T):
        rotvecs = R.from_quat(poses_body[t]).as_rotvec()            # (35, 3)
        theta[t, 1:] = rotvecs
    theta = torch.tensor(theta.reshape(T, 108), dtype=torch.float32, device=device)

    betas = torch.zeros((T, 9), dtype=torch.float32, device=device) # Neutral shape per frame
    trans_t = torch.tensor(trans, dtype=torch.float32, device=device)

    with torch.no_grad():
        verts, _, _ = smal_model(beta=betas, theta=theta, trans=trans_t)
        # verts: (T, V, 3)

    if align_per_frame:
        min_z = torch.min(verts[:, :, 2], dim=1).values.cpu().numpy()  # (T,)
        print(f"min_z: {min_z}", f"trans: {trans}")
        trans[:, 2] = trans[:, 2] - min_z + 0.35
        print(f"min_z: {min_z}", f"trans: {trans}")
    else:
        overall_min = torch.min(verts[:, :, 2]).item()
        trans[:, 2] = trans[:, 2] - overall_min

    # Build qpos|qvel and return
    ref_motion_aligned = {
        "poses_root": poses_root,  # xyzw
        "poses": poses_body,       # (T, 35, 4)
        "trans": trans,            # (T, 3), aligned
    }
    data = convert_to_qpos_qvel(ref_motion_aligned, full=True)
    return data


def main():
    ap = argparse.ArgumentParser(description='Process kinematic data with SMAL-based ground alignment')
    ap.add_argument('--input_dir', type=str, required=True, help='Input directory containing *_params.pkl files')
    ap.add_argument('--output_dir', type=str, required=True, help='Output directory for processed npy file')
    ap.add_argument('--smal_model_path', type=str, default=os.path.join(SMAL_ROOT, 'SMAL', 'my_smpl_0000_horse_new_skeleton_horse.pkl'), help='Path to SMAL model pkl')
    ap.add_argument('--filename_prefix', type=str, default='0065000_', help='Prefix of kinematic PKL files')
    ap.add_argument('--filename_suffix', type=str, default='_params.pkl', help='Suffix of kinematic PKL files')
    ap.add_argument('--max_frames', type=int, default=30, help='Max frames to load')
    ap.add_argument('--no_align_per_frame', action='store_true', help='Align using global minimum instead of per-frame')
    add_trans_scale_args(ap)
    args = ap.parse_args()
    trans_scale = resolve_trans_scale(args.trans_scale, args.median_scale_file)
    print(f"Using trans_scale = {trans_scale}")

    seq_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = process_sequence_with_smal(
        seq_dir=seq_dir,
        smal_model_path=args.smal_model_path,
        filename_prefix=args.filename_prefix,
        filename_suffix=args.filename_suffix,
        max_frames=args.max_frames,
        align_per_frame=not args.no_align_per_frame,
        trans_scale=trans_scale,
    )

    if data is None:
        print("No data produced.")
        return

    out_file = out_dir / 'processed_kinematic_data_smal.npy'
    np.save(out_file, data)
    print(f"Saved SMAL-processed data to: {out_file}  shape={data.shape}")


if __name__ == '__main__':
    main()


