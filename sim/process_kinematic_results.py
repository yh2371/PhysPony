#!/usr/bin/env python3
"""
End-to-end utility to convert kinematic params.pkl sequences (e.g. from
FastVGGT run.py --config-name test_ponymation_horse) into MimicKit-ready
motion chunks (.pkl) plus optional visual sanity checks.

High-level pipeline
-------------------
1) Discover sequence folders under a results directory.
   Each sequence folder is expected to contain multiple "*_params.pkl" files.
2) For each sequence:
   - Load per-frame SMAL/pose data from params.pkl
   - Convert to a reference motion with:
       * root_pos  (world XYZ, meters)
       * root_rot  (XYZW quaternion, world orientation)
       * body joint quaternions (XYZW, local)
3) Convert that reference motion into MimicKit "frames" with layout:
       [root_pos(3) | root_rot_expmap(3) | joint_dof_expmap(3*J)]
   and save as:
       {"loop_mode": 0, "fps": 30, "frames": np.ndarray[N, D]}
4) Optionally chunk long sequences into fixed-length windows and save each
   chunk as its own motion .pkl, suitable for direct use with
   mimickit/anim/motion_lib.MotionLib and DeepMimic environments.
5) Optionally run a visualization similar to analyze_kinematic_motion.py
   on a subset of generated chunks for sanity checking.

This script consolidates logic from:
  - physhorse_new/process_kinematic_data_smal.py
      * load_pkl_data (camera/world transform + pose6D -> quats)
      * convert_to_qpos_qvel (quats -> exp-map DOFs)
  - MimicKit/analyze_kinematic_motion.py
      * FK and skeleton visualization for artifact inspection
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


###############################################################################
# Paths and imports
###############################################################################

MIMICKIT_DIR = Path(__file__).resolve().parent

# Allow importing analyze_kinematic_motion when running from other cwd
if str(MIMICKIT_DIR) not in sys.path:
    sys.path.append(str(MIMICKIT_DIR))

try:
    import analyze_kinematic_motion as akm  # type: ignore
except Exception:
    akm = None

def extract_pose_data(frames, zero_root=False):
    """frames: (N, D). Returns root_pos (N,3), root_rot (N,3 exp_map), joint_dof (N, dof_size)."""
    root_pos = frames[:, 0:3]
    root_rot = frames[:, 3:6]
    if zero_root:
        root_pos[:,:] = 0.0
        root_rot[:,:] = 0.0
    joint_dof = frames[:, 6:]

    return root_pos, root_rot, joint_dof.reshape(-1,105)

KIN2SMAL = {
    0: 1,  1: 2,  2: 3,  3: 4,  4: 5,  5: 6,  6: 7,  7: 8,  8: 9,  9: 10,
    10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
    17: 33, 18: 34, 19: 35,        # hind‑leg mapping
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24,
    27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32,
}

def ref_motion_to_mimickit_frames(ref_motion: dict, apply_kin2smal_reindex: bool = True) -> np.ndarray:
    """
    ref_motion:
      poses_root: (T,4) XYZW world root quats
      poses:      (T,35,4) XYZW body quats in SMAL order
      trans:      (T,3)    world root positions in meters
    returns frames: (T, 3 + 3 + 35*3) in KinChar joint order

    apply_kin2smal_reindex: reorder hind-leg/tail joints from the kinematic model's
    SMAL output. Pass False for raw mocap, which is already in SMAL order.
    """
    poses_root = ref_motion["poses_root"]  # (T,4)
    poses      = ref_motion["poses"]       # (T,35,4) SMAL order
    trans      = ref_motion["trans"]       # (T,3)

    T, J, _ = poses.shape
    assert J == 35

    # Root quat -> exp‑map
    root_exp = quat_array_to_expmap(poses_root)               # (T,3)

    # SMAL body quats -> exp‑map per joint
    poses_flat = poses.reshape(T * J, 4)
    body_exp_smal = quat_array_to_expmap(poses_flat).reshape(T, J, 3)  # (T,35,3)

    if apply_kin2smal_reindex:
        # Reorder to KinChar order using KIN2SMAL mapping
        body_exp_kin = np.zeros_like(body_exp_smal)
        for kin_idx in range(35):
            smal_j = KIN2SMAL[kin_idx] - 1  # mapping is 1‑based
            body_exp_kin[:, kin_idx, :] = body_exp_smal[:, smal_j, :]
    else:
        body_exp_kin = body_exp_smal

    body_exp_flat = body_exp_kin.reshape(T, 35 * 3)

    frames = np.concatenate([trans, root_exp, body_exp_flat], axis=1).astype(np.float32)
    return frames

###############################################################################
# Core loading utilities (adapted from process_kinematic_data_smal.py)
###############################################################################

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Reads pickles written with numpy 2 (numpy._core) under numpy 1 (numpy.core)."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_params_pkl(pkl_path: Path, trans_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Load one *_params.pkl frame and return:
        poses_root: (4,)   XYZW quaternion in world frame
        poses:      (35,4) XYZW quaternions for body joints
        trans:      (3,)   root translation in world frame (meters)

    trans_scale: per-sequence metric scale (median_scale.txt) applied to the
    predicted root translation: world = R_c2w @ (trans_scale * trans) + T_c2w.
    """
    with open(pkl_path, "rb") as f:
        data = _NumpyCompatUnpickler(f).load()

    # Camera/world transforms
    M = data["rotation"]
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    if M.ndim == 3:
        M = M[0]
    assert M.shape == (4, 4), f"rotation in {pkl_path} must be 4x4, got {M.shape}"
    # Same two fixed transforms used in process_kinematic_data_smal.py
    transform_temp = np.eye(4, dtype=np.float64)
    transform_temp[:3, :3] = np.array(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    transform_temp2 = np.eye(4, dtype=np.float64)
    transform_temp2[:3, :3] = np.array(
        [
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )
    M = transform_temp2 @ M @ transform_temp
    R_c2w = M[:3, :3]
    T_c2w = M[:3, 3]

    pose6d = torch.as_tensor(data["pose"], dtype=torch.float32)
    if pose6d.shape[-1] == 6 or pose6d.shape[-1] == 36 * 6:
        pose6d = pose6d.view(36, -1)
        pose_R = rot6d_to_rotmat(pose6d).view(36, 3, 3).numpy()
    else:
        pose_R = pose6d.view(36, 3, 3).numpy()

    trans = trans_scale * torch.as_tensor(data["trans"], dtype=torch.float32).view(1, 3).numpy()
    # Rotate root orientation and position into world frame
    pose_root = R_c2w @ pose_R[0:1]
    trans = (R_c2w @ trans.T).T + T_c2w[None, :]

    pose_root_quat = R.from_matrix(pose_root).as_quat()  # xyzw
    poses_quat = R.from_matrix(pose_R[1:]).as_quat()     # xyzw for 35 joints

    return {
        "poses_root": pose_root_quat[0],  # (4,)
        "poses": poses_quat,              # (35,4)
        "trans": trans[0],                # (3,)
    }


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix (Zhou et al.)."""
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def quat_array_to_expmap(q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_xyzw: (..., 4) numpy array with XYZW quaternions.
    Returns: (..., 3) exponential map (axis-angle vector).
    """
    # SciPy Rotation uses xyzw order by default.
    rot = R.from_quat(q_xyzw)
    rotvec = rot.as_rotvec().astype(np.float32)
    return rotvec


###############################################################################
# Sequence-level aggregation and conversion to MimicKit frames
###############################################################################

def build_ref_motion_from_files(frame_files: List[Path], trans_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Load a list of *_params.pkl frames (one sequence) and return:
        poses_root: (T,4) XYZW
        poses:      (T,35,4) XYZW
        trans:      (T,3)
    """
    if not frame_files:
        raise FileNotFoundError("No *_params.pkl files provided for this sequence.")

    poses_root_list: List[np.ndarray] = []
    poses_list: List[np.ndarray] = []
    trans_list: List[np.ndarray] = []

    for pkl_path in frame_files:
        d = load_params_pkl(pkl_path, trans_scale=trans_scale)
        poses_root_list.append(d["poses_root"][None, :])
        poses_list.append(d["poses"][None, ...])
        trans_list.append(d["trans"][None, :])

    poses_root = np.concatenate(poses_root_list, axis=0).astype(np.float32)
    poses = np.concatenate(poses_list, axis=0).astype(np.float32)
    trans = np.concatenate(trans_list, axis=0).astype(np.float32)
    return {"poses_root": poses_root, "poses": poses, "trans": trans}


def save_motion_pkl(
    out_path: Path,
    frames: np.ndarray,
    fps: float = 30.0,
    loop_mode: int = 0,
) -> None:
    """
    Save frames in the simple motion dict format expected by
    mimickit/anim/motion.load_motion.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "loop_mode": int(loop_mode),
        "fps": float(fps),
        "frames": frames,
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)


def chunk_sequence_frames(
    frames: np.ndarray,
    base_name: str,
    chunk_len: int,
) -> List[Tuple[str, np.ndarray]]:
    """
    Split one long sequence frames array into non-overlapping chunks of length
    chunk_len. The last partial chunk is kept if at least half length.

    Returns list of (chunk_name, chunk_frames).
    """
    T = frames.shape[0]
    chunks: List[Tuple[str, np.ndarray]] = []

    start = 0
    while start < T:
        end = start + chunk_len
        if end > T and (T - start) < chunk_len // 2:
            # too short, drop
            break
        end = min(end, T)
        sub = frames[start:end]
        chunk_name = f"{base_name}_{start}"
        chunks.append((chunk_name, sub))
        start = end

    return chunks


###############################################################################
# Visualization helpers (leveraging analyze_kinematic_motion.py)
###############################################################################

def visualize_motion_chunk(
    motion_pkl: Path,
    char_file: Path,
    output_dir: Path,
) -> None:
    """
    Run a minimal version of analyze_kinematic_motion on one motion file:
    - Loads frames
    - Computes FK body positions
    - Renders a 3D skeleton MP4 with limb coloring and artifact overlays.
    """
    if akm is None:
        print("analyze_kinematic_motion could not be imported, skipping visualization.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    data = akm.load_motion_pkl(str(motion_pkl))
    frames = data["frames"]
    fps = data.get("fps", 30)
    if isinstance(frames, list):
        frames = np.array(frames)
    root_pos, root_rot, joint_dof = extract_pose_data(frames)

    kin_model = akm._load_kin_char(str(char_file))
    if kin_model is None:
        print(f"Could not load character XML {char_file}, skipping visualization.")
        return

    root_pos_z = root_pos.copy()
    root_rot_z = root_rot.copy()

    body_pos, joint_quats = akm.compute_body_positions_numpy(
        root_pos_z, root_rot_z, joint_dof, kin_model
    )

    # Simple inertial/geometric masks for coloring
    mask_inertial, mask_geometric = akm.detect_artifacts_robust(
        joint_quats, body_pos, fps
    )
    mask_dict = {"inertial": mask_inertial, "geometric": mask_geometric}

    edges = akm.get_skeleton_edges(kin_model)

    body_pos_vis = body_pos.copy()

    base = motion_pkl.stem
    out_mp4 = output_dir / f"{base}_analysis.mp4"

    akm.render_skeleton_mp4(
        body_pos_vis,
        edges,
        mask_dict,
        str(out_mp4),
        kin_model,
        fps=fps,
        equal_aspect=True,
        title=f"{base}: Inertial/Geo check",
    )


###############################################################################
# CLI
###############################################################################

def group_params_by_sequence(input_dir: Path, filename_suffix: str = "__params.pkl") -> Dict[str, List[Path]]:
    """
    Group *_params.pkl files in a flat directory into sequences.

    Heuristic:
    - Consider all files ending with filename_suffix (default "_params.pkl").
    - For each file, strip the suffix and then split the stem into:
        stem = "<seq_key><frame_digits>"
      where <frame_digits> is the final integer run. We use a regex to find
      the last integer; everything before that is the seq_key.
    - Files that share the same seq_key form a sequence, ordered by frame id.

    This mirrors the explicit prefix/suffix approach from
    process_kinematic_data_smal.py but does not require manually specifying
    prefixes.
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
            # Fallback: treat entire stem as sequence key with unknown frame id
            seq_key = stem
            frame_id = -1
        seq_map.setdefault(seq_key, []).append((frame_id, p))

    grouped: Dict[str, List[Path]] = {}
    for key, lst in seq_map.items():
        lst_sorted = sorted(lst, key=lambda t: (t[0] if t[0] >= 0 else 0, t[1].name))
        grouped[key] = [p for _, p in lst_sorted]

    return grouped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert FastVGGT params.pkl kinematic sequences into MimicKit motion chunks."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing *_params.pkl files (possibly from multiple sequences).",
    )
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="__params.pkl",
        help="Filename suffix to identify kinematic PKL files (default: '_params.pkl').",
    )
    parser.add_argument(
        "--output_motion_dir",
        type=Path,
        default=Path("data/motions/horse_kin_from_params_debugging"),
        help="Directory to save generated motion .pkl chunks (relative to MimicKit root by default).",
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=0,
        help="Chunk length in frames for splitting long sequences (0 or <=0 = no chunking, use full sequence).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS to record in the motion files.",
    )
    parser.add_argument(
        "--char_file",
        type=Path,
        default=MIMICKIT_DIR / "data/assets/horse/test_full_box2_full.xml",
        help="Path to horse character XML for visualization.",
    )
    parser.add_argument(
        "--vis_dir",
        type=Path,
        default=Path("analysis_results/kinematic_from_params_debugging"),
        help="Directory to write visualization MP4s.",
    )
    parser.add_argument(
        "--visualize_every",
        type=int,
        default=0,
        help="If >0, run visualization on every N-th generated chunk (0=disable).",
    )
    parser.add_argument(
        "--trans_scale",
        type=float,
        default=None,
        help="Per-sequence metric scale for the predicted root translation "
             "(median_scale.txt from align_depth.py). Default: --median_scale_file, else 1.0.",
    )
    parser.add_argument(
        "--median_scale_file",
        type=Path,
        default=None,
        help="Path to a median_scale.txt written by align_depth.py; read as "
             "--trans_scale if --trans_scale isn't given explicitly.",
    )

    args = parser.parse_args(argv)

    trans_scale = args.trans_scale
    if trans_scale is None and args.median_scale_file is not None:
        trans_scale = float(args.median_scale_file.read_text().strip())
    if trans_scale is None:
        trans_scale = 1.0
    print(f"Using trans_scale = {trans_scale}")

    input_dir: Path = args.input_dir
    out_dir: Path = args.output_motion_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    seq_map = group_params_by_sequence(input_dir, filename_suffix=args.filename_suffix)
    if not seq_map:
        print(f"No files ending with '{args.filename_suffix}' found under {input_dir}")
        return

    print(f"Discovered {len(seq_map)} sequences in {input_dir}")

    vis_counter = 0

    for seq_idx, (seq_key, frame_files) in enumerate(sorted(seq_map.items())):
        print(f"\n=== Processing sequence {seq_idx + 1}/{len(seq_map)}: '{seq_key}' ({len(frame_files)} frames) ===")

        ref_motion = build_ref_motion_from_files(frame_files, trans_scale=trans_scale)
        frames = ref_motion_to_mimickit_frames(ref_motion)

        # Decide whether to chunk or keep full sequence
        if args.chunk_len is not None and args.chunk_len > 0 and args.chunk_len < frames.shape[0]:
            chunks = chunk_sequence_frames(frames, base_name=seq_key, chunk_len=args.chunk_len)
        else:
            chunks = [(seq_key, frames)]

        print(f"  Sequence has {frames.shape[0]} frames -> {len(chunks)} chunk(s)")

        for c_idx, (chunk_name, chunk_frames) in enumerate(chunks):
            out_pkl = out_dir / f"{chunk_name}.pkl"
            save_motion_pkl(out_pkl, chunk_frames, fps=args.fps, loop_mode=0)
            print(f"    Saved motion chunk: {out_pkl}  shape={chunk_frames.shape}")

            # Optional visualization
            if args.visualize_every > 0:
                vis_counter += 1
                if vis_counter % args.visualize_every == 0:
                    try:
                        visualize_motion_chunk(out_pkl, args.char_file, args.vis_dir)
                    except Exception as e:
                        print(f"      Visualization failed for {out_pkl}: {e}")


if __name__ == "__main__":
    main()

