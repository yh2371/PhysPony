#!/usr/bin/env python3
"""
Combined visualization for one rollout sequence.

Panels:
  1) optional input RGB video
  2) 3D SMAL meshes:
       - kinematics / ref   : green
       - RL / sim           : blue
       - mocap ground truth : black
  3) aligned top-down root trajectories
  4) root height over time + variance

This version does NOT use gather_sequence_data(...).
It loads sim/ref directly from <seq_name>_states.npz and mocap directly
from the mocap .npy, using the same frame-matching logic as eval_all_sequences.py.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import imageio.v2 as imageio
import scipy.spatial.transform

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from mimickit.util.visualize_smal_states import generate_smal_mesh


# ------------------------------
# Alignment helpers
# ------------------------------

def rigid_align_no_scale(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align P -> Q with rotation + translation, no scale. P,Q: (N,3)."""
    P = np.asarray(P, dtype=np.float64)[:-20]
    Q = np.asarray(Q, dtype=np.float64)[:-20]

    P[:,2] = 0
    Q[:,2] = 0

    muP = P.mean(0)
    muQ = Q.mean(0)
    X = P - muP
    Y = Q - muQ

    S = X.T @ Y / P.shape[0]
    U, _, Vt = np.linalg.svd(S)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T

    #rotate 120 around z axis
    Rm = project_rotation_to_z(Rm)
    Rx = np.array([
        [np.cos(120), -np.sin(120), 0],
        [np.sin(120),  np.cos(120), 0],
        [0, 0, 1]
    ])
    Rm = Rx @ Rm
    Rm = Rm
    t = muQ - (Rm @ muP)
    return Rm, t


def apply_rigid(P: np.ndarray, Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    return P @ Rm.T + t

def project_rotation_to_z(R):
    """
    Constrain a 3x3 rotation matrix to rotation about the Z axis only.
    """
    theta = np.arctan2(R[1,0], R[0,0])

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    return Rz


def align_root_trajectories_to_mocap(
    sim_root: np.ndarray,
    ref_root: Optional[np.ndarray],
    mocap_root: Optional[np.ndarray],
):
    """
    Matches visualize_sequence_topdown.py:
      - T = min(lengths) - 17
      - rigid no-scale align sim/ref to mocap
      - subtract first frame so all start at origin
    """
    lengths = [sim_root.shape[0]]
    if ref_root is not None:
        lengths.append(ref_root.shape[0])
    if mocap_root is not None:
        lengths.append(mocap_root.shape[0])

    T = min(lengths)
    if T <= 0:
        raise ValueError(f"Invalid T after -17 trim: {T}")

    sim_t = sim_root[:T].astype(np.float64)
    ref_t = ref_root[:T].astype(np.float64) if ref_root is not None else None
    mocap_t = mocap_root[:T].astype(np.float64) if mocap_root is not None else None

    if mocap_t is not None:
        finite_s = np.isfinite(sim_t).all(1) & np.isfinite(mocap_t).all(1)
        if finite_s.sum() >= 3:
            R_s, t_s = rigid_align_no_scale(ref_t[finite_s], mocap_t[finite_s])
            sim_al = apply_rigid(sim_t, R_s, t_s)
        else:
            sim_al = sim_t

        if ref_t is not None:
            finite_r = np.isfinite(ref_t).all(1) & np.isfinite(mocap_t).all(1)
            if finite_r.sum() >= 3:
                R_r, t_r = rigid_align_no_scale(ref_t[finite_r], mocap_t[finite_r])
                ref_al = apply_rigid(ref_t, R_r, t_r)
            else:
                ref_al = ref_t
        else:
            ref_al = None

        mocap_al = mocap_t
    else:
        sim_al = sim_t
        ref_al = ref_t
        mocap_al = None

    sim_final = sim_al - sim_al[0:1]
    ref_final = ref_al - ref_al[0:1] if ref_al is not None else None
    mocap_final = mocap_al - mocap_al[0:1] if mocap_al is not None else None

    return sim_final, ref_final, mocap_final, T, R_s, t_s, R_r, t_r


# ------------------------------
# Motion format helpers
# ------------------------------

KIN2SMAL = {
    0: 1,  1: 2,  2: 3,  3: 4,  4: 5,  5: 6,  6: 7,  7: 8,  8: 9,  9: 10,
    10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
    17: 33, 18: 34, 19: 35,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24,
    27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32,
}


def kin2smal(body_exp_kin: np.ndarray) -> np.ndarray:
    body_exp_smal = np.zeros_like(body_exp_kin)
    for kin_idx in range(35):
        smal_j = KIN2SMAL[kin_idx] - 1
        body_exp_smal[:, smal_j, :] = body_exp_kin[:, kin_idx, :]
    return body_exp_smal


def maybe_convert_kin_to_smal(poses_body: np.ndarray, force: bool = False) -> np.ndarray:
    arr = np.asarray(poses_body)
    if force:
        return kin2smal(arr)
    if arr.ndim == 3 and arr.shape[1:] == (35, 3):
        return kin2smal(arr)
    return arr


def mirror_translations(trans, axis=1):
    trans_m = np.array(trans, copy=True)
    trans_m[..., axis] *= -1
    return trans_m


def mirror_rotation_matrices(rotmats, axis=1):
    rotmats = np.asarray(rotmats)
    S = np.eye(3)
    S[axis, axis] = -1.0
    return S @ rotmats @ S


def mirror_rotvecs(rotvec, axis=1):
    rotvec = np.asarray(rotvec)
    orig_shape = rotvec.shape
    flat = rotvec.reshape(-1, 3)
    mats = scipy.spatial.transform.Rotation.from_rotvec(flat).as_matrix()
    mats_m = mirror_rotation_matrices(mats, axis=axis)
    rotvec_m = scipy.spatial.transform.Rotation.from_matrix(mats_m).as_rotvec()
    return rotvec_m.reshape(orig_shape)


def mirror_quaternions(quat, axis=1, wxyz=False):
    quat = np.asarray(quat)
    orig_shape = quat.shape
    flat = quat.reshape(-1, 4)

    if wxyz:
        flat_xyzw = np.concatenate([flat[:, 1:], flat[:, :1]], axis=-1)
    else:
        flat_xyzw = flat

    mats = scipy.spatial.transform.Rotation.from_quat(flat_xyzw).as_matrix()
    mats_m = mirror_rotation_matrices(mats, axis=axis)
    quat_m_xyzw = scipy.spatial.transform.Rotation.from_matrix(mats_m).as_quat()

    if wxyz:
        quat_m = np.concatenate([quat_m_xyzw[:, 3:], quat_m_xyzw[:, :3]], axis=-1)
    else:
        quat_m = quat_m_xyzw

    return quat_m.reshape(orig_shape)


def mirror_rotation_repr(x, axis=1, name="rotation"):
    x = np.asarray(x)

    if x.ndim >= 2 and x.shape[-2:] == (3, 3):
        return mirror_rotation_matrices(x, axis=axis)

    if x.shape[-1] == 4:
        return mirror_quaternions(x, axis=axis, wxyz=False)

    if x.shape[-1] == 3:
        return mirror_rotvecs(x, axis=axis)

    print(f"Warning: unsupported {name} shape {x.shape}; leaving unchanged")
    return np.array(x, copy=True)


def mirror_motion_data(data, axis=1):
    return {
        "trans": mirror_translations(data["trans"], axis=axis),
        "poses_root": mirror_rotation_repr(data["poses_root"], axis=axis, name="poses_root"),
        "poses_body": mirror_rotation_repr(data["poses_body"], axis=axis, name="poses_body"),
    }


# ------------------------------
# Direct data loaders
# ------------------------------

def load_states_npz(path: Path) -> Dict[str, Optional[np.ndarray]]:
    """
    Direct rollout-state loader, matching the structure in eval_all_sequences.py.
    """
    data = np.load(path, allow_pickle=True)
    return {
        "body_pos": data["body_pos"],
        "trans": data["trans"],
        "ref_body_pos": data["ref_body_pos"] if "ref_body_pos" in data else None,
        "ref_trans": data["ref_trans"] if "ref_trans" in data else None,
        "poses_root": data["poses_root"] if "poses_root" in data else None,
        "poses_body": data["poses_body"] if "poses_body" in data else None,
        "ref_poses_root": data["ref_poses_root"] if "ref_poses_root" in data else None,
        "ref_poses_body": data["ref_poses_body"] if "ref_poses_body" in data else None,
    }


def get_frame_indices_from_image_folder(seq_images_dir: Path) -> Optional[List[int]]:
    if not seq_images_dir.exists() or not seq_images_dir.is_dir():
        return None

    exts = {"_rgb.jpg", "_rgb.png"}
    files = [f for f in seq_images_dir.iterdir() if f.name[-8:] in exts]
    if not files:
        return None

    frame_nums = []
    pat = re.compile(r"(\d+)")
    for f in files:
        m = pat.search(f.stem)
        if m:
            frame_nums.append((int(m.group(1)), f))
        else:
            frame_nums.append((-1, f))

    frame_nums.sort(key=lambda x: (x[0], x[1].name))
    if all(x[0] >= 0 for x in frame_nums):
        return [x[0] for x in frame_nums]
    return list(range(len(frame_nums)))


def derive_full_seq_name(seq_name: str) -> str:
    parts = seq_name.split("_")
    if len(parts) >= 6: # and "loco" in seq_name:
        return "_".join(parts[:-2])
    return "_".join(parts[:-1]) if len(parts) >= 3 else seq_name


def load_mocap_npy(mocap_path: Path) -> Optional[Dict]:
    if not mocap_path.exists():
        print(f"[warn] missing mocap file: {mocap_path}")
        return None
    arr = np.load(mocap_path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        return arr.item()
    return None

from scipy.spatial.transform import Rotation as R
def quat_array_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_xyzw: (..., 4) numpy array with XYZW quaternions.
    Returns: (..., 3) exponential map (axis-angle vector).
    """
    # SciPy Rotation uses xyzw order by default.
    rot = R.from_quat(q_xyzw)
    rotmat = rot.as_matrix().astype(np.float32)
    return rotmat


import torch
import mimickit.util.torch_util as torch_util

device = torch.device("cpu")

def load_mocap_motion(
    seq_name: str,
    mocap_root: Path,
    images_root: Path,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load mocap SMAL params directly, using the same frame-id / filename matching logic
    as eval_all_sequences.py.
    """
    seq_images_dir = images_root / seq_name
    frame_ids = get_frame_indices_from_image_folder(seq_images_dir)
    if not frame_ids:
        print(f"[warn] no valid frame ids found for {seq_name}")
        return None

    full_seq_name = derive_full_seq_name(seq_name)
    mocap_path = mocap_root / f"{full_seq_name}.npy"
    mocap = load_mocap_npy(mocap_path)
    if mocap is None:
        return None

    trans = mocap.get("trans", None)
    poses_root = mocap.get("poses_root", None)
    poses_body = mocap.get("poses_body", None)
    if trans is None or poses_root is None or poses_body is None:
        print(f"[warn] mocap missing required keys in {mocap_path}")
        return None

    

    trans = np.asarray(trans, dtype=np.float64)
    poses_root = np.asarray(poses_root, dtype=np.float64)
    poses_body = np.asarray(poses_body, dtype=np.float64)

    # Convert root quat wxyz -> xyzw if needed, matching eval_all_sequences.py logic.

    # SMAL body quats -> exp‑map per joint
    T = trans.shape[0]
    poses_flat = poses_body.reshape(T * 35, 4)
    body_rotmat = quat_array_to_rotmat(poses_flat).reshape(T, 35, 3,3)  # (T,35,3,3)
    root_rotmat = quat_array_to_rotmat(poses_root).reshape(T, 3,3)
    valid_ids = [i for i in frame_ids if 0 <= i < trans.shape[0]]
    if not valid_ids:
        print(f"[warn] no valid frame ids overlap mocap for {seq_name}")
        return None

    return {
        "trans": trans[valid_ids].astype(np.float32),
        "poses_root": root_rotmat[valid_ids].astype(np.float32),
        "poses_body": body_rotmat[valid_ids].astype(np.float32),
    }


def load_rollout_triplet(
    seq_name: str,
    states_path: Path,
    mocap_root: Path,
    images_root: Path,
    force_ref_kin_to_smal=False,
    force_sim_kin_to_smal=False,
    force_mocap_kin_to_smal=False,
):
    """
    Returns:
      sim_motion, ref_motion, mocap_motion
    each as dict with trans / poses_root / poses_body.
    """
    states = load_states_npz(states_path)

    for required in ["trans", "poses_root", "poses_body", "ref_trans", "ref_poses_root", "ref_poses_body"]:
        if states[required] is None:
            raise ValueError(f"Missing '{required}' in rollout states: {states_path}")

    sim_motion = {
        "trans": np.asarray(states["trans"], dtype=np.float32),
        "poses_root": np.asarray(states["poses_root"]),
        "poses_body": kin2smal(np.asarray(states["poses_body"])),#  force=force_sim_kin_to_smal),
    }

    ref_motion = {
        "trans": np.asarray(states["ref_trans"], dtype=np.float32),
        "poses_root": np.asarray(states["ref_poses_root"]),
        "poses_body": kin2smal(np.asarray(states["ref_poses_body"])),#  force=force_ref_kin_to_smal),
    }

    mocap_motion = load_mocap_motion(seq_name=seq_name, mocap_root=mocap_root, images_root=images_root)
    if mocap_motion is None:
        raise ValueError(f"Could not load mocap motion for {seq_name}")

    mocap_motion["poses_body"] = kin2smal(np.asarray(mocap_motion["poses_body"])) #) #, force=force_mocap_kin_to_smal
    #)

    return sim_motion, ref_motion, mocap_motion


def load_video_frames(images_root: Path, seq_name: str, max_frames=None):
    seq_dir = images_root / seq_name
    if not seq_dir.exists():
        return None

    exts = {"_rgb.jpg", "_rgb.png"}
    img_files = sorted([f for f in seq_dir.iterdir() if f.name[-8:] in exts])
    if not img_files:
        return None

    if max_frames is not None:
        img_files = img_files[:max_frames]

    return [imageio.imread(p) for p in img_files]


# ------------------------------
# Animation helpers
# ------------------------------

def sample_mesh_points(verts: np.ndarray, max_points: int = 1500):
    if verts.shape[0] <= max_points:
        return verts
    idx = np.linspace(0, verts.shape[0] - 1, max_points, dtype=int)
    return verts[idx]


def save_animation_with_fallback(anim, output_path, fps=30, bitrate=1800):
    codec_candidates = [
        ("mpeg4", ["-pix_fmt", "yuv420p"]),
        ("mjpeg", []),
    ]
    for codec, extra_args in codec_candidates:
        try:
            print(f"Trying codec: {codec}")
            writer = FFMpegWriter(
                fps=fps,
                codec=codec,
                bitrate=bitrate,
                extra_args=extra_args,
            )
            anim.save(output_path, writer=writer)
            print(f"Saved animation to {output_path} using codec={codec}")
            return output_path
        except Exception as e:
            print(f"Failed with codec={codec}: {e}")

    gif_path = str(Path(output_path).with_suffix(".gif"))
    print(f"Falling back to GIF: {gif_path}")
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    print(f"Saved animation to {gif_path}")
    return gif_path


# ------------------------------
# Main visualization
# ------------------------------

def trim_motion(motion, T):
    return {
        "trans": motion["trans"][:T],
        "poses_root": motion["poses_root"][:T],
        "poses_body": motion["poses_body"][:T],
    }


def visualize_sequence_combined(
    seq_name: str,
    sim_motion: Dict[str, np.ndarray],
    ref_motion: Dict[str, np.ndarray],
    mocap_motion: Dict[str, np.ndarray],
    video_frames,
    output_path: Path,
    fps: int = 10,
    smal_model_path: Optional[str] = None,
    device: str = "cpu",
    mirror_axis: int = 1,
):
    sim_top, ref_top, mocap_top, T_top, R_s, t_s, R_r, t_r = align_root_trajectories_to_mocap(
        sim_root=sim_motion["trans"],
        ref_root=ref_motion["trans"],
        mocap_root=mocap_motion["trans"],
    )

    T_pose = min(
        sim_motion["trans"].shape[0],
        ref_motion["trans"].shape[0],
        mocap_motion["trans"].shape[0],
    )

    sim_motion["trans"] = sim_top
    ref_motion["trans"] = ref_top
    mocap_motion["trans"] = mocap_top

    lengths = [T_top, T_pose]
    if video_frames is not None:
        lengths.append(len(video_frames))
    T = min(lengths)
    if T <= 0:
        raise ValueError(f"Invalid final T={T}")

    sim_motion = trim_motion(sim_motion, T)
    ref_motion = trim_motion(ref_motion, T)
    mocap_motion = trim_motion(mocap_motion, T)
    if video_frames is not None:
        video_frames = video_frames[:T]

    sim_top = sim_top[:T]
    ref_top = ref_top[:T] if ref_top is not None else None
    mocap_top = mocap_top[:T] if mocap_top is not None else None

    tracks = [
        {"label": "Kinematics", "color": "green", "motion": ref_motion, "top": ref_top},
        {"label": "RL",         "color": "blue",  "motion": sim_motion, "top": sim_top},
        {"label": "Mocap",      "color": "black", "motion": mocap_motion, "top": mocap_top},
    ]

    for tr in tracks:
        tr["root_z"] = tr["motion"]["trans"][:, 2]
        tr["root_var"] = float(np.var(tr["root_z"]))

    print(f"Generating SMAL meshes for T={T} frames...")
    mesh_results = []
    for tr in tracks:
        motion = {
            "poses_body": np.array(tr["motion"]["poses_body"], copy=True),
            "trans": np.array(tr["motion"]["trans"], copy=True),
            "poses_root": np.array(tr["motion"]["poses_root"], copy=True),
        }

        
        if tr["label"] == "Kinematics":
            motion["poses_root"] = R_r @ motion["poses_root"]
        elif tr["label"] == "RL":
            motion["poses_root"] = R_s @ motion["poses_root"]
        elif tr["label"] == "Mocap":
            #rotate -60 around z axis
            motion["poses_root"] = motion["poses_root"]
        verts, joints, _ = generate_smal_mesh(
            motion["poses_body"],
            motion["trans"],
            motion["poses_root"],
            smal_model_path=smal_model_path,
            device=device,
        )
        mesh_results.append({"verts": verts, "joints": joints})
        print(f"  mesh ok: {tr['label']}")

    all_verts = np.concatenate([m["verts"] for m in mesh_results], axis=0)
    x_min, x_max = all_verts[:, :, 0].min(), all_verts[:, :, 0].max()
    y_min, y_max = all_verts[:, :, 1].min(), all_verts[:, :, 1].max()
    z_min, z_max = all_verts[:, :, 2].min(), all_verts[:, :, 2].max()
    pad = 0.08 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)

    xs_all = np.concatenate([tr["top"][:, 0] for tr in tracks if tr["top"] is not None])
    ys_all = np.concatenate([tr["top"][:, 1] for tr in tracks if tr["top"] is not None])
    cx = 0.5 * (xs_all.min() + xs_all.max())
    cy = 0.5 * (ys_all.min() + ys_all.max())
    rr = max(xs_all.max() - xs_all.min(), ys_all.max() - ys_all.min(), 1e-6)
    half = 0.55 * rr

    all_root_z = np.concatenate([tr["root_z"] for tr in tracks], axis=0)
    hz_min, hz_max = all_root_z.min(), all_root_z.max()
    hz_pad = 0.1 * max(hz_max - hz_min, 1e-6)

    # Layout:
    # Row 1: video | topdown | height
    # Row 2: overlapped 3D | kin 3D | RL 3D | mocap 3D
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 12, hspace=0.28, wspace=0.20)

    if video_frames is not None:
        ax_video  = fig.add_subplot(gs[0, 0:4])
    else:
        ax_video = fig.add_subplot(gs[0, 0:4])
    ax_top    = fig.add_subplot(gs[0, 4:8])
    ax_height = fig.add_subplot(gs[0, 8:12])

    ax_3d_all   = fig.add_subplot(gs[1, 0:3], projection="3d")
    ax_3d_kin   = fig.add_subplot(gs[1, 3:6], projection="3d")
    ax_3d_rl    = fig.add_subplot(gs[1, 6:9], projection="3d")
    ax_3d_mocap = fig.add_subplot(gs[1, 9:12], projection="3d")

    def setup_3d_axis(ax, title):
        ax.set_xlim([x_min - pad, x_max + pad])
        ax.set_ylim([y_min - pad, y_max + pad])
        ax.set_zlim([z_min - pad, z_max + pad])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.view_init(elev=35, azim=-50)

    def draw_single_mesh(ax, verts, color, alpha=0.75, max_points=1500):
        v = sample_mesh_points(verts, max_points=max_points)
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=color, s=1, alpha=alpha)

    def animate(frame_idx):
        for ax in [ax_video, ax_top, ax_height, ax_3d_all, ax_3d_kin, ax_3d_rl, ax_3d_mocap]:
            ax.clear()

        # -------- row 1 --------
        if video_frames is not None:
            ax_video.imshow(video_frames[frame_idx])
            ax_video.set_title("Input Video")
            ax_video.axis("off")
        else:
            ax_video.text(
                0.5, 0.5, "No input video",
                ha="center", va="center", fontsize=16
            )
            ax_video.set_title("Input Video")
            ax_video.axis("off")

        ax_top.set_aspect("equal", adjustable="box")
        ax_top.set_xlim(cx - half, cx + half)
        ax_top.set_ylim(cy - half, cy + half)
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Y (m)")
        ax_top.set_title("Aligned Top-Down Root Trajectories")
        ax_top.grid(True, alpha=0.3)

        for tr in tracks:
            cur = tr["top"][:frame_idx + 1]
            if len(cur) > 1:
                ax_top.plot(cur[:, 0], cur[:, 1], color=tr["color"], linewidth=2.0, label=tr["label"])
            ax_top.scatter(cur[0, 0], cur[0, 1], color=tr["color"], marker="o", s=35, zorder=3)
            ax_top.scatter(cur[-1, 0], cur[-1, 1], color=tr["color"], marker="s", s=40, zorder=4)


        ax_height.set_xlim([0, T - 1])
        ax_height.set_ylim([hz_min - hz_pad, hz_max + hz_pad])
        ax_height.set_xlabel("Frame")
        ax_height.set_ylabel("Root Height (m)")
        ax_height.set_title("Root Height Over Time")
        ax_height.grid(True, alpha=0.3)

        xs_frame = np.arange(frame_idx + 1)
        for tr in tracks:
            label = f"{tr['label']} (var={tr['root_var']:.4f})"
            if len(xs_frame) > 1:
                ax_height.plot(
                    xs_frame,
                    tr["root_z"][:frame_idx + 1],
                    color=tr["color"],
                    linewidth=2,
                    alpha=0.9,
                    label=label,
                )
            ax_height.scatter([frame_idx], [tr["root_z"][frame_idx]], color=tr["color"], s=40, zorder=5)


        # -------- row 2 --------
        setup_3d_axis(ax_3d_all, "Aligned 3D View: All")
        setup_3d_axis(ax_3d_kin, "Ours w/o Motion Refinement")
        setup_3d_axis(ax_3d_rl, "Ours w/ Motion Refinement")
        setup_3d_axis(ax_3d_mocap, "Ground Truth Mocap")

        # overlapped
        for i, tr in enumerate(tracks):
            verts = mesh_results[i]["verts"][frame_idx]
            alpha = 0.75 if tr["label"] != "Mocap" else 0.55
            draw_single_mesh(ax_3d_all, verts, tr["color"], alpha=alpha, max_points=1500)

        # individual
        draw_single_mesh(ax_3d_kin,   mesh_results[0]["verts"][frame_idx], tracks[0]["color"], alpha=0.85, max_points=1500)
        draw_single_mesh(ax_3d_rl,    mesh_results[1]["verts"][frame_idx], tracks[1]["color"], alpha=0.85, max_points=1500)
        draw_single_mesh(ax_3d_mocap, mesh_results[2]["verts"][frame_idx], tracks[2]["color"], alpha=0.75, max_points=1500)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Kinematics'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='RL'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Mocap'),
        ]

        return []

    anim = FuncAnimation(fig, animate, frames=T, interval=1000 / fps, blit=False, repeat=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_animation_with_fallback(anim, str(output_path), fps=fps, bitrate=1800)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Combined SMAL + aligned topdown + root height + optional video visualization."
    )
    parser.add_argument("--seq_name", type=str, required=True, help="Sequence name (stem without _states)")
    parser.add_argument("--rollout_dir", type=Path, default=Path("output/rollouts_mocap2/rollouts"), help="Directory containing <seq_name>_states.npz")
    parser.add_argument("--char_file", type=Path, default=Path("data/assets/horse/test_full_box2_full.xml"), help="Character XML used for evaluation")
    parser.add_argument("--mocap_root", type=Path, default=Path("mimickit/mocap_full"), help="Folder with full-sequence mocap .npy")
    parser.add_argument("--images_root", type=Path, default=Path("data/motions/horse_chunked_ID4_full_val_video_chunks"), help="Root folder with one subfolder per sequence containing RGB frames")
    parser.add_argument("--out_dir", type=Path, default=Path("output/vis_seq"), help="Output directory")
    parser.add_argument("--fps", type=int, default=20, help="Animation fps")
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="Timestep in seconds")
    parser.add_argument("--device", type=str, default="cpu", help="Device for SMAL generation")
    parser.add_argument("--smal-model-path", type=str, default=None, help="Optional SMAL model path")
    parser.add_argument("--mirror-axis", type=int, default=1, help="Mirror axis")
    parser.add_argument("--force-ref-kin-to-smal", action="store_true")
    parser.add_argument("--force-sim-kin-to-smal", action="store_true")
    parser.add_argument("--force-mocap-kin-to-smal", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    states_path = args.rollout_dir / f"{args.seq_name}_states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"Could not find states file: {states_path}")

    sim_motion, ref_motion, mocap_motion = load_rollout_triplet(
        seq_name=args.seq_name,
        states_path=states_path,
        mocap_root=args.mocap_root,
        images_root=args.images_root,
        force_ref_kin_to_smal=True, #args.force_ref_kin_to_smal,
        force_sim_kin_to_smal=True, #args.force_sim_kin_to_smal,
        force_mocap_kin_to_smal=False, #args.force_mocap_kin_to_smal,
    )

    video_frames = None if args.no_video else load_video_frames(args.images_root, args.seq_name)

    output_path = args.out_dir / f"{args.seq_name}_combined.mp4"
    visualize_sequence_combined(
        seq_name=args.seq_name,
        sim_motion=sim_motion,
        ref_motion=ref_motion,
        mocap_motion=mocap_motion,
        video_frames=video_frames,
        output_path=output_path,
        fps=args.fps,
        smal_model_path=args.smal_model_path,
        device=args.device,
        mirror_axis=args.mirror_axis,
    )

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()