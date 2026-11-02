#!/usr/bin/env python3
"""
Automated evaluation over many simulated rollouts.

Updated with:
- Procrustes Aligned MPJPE (PA-MPJPE) in mm.
- Root Translation Error (RTE) in meters.
- Jitter in 10^-3 m/s^3.
- Foot sliding as averaged displacement in cm during contact.
- Metrics computed for sim_vs_mocap, kin_vs_mocap, and sim_vs_kin where applicable.

Units are explicitly noted in metric keys.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np

# Ensure mimickit dir is in path for anim/util imports when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def rigid_align_no_scale(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align P->Q with rotation+translation (no scale). P,Q: (N,3) finite."""
    muP = P.mean(0)
    muQ = Q.mean(0)
    X = P - muP
    Y = Q - muQ
    S = X.T @ Y / P.shape[0]
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    return R, t


def rigid_align_with_scale(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Procrustes alignment: Aligns P to Q using rotation, translation, and scale.
    P, Q: (N, 3). Returns (R, t, s) such that Q ~ s * (P @ R.T) + t.
    """
    muP = P.mean(0)
    muQ = Q.mean(0)
    X = P - muP
    Y = Q - muQ

    # Scale
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    if normX < 1e-8:
        return np.eye(3), muQ - muP, 1.0
    
    X_n = X / normX
    Y_n = Y / normY

    # Rotation
    S = X_n.T @ Y_n
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    s = normY / normX
    t = muQ - s * (R @ muP)
    return R, t, s


def apply_rigid(P: np.ndarray, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    return s * (P @ R.T) + t


def velocity(P: np.ndarray, dt: float) -> np.ndarray:
    return (P[1:] - P[:-1]) / dt


def accel(P: np.ndarray, dt: float) -> np.ndarray:
    V = velocity(P, dt)
    return (V[1:] - V[:-1]) / dt


def jerk(P: np.ndarray, dt: float) -> np.ndarray:
    A = accel(P, dt)
    return (A[1:] - A[:-1]) / dt


def pa_mpjpe_mm(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Procrustes Aligned MPJPE (single R, t, s for entire sequence) in mm.
    pred/target: (T,J,3)
    """
    T = min(pred.shape[0], target.shape[0])
    J = min(pred.shape[1], target.shape[1])
    P = pred[:T, :J].reshape(-1, 3)
    Q = target[:T, :J].reshape(-1, 3)
    finite = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    if finite.sum() < 3:
        return float("nan")
    
    R, t, s = rigid_align_with_scale(P[finite], Q[finite])
    P_al = apply_rigid(P, R, t, s)
    # Result in mm (assuming input is in meters)
    err = np.linalg.norm(P_al - Q, axis=1).mean() * 1000.0
    return float(err)


def pa_mpjve_mm_per_s(pred: np.ndarray, target: np.ndarray, dt: float) -> float:
    """
    Procrustes Aligned Mean Per-Joint Velocity Error in mm/s.
    Uses a single R, t, s for the whole sequence (same as pa_mpjpe_mm),
    then compares joint velocities.
    """
    if dt <= 0.0:
        return float("nan")

    T = min(pred.shape[0], target.shape[0])
    J = min(pred.shape[1], target.shape[1])
    if T < 2 or J < 1:
        return float("nan")

    P = pred[:T, :J].reshape(-1, 3)
    Q = target[:T, :J].reshape(-1, 3)
    finite = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    if finite.sum() < 3:
        return float("nan")

    # Align positions first (same Procrustes as pa_mpjpe_mm)
    R, t, s = rigid_align_with_scale(P[finite], Q[finite])
    P_al = apply_rigid(P, R, t, s)

    # Reshape back to (T, J, 3) and compute velocities
    P_al = P_al.reshape(T, J, 3)
    Q = Q.reshape(T, J, 3)
    Vp = velocity(P_al, dt)   # (T-1, J, 3)
    Vq = velocity(Q, dt)

    # Mean L2 velocity error over all joints and frames, in mm/s
    verr = np.linalg.norm(Vp - Vq, axis=-1).mean() * 1000.0
    return float(verr)


def compute_jitter(body_pos: np.ndarray, dt: float) -> float:
    """
    Jitter (10^-3 m/s^3) - average magnitude of jerk across all joints.
    body_pos: (T, J, 3)
    """
    if body_pos.shape[0] < 4:
        return 0.0
    # Jerk: (T-3, J, 3)
    j = jerk(body_pos, dt)
    jitter_val = np.linalg.norm(j, axis=-1).mean()
    return float(jitter_val * 1e-3)


def compute_foot_sliding_cm(
    body_pos: np.ndarray,
    foot_ids: List[int],
    dt: float,
    ground_z: float = 0.0,
    near_ground_tol: float = 0.1,
) -> float:
    """
    Averaged displacement (cm) of foot vertices during contact with the ground.
    """
    if not foot_ids or body_pos.shape[0] < 2:
        print("!!!")
        return 0.0

    P = body_pos[:, foot_ids, :]                 # (T, F, 3)
    # Positions at t and t+1
    P_curr = P[:-1]                              # (T-1, F, 3)
    P_next = P[1:]                               # (T-1, F, 3)
    
    # Contact condition based on height at t
    near_ground = np.abs(P_curr[..., 2] - ground_z) <= near_ground_tol # (T-1, F)

    
    if not near_ground.any():
        print("???")
        return 0.0
    
    # Displacement in horizontal plane (XY)
    disp_xy = np.linalg.norm(P_next[..., :2] - P_curr[..., :2], axis=-1) # (T-1, F)
    
    # Average displacement in cm for frames in contact
    sliding_cm = disp_xy[near_ground].mean() * 100.0
    return float(sliding_cm)


def compute_penetration_mm(
    body_pos: np.ndarray,
    foot_ids: List[int],
    ground_z: float = 0.0,
) -> float:
    if not foot_ids:
        return 0.0
    P = body_pos[:, foot_ids, :]
    pen_depth = np.maximum(0.0, ground_z - P[..., 2])
    if not (pen_depth > 0).any():
        return 0.0
    return float(pen_depth[pen_depth > 0].mean() * 1000.0)


def compute_floating_cm(
    body_pos: np.ndarray,
    foot_ids: List[int],
    ground_z: float = 0.0,
    float_tol: float = 0.1,
) -> float:
    """
    Floating metric: mean amount (in cm) that feet are above the ground plane
    beyond a tolerance, aggregated over all frames and selected feet.
    """
    if not foot_ids:
        return 0.0
    P = body_pos[:, foot_ids, :]  # (T, F, 3)
    # Positive height above ground minus tolerance
    float_amt = np.maximum(0.0, (P[..., 2] - ground_z) - float_tol)
    if not (float_amt > 0).any():
        return 0.0
    return float(float_amt[float_amt > 0].mean() * 100.0)


def load_states_npz(path: Path) -> Dict[str, Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    return {
        "body_pos": data["body_pos"],      # (T,B,3)
        "trans": data["trans"],            # (T,3)
        "ref_body_pos": data["ref_body_pos"] if "ref_body_pos" in data else None,
        "ref_trans": data["ref_trans"] if "ref_trans" in data else None,
        "poses_root": data["poses_root"] if "poses_root" in data else None,
        "poses_body": data["poses_body"] if "poses_body" in data else None,
        "ref_poses_root": data["ref_poses_root"] if "ref_poses_root" in data else None,
        "ref_poses_body": data["ref_poses_body"] if "ref_poses_body" in data else None,
    }

# SMAL-to-Kin joint reordering
KIN2SMAL = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
    17: 33, 18: 34, 19: 35,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24,
    27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32,
}

def _quat_xyzw_to_expmap(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(np.asarray(q, dtype=np.float64))
    return rot.as_rotvec().astype(np.float32)

def smal_params_to_kin_body_positions(
    trans: np.ndarray,
    poses_root: np.ndarray,
    poses: np.ndarray,
    char_file: Path,
) -> np.ndarray:
    import torch
    from anim.kin_char_model import KinCharModel
    import util.torch_util as torch_util

    device = torch.device("cpu")
    kin = KinCharModel(device)
    kin.load_char_file(str(char_file))

    T = trans.shape[0]
    root_pos = np.asarray(trans, dtype=np.float32)
    root_exp = _quat_xyzw_to_expmap(poses_root)

    poses_flat = poses.reshape(T * 35, 4)
    body_exp_smal = _quat_xyzw_to_expmap(poses_flat).reshape(T, 35, 3)
    body_exp_kin = np.zeros_like(body_exp_smal)
    for kin_idx in range(35):
        smal_j = KIN2SMAL[kin_idx] - 1
        body_exp_kin[:, kin_idx, :] = body_exp_smal[:, smal_j, :]
    joint_dof = body_exp_kin.reshape(T, 105)

    rp = torch.tensor(root_pos, dtype=torch.float32, device=device)
    rr = torch_util.exp_map_to_quat(torch.tensor(root_exp, dtype=torch.float32, device=device))
    jd = torch.tensor(joint_dof, dtype=torch.float32, device=device)
    jr = kin.dof_to_rot(jd)
    jr = torch_util.quat_pos(jr)

    body_pos, _ = kin.forward_kinematics(rp, rr, jr)
    return body_pos.detach().cpu().numpy().astype(np.float32)


def get_frame_indices_from_image_folder(seq_images_dir: Path) -> Optional[List[int]]:
    print("seq_images_dir", seq_images_dir)
    if not seq_images_dir.exists() or not seq_images_dir.is_dir():
        print("here?")
        return None
    exts = {"_rgb.jpg", "_rgb.png"}
    files = [f for f in seq_images_dir.iterdir() if f.name[-8:] in exts]
    if not files:
        return None
    frame_nums: List[Tuple[int, Path]] = []
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


def load_mocap_npy(mocap_path: Path) -> Optional[Dict]:
    if not mocap_path.exists():
        print("missing mocap file", mocap_path)
        return None
    arr = np.load(mocap_path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        return arr.item()
    return None


def load_mocap_smal_and_convert(
    mocap_path: Path,
    frame_ids: List[int],
    char_file: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load mocap .npy with SMAL params and convert selected frames to kin body positions.
    Expects keys:
      trans: (T,3)
      poses_root: (T,4) (wxyz or xyzw)
      poses_body: (T,35,4) xyzw in SMAL joint order
    """
    data = load_mocap_npy(mocap_path)
    if data is None:
        return None

    trans = data.get("trans")
    poses_root = data.get("poses_root")
    poses = data.get("poses_body")
    if trans is None or poses_root is None or poses is None:
        return None

    trans = np.asarray(trans, dtype=np.float64)
    poses_root = np.asarray(poses_root, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)

    # Quat order: mocap often stores wxyz; convert to xyzw if needed.
    if poses_root.shape[-1] == 4 and np.abs(poses_root[:, 0]).mean() > np.abs(poses_root[:, -1]).mean():
        w, x, y, z = poses_root[:, 0], poses_root[:, 1], poses_root[:, 2], poses_root[:, 3]
        poses_root = np.stack([x, y, z, w], axis=-1)

    T_full = trans.shape[0]
    valid_ids = [i for i in frame_ids if 0 <= i < T_full]
    if not valid_ids:
        return None

    body_pos = smal_params_to_kin_body_positions(
        trans[valid_ids], poses_root[valid_ids], poses[valid_ids], char_file
    )
    return body_pos, trans[valid_ids].astype(np.float32)


def foot_body_ids_from_names(body_names: List[str]) -> List[int]:
    foot_ids: List[int] = []
    for i, name in enumerate(body_names):
        n = name.lower()
        if any(s in n for s in ["foot"]):
            foot_ids.append(i)
    if not foot_ids:
        for i, name in enumerate(body_names):
            if "shin" in name.lower():
                foot_ids.append(i)
    return foot_ids


def gather_sequence_data(
    seq_name: str,
    states_path: Path,
    char_file: Path,
    dt: float,
    mocap_root: Optional[Path],
    images_root: Optional[Path] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load sim / kin / mocap data for one sequence.
    Returns a dict with:
      sim_body_pos, sim_root, ref_body_pos, ref_root,
      mocap_body_pos, mocap_root_traj.
    """
    print("states_path", states_path)
    states = load_states_npz(states_path)
    sim_body_pos = np.asarray(states["body_pos"], dtype=np.float32)
    sim_root = np.asarray(states["trans"], dtype=np.float32)
    ref_body_pos = (
        np.asarray(states["ref_body_pos"], dtype=np.float32)
        if states["ref_body_pos"] is not None
        else None
    )
    ref_root = (
        np.asarray(states["ref_trans"], dtype=np.float32)
        if states["ref_trans"] is not None
        else None
    )

    mocap_body_pos: Optional[np.ndarray] = None
    mocap_root_traj: Optional[np.ndarray] = None
    print("mocap_root", mocap_root)
    print("images_root", images_root)
    print("seq_name", seq_name)
    if mocap_root is not None:
        if images_root is not None:
            seq_images_dir = images_root / seq_name
            frame_ids = get_frame_indices_from_image_folder(seq_images_dir)
            print("frame_ids", frame_ids)
            if frame_ids:
                parts = seq_name.split("_")
                print(len(parts), seq_name)
                if len(parts) >= 5 and "loco"  in seq_name:
                    full_seq_name = "_".join(parts[:-2])
                else:
                    full_seq_name = "_".join(parts[:-1]) if len(parts) >= 3 else seq_name
                mocap_path = mocap_root / f"{full_seq_name}.npy"
                print(mocap_path, "exists", mocap_path.exists())
                result = load_mocap_smal_and_convert(mocap_path, frame_ids, char_file)
                if result:
                    mocap_body_pos, mocap_root_traj = result

    return {
        "sim_body_pos": sim_body_pos,
        "sim_root": sim_root,
        "ref_body_pos": ref_body_pos,
        "ref_root": ref_root,
        "mocap_body_pos": mocap_body_pos,
        "mocap_root_traj": mocap_root_traj,
    }


def evaluate_one(
    seq_name: str,
    states_path: Path,
    char_file: Path,
    dt: float,
    mocap_root: Optional[Path],
    images_root: Optional[Path] = None,
) -> Dict[str, float]:
    data = gather_sequence_data(
        seq_name=seq_name,
        states_path=states_path,
        char_file=char_file,
        dt=dt,
        mocap_root=mocap_root,
        images_root=images_root,
    )
    sim_body_pos = data["sim_body_pos"]
    sim_root = data["sim_root"]
    ref_body_pos = data["ref_body_pos"]
    ref_root = data["ref_root"]
    mocap_body_pos = data["mocap_body_pos"]
    mocap_root_traj = data["mocap_root_traj"]

    metrics: Dict[str, float] = {}

    # Pose metrics: PA-MPJPE (mm) and PA-MPJVE (mm/s)
    if ref_body_pos is not None:
        metrics["pa_mpjpe_sim_vs_kin_mm"] = pa_mpjpe_mm(sim_body_pos, ref_body_pos)
        metrics["pa_mpjve_sim_vs_kin_mm_per_s"] = pa_mpjve_mm_per_s(sim_body_pos, ref_body_pos, dt)
    if mocap_body_pos is not None:
        metrics["pa_mpjpe_sim_vs_mocap_mm"] = pa_mpjpe_mm(sim_body_pos, mocap_body_pos)
        metrics["pa_mpjve_sim_vs_mocap_mm_per_s"] = pa_mpjve_mm_per_s(sim_body_pos, mocap_body_pos, dt)
        if ref_body_pos is not None:
            metrics["pa_mpjpe_kin_vs_mocap_mm"] = pa_mpjpe_mm(ref_body_pos, mocap_body_pos)
            metrics["pa_mpjve_kin_vs_mocap_mm_per_s"] = pa_mpjve_mm_per_s(ref_body_pos, mocap_body_pos, dt)

    # Trajectory metrics: RTE (m)
    def get_rte_m(p, target):
        T = min(p.shape[0], target.shape[0])
        P_s, Q_s = p[:T].astype(np.float64), target[:T].astype(np.float64)
        finite = np.isfinite(P_s).all(1) & np.isfinite(Q_s).all(1)
        if finite.sum() < 3: return float("nan")
        R, t = rigid_align_no_scale(P_s[finite], Q_s[finite])
        P_al = apply_rigid(P_s, R, t)
        return float(np.linalg.norm(P_al - Q_s, axis=1).mean())

    if ref_root is not None:
        metrics["rte_sim_vs_kin_m"] = get_rte_m(sim_root, ref_root)
    if mocap_root_traj is not None:
        metrics["rte_sim_vs_mocap_m"] = get_rte_m(sim_root, mocap_root_traj)
        if ref_root is not None:
            metrics["rte_kin_vs_mocap_m"] = get_rte_m(ref_root, mocap_root_traj)

    # Jitter (10^-3 m/s^3)
    metrics["sim_jitter_10m_fps3"] = compute_jitter(sim_body_pos, dt)
    if ref_body_pos is not None:
        metrics["kin_jitter_10m_fps3"] = compute_jitter(ref_body_pos, dt)

    # Foot sliding (cm), penetration (mm), and floating (cm)
    import torch
    from anim.kin_char_model import KinCharModel
    kin = KinCharModel(torch.device("cpu"))
    kin.load_char_file(str(char_file))
    foot_ids = foot_body_ids_from_names(list(kin.get_body_names()))
    metrics["sim_foot_sliding_cm"] = compute_foot_sliding_cm(sim_body_pos, foot_ids, dt)
    metrics["sim_foot_penetration_mm"] = compute_penetration_mm(sim_body_pos, foot_ids)
    metrics["sim_foot_floating_cm"] = compute_floating_cm(sim_body_pos, foot_ids)
    if ref_body_pos is not None:
        print("kin")
        metrics["kin_foot_sliding_cm"] = compute_foot_sliding_cm(ref_body_pos, foot_ids, dt)
        metrics["kin_foot_penetration_mm"] = compute_penetration_mm(ref_body_pos, foot_ids)
        metrics["kin_foot_floating_cm"] = compute_floating_cm(ref_body_pos, foot_ids)

    return metrics


def visualize_sequence_gif(
    seq_name: str,
    data: Dict[str, Optional[np.ndarray]],
    images_dir: Path,
    out_path: Path,
    dt: float,
    max_frames: int = 300,
) -> None:
    """
    Create an animated GIF with:
      - RGB video frame
      - 3D sim vs kin vs mocap skeletons (aligned to mocap)
      - Top-down root trajectories.
    """
    sim_body_pos = data["sim_body_pos"]
    sim_root = data["sim_root"]
    ref_body_pos = data["ref_body_pos"]
    ref_root = data["ref_root"]
    mocap_body_pos = data["mocap_body_pos"]
    mocap_root_traj = data["mocap_root_traj"]

    if mocap_body_pos is None:
        mocap_body_pos = ref_body_pos
        mocap_root_traj = ref_root

    print("images_dir", images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        return

    # List image files and sort by name (assumes consistent naming).
    exts = {"_rgb.jpg", "_rgb.png"}
    img_files = sorted(f for f in images_dir.iterdir() if f.name[-8:] in exts)
    if not img_files:
        return
    print("found", len(img_files), "images")
    # Time length limited by all sources.
    T = min(
        sim_body_pos.shape[0],
        mocap_body_pos.shape[0],
        len(img_files),
        max_frames,
    )
    if T < 2:
        return

    # Align sim (and optionally kin) to mocap using a single rigid transform.
    J = min(sim_body_pos.shape[1], mocap_body_pos.shape[1])
    P = sim_body_pos[:T, :J].reshape(-1, 3)
    Q = mocap_body_pos[:T, :J].reshape(-1, 3)
    finite = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    if finite.sum() < 3:
        return
    R, t = rigid_align_no_scale(P[finite], Q[finite])

    sim_body_al = apply_rigid(
        sim_body_pos[:T].reshape(-1, 3), R, t
    ).reshape(T, -1, 3)
    sim_root_al = apply_rigid(sim_root[:T], R, t)

    ref_body_al = None
    ref_root_al = None
    if ref_body_pos is not None:
        ref_body_al = apply_rigid(
            ref_body_pos[:T].reshape(-1, 3), R, t
        ).reshape(T, -1, 3)
    if ref_root is not None:
        ref_root_al = apply_rigid(ref_root[:T], R, t)

    # Mocap root: prefer mocap_root_traj, otherwise joint 0.
    if mocap_root_traj is not None:
        mocap_root = mocap_root_traj[:T]
    else:
        mocap_root = mocap_body_pos[:T, 0, :]

    # Precompute global bounds for 3D view.
    pts_list = [sim_body_al[:T].reshape(-1, 3), mocap_body_pos[:T].reshape(-1, 3)]
    if ref_body_al is not None:
        pts_list.append(ref_body_al[:T].reshape(-1, 3))
    all_pts = np.concatenate(pts_list, axis=0)
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    center = 0.5 * (mn + mx)
    radius = 0.25 * float(np.max(mx - mn))
    if radius < 1e-6:
        radius = 1.0

    # Import heavy deps lazily.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import imageio.v2 as imageio
    except ImportError:
        return

    frames = []
    fps = max(1, int(round(1.0 / dt))) if dt > 0 else 30

    from PIL import Image
    for t_idx in range(T):
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

        # RGB frame
        ax_img = fig.add_subplot(gs[0, 0])
        img = imageio.imread(img_files[t_idx])
        img = Image.fromarray(img).resize((50, 50))
        img = np.array(img)
        ax_img.imshow(img)
        ax_img.set_title(f"{seq_name}  frame {t_idx}")
        ax_img.axis("off")

        # 3D skeletons
        ax3d = fig.add_subplot(gs[0, 1], projection="3d")
        ax3d.scatter(
            sim_body_al[t_idx, :, 0],
            sim_body_al[t_idx, :, 1],
            sim_body_al[t_idx, :, 2],
            c="r", s=8, label="sim",
        )
        if ref_body_al is not None:
            ax3d.scatter(
                ref_body_al[t_idx, :, 0],
                ref_body_al[t_idx, :, 1],
                ref_body_al[t_idx, :, 2],
                c="b", s=8, label="kin",
            )
        ax3d.scatter(
            mocap_body_pos[t_idx, :J, 0],
            mocap_body_pos[t_idx, :J, 1],
            mocap_body_pos[t_idx, :J, 2],
            c="g", s=8, label="mocap",
        )
        ax3d.set_xlim(center[0] - radius, center[0] + radius)
        ax3d.set_ylim(center[1] - radius, center[1] + radius)
        ax3d.set_zlim(center[2] , center[2] + 2*radius)
        ax3d.set_title("3D: sim vs kin vs mocap")
        ax3d.legend(loc="upper right")

        # Top-down root trajectories
        ax_td = fig.add_subplot(gs[1, :])
        ax_td.plot(
            mocap_root[:T, 0], mocap_root[:T, 1], "g-", label="mocap"
        )
        ax_td.plot(
            sim_root_al[:T, 0], sim_root_al[:T, 1], "r-", label="sim"
        )
        if ref_root_al is not None:
            ax_td.plot(
                ref_root_al[:T, 0], ref_root_al[:T, 1], "b-", label="kin"
            )
        ax_td.scatter(mocap_root[t_idx, 0], mocap_root[t_idx, 1], c="g", s=20)
        ax_td.scatter(sim_root_al[t_idx, 0], sim_root_al[t_idx, 1], c="r", s=20)
        if ref_root_al is not None:
            ax_td.scatter(ref_root_al[t_idx, 0], ref_root_al[t_idx, 1], c="b", s=20)
        ax_td.set_aspect("equal", adjustable="box")
        ax_td.set_title("Top-down root trajectories")
        ax_td.legend(loc="upper right")

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frames.append(buf.reshape(h, w, 3))
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def aggregate(all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for m in all_metrics.values() for k in m.keys()})
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [m[k] for m in all_metrics.values() if k in m and np.isfinite(m[k])]
        if not vals: continue
        arr = np.asarray(vals, dtype=float)
        summary[k] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "count": int(arr.shape[0]),
        }
    return summary


# Group metric keys by unit so each subplot's y-axis is comparable.
_PLOT_GROUPS = [
    ("Pose error (mm)", ["pa_mpjpe_sim_vs_kin_mm", "pa_mpjpe_sim_vs_mocap_mm", "pa_mpjpe_kin_vs_mocap_mm",
                          "sim_foot_penetration_mm"]),
    ("Pose velocity error (mm/s)", ["pa_mpjve_sim_vs_kin_mm_per_s", "pa_mpjve_sim_vs_mocap_mm_per_s",
                                     "pa_mpjve_kin_vs_mocap_mm_per_s"]),
    ("Trajectory error (m)", ["rte_sim_vs_kin_m", "rte_sim_vs_mocap_m", "rte_kin_vs_mocap_m"]),
    ("Foot sliding / floating (cm)", ["sim_foot_sliding_cm", "kin_foot_sliding_cm",
                                       "sim_foot_floating_cm", "kin_foot_floating_cm"]),
    ("Jitter (1e-3 m/s^3)", ["sim_jitter_10m_fps3", "kin_jitter_10m_fps3"]),
]


def plot_summary_bar_chart(summary: Dict[str, Dict[str, float]], out_path: Path) -> Optional[Path]:
    """Mean+/-std bar chart per metric group, skipping groups with no data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(title, [k for k in keys if k in summary]) for title, keys in _PLOT_GROUPS]
    groups = [(title, keys) for title, keys in groups if keys]
    if not groups:
        return None

    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 4.5), squeeze=False)
    for ax, (title, keys) in zip(axes[0], groups):
        means = [summary[k]["mean"] for k in keys]
        stds = [summary[k]["std"] for k in keys]
        ax.bar(range(len(keys)), means, yerr=stds, capsize=4, color="#4c72b0")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
        ax.set_title(title, fontsize=9)
        n = summary[keys[0]]["count"]
        ax.set_xlabel(f"n={n}", fontsize=8)
    fig.suptitle("Stage-4 rollout evaluation summary (mean +/- std)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def log_to_wandb(summary: Dict[str, Dict[str, float]], all_metrics: Dict[str, Dict[str, float]],
                  project: str, run_name: Optional[str], plot_path: Optional[Path]) -> None:
    """Offline-safe wandb logging: scalar summary stats + a per-sequence table + the bar chart."""
    import wandb

    wandb.init(project=project, name=run_name, config={"num_sequences": len(all_metrics)})
    log_dict = {}
    for key, stats in summary.items():
        for stat_name, val in stats.items():
            log_dict[f"eval/{key}_{stat_name}"] = val
    if plot_path is not None:
        log_dict["eval/summary_plot"] = wandb.Image(str(plot_path))

    seq_names = sorted(all_metrics.keys())
    metric_keys = sorted({k for m in all_metrics.values() for k in m.keys()})
    table = wandb.Table(columns=["sequence"] + metric_keys)
    for seq in seq_names:
        row = [seq] + [all_metrics[seq].get(k, float("nan")) for k in metric_keys]
        table.add_data(*row)
    log_dict["eval/per_sequence_table"] = table

    wandb.log(log_dict)
    wandb.finish()
    return


def main():
    ap = argparse.ArgumentParser(description="Evaluate rollouts for all sequences.")
    ap.add_argument("--rollout_dir", type=Path, required=True, help="Folder containing *_states.npz.")
    ap.add_argument("--char_file", type=Path, required=True, help="Character XML.")
    ap.add_argument("--out_dir", type=Path, default=Path("output/eval_all"), help="Output directory for metrics.")
    ap.add_argument("--mocap_root", type=Path, default=None, help="Folder with mocap .npy files (SMAL params).")
    ap.add_argument("--images_root", type=Path, default=None, help="Folder of subfolders for frame selection / visualization.")
    ap.add_argument("--viz_out_dir", type=Path, default=None, help="If set, writes per-sequence GIFs with RGB + 3D + top-down views.")
    ap.add_argument("--dt", type=float, default=1.0/30.0, help="Timestep.")
    ap.add_argument("--wandb", action="store_true", help="Also log the summary (+ per-sequence table, + bar chart) to wandb. Set WANDB_MODE=offline to avoid needing network/login.")
    ap.add_argument("--wandb_project", type=str, default="physpony_eval", help="wandb project name (only used with --wandb).")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name (only used with --wandb).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(args.rollout_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {args.rollout_dir}"); return

    all_metrics: Dict[str, Dict[str, float]] = {}
    for f in npz_files:
        seq_name = f.stem.replace("_states", "")
        try:
            all_metrics[seq_name] = evaluate_one(
                seq_name=seq_name,
                states_path=f,
                char_file=args.char_file,
                dt=args.dt,
                mocap_root=args.mocap_root,
                images_root=args.images_root,
            )
            print(f"[ok] {seq_name}")

            # Optional visualization
            if (
                args.viz_out_dir is not None
                and args.images_root is not None
                and args.mocap_root is not None
            ):
                data = gather_sequence_data(
                    seq_name=seq_name,
                    states_path=f,
                    char_file=args.char_file,
                    dt=args.dt,
                    mocap_root=args.mocap_root,
                    images_root=args.images_root,
                )
                viz_dir = args.viz_out_dir
                viz_dir.mkdir(parents=True, exist_ok=True)
                images_dir = args.images_root / seq_name
                out_gif = viz_dir / f"{seq_name}_viz.gif"
                visualize_sequence_gif(
                    seq_name=seq_name,
                    data=data,
                    images_dir=images_dir,
                    out_path=out_gif,
                    dt=args.dt,
                )
        except Exception as e:
            print(f"[fail] {seq_name}: {e}")

    summary = aggregate(all_metrics)
    with open(args.out_dir / "per_sequence_metrics.json", "w") as fp:
        json.dump(all_metrics, fp, indent=2)
    with open(args.out_dir / "summary_metrics.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    plot_path = plot_summary_bar_chart(summary, args.out_dir / "summary_plot.png")
    if plot_path is not None:
        print(f"Wrote summary plot to {plot_path}")

    if args.wandb:
        log_to_wandb(summary, all_metrics, project=args.wandb_project,
                      run_name=args.wandb_run_name, plot_path=plot_path)
        print(f"Logged summary to wandb project '{args.wandb_project}'")

    print(f"Evaluation complete. Results in {args.out_dir}")

if __name__ == "__main__":
    main()