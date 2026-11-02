#!/usr/bin/env python3
import os
import re
import glob
import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from visualize_global import rot6d_to_rotmat
from process_kinematic_data_smal import convert_to_qpos_qvel
from scale_utils import add_trans_scale_args, resolve_trans_scale
from coord_utils import fix_camera_axes

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -----------------------
# SMAL & Torch Imports
# -----------------------
try:
    import torch
    if "SMAL" not in globals():
        # Dummy placeholder if not found, to avoid crash on import
        class SMAL: pass
except ImportError:
    print("[Warning] Torch not found. Joint metrics will be skipped.")
    torch = None
    SMAL = None

# -----------------------
# Math & Geometry Helpers
# -----------------------

def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)

def quat_to_rotmat(q: np.ndarray, wxyz: bool = True) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if not wxyz:
        q = q[..., [3,0,1,2]]
    q = normalize_quat(q)
    w,x,y,z = q[...,0], q[...,1], q[...,2], q[...,3]
    ww,xx,yy,zz = w*w, x*x, y*y, z*z
    wx,wy,wz = w*x, w*y, w*z
    xy,xz,yz = x*y, x*z, y*z
    R = np.stack([
        1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)
    ], axis=-1).reshape(q.shape[:-1]+(3,3))
    return R

def rigid_align_no_scale(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    muP = P.mean(0); muQ = Q.mean(0)
    X = P - muP; Y = Q - muQ
    S = X.T @ Y / P.shape[0]
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    return R, t

def apply_rigid(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (P @ R.T) + t

def velocity(P: np.ndarray, dt: float=1.0) -> np.ndarray:
    return (P[1:] - P[:-1]) / dt

def align_to_start(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Aligns P to Q such that:
    1. The start point P[0] matches Q[0].
    2. The initial direction (P[1]-P[0]) aligns with (Q[1]-Q[0]) in the XY plane.
    """
    P_aligned = P.copy()
    Q_aligned = Q.copy()
    
    # 1. Translation: match start points
    t = Q_aligned[0] - P_aligned[0]
    P_aligned += t
    
    
    
        
        
        
        
    return P_aligned

# -----------------------
# SMAL Integration
# -----------------------

def load_smal_layer(model_path: str, device='cpu'):
    from visualize_global import SMALWrapper
    return SMALWrapper(model_path, device=device)

def compute_smal_joints(smal_layer, poses_np, betas_np, trans_np=None):
    if smal_layer is None or torch is None:
        return None

    device = smal_layer.device
    
    betas = torch.as_tensor(betas_np, dtype=torch.float32)
    pose  = torch.as_tensor(poses_np,  dtype=torch.float32).view(-1, 36, 6)
    pose  = rot6d_to_rotmat(pose).view(-1, 36, 3, 3)
    trans = torch.as_tensor(trans_np, dtype=torch.float32)

    with torch.no_grad():
        verts, joints = smal_layer(betas, pose, trans)
    
    return joints.cpu().numpy()

# -----------------------
# File I/O
# -----------------------

def read_pkl(p: Path) -> Dict:
    with open(p, "rb") as f:
        return pickle.load(f)

def parse_frame_id(name: str) -> Optional[int]:
    """ Extracts the last number before __params.pkl """
    m = re.search(r"(\d+)__params\.pkl$", name)
    if m:
        return int(m.group(1))
    return None

def load_sequence_from_results(
    results_dir: Path,
    seq_name: str,
    smal_layer,
    pkl_rot_key: str = "rotation",
    assume_c2w: bool = True,
    trans_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Looks in `results_dir` for files starting with `seq_name`.
    Example: if seq_name="camel_01", looks for "camel_01*__params.pkl"
    """
    
    patt = os.path.join(str(results_dir), f"{seq_name}*__params.pkl")
    files = [Path(x) for x in glob.glob(patt)]
    
    files = [f for f in files if parse_frame_id(f.name) is not None]
    files.sort(key=lambda p: parse_frame_id(p.name))

    if not files:
        return None

    frame_ids = []
    R_list, T_list, trans_list = [], [], []
    poses_list, betas_list = [], []

    all_info = []
    ref_data = {'poses_root': [], 'poses': [], 'trans': []}
    all_frames = {'fps': 30.0, 'frames': [], 'loop_mode': "0"}

    for p in files:
        fid = parse_frame_id(p.name)
        d = read_pkl(p)

        M = np.asarray(d.get(pkl_rot_key, np.eye(4)))
        if M.ndim == 3: M = M[0]
        
        if assume_c2w:
            R_c = M[:3,:3]
            T_c = M[:3, 3]
        else:
            R_w2c = M[:3,:3]
            t_w2c = M[:3, 3]
            R_c = R_w2c.T
            T_c = -R_w2c.T @ t_w2c
        
        tr = trans_scale * np.asarray(d.get("trans", [0,0,0])).reshape(3)
        
        if 'pose' in d:
            poses_list.append(np.asarray(d['pose']).flatten())
        
        if 'betas' in d:
            betas_list.append(np.asarray(d['betas']).flatten())
        elif 'shape' in d:
            betas_list.append(np.asarray(d['shape']).flatten())
        else:
            if len(betas_list) > 0: betas_list.append(betas_list[-1]) 
            else: betas_list.append(np.zeros(10))

        scale = np.linalg.norm(R_c[2]) 

        Rx_pos90 = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
        R_c = Rx_pos90.T @ R_c
        T_c = Rx_pos90.T @ T_c

        M[:3,:3] = R_c
        M[:3, 3] = T_c

        M = fix_camera_axes(M)

        R_c = M[:3,:3]
        T_c = M[:3, 3]

        temp_pose = rot6d_to_rotmat(torch.from_numpy(d['pose'].reshape(36,6)))
        pose_root = R_c @ np.asarray(temp_pose[0:1]).reshape(3, 3) / scale
        poses = np.asarray(temp_pose[1:]).reshape(35, 3, 3)
        trans = R_c @ (trans_scale * np.asarray(d['trans']).reshape(3, 1)) + T_c.reshape(3, 1)

        from scipy.spatial.transform import Rotation as R
        poses_root_quat = R.from_matrix(pose_root).as_quat()  # xyzw
        poses_quat = R.from_matrix(poses).as_quat()     # xyzw for 35 joints
        ref_data['poses_root'].append(poses_root_quat)
        ref_data['poses'].append(poses_quat)
        ref_data['trans'].append(d['trans'])

        
        frame_ids.append(fid)
        R_list.append(R_c)
        T_list.append(T_c)
        trans_list.append(tr)
        all_info.append([R_c, T_c, betas_list[-1], poses_list[-1], trans_list[-1]])

    pickle_dir = os.environ.get("PHYSPONY_PFERD_SIM_DIR", "")
    # Note: optional debug side-output; only written if the directory exists.
    if pickle_dir and os.path.exists(pickle_dir):
        frame_data = convert_to_qpos_qvel(ref_data)[:,:111]
        all_frames['frames'] = frame_data
        pickle.dump(all_frames, open(pickle_dir + seq_name + ".pkl", "wb"))

    data = {
        "frame_ids": frame_ids,
        "R_c2w": np.stack(R_list),
        "T_c2w": np.stack(T_list),
        "trans_pred_cam": np.stack(trans_list),
    }
    return data

def load_gt_mocap(npy_path: Path, quat_order: str="wxyz") -> Dict[str, Any]:
    if not npy_path.exists():
        return None
    
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()
    
    poses_root = np.asarray(data["poses_root"], dtype=np.float64) 
    trans = np.asarray(data["trans"], dtype=np.float64)           
    R_root = quat_to_rotmat(poses_root, wxyz=(quat_order=="wxyz"))
    
    out = {
        "trans": trans,
        "R_root": R_root,
    }

    j3d = None
    for k in ['joints3d', 'joints', 'gt_joints']:
        if k in data:
            j3d = np.asarray(data[k])
            break
    if j3d is not None:
        out["joints"] = j3d

    return out

# -----------------------
# Smoothing Function
# -----------------------

def smooth_trajectory(trajectory: np.ndarray, window_length: int = 20, polyorder: int = 1) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth the trajectory.
    """
    smoothed_traj = np.copy(trajectory)
    for i in range(3):  # Apply smoothing to each axis (X, Y, Z)
        smoothed_traj[:, i] = savgol_filter(trajectory[:, i], window_length, polyorder)
    
    return smoothed_traj

# -----------------------
# Metrics
# -----------------------

def compute_metrics(pred_data: Dict, gt_data: Dict, dt: float = 1.0) -> Dict[str, float]:
    """
    Compute metrics for the predicted and ground truth data, considering both original and smoothed trajectories.
    """
    # Slice arrays 
    R_c2w = pred_data["R_c2w"]
    T_c2w = pred_data["T_c2w"]
    t_cam = pred_data["trans_pred_cam"]
    
    # Transform Pred Root to World 
    P_pred_world = np.einsum("nij,nj->ni", R_c2w, t_cam) + T_c2w 
    
    # GT Root
    P_gt_cam = gt_data["trans"][np.array(pred_data["frame_ids"])] 
    P_gt_world = P_gt_cam 

    smoothed_P_pred_world = smooth_trajectory(P_pred_world)

    # Rigid Alignment (Global Procrustes for Metrics)
    R_align, t_align = rigid_align_no_scale(P_pred_world, P_gt_world)
    P_pred_aligned = apply_rigid(P_pred_world, R_align, t_align)
    P_pred_aligned_smoothed = apply_rigid(smoothed_P_pred_world, R_align, t_align)

    # Start Alignment (For Visualization - top down plot)
    P_pred_start = align_to_start(P_pred_aligned, P_gt_world)
    P_pred_smoothed_start = align_to_start(P_pred_aligned_smoothed, P_gt_world)

    # Metrics Calculation (Original - using global alignment)
    err = np.linalg.norm(P_pred_aligned - P_gt_world, axis=1).mean()
    path_len = np.linalg.norm(P_gt_world[-1] - P_gt_world[0])
    rte_pct = 100.0 * err / max(path_len, 1e-9)
    v_err = np.linalg.norm(velocity(P_pred_aligned, dt) - velocity(P_gt_world, dt), axis=1).mean()

    # Metrics Calculation (Smoothed - using global alignment)
    err_smoothed = np.linalg.norm(P_pred_aligned_smoothed - P_gt_world, axis=1).mean()
    rte_pct_smoothed = 100.0 * err_smoothed / max(path_len, 1e-9)
    v_err_smoothed = np.linalg.norm(velocity(P_pred_aligned_smoothed, dt) - velocity(P_gt_world, dt), axis=1).mean()

    # Compute the new metric - Egocentric-Frame Root Velocity Error (ERVE) before and after smoothing
    erve_before_smoothing = compute_egocentric_root_velocity_error(P_pred_world, P_gt_world, dt)
    erve_after_smoothing = compute_egocentric_root_velocity_error(smoothed_P_pred_world, P_gt_world, dt)
    

    metrics = {
        "RTE_m": float(err),
        "RTE_pct": float(rte_pct),
        "VelErr": float(v_err) * 1000,
        "RTE_m_smoothed": float(err_smoothed),
        "RTE_pct_smoothed": float(rte_pct_smoothed),
        "VelErr_smoothed": float(v_err_smoothed) * 1000,
        "ERVE_before_smoothing": float(erve_before_smoothing) * 1000,  # In mm/frame
        "ERVE_after_smoothing": float(erve_after_smoothing) * 1000,    # In mm/frame
    }

    vis_data = {
        "P_pred": P_pred_world,
        "P_gt": P_gt_world,
        "P_pred_aligned": P_pred_aligned,
        "P_pred_aligned_smoothed": P_pred_aligned_smoothed,
        "P_pred_start": P_pred_start,
        "P_pred_smoothed_start": P_pred_smoothed_start
    }
    
    return metrics, vis_data


def compute_egocentric_root_velocity_error(P_pred_world: np.ndarray, P_gt_world: np.ndarray, dt: float = 1.0) -> float:
    """
    Compute the Egocentric-Frame Root Velocity Error (ERVE) between predicted and ground truth data.
    This is the error in velocity between the root of the agent in world coordinates, transformed to an 
    egocentric frame relative to the root position.
    """
    # Extract root positions and velocities
    print(P_pred_world.shape, P_gt_world.shape)
    # Compute velocities of the root in world coordinates
    V_pred_world = velocity(P_pred_world, dt)
    V_gt_world = velocity(P_gt_world, dt)
    
    # Compute the relative velocity in the egocentric frame
    print(P_pred_world.shape)
    V_pred_egocentric = V_pred_world - velocity(P_pred_world[0:2], dt)  # Root relative velocity
    V_gt_egocentric = V_gt_world - velocity(P_gt_world[0:2], dt)        # Root relative velocity
    
    # Compute the velocity error (Euclidean norm)
    v_err_egocentric = np.linalg.norm(V_pred_egocentric - V_gt_egocentric, axis=1).mean()
    
    return v_err_egocentric

# -----------------------
# Visualization
# -----------------------

def plot_sequence(vis_data: Dict, save_path: Path, title: str):
    """
    Plot both original and smoothed trajectories for comparison.
    """
    P_gt = vis_data["P_gt"]
    # For ax1: Use global alignment (or world if preferred, but existing code used P_pred_al_smoothed)
    # The existing code used: P_gt, P_pred (red), P_pred_al_smoothed (green).
    # We keep ax1 as is (Comparison of GT vs Raw World vs Smoothed Aligned)
    P_pred = vis_data["P_pred"]
    P_pred_al_smoothed = vis_data["P_pred_aligned_smoothed"]

    # For ax2: Use the new Start-Aligned trajectories
    P_pred_start = vis_data["P_pred_start"]
    P_pred_smoothed_start = vis_data["P_pred_smoothed_start"]

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(P_gt[:, 0], P_gt[:, 1], 'k-', label="GT", linewidth=2)
    ax1.plot(P_pred[:, 0], P_pred[:, 1], 'r--', label="Pred", linewidth=2)
    ax1.plot(P_pred_al_smoothed[:, 0], P_pred_al_smoothed[:, 1], 'g-', label="Smoothed Pred (Global Align)", linewidth=2)
    ax1.set_title(f"{title} - World / Global Align")
    ax1.legend()
    ax1.axis('equal')

    ax2 = fig.add_subplot(1, 2, 2) 
    ax2.plot(P_gt[:,0], P_gt[:,1], 'k-', label="GT") 
    # Use the start-aligned data here
    ax2.plot(P_pred_start[:, 0], P_pred_start[:, 1], label="w/o RL", linewidth=2)
    ax2.plot(P_pred_smoothed_start[:, 0], P_pred_smoothed_start[:, 1], label="w/ RL", linewidth=2)

    ax2.set_title("Aligned to 1st Frame (Start)")
    ax2.legend() 
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# -----------------------
# Main
# -----------------------

def main():
    _model_dir = os.path.dirname(os.path.abspath(__file__))          # <repo>/model
    _repo = os.path.dirname(_model_dir)                               # <repo>
    ap = argparse.ArgumentParser()
    # Path with subfolders (e.g. val3/camel01, val3/horse02) to identify Sequence Names
    ap.add_argument("--data_root", default=os.path.join(_repo, "data", "magicpony", "processed_data", "val3"))
    # Path where mixed pkl files are actually stored
    ap.add_argument("--results_dir", default=os.path.join(_repo, "checkpoints", "ponymation", "pretrained_horse_stage2_rot6d_withrendering_smoothing_extra_global", "test_results_checkpoint85000") + os.sep)
    # Path to GT (mocap sequences bundled with the sim module)
    ap.add_argument("--mocap_root", default=os.path.join(_repo, "sim", "mimickit", "mocap_full") + os.sep)
    # Output
    ap.add_argument("--out_dir", default="./pferd_ID1_loco")
    # SMAL
    ap.add_argument("--smal_model", default=os.path.join(_model_dir, "model", "geometry", "SMAL", "my_smpl_0000_horse_new_skeleton_horse.pkl"))
    ap.add_argument("--dt", type=float, default=0.033)
    add_trans_scale_args(ap)

    args = ap.parse_args()
    trans_scale = resolve_trans_scale(args.trans_scale, args.median_scale_file)
    print(f"Using trans_scale = {trans_scale}")


    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize SMAL
    smal_layer = None
    if args.smal_model:
        smal_layer = load_smal_layer(args.smal_model)
    else:
        print("[Info] No --smal_model provided. Skipping joint metrics.")

    # 2. Get list of sequence names from data_root
    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    mocap_root = Path(args.mocap_root)
    
    # Identify sequences by looking at subfolders in data_root
    if not data_root.exists():
        print(f"[Error] data_root {data_root} does not exist.")
        return

    seq_names = [d.name for d in data_root.iterdir() if d.is_dir()]
    seq_names = sorted(seq_names)
    
    print(f"Found {len(seq_names)} target sequences in {data_root}.")
    print(f"Searching for .pkl files in {results_dir}...")
    
    agg_metrics = {}
    
    for seq_name in seq_names:
        if "20201128_ID_1_0007_23604_01049" in seq_name:
            continue
       
        temp_seq_name = "_".join(seq_name.split("_")[:-2])
        # Load GT
        mocap_file = mocap_root / f"{temp_seq_name}.npy"
        
        if not mocap_file.exists():
            continue
        
        gt_data = load_gt_mocap(mocap_file)
        
        # Load Preds from Results Directory using seq_name as prefix
        pred_data = load_sequence_from_results(
            results_dir=results_dir,
            seq_name=seq_name,
            smal_layer=smal_layer,
            trans_scale=trans_scale,
        )
        
        if not pred_data:
            continue
            
        res = compute_metrics(pred_data, gt_data, dt=args.dt)
        if not res: continue
        
        metrics, vis_data = res 
        # Save results
        out_subdir = out_root / seq_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        plot_sequence(vis_data, out_subdir / "traj.png", seq_name)
        
        row = [f"{k}={v:.2f}" for k,v in metrics.items()]
        print(f" -> {seq_name}: " + ", ".join(row))
        
        for k,v in metrics.items():
            agg_metrics.setdefault(k, []).append(v)
            
    # Summary
    print("\n" + "="*40)
    print(f"Aggregated Results ({len(seq_names)} seqs scanned)")
    print("="*40)
    for k, vals in agg_metrics.items():
        v = np.array(vals)
        print(f"{k:<15} | {v.mean():.4f} +/- {v.std():.4f}")
        
    with open(out_root / "metrics.txt", "w") as f:
        for k, vals in agg_metrics.items():
            v = np.array(vals)
            f.write(f"{k}: {v.mean():.4f} +/- {v.std():.4f}\n")


if __name__ == "__main__":
    main()