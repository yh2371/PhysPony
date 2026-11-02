#!/usr/bin/env python3
import os, re, glob, pickle, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from scale_utils import add_trans_scale_args, resolve_trans_scale

# -----------------------
# Helpers
# -----------------------
def read_pkl(p: Path) -> Dict:
    with open(p, "rb") as f:
        return pickle.load(f)

def parse_frame_id_from_name(name: str) -> Optional[int]:
    """
    Supports filenames like:
      <prefix>{frame_num}__params.pkl
    Extract the {frame_num} by regex.
    """
    m = re.search(r"(\d+)__params\.pkl$", name)
    if m:
        return int(m.group(1))
    return None

def list_pkls(data_root: str, prefix: str) -> List[Path]:
    # matches: prefix + "{frame}__params.pkl"
    patt = os.path.join(data_root, f"{prefix}*__params.pkl")
    files = [Path(x) for x in glob.glob(patt)]
    files = [f for f in files if parse_frame_id_from_name(f.name) is not None]
    files.sort(key=lambda p: parse_frame_id_from_name(p.name))
    return files

def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)

def quat_to_rotmat(q: np.ndarray, wxyz: bool = True) -> np.ndarray:
    """
    q: (...,4)
    wxyz True => [w,x,y,z], else [x,y,z,w]
    """
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
    """Align P->Q with rotation+translation (no scale). P,Q: (N,3) finite."""
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

def accel(P: np.ndarray, dt: float=1.0) -> np.ndarray:
    V = velocity(P, dt)
    return (V[1:] - V[:-1]) / dt

def rte_percent(P_aligned: np.ndarray, Q: np.ndarray) -> float:
    err = np.linalg.norm(P_aligned - Q, axis=1).mean()
    disp = np.linalg.norm(Q[-1] - Q[0])
    disp = max(disp, 1e-9)
    return 100.0 * err / disp

def egocentric_vel(P: np.ndarray, R_local: np.ndarray, dt: float=1.0) -> np.ndarray:
    """
    P: (N,3) world positions
    R_local: (N,3,3) world orientation defining local frame at each frame
    returns velocities expressed in local frame, shape (N-1,3)
    """
    v = velocity(P, dt)                 # (N-1,3)
    Ruse = R_local[:-1]                 # (N-1,3,3)
    vloc = np.einsum("nij,nj->ni", np.transpose(Ruse, (0,2,1)), v)
    return vloc

# -----------------------
# Core extraction
# -----------------------
def extract_from_pkls(
    data_root: str,
    prefix: str,
    pkl_rotation_key: str = "rotation",
    assume_rotation_is_c2w: bool = True,
    trans_scale: float = 1.0,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      frame_ids: list
      R_c2w: (N,3,3)
      T_c2w: (N,3)
      trans_pred_cam: (N,3) predicted root translation in camera coords (after your scaling in pkl)
    """
    pkls = list_pkls(data_root, prefix)
    if not pkls:
        raise RuntimeError(f"No pkls found under {data_root} with prefix '{prefix}'")

    frame_ids = []
    R_list, T_list, trans_list = [], [], []
    for p in pkls:
        fid = parse_frame_id_from_name(p.name)
        d = read_pkl(p)

        M = d[pkl_rotation_key]
        M = np.asarray(M)
        if M.ndim == 3:
            M = M[0]
        assert M.shape == (4,4), f"{p} rotation must be 4x4"

        if assume_rotation_is_c2w:
            R_c2w = M[:3,:3].astype(np.float64)
            T_c2w = M[:3, 3].astype(np.float64)
        else:
            # if M is w2c, invert
            R_w2c = M[:3,:3].astype(np.float64)
            t_w2c = M[:3, 3].astype(np.float64)
            R_c2w = R_w2c.T
            T_c2w = -R_w2c.T @ t_w2c


        trans_pred_cam = np.asarray(d["trans"], dtype=np.float64).reshape(3) * trans_scale

        frame_ids.append(fid)
        R_list.append(R_c2w)
        T_list.append(T_c2w)
        trans_list.append(trans_pred_cam)

    return frame_ids, np.stack(R_list), np.stack(T_list), np.stack(trans_list)

def load_mocap_root_cam(mocap_npy: str, quat_order: str="wxyz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      R_root_cam: (T,3,3) root orientation in camera coords (if you need it)
      trans_root_cam: (T,3)
    """
    data = np.load(mocap_npy, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()
    poses_root = np.asarray(data["poses_root"], dtype=np.float64)  # (T,4)
    trans = np.asarray(data["trans"], dtype=np.float64)           # (T,3)
    R_root = quat_to_rotmat(poses_root, wxyz=(quat_order=="wxyz"))
    return R_root, trans

def to_world(R_c2w: np.ndarray, T_c2w: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    """
    R_c2w: (N,3,3), T_c2w: (N,3), p_cam: (N,3)
    returns p_world: (N,3)
    """
    return np.einsum("nij,nj->ni", R_c2w, p_cam) + T_c2w

# -----------------------
# Plot + metrics
# -----------------------
def plot_topdown_and_3d(P_gt: np.ndarray, P_pred: np.ndarray, P_cam: np.ndarray, out_png: str):
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2, projection="3d")

    ax1.plot(P_gt[:,0], P_gt[:,1], label="GT root (world)", linewidth=2)
    ax1.plot(P_pred[:,0], P_pred[:,1], label="Pred root (world)", linewidth=2)
    ax1.plot(P_cam[:,0], P_cam[:,1], label="Camera (world)", alpha=0.8)

    ax1.set_title("Top-down (X-Y)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.axis("equal"); ax1.legend()

    ax2.plot(P_gt[:,0], P_gt[:,1], P_gt[:,2], label="GT root (world)", linewidth=2)
    ax2.plot(P_pred[:,0], P_pred[:,1], P_pred[:,2], label="Pred root (world)", linewidth=2)
    ax2.plot(P_cam[:,0], P_cam[:,1], P_cam[:,2], label="Camera (world)", alpha=0.8)
    ax2.set_title("3D")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z"); ax2.legend()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_topdown_and_3d_after_alignment(
    P_gt: np.ndarray,
    P_pred: np.ndarray,
    P_cam: np.ndarray,
    out_png: str,
    align_cam_too: bool = True,
):
    """
    Rigid-align P_pred -> P_gt (no scale), then plot:
      - GT
      - aligned pred
      - (optionally) camera trajectory aligned by same transform (for consistent view)
    """
    finite = np.isfinite(P_gt).all(1) & np.isfinite(P_pred).all(1)
    if finite.sum() < 3:
        raise RuntimeError("Not enough valid points to rigid-align trajectories.")

    R, t = rigid_align_no_scale(P_pred[finite], P_gt[finite])
    P_pred_al = apply_rigid(P_pred, R, t)

    if align_cam_too:
        P_cam_al = apply_rigid(P_cam, R, t)
    else:
        P_cam_al = P_cam

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.plot(P_gt[:, 0], P_gt[:, 1], label="GT root (world)", linewidth=2)
    ax1.plot(P_pred[:, 0], P_pred[:, 1], label="Pred root (aligned)", linewidth=2)
    ax1.plot(P_cam_al[:, 0], P_cam_al[:, 1], label="Camera (aligned)" if align_cam_too else "Camera", alpha=0.8)
    ax1.set_title("Top-down (X-Y) after rigid alignment (no scale)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y")
    ax1.axis("equal")
    ax1.legend()

    ax2.plot(P_gt[:, 0], P_gt[:, 1], P_gt[:, 2], label="GT root (world)", linewidth=2)
    ax2.plot(P_pred_al[:, 0], P_pred_al[:, 1], P_pred_al[:, 2], label="Pred root (aligned)", linewidth=2)
    ax2.plot(P_cam_al[:, 0], P_cam_al[:, 1], P_cam_al[:, 2], label="Camera (aligned)" if align_cam_too else "Camera", alpha=0.8)
    ax2.set_title("3D after rigid alignment (no scale)")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def compute_root_motion_metrics(P_pred: np.ndarray, P_gt: np.ndarray, R_local: np.ndarray, dt: float=1.0) -> Dict[str,float]:
    """
    P_pred/P_gt: (N,3) in SAME frame (world)
    R_local: (N,3,3) local frame rotations (use GT root/world orientation proxy; if not available, use camera R_c2w)
    """
    finite = np.isfinite(P_pred).all(1) & np.isfinite(P_gt).all(1)
    Pp = P_pred[finite]; Pg = P_gt[finite]; Rl = R_local[finite]

    # rigid-align pred->gt w/o scaling (for RTE/ERVE like WHAM-style)
    R, t = rigid_align_no_scale(Pp, Pg)
    Pp_al = apply_rigid(Pp, R, t)

    rte = rte_percent(Pp_al, Pg)

    v_err = np.linalg.norm(velocity(Pp_al, dt) - velocity(Pg, dt), axis=1).mean()
    a_err = np.linalg.norm(accel(Pp_al, dt) - accel(Pg, dt), axis=1).mean()

    vP_loc = egocentric_vel(Pp_al, Rl, dt)
    vG_loc = egocentric_vel(Pg,    Rl, dt)
    erve_val = np.linalg.norm(vP_loc - vG_loc, axis=1).mean()

    return {
        "RTE_percent": float(rte),
        "RootVelErr_world": float(v_err)*1000,
        "RootAccelErr_world": float(a_err)*1000,
        "ERVE_egovel": float(erve_val)*1000,
    }

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder with pkls/images")
    ap.add_argument("--prefix", required=True, help="same prefix you used in PKL_PATTERN, without {frame_num}")
    ap.add_argument("--mocap_npy", required=True, help="mocap/<seq>.npy containing poses_root/trans")

    ap.add_argument("--quat_order", choices=["wxyz","xyzw"], default="wxyz")
    ap.add_argument("--assume_rot", choices=["c2w","w2c"], default="c2w",
                    help="interpret pkl['rotation'] as camera-to-world or world-to-camera")
    ap.add_argument("--frame_offset", type=int, default=0,
                    help="mocap_index = frame_id - frame_offset")
    ap.add_argument("--dt", type=float, default=1.0)

    ap.add_argument("--out_dir", required=True)
    add_trans_scale_args(ap)
    args = ap.parse_args()
    trans_scale = resolve_trans_scale(args.trans_scale, args.median_scale_file)
    print(f"Using trans_scale = {trans_scale}")

    frame_ids, R_c2w, T_c2w, trans_pred_cam = extract_from_pkls(
        data_root=args.data_root,
        trans_scale=trans_scale,
        prefix=args.prefix,
        pkl_rotation_key="rotation",
        assume_rotation_is_c2w=(args.assume_rot=="c2w"),
    )

    R_root_cam, trans_gt_cam_all = load_mocap_root_cam(args.mocap_npy, quat_order=args.quat_order)

    # align mocap indices to frames
    N = len(frame_ids)
    trans_gt_cam = np.zeros((N,3), dtype=np.float64)
    for i, fid in enumerate(frame_ids):
        t = fid - args.frame_offset
        if 0 <= t < trans_gt_cam_all.shape[0]:
            trans_gt_cam[i] = trans_gt_cam_all[t]
        else:
            trans_gt_cam[i] = np.nan

    # Camera trajectory in world: camera origin transformed into world => T_c2w
    P_cam_world = T_c2w.copy()

    # Root trajectories in world
    P_pred_world = to_world(R_c2w, T_c2w, trans_pred_cam)
    Rx_pos90 = np.array([
    [1, 0, 0],
    [0, 0,-1],
    [0, 1, 0],
], dtype=np.float64)
    P_pred_world = (Rx_pos90 @ P_pred_world.T).T
    P_gt_world   = to_world(R_c2w, T_c2w, trans_gt_cam)

    out_png = os.path.join(args.out_dir, "topdown_global_traj.png")
    plot_topdown_and_3d(P_gt_world, P_pred_world, P_cam_world, out_png)
    print(f"[ok] wrote {out_png}")
    out_png_aligned = os.path.join(args.out_dir, "topdown_global_traj_aligned.png")
    plot_topdown_and_3d_after_alignment(P_gt_world, P_pred_world, P_cam_world, out_png_aligned, align_cam_too=True)
    print(f"[ok] wrote {out_png_aligned}")

    # For local frame rotations in ERVE, a decent proxy is the camera-to-world rotation
    # (if you have GT root world rotations, use those instead).
    metrics = compute_root_motion_metrics(P_pred_world, P_gt_world, R_local=R_c2w, dt=args.dt)
    print("=== Root motion metrics (rigid-aligned, no scale) ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.6f}")

if __name__ == "__main__":
    main()
