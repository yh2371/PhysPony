#!/usr/bin/env python3
# visualize_global_skeleton.py
import os
import math
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from scale_utils import add_trans_scale_args, resolve_trans_scale

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Zhou et al., CVPR'19."""
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)

# ---------- User-config ----------
PKL_PATTERN = "{checkpoint_id:07d}_{frame_num: 10d}_params.pkl"
PRED_IMG_PATTERN = "{checkpoint_id:07d}_{frame_num: 10d}_image_pred.png"
GT_IMG_PATTERN   = "{checkpoint_id:07d}_{frame_num: 10d}_image_gt.png"


SMAL_MODEL_PATH = os.environ.get(
    "PHYSPONY_SMAL_PKL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "model", "geometry", "SMAL", "my_smpl_0000_horse_new_skeleton_horse.pkl"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 120
FIG_W = 16
FIG_H = 10   # taller to fit extra row
ROOT_IDX = 0 # change if your root joint index differs

# ---------- SMAL wrapper ----------
class SMALWrapper(nn.Module):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        from model.geometry.SMAL.smal_torch.smal_torch import SMAL  # type: ignore
        self.smal = SMAL(model_path, device=self.device)
        self.parents = None
        for attr in ["kintree_table", "parents", "parent"]:
            if hasattr(self.smal, attr):
                tbl = getattr(self.smal, attr)
                try:
                    arr = np.array(tbl).squeeze()
                    if arr.ndim == 2 and arr.shape[0] == 2:
                        arr = arr[0]
                    self.parents = arr.tolist()
                except Exception:
                    pass
                break

    @torch.no_grad()
    def forward(self, betas: torch.Tensor, pose_R: torch.Tensor, trans: torch.Tensor):
        """pose_R: [1, 36, 3, 3] rotation matrices per joint."""
        outs = self.smal(beta=betas.to(self.device), theta=pose_R.to(self.device), trans=trans.to(self.device))
        if isinstance(outs, (tuple, list)):
            verts = outs[0]
            joints = outs[1] if len(outs) > 1 else None
        else:
            verts = outs.verts if hasattr(outs, "verts") else outs[0]
            joints = outs.joints if hasattr(outs, "joints") else None
        if joints is None:
            raise RuntimeError("SMAL forward didn't return joints; please expose joints.")
        return verts, joints

# ---------- IO helpers ----------
def load_pkl(pkl_path: Path, trans_scale: float = 1.0):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    M = data["rotation"]
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    if M.ndim == 3:
        M = M[0]
    assert M.shape == (4, 4), f"rotation in {pkl_path} must be 4x4, got {M.shape}"

    transform_temp = np.eye(4, dtype=np.float32)
    transform_temp[:3, :3] = np.array([[-1, 0, 0],
                       [ 0,-1, 0],
                       [ 0, 0, 1]], dtype=np.float32) # dessie cam preset, modify as needed
    transform_temp2 = np.eye(4, dtype=np.float32)
    transform_temp2[:3, :3] = np.array([[1, 0, 0],
                                       [0,0,1],
                                       [0,1,0]], dtype=np.float32) # align to z-up modify as needed

    M = transform_temp2 @ M @ transform_temp

    

    # If you have true camera->world in M, use it. Current code keeps identity with a flip.
    R_c2w = M[:3, :3]
    T_c2w = M[:3, 3]

    betas = torch.as_tensor(data["betas"], dtype=torch.float32).view(1, -1)
    pose6d = torch.as_tensor(data["pose"], dtype=torch.float32)
    if pose6d.shape[-1] == 6 or pose6d.shape[-1] == 216:
        pose6d = pose6d.view(36, -1)
        pose_R = rot6d_to_rotmat(pose6d).view(1, 36, 3, 3)
    else:
        pose_R = pose6d.view(1, 36, 3, 3)
    trans  = torch.as_tensor(data["trans"], dtype=torch.float32).view(1, 3)

    return R_c2w, T_c2w, betas, pose_R, trans * trans_scale

def load_img(path: Path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

# ---------- Geometry helpers ----------
def cam_to_world(points_cam: torch.Tensor, R_c2w: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    P = points_cam[0].detach().cpu().numpy()  # [J,3]
    return (R_c2w @ P.T).T + T_c2w[None, :]

def camera_center_and_zdir(R_c2w: np.ndarray, T_c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Camera center C and Z-axis (forward) direction in world (from c2w)."""
    C = T_c2w.copy()
    zdir = (R_c2w @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    # normalize for plotting
    n = np.linalg.norm(zdir) + 1e-9
    return C, zdir / n

def get_bones_from_parents(parents: Optional[List[int]], J: int) -> List[Tuple[int, int]]:
    edges = []
    if parents is not None:
        for i in range(J):
            p = parents[i] if i < len(parents) else -1
            if p >= 0 and p < J and p != i:
                edges.append((i, p))
    else:
        edges = [(i, i - 1) for i in range(1, J)]
    return edges

def set_axes_equal(ax, X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    max_range = (np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()) * 0.5
    mid = np.array([x.mean(), y.mean(), z.mean()])
    if not np.isfinite(max_range) or max_range < 1e-6:
        max_range = 0.1
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

def draw_skeleton_world(ax, joints_w: np.ndarray, bones: List[Tuple[int,int]]):
    ax.cla()
    for (i, j) in bones:
        p, q = joints_w[i], joints_w[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=2)
    ax.scatter(joints_w[:, 0], joints_w[:, 1], joints_w[:, 2], s=10)
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_axes_equal(ax, joints_w)
    ax.set_title("SMAL skeleton (world)")

# ---------- Main routine ----------
def make_video(
    data_root: str,
    smal_model_path: str,
    frame_nums:str,# List[int],
    out_mp4: str,
    fps: int = 15,
    device: str = DEVICE,
    checkpoint_id: int = 20000,
    save_path: str = "all_kin.pkl",
    trans_scale: float = 1.0,
):
    data_root2 = Path(data_root)
    smal = SMALWrapper(smal_model_path, device=device).eval()

    cache = []
    all_pts_for_bounds = []

    # Pre-pass: load & compute world data per frame
    root_traj = []   # [T,3]
    cam_traj  = []   # [T,3]
    zdir_traj = []   # [T,3] unit vectors

    all_info = []
    import glob
    print(frame_nums)
    for fn in sorted(glob.glob(data_root+"/*image_gt.png")):
        if not frame_nums in fn:
            continue
        print(fn)
        pkl_path  = fn.replace("image_gt.png", "params.pkl") #PKL_PATTERN.format(checkpoint_id=checkpoint_id, frame_num=fn)
        pred_path = fn.replace("image_gt.png", "image_pred.png") #PRED_IMG_PATTERN.format(checkpoint_id=checkpoint_id, frame_num=fn)
        gt_path   = fn #data_root / fn#.replace("image_gt.png", "params.pkl") #GT_IMG_PATTERN.format(checkpoint_id=checkpoint_id, frame_num=fn)

        if not os.path.exists(pkl_path): #.exists():
            print(f"[warn] missing {pkl_path}, skipping")
            continue
        if not os.path.exists(pred_path) or not os.path.exists(gt_path): #.exists():
            print(f"[warn] missing pred/gt image for frame {fn}, skipping")
            continue

        R_c2w, T_c2w, betas, pose_R, trans = load_pkl(pkl_path, trans_scale=trans_scale)
        verts_cam, joints_cam = smal(betas, pose_R, trans)
        joints_w = cam_to_world(joints_cam, R_c2w, T_c2w)
        all_info.append((R_c2w, T_c2w, betas, pose_R, trans, joints_w))
        # camera center & z-direction
        C, zdir = camera_center_and_zdir(R_c2w, T_c2w)

        # choose a root joint (default ROOT_IDX)
        root_w = joints_w[ROOT_IDX]

        pred_img = load_img(pred_path)
        gt_img   = load_img(gt_path)

        cache.append((fn, pred_img, gt_img, joints_w, C, zdir, root_w))
        all_pts_for_bounds.append(joints_w)
        root_traj.append(root_w)
        cam_traj.append(C)
        zdir_traj.append(zdir)

    if not cache:
        raise RuntimeError("No frames found to render.")

    root_traj = np.stack(root_traj, axis=0)  # [T,3]
    cam_traj  = np.stack(cam_traj,  axis=0)  # [T,3]
    zdir_traj = np.stack(zdir_traj, axis=0)  # [T,3]

    pickle.dump(all_info, open(save_path+".pkl", "wb"))


    # Build bone edges
    J = cache[0][3].shape[0]
    bones = get_bones_from_parents(smal.parents, J)

    # Figure & writer (2 rows x 3 cols)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.6], width_ratios=[1, 1, 1.2], hspace=0.25, wspace=0.1)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt   = fig.add_subplot(gs[0, 1])
    ax_3d   = fig.add_subplot(gs[0, 2], projection='3d')

    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[1, 2])

    writer = FFMpegWriter(fps=fps, bitrate=6000)
    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Global bounds for stable axes
    X_all = np.concatenate(all_pts_for_bounds + [root_traj, cam_traj], axis=0)
    mn, mx = X_all.min(axis=0), X_all.max(axis=0)
    max_range = (np.max(mx - mn) * 0.4) or 0.1
    mid = (mn + mx) / 2.0

    def set3d_bounds(ax):
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range*0.5, mid[2]+max_range*0.5)

    def set2d_equal(ax, xs, ys):
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
        r = max(xmax-xmin, ymax-ymin) * 0.55 + 1e-6
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_aspect('equal', adjustable='box')

    with writer.saving(fig, str(out_path), DPI):
        for t, (fn, pred_img, gt_img, joints_w, C, zdir, root_w) in enumerate(cache):
            # Left: pred
            ax_pred.cla()
            ax_pred.imshow(pred_img)
            ax_pred.set_title(f"pred: {fn}")
            ax_pred.axis("off")

            # Middle: gt
            ax_gt.cla()
            ax_gt.imshow(gt_img)
            ax_gt.set_title(f"gt: {fn}")
            ax_gt.axis("off")

            # Right top: 3D skeleton + camera + trajectories
            ax_3d.cla()
            # skeleton
            for (i, j) in bones:
                p, q = joints_w[i], joints_w[j]
                ax_3d.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=2)
            ax_3d.scatter(joints_w[:, 0], joints_w[:, 1], joints_w[:, 2], s=10)

            # trajectories up to t
            ax_3d.plot(cam_traj[:t+1,0], cam_traj[:t+1,1], cam_traj[:t+1,2], linestyle='-', linewidth=2, alpha=0.8, label='cam path')
            ax_3d.plot(root_traj[:t+1,0], root_traj[:t+1,1], root_traj[:t+1,2], linestyle='--', linewidth=2, alpha=0.8, label='root path')

            # current camera center & z-axis arrow
            arrow_len = 0.15 * (2*max_range)
            ax_3d.scatter([C[0]], [C[1]], [C[2]], s=40)
            ax_3d.plot([C[0], C[0] + zdir[0]*arrow_len],
                       [C[1], C[1] + zdir[1]*arrow_len],
                       [C[2], C[2] + zdir[2]*arrow_len], linewidth=3)

            ax_3d.view_init(elev=20, azim=-60)
            ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
            ax_3d.set_title("World Frame: skeleton + camera (C & z-dir)")
            set3d_bounds(ax_3d)
            # optional legend (can clutter): ax_3d.legend(loc='upper right')

            # Bottom row: XY / XZ / YZ projections (root & camera)
            # XY
            ax_xy.cla()
            ax_xy.plot(cam_traj[:t+1,0], cam_traj[:t+1,1], '-', linewidth=2, alpha=0.9, label='cam')
            ax_xy.plot(root_traj[:t+1,0], root_traj[:t+1,1], '--', linewidth=2, alpha=0.9, label='root')
            ax_xy.scatter([cam_traj[t,0]], [cam_traj[t,1]], s=30)
            ax_xy.scatter([root_traj[t,0]], [root_traj[t,1]], s=30)
            ax_xy.set_title("XY Plane Trajectory")
            ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
            set2d_equal(ax_xy, X_all[:,0], X_all[:,1])
            # XZ
            ax_xz.cla()
            ax_xz.plot(cam_traj[:t+1,0], cam_traj[:t+1,2], '-', linewidth=2, alpha=0.9, label='cam')
            ax_xz.plot(root_traj[:t+1,0], root_traj[:t+1,2], '--', linewidth=2, alpha=0.9, label='root')
            ax_xz.scatter([cam_traj[t,0]], [cam_traj[t,2]], s=30)
            ax_xz.scatter([root_traj[t,0]], [root_traj[t,2]], s=30)
            ax_xz.set_title("XZ Plane Trajectory")
            ax_xz.set_xlabel("X (m)"); ax_xz.set_ylabel("Z (m)")
            set2d_equal(ax_xz, X_all[:,0], X_all[:,2])
            # YZ
            ax_yz.cla()
            ax_yz.plot(cam_traj[:t+1,1], cam_traj[:t+1,2], '-', linewidth=2, alpha=0.9, label='cam')
            ax_yz.plot(root_traj[:t+1,1], root_traj[:t+1,2], '--', linewidth=2, alpha=0.9, label='root')
            ax_yz.scatter([cam_traj[t,1]], [cam_traj[t,2]], s=30)
            ax_yz.scatter([root_traj[t,1]], [root_traj[t,2]], s=30)
            ax_yz.set_title("YZ Plane Trajectory")
            ax_yz.set_xlabel("Y (m)"); ax_yz.set_ylabel("Z (m)")
            set2d_equal(ax_yz, X_all[:,1], X_all[:,2])

            # show legend just once (bottom-left)
            ax_xy.legend(loc='upper right', fontsize=8)

            fig.tight_layout()
            writer.grab_frame()

    print(f"[ok] wrote {out_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Render pred/gt + SMAL world-space skeleton + camera visuals to MP4")
    ap.add_argument("--data_root", type=str, required=True, help="Directory containing frame_num.pkl and *_rgb images")
    ap.add_argument("--smal_model", type=str, default=SMAL_MODEL_PATH, help="Path to SMAL model pkl")
    ap.add_argument("--out", type=str, default="out.mp4", help="Output MP4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", type=str, default=DEVICE)
    ap.add_argument("--checkpoint_id", type=int, default=85000, help="Checkpoint ID")
    ap.add_argument("--save_path", type=str, default="all_kin", help="Checkpoint ID")
    add_trans_scale_args(ap)
    args = ap.parse_args()
    trans_scale = resolve_trans_scale(args.trans_scale, args.median_scale_file)
    print(f"Using trans_scale = {trans_scale}")

    frames = args.out[:-4]

    make_video(
        data_root=args.data_root,
        smal_model_path=args.smal_model,
        frame_nums=frames,
        out_mp4=args.out,
        fps=args.fps,
        device=args.device,
        checkpoint_id=args.checkpoint_id,
        save_path=frames,
        trans_scale=trans_scale,
    )
