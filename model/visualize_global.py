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

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


# --- Import your SMAL implementation ---
# Adjust this import/path if needed for your repo layout

# ---------- User-config ----------
# Path pattern helpers. Edit these to match your files.
prefix = "20201128_ID_1_0004_23415_1500_0" #"0085000_      "
PKL_PATTERN = prefix + "{frame_num}__params.pkl"                 # e.g., "12.pkl"
PRED_IMG_PATTERN = prefix + "{frame_num}__image_pred.png"   # e.g., "12_pred_rgb.jpg"
GT_IMG_PATTERN   = prefix + "{frame_num}__image_gt.png"     # e.g., "12_gt_rgb.jpg"

# If you have a known joint connectivity (parent array), you can hardcode it here.
# Otherwise we will try to read smal.kintree_table.
SMAL_MODEL_PATH = os.environ.get(
    "PHYSPONY_SMAL_PKL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "model", "geometry", "SMAL", "my_smpl_0000_horse_new_skeleton_horse.pkl"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DPI = 120
FIG_W = 16
FIG_H = 6

# ---------- SMAL wrapper ----------
class SMALWrapper(nn.Module):
    """
    Thin wrapper that:
      - loads SMAL model file,
      - returns joints (and/or verts) in CAMERA space given (betas, pose, trans).
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        # Lazy import here so the script doesn't crash if the path is wrong at import time
        from model.geometry.SMAL.smal_torch.smal_torch import SMAL  # type: ignore
        self.smal = SMAL(model_path, device=self.device)
        # Try to read a parent table if available (like SMPL's kintree_table)
        self.parents = None
        for attr in ["kintree_table", "parents", "parent"]:
            if hasattr(self.smal, attr):
                tbl = getattr(self.smal, attr)
                try:
                    arr = np.array(tbl).squeeze()
                    if arr.ndim == 2 and arr.shape[0] == 2:
                        # SMPL-style 2xN
                        arr = arr[0]
                    self.parents = arr.tolist()
                except Exception:
                    pass
                break

    @torch.no_grad()
    def forward(self, betas: torch.Tensor, pose: torch.Tensor, trans: torch.Tensor):
        """
        betas: [..., nb]  (e.g., 9)
        pose:  [..., 108] axis-angle for 36 joints (3*36)
        trans: [..., 3]   root translation in camera coords
        returns:
          verts_cam: [N, V, 3]
          joints_cam: [N, J, 3]
        """
        # SMAL expects: beta, theta, trans; returns verts, joints(?), faces/aux(?)
        outs = self.smal(beta=betas.to(self.device), theta=pose.to(self.device), trans=trans.to(self.device))
        if isinstance(outs, (tuple, list)):
            verts = outs[0]
            joints = outs[1] if len(outs) > 1 else None
        else:
            # some impls return object with attributes
            verts = outs.verts if hasattr(outs, "verts") else outs[0]
            joints = outs.joints if hasattr(outs, "joints") else None

        if joints is None:
            # Fallback: try to compute "joints" as a subset (this is a last resort)
            raise RuntimeError(
                "SMAL forward didn't return joints. Please expose joints or a joint regressor from your SMAL model."
            )
        return verts, joints

# ---------- IO helpers ----------
def load_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # Expect keys: rotation (4x4), betas, pose, trans
    M = data["rotation"]  # ndarray or torch, shape (4,4) or (N,4,4)
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    if M.ndim == 3:
        M = M[0]  # in case a batch dimension sneaks in
    assert M.shape == (4, 4), f"rotation in {pkl_path} must be 4x4, got {M.shape}"
    R_c2w =  M[:3, :3].astype(np.float64)
    T_c2w =  M[:3, 3].astype(np.float64)

    betas = data["betas"]
    pose  = data["pose"]
    trans = data["trans"]
    print(data.keys())
    # Convert to torch 2D batch (N=1)
    betas = torch.as_tensor(betas, dtype=torch.float32).view(1, -1)
    pose  = torch.as_tensor(pose,  dtype=torch.float32).view(36, -1)
    pose  = rot6d_to_rotmat(pose).view(1, 36, 3, 3)
    trans = torch.as_tensor(trans, dtype=torch.float32).view(1, 3)
    print(trans)
    return R_c2w, T_c2w, betas, pose, trans

def load_img(path: Path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

# ---------- Geometry helpers ----------
def cam_to_world(points_cam: torch.Tensor, R_c2w: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    """
    points_cam: [N,J,3] on CUDA/CPU torch
    R_c2w: (3,3) numpy, T_c2w: (3,) numpy
    returns: [J,3] numpy in world
    """
    P = points_cam[0].detach().cpu().numpy()  # [J,3]
    Pw = (R_c2w @ P.T).T + T_c2w[None, :]
    return Pw

def get_bones_from_parents(parents: Optional[List[int]], J: int) -> List[Tuple[int, int]]:
    """
    Build bone edges from a parent list: edge from (i, parent[i]) for i>0 and parent>=0
    """
    edges = []
    if parents is not None:
        for i in range(J):
            p = parents[i] if i < len(parents) else -1
            if p >= 0 and p < J and p != i:
                edges.append((i, p))
    else:
        # Fallback: chain
        edges = [(i, i - 1) for i in range(1, J)]
    return edges

# ---------- Visualization ----------
def set_axes_equal(ax, X):
    """
    Set 3D plot axes to equal scale based on points X [N,3].
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    max_range = (np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()) * 0.5
    mid = np.array([x.mean(), y.mean(), z.mean()])
    if max_range < 1e-6:
        max_range = 0.1
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

def draw_skeleton_world(ax, joints_w: np.ndarray, bones: List[Tuple[int,int]]):
    ax.cla()
    # plot bones
    for (i, j) in bones:
        p, q = joints_w[i], joints_w[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=2)
    # plot joints
    ax.scatter(joints_w[:, 0], joints_w[:, 1], joints_w[:, 2], s=10)
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    set_axes_equal(ax, joints_w)
    ax.set_title("SMAL skeleton (world)")

# ---------- Main routine ----------
def make_video(
    data_root: str,
    smal_model_path: str,
    frame_nums: List[int],
    out_mp4: str,
    fps: int = 15,
    device: str = DEVICE,
):
    """
    data_root: directory containing the pkl + images for the sequence
    smal_model_path: path to SMAL model .pkl
    frame_nums: list of frame numbers to visualize (e.g., range)
    out_mp4: output video path
    """
    data_root = Path(data_root)
    smal = SMALWrapper(smal_model_path, device=device).eval()

    # Preload everything to get consistent axis bounds
    all_joints_w = []

    cache = []
    all_info = []
    for fn in frame_nums:
        pkl_path  = data_root / PKL_PATTERN.format(frame_num=fn)
        pred_path = data_root / PRED_IMG_PATTERN.format(frame_num=fn)
        gt_path   = data_root / GT_IMG_PATTERN.format(frame_num=fn)

        if not pkl_path.exists():
            print(f"[warn] missing {pkl_path}, skipping")
            continue
        if not pred_path.exists() or not gt_path.exists():
            print(f"[warn] missing pred/gt image for frame {fn}, skipping")
            continue

        R_c2w, T_c2w, betas, pose, trans = load_pkl(pkl_path)
        all_info.append((R_c2w, T_c2w, betas, pose, trans))
        verts_cam, joints_cam = smal(betas, pose, trans)  # [1,V,3], [1,J,3]
        joints_w = cam_to_world(joints_cam, R_c2w, T_c2w) # [J,3]

        pred_img = load_img(pred_path)
        gt_img   = load_img(gt_path)

        cache.append((fn, pred_img, gt_img, joints_w))
        all_joints_w.append(joints_w)

    pickle.dump(all_info, open("all_kin_info.pkl", "wb"))
    print("saved all_kin_info.pkl")

    if not cache:
        raise RuntimeError("No frames found to render.")

    # Build bone edges
    J = cache[0][3].shape[0]
    bones = get_bones_from_parents(smal.parents, J)

    # Figure & writer
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt   = fig.add_subplot(gs[0, 1])
    ax_3d   = fig.add_subplot(gs[0, 2], projection='3d')

    writer = FFMpegWriter(fps=fps, bitrate=6000)
    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Global axis bounds (optional: use all joints to set a consistent cube)
    X_all = np.concatenate(all_joints_w, axis=0)
    global_bounds = (X_all.min(axis=0), X_all.max(axis=0))

    with writer.saving(fig, str(out_path), DPI):
        for fn, pred_img, gt_img, joints_w in cache:
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

            # Right: 3D skeleton
            draw_skeleton_world(ax_3d, joints_w, bones)
            # lock to global bounds for stability
            mn, mx = global_bounds
            max_range = (np.max(mx - mn) * 0.5) or 0.1
            mid = (mn + mx) / 2.0
            ax_3d.set_xlim(mid[0]-max_range, mid[0]+max_range)
            ax_3d.set_ylim(mid[1]-max_range, mid[1]+max_range)
            ax_3d.set_zlim(mid[2]-max_range, mid[2]+max_range)

            fig.tight_layout()
            writer.grab_frame()

    print(f"[ok] wrote {out_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Render pred/gt + SMAL world-space skeleton to MP4")
    ap.add_argument("--data_root", type=str, required=True, help="Directory containing frame_num.pkl and *_rgb.jpg")
    ap.add_argument("--smal_model", type=str, default=SMAL_MODEL_PATH, help="Path to SMAL model pkl")
    ap.add_argument("--start", type=int, required=True, help="Start frame number (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="End frame number (inclusive)")
    ap.add_argument("--out", type=str, default="out.mp4", help="Output MP4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", type=str, default=DEVICE)
    args = ap.parse_args()

    frames = list(range(args.start, args.end + 1))
    make_video(
        data_root=args.data_root,
        smal_model_path=args.smal_model,
        frame_nums=frames,
        out_mp4=args.out,
        fps=args.fps,
        device=args.device
    )
