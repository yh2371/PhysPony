#!/usr/bin/env python3
import os
import sys
import csv
import glob
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
from torchmin import minimize

# ----------------------------
# Scale estimation utilities
# ----------------------------

def gmof(x, sigma=100):
    """Geman–McClure robust error."""
    x2 = x ** 2
    s2 = sigma ** 2
    return (s2 * x2) / (s2 + x2)

def est_scale_iterative(slam_depth, pred_depth, iters=10, msk=None, far_thresh=10.0):
    """Depth-align by iterative median and thresholding (scale * slam ≈ pred)."""
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # initial robust set
    robust = (msk < 0.5) & (pred_depth > 0) & (pred_depth < far_thresh) & (slam_depth > 0)
    s = pred_depth / np.maximum(slam_depth, 1e-9)
    if not np.any(robust):
        return 1.0
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(iters):
        slam_scaled = slam_depth * scale
        robust = (msk < 0.5) & (slam_scaled > 0) & (slam_scaled < far_thresh) & (pred_depth > 0) & (pred_depth < far_thresh)
        if not np.any(robust):
            break
        s_est = s[robust]
        scale = np.median(s_est)

    return float(scale)

def est_scale_gmof(slam_depth, pred_depth, lr=1e-1, sigma=0.5, iters=300, msk=None, far_thresh=10.0):
    """Depth-align by robust least squares (scale * slam ≈ pred)."""
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk < 0.5) & (pred_depth > 0) & (pred_depth < far_thresh) & (slam_depth > 0)
    if not np.any(robust):
        return 1.0

    pm = torch.from_numpy(pred_depth[robust]).float()
    sm = torch.from_numpy(slam_depth[robust]).float()

    scale = torch.tensor([1.0], requires_grad=True)
    optim = torch.optim.Adam([scale], lr=lr)

    for _ in range(iters):
        loss = sm * scale - pm
        loss = gmof(loss, sigma=sigma).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    return float(scale.detach().cpu().item())

def est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=None, far_thresh=10.0):
    """Iterative median warmup + BFGS robust refinement (scale * slam ≈ pred)."""
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: iterative median
    s = pred_depth / np.maximum(slam_depth, 1e-9)
    robust = (msk < 0.5) & (pred_depth > 0) & (pred_depth < far_thresh) & (slam_depth > 0)
    if not np.any(robust):
        return 1.0
    scale = np.median(s[robust])

    for _ in range(10):
        slam_scaled = slam_depth * scale
        robust = (msk < 0.5) & (slam_scaled > 0) & (slam_scaled < far_thresh) & (pred_depth > 0) & (pred_depth < far_thresh)
        if not np.any(robust):
            break
        s_est = s[robust]
        scale = np.median(s_est)

    # Stage 2: BFGS on GMOF
    slam_scaled = slam_depth * scale  # for mask reuse
    robust = (msk < 0.5) & (slam_scaled > 0) & (slam_scaled < far_thresh) & (pred_depth > 0) & (pred_depth < far_thresh)
    if not np.any(robust):
        return float(scale)

    pm = torch.from_numpy(pred_depth[robust]).float()
    sm = torch.from_numpy(slam_depth[robust]).float()

    def f(x):
        # x is tensor([scale])
        loss = sm * x[0] - pm
        return gmof(loss, sigma=sigma).mean()

    x0 = torch.tensor([float(scale)], dtype=torch.float32)
    result = minimize(f, x0, method='bfgs')
    return float(result.x.detach().cpu().item())

# Optional: shift/scale-shift helpers if you later want them
def scale_shift_align(smpl_depth, pred_depth, sigma=0.5):
    smpl = torch.from_numpy(smpl_depth).float()
    pred = torch.from_numpy(pred_depth).float()
    def f(x):
        loss = smpl - (pred * x[0] + x[1])
        return gmof(loss, sigma=sigma).mean()
    x0 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    result = minimize(f, x0, method='bfgs')
    return result.x.detach().cpu().numpy()

def shift_align(smpl_depth, pred_depth, sigma=0.5):
    smpl = torch.from_numpy(smpl_depth).float()
    pred = torch.from_numpy(pred_depth).float()
    def f(x):
        loss = smpl - (pred + x[0])
        return gmof(loss, sigma=sigma).mean()
    x0 = torch.tensor([0.0], dtype=torch.float32)
    result = minimize(f, x0, method='bfgs')
    return result.x.detach().cpu().numpy()

# ----------------------------
# I/O utilities
# ----------------------------

def load_optional_mask(mask_path: Path) -> Optional[np.ndarray]:
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        return m
    return None

def safe_load_depth(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = np.load(str(path))
        if arr is None:
            return None
        arr = np.array(arr, dtype=np.float32)
        # clean nans/infs
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception as e:
        print(f"[WARN] Failed to load depth {path}: {e}")
        return None

# ----------------------------
# Main processing
# ----------------------------

def compute_scale_for_frame(
    slam_path: Path,
    pred_path: Path,
    mask_path: Optional[Path],
    method: str = "hybrid",
    far_thresh: float = 10.0,
    sigma: float = 0.5,
) -> Optional[float]:
    slam = safe_load_depth(slam_path)
    pred = safe_load_depth(pred_path)
    if slam is None or pred is None:
        return None
    slam = slam[0, 0]

    # Resize slam to pred if shapes mismatch (keep your convention consistent)
    if slam.shape != pred.shape:
        slam = cv2.resize(slam, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    if len(pred.shape) == 3:
        pred = pred[:,:,0]

    msk = load_optional_mask(mask_path) if mask_path else None

    if method == "iter":
        scale = est_scale_iterative(slam, pred, iters=10, msk=msk, far_thresh=far_thresh)
    elif method == "gmof":
        scale = est_scale_gmof(slam, pred, lr=1e-1, sigma=sigma, iters=300, msk=msk, far_thresh=far_thresh)
    elif method == "hybrid":
        scale = est_scale_hybrid(slam, pred, sigma=sigma, msk=msk, far_thresh=far_thresh)
    else:
        raise ValueError(f"Unknown method: {method}")

    return scale

def process_sequence_dir(
    seq_dir: Path,
    slam_suffix: str = "_metric_depth.npy",
    pred_suffix: str = "_vggt_depth.npy",
    rotation_suffix: str = "_rotation.npy",
    mask_suffix: Optional[str] = "_mask.png",
    method: str = "hybrid",
    far_thresh: float = 10.0,
    sigma: float = 0.5,
    save_aligned: bool = False,
    align_target: str = "pred",  # "pred" means find scale so (scale*slam) ~ pred; alternatively "slam" for (scale*pred) ~ slam
) -> None:
    """
    Process a single sequence directory:
    - Finds all pred depth files
    - Pairs them with slam depth and optional mask
    - Computes per-frame scale
    - Writes scales.csv and sequence_scale.txt
    - Optionally saves aligned depth as *_pred_aligned.npy or *_slam_aligned.npy
    """
    pred_files = sorted(seq_dir.glob(f"*{pred_suffix}"))
    if not pred_files:
        return

    rows = [("frame_stem", "scale")]
    scales = []
    
    for pred_path in pred_files:
        stem = pred_path.name.replace(pred_suffix, "")

        slam_path = seq_dir / f"{stem}{slam_suffix}"

        if mask_suffix:
            mask_path = seq_dir / f"{stem}{mask_suffix}"
        else:
            mask_path = None

        if not slam_path.exists():
            print(f"[WARN] Missing slam depth for {pred_path.name}, expected {slam_path.name}")
            continue

        scale = compute_scale_for_frame(
            slam_path, pred_path, mask_path,
            method=method, far_thresh=far_thresh, sigma=sigma
        )
        if scale is None or not np.isfinite(scale):
            print(f"[WARN] Scale failed for {pred_path.name}")
            continue

        rows.append((stem, f"{scale:.8f}"))
        scales.append(scale)

        if save_aligned:
            slam = safe_load_depth(slam_path)
            pred = safe_load_depth(pred_path)
            if slam is None or pred is None:
                continue
            # resize slam to pred if needed
            if slam.shape != pred.shape:
                slam = cv2.resize(slam, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

            if align_target == "pred":
                # scale * slam ≈ pred  -> produce pred_aligned from slam
                aligned = slam * scale
                out_path = seq_dir / f"{stem}_pred_aligned.npy"
            else:
                # scale * pred ≈ slam  -> produce slam_aligned from pred
                aligned = pred * scale
                out_path = seq_dir / f"{stem}_slam_aligned.npy"

            np.save(out_path, aligned.astype(np.float32))

    if len(scales) == 0:
        return

    # Write per-sequence median scale
    median_scale = float(np.median(np.array(scales)))

    for pred_path in pred_files:
        stem = pred_path.name.replace(pred_suffix, "")
        rotation_path = seq_dir / f"{stem}{rotation_suffix}"
        scaled_rotation_path = seq_dir / f"{stem}_scaled_rotation.npy"
        try:
            transform = np.load(rotation_path)
        except:
            print("missing", rotation_path)
            continue
        # Scale only the translation into metric units.
        scaled_transform = transform.copy()
        scaled_transform[:3, 3] = scaled_transform[:3, 3] * median_scale
        np.save(scaled_rotation_path, scaled_transform)

    (seq_dir / "median_scale.txt").write_text(f"{median_scale:.8f}\n")
    print(f"[OK] {seq_dir}: {len(scales)} frames, median scale = {median_scale:.6f}")

def find_sequence_dirs(root: Path) -> List[Path]:
    """
    Define 'sequence dir' as any directory that contains at least one *_pred_depth.npy.
    Modify this if you prefer to key off *_rgb.jpg or other patterns.
    """
    seq_dirs = set()
    for p in root.rglob("*_vggt_depth.npy"):
        seq_dirs.add(p.parent)
    return sorted(seq_dirs)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch scale alignment across all sequences.")
    parser.add_argument("root_dir", type=str, help="Root directory containing sequence subfolders.")
    parser.add_argument("--slam_suffix", type=str, default="_metric_depth.npy",
                        help="Suffix for metric (reference) depth, e.g., UniDepthV2 outputs.")
    parser.add_argument("--pred_suffix", type=str, default="_vggt_depth.npy",
                        help="Suffix for predicted up-to-scale depth, e.g., VGGT outputs.")
    parser.add_argument("--mask_suffix", type=str, default="_mask.png",
                        help="Optional mask suffix (set to '' to disable).")
    parser.add_argument("--method", type=str, default="hybrid", choices=["iter", "gmof", "hybrid"],
                        help="Scale estimation method.")
    parser.add_argument("--far_thresh", type=float, default=10.0, help="Max depth to consider (meters).")
    parser.add_argument("--sigma", type=float, default=0.5, help="GMOF sigma.")
    parser.add_argument("--save_aligned", action="store_true",
                        help="Save aligned depth arrays per-frame.")
    parser.add_argument("--align_target", type=str, default="pred", choices=["pred", "slam"],
                        help="pred: scale*slam≈pred -> save *_pred_aligned.npy; slam: scale*pred≈slam -> save *_slam_aligned.npy.")
    args = parser.parse_args()

    root = Path(args.root_dir)
    mask_suffix = None if (args.mask_suffix is None or args.mask_suffix.strip() == "") else args.mask_suffix

    seq_dirs = find_sequence_dirs(root)
    if not seq_dirs:
        print("[INFO] No sequence directories found (no files matching *_pred_depth.npy).")
        sys.exit(0)

    for seq in seq_dirs:
        process_sequence_dir(
            seq_dir=seq,
            slam_suffix=args.slam_suffix,
            pred_suffix=args.pred_suffix,
            mask_suffix=mask_suffix,
            method=args.method,
            far_thresh=args.far_thresh,
            sigma=args.sigma,
            save_aligned=args.save_aligned,
            align_target=args.align_target,
        )

if __name__ == "__main__":
    main()
