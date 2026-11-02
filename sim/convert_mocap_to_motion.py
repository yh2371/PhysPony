#!/usr/bin/env python
"""
Convert a raw PFERD mocap capture (mimickit/mocap_full/<capture>.npy, 60 fps,
SMAL joint order) into a MimicKit motion .pkl.

Usage:
    python convert_mocap_to_motion.py --capture 20201129_ID_4_0006 \
        --out data/motions/horse_mocap_long/20201129_ID_4_0006.pkl
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_kinematic_results import ref_motion_to_mimickit_frames, save_motion_pkl


def load_mocap_ref_motion(mocap_path, target_fps=30, source_fps=60):
    data = np.load(mocap_path, allow_pickle=True).item()
    trans = np.asarray(data["trans"], dtype=np.float64)
    poses_root = np.asarray(data["poses_root"], dtype=np.float64)
    poses = np.asarray(data["poses_body"], dtype=np.float64)

    # Convert wxyz quaternions to the xyzw order scipy expects.
    if poses_root.shape[-1] == 4 and np.abs(poses_root[:, 0]).mean() > np.abs(poses_root[:, -1]).mean():
        w, x, y, z = poses_root[:, 0], poses_root[:, 1], poses_root[:, 2], poses_root[:, 3]
        poses_root = np.stack([x, y, z, w], axis=-1)
        pw, px, py, pz = poses[..., 0], poses[..., 1], poses[..., 2], poses[..., 3]
        poses = np.stack([px, py, pz, pw], axis=-1)

    stride = max(1, round(source_fps / target_fps))
    return {
        "poses_root": poses_root[::stride].astype(np.float32),
        "poses": poses[::stride].astype(np.float32),
        "trans": trans[::stride].astype(np.float32),
    }, stride


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="Capture name, e.g. 20201129_ID_4_0006 (matches mocap_full/<capture>.npy).")
    ap.add_argument("--mocap_root", default=None, help="Default: sim/mimickit/mocap_full relative to this script.")
    ap.add_argument("--out", required=True, help="Output motion .pkl path.")
    ap.add_argument("--target_fps", type=float, default=30.0)
    ap.add_argument("--source_fps", type=float, default=60.0, help="mocap_full is recorded at 60fps per the published PFERD dataset paper (Nature Scientific Data / PubMed 38750064); verified directly via gait-cycle autocorrelation (a canter segment's stride frequency is only physically plausible, ~2.3Hz, at 60fps - 120fps implies ~4.6Hz, impossible for a horse). An earlier internal doc's '120fps (halved from 240fps)' claim was not independently verified and appears to be wrong.")
    args = ap.parse_args()

    mocap_root = args.mocap_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mimickit", "mocap_full")
    mocap_path = os.path.join(mocap_root, f"{args.capture}.npy")

    ref_motion, stride = load_mocap_ref_motion(mocap_path, args.target_fps, args.source_fps)
    # apply_kin2smal_reindex=False: this input is raw mocap, not stage-3
    # kinematic-model output - see ref_motion_to_mimickit_frames docstring
    # for why the reindex must not be applied here (it scrambles the
    # right-hind leg / tail joints).
    frames = ref_motion_to_mimickit_frames(ref_motion, apply_kin2smal_reindex=False)
    save_motion_pkl(__import__("pathlib").Path(args.out), frames, fps=args.target_fps, loop_mode=0)
    print(f"{args.capture}: {ref_motion['trans'].shape[0]} frames @ {args.target_fps}fps "
          f"({ref_motion['trans'].shape[0]/args.target_fps:.1f}s, stride={stride}) -> {args.out}")


if __name__ == "__main__":
    main()
