"""Computes KT-PCK@0.1 from a PhysPony test_results_* directory's *_params.pkl
files (needs only vertex_2d/vertex_vis -- no torch model import).

Usage:
    python model/eval/eval_kt_pck.py \
        --results_dir <checkpoint_dir>/test_results_<ckpt>_pascalkp \
        --keypoints data/eval/pascal_kp/keypoints.json \
        --out_json <results_dir>/kt_pck.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kp_transfer import kp_transfer_pck


def stem_from_pkl_path(pkl_path):
    base = os.path.basename(pkl_path)
    assert base.startswith('images_') and base.endswith('__params.pkl'), base
    return base[len('images_'):-len('__params.pkl')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--keypoints', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    with open(args.keypoints) as f:
        gt_keypoints = json.load(f)

    pkl_paths = sorted(glob.glob(os.path.join(args.results_dir, 'images_*__params.pkl')))
    print(f"Found {len(pkl_paths)} result pkls in {args.results_dir}")

    kps_list, verts_2d_list, verts_vis_list = [], [], []
    missing_gt, missing_vis = 0, 0
    for pkl_path in pkl_paths:
        stem = stem_from_pkl_path(pkl_path)
        if stem not in gt_keypoints:
            missing_gt += 1
            continue
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        if d.get('vertex_vis') is None:
            missing_vis += 1
            continue
        kps_list.append(torch.tensor(gt_keypoints[stem]['kp'], dtype=torch.float32))
        verts_2d_list.append(torch.as_tensor(d['vertex_2d'], dtype=torch.float32))
        verts_vis_list.append(torch.as_tensor(d['vertex_vis'], dtype=torch.float32))

    if missing_gt:
        print(f"  skipped {missing_gt} pkls with no matching GT keypoints")
    if missing_vis:
        print(f"  skipped {missing_vis} pkls with no vertex_vis")
    assert len(kps_list) >= 2, f"need >=2 images with GT+vis, got {len(kps_list)}"

    result = kp_transfer_pck(kps_list, verts_2d_list, verts_vis_list, img_size=args.img_size)
    print(f"num_images={result['num_images']} num_pairs={result['num_pairs']}")
    print(f"PCK@0.05 = {result['pck05']*100:.2f}")
    print(f"PCK@0.10 = {result['pck10']*100:.2f}")

    out_json = args.out_json or os.path.join(args.results_dir, 'kt_pck.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == '__main__':
    main()
