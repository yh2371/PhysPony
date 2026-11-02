#!/usr/bin/env python3
import os
import glob
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

from unidepth.models import UniDepthV2


def run_unidepth_on_folder(root_dir: str, device: str = "cuda:0"):
    """
    Runs UniDepthV2 on all *_rgb.jpg images in subdirectories of root_dir.
    Saves both *_metric_depth.png (16-bit visualization) and *_metric_depth.npy (raw values).
    """
    # Load UniDepthV2 model (metric depth)
    type_ = "l"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level (only V2)

    # set interpolation mode (only V2)
    model.interpolation_mode = "bilinear"

    device = torch.device(device)
    model = model.to(device).eval()
    # Collect all images
    image_paths = glob.glob(os.path.join(root_dir, "**", "*_rgb.jpg"), recursive=True)
    print(f"Found {len(image_paths)} images.")
    from tqdm import tqdm

    for img_path in tqdm(image_paths):
        out_png = img_path.replace("_rgb.jpg", "_metric_depth.png")
        out_npy = img_path.replace("_rgb.jpg", "_metric_depth.npy")
        if os.path.exists(out_png) and os.path.exists(out_npy):
            print(f"Skipping {img_path}, already processed.")
            continue

        # Load the RGB image and the normalization will be taken care of by the model
        rgb = torch.from_numpy(np.array(Image.open(img_path))).permute(2, 0, 1) # C, H, W

        predictions = model.infer(rgb)

        # Metric Depth Estimation
        depth = predictions["depth"].cpu().numpy()
        np.save(out_npy, depth)

        # Normalize for visualization as 16-bit PNG
        dmin, dmax = np.min(depth), np.max(depth)
        if dmax - dmin > 1e-6:
            depth_vis = (65535 * (depth - dmin) / (dmax - dmin)).astype(np.uint16)
        else:
            depth_vis = np.zeros_like(depth, dtype=np.uint16)

        cv2.imwrite(out_png, depth_vis)

        print(f"Saved {out_png} and {out_npy}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="Path to root directory containing *_rgb.jpg images")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    run_unidepth_on_folder(args.root_dir, args.device)
