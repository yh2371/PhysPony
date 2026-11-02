"""Materializes a PhysPony ImageDataset-loadable folder + keypoints.json from
Dessie's coco_pascal_nomask.jsons GT keypoint annotations.

Usage:
    python model/eval/prepare_pascal_eval_set.py \
        --jsons /path/to/coco_pascal_nomask.jsons \
        --images_dir /path/to/coco_pascal/images \
        --out_dir data/eval/pascal_kp
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pascal_kp_data import load_coco_pascal_nomask, conform_acsm16_to_lassie14


def write_box_txt(fpath, size=256):
    # frame_id crop_x0 crop_y0 crop_w crop_h full_w full_h <unused>
    with open(fpath, 'w') as f:
        f.write(f"0 0 0 {size} {size} {size} {size} 0\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsons', required=True)
    ap.add_argument('--images_dir', default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--img_size', type=int, default=256)
    args = ap.parse_args()

    entries = load_coco_pascal_nomask(args.jsons, images_dir=args.images_dir)
    print(f"Loaded {len(entries)} pascal-sourced entries from {args.jsons}")

    images_out = os.path.join(args.out_dir, 'images')
    os.makedirs(images_out, exist_ok=True)

    keypoints = {}
    written = 0
    for img_id, entry in entries.items():
        src_path = entry['image_path']
        if not os.path.isfile(src_path):
            print(f"  skip {img_id}: image not found at {src_path}")
            continue
        stem = os.path.splitext(entry['basename'])[0]

        img = Image.open(src_path).convert('RGB')
        if img.size != (args.img_size, args.img_size):
            img = img.resize((args.img_size, args.img_size), Image.BILINEAR)
        img.save(os.path.join(images_out, f"{stem}_rgb.jpg"), quality=95)

        mask = Image.new('RGB', (args.img_size, args.img_size), (255, 255, 255))
        mask.save(os.path.join(images_out, f"{stem}_mask.png"))

        np.save(os.path.join(images_out, f"{stem}_rotation.npy"), np.eye(4, dtype=np.float32))
        write_box_txt(os.path.join(images_out, f"{stem}_box.txt"), size=args.img_size)

        kp14 = conform_acsm16_to_lassie14(entry['kp16'])
        keypoints[stem] = {'kp': kp14}
        written += 1

    keypoints_path = os.path.join(args.out_dir, 'keypoints.json')
    with open(keypoints_path, 'w') as f:
        json.dump(keypoints, f)

    print(f"Wrote {written} images to {images_out}")
    print(f"Wrote keypoints for {len(keypoints)} images to {keypoints_path}")


if __name__ == '__main__':
    main()
