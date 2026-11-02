#!/usr/bin/env python3
"""
Build the per-sequence inputs that stage 4 (physics refinement) consumes,
straight from the kinematic model's test results:

    <out_dir>/motions/<seq>.pkl         MimicKit reference motion
    <out_dir>/envs/env_<seq>.yaml       test env  (full clip, fixed start)
    <out_dir>/envs_rsi/env_<seq>.yaml   train env (random start, <= 10 s episodes)
    <out_dir>/scales.json               per-sequence metric scale that was used

Inputs:
    --params_dir  flat directory of `<seq>_<frame>__params.pkl` written by
                  `scripts/stage3_ponymation.sh test`
    --data_dir    the stage 1/2 data directory, one sub-directory per sequence.
                  The metric scale is read from `<seq>/median_scale.txt`
                  (written by perception/unidepth/align_depth.py); if that is
                  missing it is recovered as the per-frame median of
                  ||scaled_rotation[:3, 3]|| / ||rotation[:3, 3]||, which
                  reproduces median_scale.txt exactly where both exist.
    --scales_json optional {seq: scale} that takes precedence over --data_dir.

The env yamls are the tracked templates
data/envs/deepmimic_horse_trajectory_env_long_mocap{,_rsi}.yaml with only
`motion_file` and `episode_length` substituted: the test episode covers the
whole clip, the RSI (training) episode is capped at --rsi_max_seconds.

Usage (env: physpony-sim, from anywhere):
    python sim/prepare_refinement_inputs.py \
        --params_dir <stage3 test_results dir> --data_dir <stage1/2 data dir> \
        --out_dir outputs/refine [--seqs 0000091 0000058] [--decimate 4]

`--decimate 4` is for 120 Hz PFERD captures; in-the-wild video is 30 fps.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from process_kinematic_results import (
    build_ref_motion_from_files,
    ref_motion_to_mimickit_frames,
    save_motion_pkl,
)

SIM_DIR = Path(__file__).resolve().parent
TEST_TEMPLATE = SIM_DIR / "data/envs/deepmimic_horse_trajectory_env_long_mocap.yaml"
RSI_TEMPLATE = SIM_DIR / "data/envs/deepmimic_horse_trajectory_env_long_mocap_rsi.yaml"
PARAMS_SUFFIX = "__params.pkl"


def find_sequences(params_dir: Path) -> Dict[str, List[Path]]:
    """`<seq>_<frame>__params.pkl` -> {seq: [files sorted by frame]}."""
    seqs: Dict[str, List[tuple]] = {}
    for p in params_dir.iterdir():
        if not p.name.endswith(PARAMS_SUFFIX):
            continue
        m = re.match(r"(.+)_(\d+)$", p.name[:-len(PARAMS_SUFFIX)])
        if m:
            seqs.setdefault(m.group(1), []).append((int(m.group(2)), p))
    return {k: [p for _, p in sorted(v)] for k, v in seqs.items()}


def resolve_scale(seq: str, data_dir: Optional[Path], scales: Dict[str, float]) -> float:
    if seq in scales:
        return float(scales[seq])
    if data_dir is None:
        raise FileNotFoundError(f"{seq}: no scale in --scales_json and no --data_dir")
    seq_dir = data_dir / seq
    txt = seq_dir / "median_scale.txt"
    if txt.is_file():
        return float(txt.read_text().strip())
    ratios = []
    for scaled in sorted(seq_dir.glob("*_scaled_rotation.npy")):
        raw = seq_dir / scaled.name.replace("_scaled_rotation.npy", "_rotation.npy")
        if not raw.is_file():
            continue
        t_raw = np.linalg.norm(np.load(raw)[:3, 3])
        if t_raw > 1e-8:
            ratios.append(np.linalg.norm(np.load(scaled)[:3, 3]) / t_raw)
    if not ratios:
        raise FileNotFoundError(
            f"{seq}: no median_scale.txt or *_scaled_rotation.npy under {seq_dir} "
            "(run scripts/stage2_metric_depth_align.sh first)")
    return float(np.median(ratios))


def write_env(template: Path, out: Path, motion_file: Path, episode_length: float, note: str) -> None:
    text = template.read_text()
    text, n_len = re.subn(r"(?m)^(\s*)episode_length:.*$",
                          lambda m: f"{m.group(1)}episode_length: {episode_length} # {note}", text)
    text, n_mot = re.subn(r"(?m)^(\s*)motion_file:.*$",
                          lambda m: f'{m.group(1)}motion_file: "{motion_file}"', text)
    if n_len != 1 or n_mot != 1:
        raise ValueError(f"{template}: expected one episode_length and one motion_file line")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--scales_json", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seqs", nargs="*", default=None, help="default: every sequence in --params_dir")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--decimate", type=int, default=1, help="keep every Nth frame (4 for 120 Hz PFERD)")
    ap.add_argument("--rsi_max_seconds", type=float, default=10.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out_dir.resolve()
    scales = json.loads(args.scales_json.read_text()) if args.scales_json else {}
    found = find_sequences(args.params_dir)
    seqs = args.seqs or sorted(found)
    used_scales_path = out_dir / "scales.json"
    used_scales = json.loads(used_scales_path.read_text()) if used_scales_path.is_file() else {}

    failed = []
    for seq in seqs:
        motion = out_dir / "motions" / f"{seq}.pkl"
        test_env = out_dir / "envs" / f"env_{seq}.yaml"
        rsi_env = out_dir / "envs_rsi" / f"env_{seq}.yaml"
        if not args.overwrite and motion.is_file() and test_env.is_file() and rsi_env.is_file():
            print(f"[prepare] {seq}: exists, skipped (--overwrite to rebuild)")
            continue
        if seq not in found:
            print(f"[prepare] {seq}: no *{PARAMS_SUFFIX} in {args.params_dir}")
            failed.append(seq)
            continue
        try:
            scale = resolve_scale(seq, args.data_dir, scales)
        except FileNotFoundError as e:
            print(f"[prepare] {e}")
            failed.append(seq)
            continue

        frames = ref_motion_to_mimickit_frames(build_ref_motion_from_files(found[seq], trans_scale=scale))
        frames = frames[::args.decimate]
        save_motion_pkl(motion, frames, fps=args.fps, loop_mode=0)

        n = frames.shape[0]
        clip_s = round(n / args.fps, 2)
        write_env(TEST_TEMPLATE, test_env, motion, clip_s, f"seconds ({n} frames @ {args.fps:g} fps)")
        write_env(RSI_TEMPLATE, rsi_env, motion, min(args.rsi_max_seconds, clip_s),
                  f"seconds (clip is {clip_s:.2f}s)")
        used_scales[seq] = scale
        print(f"[prepare] {seq}: {len(found[seq])} params -> {n} frames, scale {scale:.6f}")

    used_scales_path.parent.mkdir(parents=True, exist_ok=True)
    used_scales_path.write_text(json.dumps(used_scales, indent=1) + "\n")
    if failed:
        raise SystemExit(f"[prepare] failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
