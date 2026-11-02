#!/usr/bin/env python3
"""
Run policy rollouts sequentially for each motion .pkl file (no SLURM).

Use this when you want to run rollouts locally, one sequence after another,
instead of submitting batch jobs via create_sim_batches.py (which is used for
training / distributed rollout).

Policy model can be specified in two ways:
  1. Per-motion from model_dir: model path = model_dir / pattern.format(stem=motion_stem)
  2. Single policy_model: same model for all motions

Usage (per-motion models):
    python run_rollouts_sequential.py \
        --motion_dir data/motions/horse_kin_from_params \
        --model_dir output/per_sequence_models \
        --base_args_file args/deepmimic_horse_ppo_base.txt \
        --out_dir output/rollouts_sequential

Usage (single model):
    python run_rollouts_sequential.py \
        --motion_dir data/motions/horse_kin_from_params \
        --policy_model output/model.pt \
        --base_args_file args/deepmimic_horse_ppo_base.txt \
        --out_dir output/rollouts_sequential
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def run_rollout(
    python_exe: str,
    mimickit_root: Path,
    base_args_file: str,
    motion_file: str,
    policy_model: str,
    rollout_dir: Path,
    log_dir: Path,
    test_episodes: int = 1,
) -> bool:
    """Run a single rollout via run.py subprocess. Returns True on success."""
    stem = Path(motion_file).stem
    save_states_path = rollout_dir / f"{stem}_states.npz"
    log_file = log_dir / f"{stem}.log"

    cmd = [
        python_exe,
        str(mimickit_root / "mimickit" / "run.py"),
        "--arg_file", base_args_file,
        "--mode", "test",
        "--motion_file", motion_file,
        "--model_file", policy_model,
        "--test_episodes", str(test_episodes),
        "--save_states_path", str(save_states_path),
        "--log_file", str(log_file),
    ]

    print(f"  Running: {motion_file} -> model: {policy_model}")
    result = subprocess.run(cmd, cwd=str(mimickit_root))
    return result.returncode == 0


def resolve_model_path(
    motion_file: str,
    model_dir: Path,
    model_pattern: str,
) -> Path:
    """Resolve model path from motion name using pattern. {stem} = motion filename without extension."""
    stem = Path(motion_file).stem
    filename = model_pattern.format(stem=stem)
    return model_dir / filename


def main():
    parser = argparse.ArgumentParser(
        description="Run policy rollouts sequentially for each motion .pkl (no SLURM)."
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        required=True,
        help="Directory containing motion .pkl files.",
    )
    parser.add_argument(
        "--base_args_file",
        type=str,
        required=True,
        help="Base args file for MimicKit run.py.",
    )
    parser.add_argument(
        "--policy_model",
        type=str,
        default="",
        help="Single trained policy .pt to use for all motions. Mutually exclusive with --model_dir.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Directory of per-motion model files. Model path = model_dir / pattern, where pattern uses {stem}=motion name.",
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        default="sim_{stem}.pt",
        help="Filename pattern for model lookup when using --model_dir. Use {stem} for motion name. Default: {stem}.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/rollouts_sequential",
        help="Output directory for rollout states and logs.",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=1,
        help="Number of test episodes per motion (1 recommended for per-sequence evaluation).",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )

    args = parser.parse_args()

    if bool(args.policy_model) == bool(args.model_dir):
        parser.error("Exactly one of --policy_model or --model_dir must be specified.")
    use_model_dir = bool(args.model_dir)

    motion_dir = Path(args.motion_dir)
    out_dir = Path(args.out_dir)
    rollout_dir = out_dir / "rollouts"
    log_dir = out_dir / "logs"

    rollout_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    motion_files = sorted(glob.glob(str(motion_dir / "*.pkl")))
    if not motion_files:
        print(f"No .pkl motion files found under {motion_dir}")
        sys.exit(1)

    mimickit_root = Path(__file__).resolve().parent
    base_args = str(Path(args.base_args_file).resolve())
    model_dir = Path(args.model_dir).resolve() if use_model_dir else None

    print(f"Running {len(motion_files)} rollouts sequentially...")
    print(f"  Motion dir: {motion_dir}")
    if use_model_dir:
        print(f"  Model dir: {model_dir} (pattern: {args.model_pattern})")
    else:
        print(f"  Policy: {args.policy_model}")
    print(f"  Output: {out_dir}")
    print()

    success_count = 0
    attempted_count = 0
    for i, motion_file in enumerate(motion_files, start=1):
        stem = Path(motion_file).stem
        if use_model_dir:
            model_path = resolve_model_path(motion_file, model_dir, args.model_pattern)
            if not model_path.exists():
                print(f"[{i}/{len(motion_files)}] {stem} SKIP: model not found at {model_path}")
                continue
            policy_model = str(model_path)
        else:
            policy_model = args.policy_model

        attempted_count += 1
        print(f"[{i}/{len(motion_files)}] {stem}")
        ok = run_rollout(
            python_exe=args.python_exe,
            mimickit_root=mimickit_root,
            base_args_file=base_args,
            motion_file=motion_file,
            policy_model=policy_model,
            rollout_dir=rollout_dir,
            log_dir=log_dir,
            test_episodes=args.test_episodes,
        )
        if ok:
            success_count += 1
        else:
            print(f"  FAILED (exit code != 0)")
        print()

    print(f"Done. {success_count}/{attempted_count} rollouts succeeded.")
    print(f"States saved to: {rollout_dir}")
    print(f"Logs saved to: {log_dir}")
    sys.exit(0 if success_count == attempted_count else 1)


if __name__ == "__main__":
    main()
