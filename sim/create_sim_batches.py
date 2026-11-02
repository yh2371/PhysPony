#!/usr/bin/env python3
"""
Create SLURM batch scripts to run MimicKit simulations for each motion .pkl file
without generating per-sequence env/args files.

Design:
- You provide:
    - A directory of motion files (e.g. data/motions/horse_kin_from_params).
    - A single base args file (e.g. args/deepmimic_horse_ppo_base.txt).
    - A trained policy model file (or pattern) to use for all runs.
- For each motion file, we call:

    python mimickit/run.py \
        --arg_file <BASE_ARGS> \
        --motion_file <MOTION_PKL> \
        --out_model_file <OUT_MODEL> \
        --log_file <LOG_FILE> \
        --mode test \
        --model_file <POLICY_MODEL>

  using MimicKit's ArgParser override semantics (CLI keys override arg_file keys).

- Motion files are grouped into batches of N (default 4) per SLURM script.
- Each SLURM script launches its N commands in parallel on a single GPU,
  similar to phys_001.sh.
"""

import os
import sys
import glob
from pathlib import Path
from typing import List


# --- portable defaults (override via environment) ---
SIM_DIR = os.path.dirname(os.path.abspath(__file__))                 # <repo>/sim
SIM_CONDA_ENV = os.environ.get(
    "PHYSPONY_SIM_CONDA",
    os.environ.get("CONDA_PREFIX", ""),  # the active env, normally physpony-sim
)
DEFAULT_PYTHON_EXE = os.environ.get(
    "PHYSPONY_SIM_PYTHON", os.path.join(SIM_CONDA_ENV, "bin", "python")
)


SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x.%j.txt
#SBATCH --error={log_dir}/%x.%j.txt
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END

# ----------------------------
# Environment setup
# ----------------------------
# Override the sim conda env via PHYSPONY_SIM_CONDA. Set WANDB_API_KEY /
# SSL_CERT_FILE in your shell if you need them.
export LD_LIBRARY_PATH={conda_env}/lib
export CUDA_HOME={conda_env}/
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ----------------------------
# Working directory & logs
# ----------------------------
cd {sim_dir}
mkdir -p {log_dir}

"""


def build_command(
    python_exe: str,
    base_args_file: str,
    motion_file: str,
    policy_model: str,
    out_model_dir: str,
    log_file: str,
    save_states_path: str,
    test_episodes: int,
) -> str:
    """
    Build a single run.py command line for one motion file.
    We rely on util.arg_parser.ArgParser semantics:
    - CLI args are parsed first and can override keys in the arg_file.
    """
    stem = Path(motion_file).stem
    out_model_file = os.path.join(out_model_dir, f"sim_{stem}.pt")

    cmd = (
        f"{python_exe} mimickit/run.py "
        f"--arg_file {base_args_file} "
        f"--mode test "
        f"--test_episodes {int(test_episodes)} "
        f"--motion_file {motion_file} "
        f"--model_file {policy_model} "
        f"--save_states_path {save_states_path} "
        f"--out_model_file {out_model_file} "
        f"--log_file {log_file}"
    )
    return cmd


def create_batch_scripts(
    motion_dir: Path,
    base_args_file: Path,
    policy_model: Path,
    python_exe: Path,
    out_root: Path,
    batch_size: int = 4,
    save_states_dir: Path = Path("output/rollouts"),
    test_episodes: int = 1,
) -> None:
    motion_files = sorted(glob.glob(str(motion_dir / "*.pkl")))
    if not motion_files:
        print(f"No .pkl motion files found under {motion_dir}")
        return

    print(f"Found {len(motion_files)} motion files in {motion_dir}")

    job_dir = out_root / "jobs"
    log_dir = out_root / "logs"
    out_model_dir = out_root / "sim_models"
    rollout_dir = out_root / "rollouts"

    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    batches: List[List[str]] = []
    for i in range(0, len(motion_files), batch_size):
        batches.append(motion_files[i : i + batch_size])

    print(f"Creating {len(batches)} SLURM job scripts with up to {batch_size} motions each.")

    for job_idx, batch in enumerate(batches, start=1):
        job_name = f"phys_sim_{job_idx:03d}"
        script_path = job_dir / f"job_{job_idx:03d}.sh"

        script_lines = [SLURM_HEADER.format(
            job_name=job_name, log_dir=log_dir,
            sim_dir=SIM_DIR, conda_env=SIM_CONDA_ENV,
        )]
        script_lines.append("COMMANDS=(")

        for k, motion_file in enumerate(batch, start=1):
            stem = Path(motion_file).stem
            log_file = log_dir / f"{stem}_job{job_idx:03d}.log"
            save_states_path = rollout_dir / f"{stem}_states.npz"
            cmd = build_command(
                python_exe=str(python_exe),
                base_args_file=str(base_args_file),
                motion_file=motion_file,
                policy_model=str(policy_model),
                out_model_dir=str(out_model_dir),
                log_file=str(log_file),
                save_states_path=str(save_states_path),
                test_episodes=test_episodes,
            )
            # store as a literal string inside array
            script_lines.append(f"  '{cmd}'")

        script_lines.append(")")
        script_lines.append("")
        script_lines.extend(
            [
                'echo "Launching ${#COMMANDS[@]} commands in parallel..."',
                "date",
                "",
                "pids=()",
                "cleanup() {",
                '  echo "Received termination signal. Killing child processes..."',
                '  for p in "${pids[@]}"; do',
                '    kill -TERM "$p" 2>/dev/null || true',
                "  done",
                "}",
                "trap cleanup SIGINT SIGTERM",
                "",
                "idx=1",
                'for cmd in "${COMMANDS[@]}"; do',
                '  echo "[Job ${idx}] $(date) Starting: $cmd"',
                f'  bash -lc "$cmd" &',
                "  pids+=($!)",
                "  idx=$((idx+1))",
                "done",
                "",
                "fail=0",
                'for i in "${!pids[@]}"; do',
                '  pid="${pids[$i]}"',
                '  wait "$pid" || fail=$((fail+1))',
                '  echo "[Job $((i+1))] finished with status $?"',
                "done",
                "",
                'echo "All jobs finished. Failures: $fail/${#pids[@]}"',
                "date",
                "exit $fail",
                "",
            ]
        )

        with open(script_path, "w") as f:
            f.write("\n".join(script_lines))
        os.chmod(script_path, 0o755)
        print(f"  Created {script_path}")

    # Create a helper submission script
    submit_path = job_dir / "submit_all_sim_jobs.sh"
    with open(submit_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write('echo "Submitting all simulation jobs..."\n')
        f.write("job_ids=()\n\n")
        for job_idx in range(1, len(batches) + 1):
            f.write(f"echo \"Submitting job {job_idx}/{len(batches)}...\"\n")
            f.write(f"job_id=$(sbatch job_{job_idx:03d}.sh | awk '{{print $4}}')\n")
            f.write("job_ids+=($job_id)\n")
            f.write(f"echo \"Job {job_idx} submitted with ID: $job_id\"\n\n")
        f.write('echo "All jobs submitted. IDs: ${job_ids[*]}"\n')
        f.write('echo "Monitor with: squeue -u $USER"\n')
    os.chmod(submit_path, 0o755)
    print(f"\nCreated submission helper: {submit_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create SLURM batches for MimicKit simulation over motion pkls.")
    parser.add_argument("--motion_dir", type=str, required=True, help="Directory containing motion .pkl files.")
    parser.add_argument("--base_args_file", type=str, required=True, help="Base args file for MimicKit run.py.")
    parser.add_argument("--policy_model", type=str, required=True, help="Trained policy model .pt to use for testing.")
    parser.add_argument(
        "--python_exe",
        type=str,
        default=DEFAULT_PYTHON_EXE,
        help="Python executable to use inside SLURM jobs (env PHYSPONY_SIM_PYTHON).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/simulation_batches",
        help="Root directory for generated job scripts, logs, and sim models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of motions per SLURM job (4 matches your desired setup).",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=1,
        help="Number of test episodes per motion (1 recommended for per-sequence evaluation).",
    )

    args = parser.parse_args()

    create_batch_scripts(
        motion_dir=Path(args.motion_dir),
        base_args_file=Path(args.base_args_file),
        policy_model=Path(args.policy_model),
        python_exe=Path(args.python_exe),
        out_root=Path(args.out_dir),
        batch_size=args.batch_size,
        test_episodes=args.test_episodes,
    )


if __name__ == "__main__":
    main()

