#!/usr/bin/env python3
"""
Visualize a single sequence:

- Top‑down root trajectories (x–y) for:
    * mocap (black)
    * sim (orange)
    * kinematic reference (blue, if available)
  with:
    * equal axis scaling
    * no legend
    * circle marker at start
    * square marker at end
    * RTE alignment (Procrustes without scale) applied first to match global rotation.
    * First frames overlapped at the origin afterwards.

- Also samples 6 RGB frames from the sequence image folder and saves them as a grid.

Assumes:
- Rollout states saved by TestStateCollector as <seq_name>_states.npz
- Mocap SMAL params + images layout compatible with mimickit/eval_all_sequences.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure we can import the evaluation helpers
THIS_DIR = Path(__file__).resolve().parent
MIMICKIT_DIR = THIS_DIR / "mimickit"
if str(MIMICKIT_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_DIR))

# Import necessary alignment functions alongside the gather function
from eval_all_sequences import (
    gather_sequence_data, 
    rigid_align_no_scale, 
    apply_rigid
)  # type: ignore


def plot_topdown_trajectories(
    seq_name: str,
    sim_root: np.ndarray,
    ref_root: np.ndarray,
    mocap_root: np.ndarray,
    out_path: Path,
) -> None:
    """
    Top‑down (x–y) trajectories with:
      - black: mocap
      - orange: sim
      - blue: kinematic
      - circle start, square end
      - equal axis, no legend.
    Trajectories are first RTE-aligned to the mocap root, and then 
    translated so their first frames overlap at the origin.
    """
    # 1. Determine common time range
    lengths = [sim_root.shape[0]]
    if ref_root is not None: lengths.append(ref_root.shape[0])
    if mocap_root is not None: lengths.append(mocap_root.shape[0])

    if not lengths:
        print(f"[warn] No trajectories to plot for {seq_name}")
        return
    T = min(lengths) - 17 
    if T == 0:
        print(f"[warn] Trajectories are empty for {seq_name}")
        return

    sim_t = sim_root[:T].astype(np.float64)
    ref_t = ref_root[:T].astype(np.float64) if ref_root is not None else None
    mocap_t = mocap_root[:T].astype(np.float64) if mocap_root is not None else None

    # 2. RTE Alignment (Procrustes without scale)
    if mocap_t is not None:
        # Align sim to mocap
        finite_s = np.isfinite(sim_t).all(1) & np.isfinite(mocap_t).all(1)
        if finite_s.sum() >= 3:
            R_s, t_s = rigid_align_no_scale(sim_t[finite_s], mocap_t[finite_s])
            sim_al = apply_rigid(sim_t, R_s, t_s)
        else:
            sim_al = sim_t

        # Align ref to mocap
        if ref_t is not None:
            finite_r = np.isfinite(ref_t).all(1) & np.isfinite(mocap_t).all(1)
            if finite_r.sum() >= 3:
                R_r, t_r = rigid_align_no_scale(ref_t[finite_r], mocap_t[finite_r])
                ref_al = apply_rigid(ref_t, R_r, t_r)
            else:
                ref_al = ref_t
        else:
            ref_al = None

        mocap_al = mocap_t
    else:
        # If no mocap, keep original
        sim_al = sim_t
        ref_al = ref_t
        mocap_al = None
    # 3. Overlap first frames (translate to origin)
    trajs = []
    if mocap_al is not None:
        mocap_final = mocap_al - mocap_al[0:1]
        mocap_final = mocap_final[:T]
        trajs.append(("mocap", mocap_final, "k"))

    sim_final = sim_al - sim_al[0:1]
    sim_final = sim_final[:T]
    trajs.append(("sim", sim_final, "orange"))

    if ref_al is not None:
        ref_final = ref_al - ref_al[0:1]
        ref_final = ref_final[:T]
        trajs.append(("kin", ref_final, "b"))

    # 4. Plotting
    xs, ys = [], []
    for name, traj, color in trajs:
        xs.append(traj[:, 0])
        ys.append(traj[:, 1])

    xs_all = np.concatenate(xs)
    ys_all = np.concatenate(ys)
    x_min, x_max = xs_all.min(), xs_all.max()
    y_min, y_max = ys_all.min(), ys_all.max()
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    r = max(x_max - x_min, y_max - y_min)
    if r < 1e-6:
        r = 1.0
    half = 0.5 * r * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))

    for name, traj, color in trajs:
        x = traj[:, 0]
        y = traj[:, 1]
        ax.plot(x, y, color=color, linewidth=2.0)
        # start (circle), end (square)
        ax.scatter(x[0], y[0], color=color, marker="o", s=40, zorder=3)
        ax.scatter(x[-1], y[-1], color=color, marker="s", s=40, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{seq_name} – top‑down root trajectories")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sample_rgb_frames(
    seq_name: str,
    images_dir: Path,
    out_path: Path,
    num_samples: int = 6,
) -> None:
    """Sample num_samples RGB frames from images_dir and save as a grid image."""
    import imageio.v2 as imageio
    print(images_dir, images_dir.exists())
    exts = {"_rgb.png", "_rgb.jpg"}
    img_files = sorted(f for f in images_dir.iterdir() if f.name[-8:] in exts)
    if not img_files:
        print(f"[warn] No images found in {images_dir}; skipping RGB grid.")
        return

    num = min(num_samples, len(img_files))
    idxs = np.linspace(0, len(img_files) - 17, num=num, dtype=int)
    chosen = [img_files[i] for i in idxs]

    # Load images
    imgs = [imageio.imread(p) for p in chosen]

    cols = num
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows),gridspec_kw={'wspace': 0.02})
    if cols == 1:
        axes = [axes]

    for ax, img, p in zip(axes, imgs, chosen):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.name, fontsize=8)

    fig.suptitle(f"{seq_name} – sampled RGB frames", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize top‑down trajectories and sampled RGB frames for a single sequence."
    )
    parser.add_argument("--seq_name", type=str, required=True, help="Sequence name (stem without _states).")
    parser.add_argument(
        "--rollout_dir",
        type=Path,
        required=True,
        help="Directory containing <seq_name>_states.npz.",
    )
    parser.add_argument(
        "--char_file",
        type=Path,
        required=True,
        help="Character XML (same as used for evaluation).",
    )
    parser.add_argument(
        "--mocap_root",
        type=Path,
        required=True,
        help="Folder with full‑sequence mocap .npy (SMAL params) compatible with eval_all_sequences.",
    )
    parser.add_argument(
        "--images_root",
        type=Path,
        required=True,
        help="Root folder with one subfolder per sequence containing RGB frames.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output/vis_seq"),
        help="Output directory for visualization images.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 30.0,
        help="Timestep in seconds (used only for consistency with gather_sequence_data).",
    )
    parser.add_argument(
        "--num_samples_rgb",
        type=int,
        default=6,
        help="Number of RGB frames to sample for the sequence.",
    )
    args = parser.parse_args()

    states_path = args.rollout_dir / f"{args.seq_name}_states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"Could not find states file: {states_path}")

    data = gather_sequence_data(
        seq_name=args.seq_name,
        states_path=states_path,
        char_file=args.char_file,
        dt=args.dt,
        mocap_root=args.mocap_root,
        images_root=args.images_root,
    )

    sim_root = data["sim_root"]
    ref_root = data["ref_root"]
    mocap_root_traj = data["mocap_root_traj"]

    # Top‑down trajectories
    topdown_path = args.out_dir / f"{args.seq_name}_topdown.png"
    plot_topdown_trajectories(
        seq_name=args.seq_name,
        sim_root=sim_root,
        ref_root=ref_root,
        mocap_root=mocap_root_traj,
        out_path=topdown_path,
    )

    # Sample RGB frames
    images_dir = args.images_root / args.seq_name
    rgb_grid_path = args.out_dir / f"{args.seq_name}_rgb_samples.png"
    sample_rgb_frames(
        seq_name=args.seq_name,
        images_dir=images_dir,
        out_path=rgb_grid_path,
        num_samples=args.num_samples_rgb,
    )

    print(f"Saved top‑down plot to {topdown_path}")
    print(f"Saved RGB grid to {rgb_grid_path}")


if __name__ == "__main__":
    main()