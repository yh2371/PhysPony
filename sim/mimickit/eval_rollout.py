#!/usr/bin/env python3
"""
Evaluate a trained DeepMimic policy by:
1) Running test rollouts.
2) Storing rollout states (via TestStateCollector, in SMAL-style rotation matrices).
3) Computing errors w.r.t. kinematic reference (from the env) and optional mocap GT.
4) Visualizing a 3D skeleton comparison (rollout vs reference vs mocap).
5) Optionally exporting OBJ files for the rollout skeleton sequence.

This script assumes:
- The environment is a DeepMimicEnv (or subclass) that the agent can build.
- The agent's BaseAgent.test_model has been extended to accept save_states_path
  and uses util.test_state_collector.TestStateCollector.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
from anim.kin_char_model import KinCharModel


def _compute_mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    """
    pred, target: (T, J, 3)
    """
    T = min(pred.shape[0], target.shape[0])
    pred = pred[:T]
    target = target[:T]
    err = np.linalg.norm(pred - target, axis=-1)  # (T,J)
    return float(err.mean())


def _load_states_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    # Tracked rollout
    poses_body = data["poses_body"]  # (T, J, 3, 3)
    body_pos = data["body_pos"]      # (T, B, 3)
    trans = data["trans"]            # (T, 3)
    poses_root = data["poses_root"]  # (T, 3, 3)

    # Reference (kinematic) if present
    ref_body_pos = data.get("ref_body_pos", None)
    ref_poses_body = data.get("ref_poses_body", None)
    ref_trans = data.get("ref_trans", None)
    ref_poses_root = data.get("ref_poses_root", None)

    return {
        "poses_body": poses_body,
        "body_pos": body_pos,
        "trans": trans,
        "poses_root": poses_root,
        "ref_body_pos": ref_body_pos,
        "ref_poses_body": ref_poses_body,
        "ref_trans": ref_trans,
        "ref_poses_root": ref_poses_root,
    }


def _load_mocap_joints(mocap_path: Path) -> Optional[np.ndarray]:
    """
    Load mocap GT joints if provided.
    Expects a .npy or .npz file with a dict-like payload containing one of:
      - 'joints3d'
      - 'joints'
      - 'gt_joints'
    with shape (T, J, 3).
    """
    if mocap_path is None or not mocap_path.exists():
        return None

    arr = np.load(mocap_path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        data = arr.item()
    else:
        data = arr

    for key in ["joints3d", "joints", "gt_joints"]:
        if key in data:
            joints = np.asarray(data[key], dtype=np.float32)
            if joints.ndim == 3 and joints.shape[-1] == 3:
                return joints
    # If nothing suitable found, return None
    return None


def _get_skeleton_edges(char_file: Path) -> Tuple[KinCharModel, np.ndarray]:
    """
    Build a KinCharModel from XML and return (model, edges_array[K,2]).
    """
    device = torch.device("cpu")
    model = KinCharModel(device)
    model.load_char_file(str(char_file))

    edges = []
    num_bodies = model.get_num_joints()
    for j in range(1, num_bodies):
        p = model.get_parent_id(j)
        if hasattr(p, "item"):
            p = p.item()
        edges.append((int(p), j))
    return model, np.asarray(edges, dtype=np.int64)


def _set_equal_3d(ax, pts: np.ndarray):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    span = np.maximum(mx - mn, 1e-6)
    half = float(span.max()) * 0.6
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect([1, 1, 1])


def visualize_skeletons(
    body_pos_rollout: np.ndarray,
    body_pos_ref: Optional[np.ndarray],
    body_pos_mocap: Optional[np.ndarray],
    edges: np.ndarray,
    out_mp4: Path,
    fps: float = 30.0,
):
    """
    Create a 3D animation comparing:
      - rollout skeleton (blue)
      - kinematic reference (orange), if available
      - mocap GT (green), if available
    Only uses body positions and skeleton edges.
    """
    T = body_pos_rollout.shape[0]
    if body_pos_ref is not None:
        T = min(T, body_pos_ref.shape[0])
    if body_pos_mocap is not None:
        T = min(T, body_pos_mocap.shape[0])

    body_pos_rollout = body_pos_rollout[:T]
    if body_pos_ref is not None:
        body_pos_ref = body_pos_ref[:T]
    if body_pos_mocap is not None:
        body_pos_mocap = body_pos_mocap[:T]

    all_pts = [body_pos_rollout.reshape(-1, 3)]
    if body_pos_ref is not None:
        all_pts.append(body_pos_ref.reshape(-1, 3))
    if body_pos_mocap is not None:
        all_pts.append(body_pos_mocap.reshape(-1, 3))
    all_pts = np.concatenate(all_pts, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    writer = FFMpegWriter(fps=fps, bitrate=6000)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out_mp4), dpi=120):
        for t in range(T):
            ax.cla()

            P_roll = body_pos_rollout[t]
            # rollout: blue
            for (i, j) in edges:
                pi, pj = P_roll[i], P_roll[j]
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]], color="tab:blue", linewidth=2)
            ax.scatter(P_roll[:, 0], P_roll[:, 1], P_roll[:, 2], color="tab:blue", s=8, label="rollout")

            # reference: orange
            if body_pos_ref is not None:
                P_ref = body_pos_ref[t]
                for (i, j) in edges:
                    pi, pj = P_ref[i], P_ref[j]
                    ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]], color="tab:orange", linewidth=1.5, alpha=0.8)
                ax.scatter(P_ref[:, 0], P_ref[:, 1], P_ref[:, 2], color="tab:orange", s=6, alpha=0.8, label="ref")

            # mocap: green
            if body_pos_mocap is not None:
                P_gt = body_pos_mocap[t]
                for (i, j) in edges:
                    pi, pj = P_gt[i], P_gt[j]
                    ax.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]], color="tab:green", linewidth=1.5, alpha=0.8)
                ax.scatter(P_gt[:, 0], P_gt[:, 1], P_gt[:, 2], color="tab:green", s=6, alpha=0.8, label="mocap")

            ax.set_title(f"Skeleton comparison – frame {t}")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            _set_equal_3d(ax, all_pts)
            ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            writer.grab_frame()

    print(f"[eval_rollout] Saved skeleton comparison video to {out_mp4}")


def export_obj_sequence(
    body_pos_rollout: np.ndarray,
    edges: np.ndarray,
    out_dir: Path,
    prefix: str = "rollout",
):
    """
    Export rollout skeleton as a sequence of OBJ files (one per frame).
    Each OBJ has vertices for all bodies and 'l' lines for bones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    T, B, _ = body_pos_rollout.shape
    for t in range(T):
        pts = body_pos_rollout[t]
        out_path = out_dir / f"{prefix}_{t:05d}.obj"
        with open(out_path, "w") as f:
            f.write("# rollout skeleton frame {}\n".format(t))
            for v in pts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for (i, j) in edges:
                # OBJ is 1-based
                f.write(f"l {i+1} {j+1}\n")
    print(f"[eval_rollout] Exported {T} OBJ frames to {out_dir}")


def run_rollout_and_analyze(
    env_config: Path,
    agent_config: Path,
    model_file: Path,
    device: str,
    num_envs: int,
    num_episodes: int,
    save_states_path: Path,
    char_file: Path,
    mocap_file: Optional[Path],
    out_dir: Path,
    export_obj: bool,
):
    # Build env and agent (single-process evaluation)
    visualize = False
    env = env_builder.build_env(str(env_config), num_envs, device, visualize)
    agent = agent_builder.build_agent(str(agent_config), env, device)

    if model_file is not None and model_file != Path(""):
        agent.load(str(model_file))

    # Run test and collect states
    print("[eval_rollout] Running test_model to collect rollout states...")
    agent.test_model(num_episodes=num_episodes, save_states_path=str(save_states_path))

    # Load collected states
    print(f"[eval_rollout] Loading collected states from {save_states_path}")
    states = _load_states_npz(save_states_path)
    body_pos_rollout = states["body_pos"]          # (T,B,3)
    ref_body_pos = states["ref_body_pos"]         # may be None

    # Load mocap joints if provided
    body_pos_mocap = _load_mocap_joints(mocap_file) if mocap_file is not None else None

    # Compute simple MPJPE metrics
    metrics = {}
    if ref_body_pos is not None:
        metrics["mpjpe_ref"] = _compute_mpjpe(body_pos_rollout, ref_body_pos)
    if body_pos_mocap is not None:
        metrics["mpjpe_mocap"] = _compute_mpjpe(body_pos_rollout, body_pos_mocap)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "rollout_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Rollout evaluation metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"[eval_rollout] Metrics written to {metrics_path}")

    # Build skeleton edges from character XML
    _, edges = _get_skeleton_edges(char_file)

    # Visualize skeletons
    vis_mp4 = out_dir / "skeleton_comparison.mp4"
    visualize_skeletons(
        body_pos_rollout=body_pos_rollout,
        body_pos_ref=ref_body_pos,
        body_pos_mocap=body_pos_mocap,
        edges=edges,
        out_mp4=vis_mp4,
        fps=30.0,
    )

    # Optional OBJ export
    if export_obj:
        obj_dir = out_dir / "objs_rollout"
        export_obj_sequence(body_pos_rollout, edges, obj_dir, prefix="rollout")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepMimic rollout against kinematic reference and mocap.")
    parser.add_argument("--env_config", type=Path, required=True, help="Path to env YAML.")
    parser.add_argument("--agent_config", type=Path, required=True, help="Path to agent YAML.")
    parser.add_argument("--model_file", type=Path, required=True, help="Trained model .pt file.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollout episodes.")
    parser.add_argument("--save_states", type=Path, default=Path("output/rollout_states.npz"))
    parser.add_argument("--char_file", type=Path, required=True, help="Character XML used in the env.")
    parser.add_argument("--mocap_file", type=Path, default=None, help="Optional mocap GT .npy/.npz file.")
    parser.add_argument("--out_dir", type=Path, default=Path("output/rollout_eval"))
    parser.add_argument("--export_obj", action="store_true", help="Export rollout skeleton sequence as OBJ files.")

    args = parser.parse_args()

    run_rollout_and_analyze(
        env_config=args.env_config,
        agent_config=args.agent_config,
        model_file=args.model_file,
        device=args.device,
        num_envs=args.num_envs,
        num_episodes=args.episodes,
        save_states_path=args.save_states,
        char_file=args.char_file,
        mocap_file=args.mocap_file,
        out_dir=args.out_dir,
        export_obj=args.export_obj,
    )


if __name__ == "__main__":
    main()

