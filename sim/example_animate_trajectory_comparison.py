#!/usr/bin/env python
"""
Animate trajectory comparison from one or two npz files.

Supported modes
---------------
1) One npz file:
   - visualizes [reference, tracked]
   - useful when a single file contains both ref_* and tracked fields

2) Two npz files:
   - visualizes [reference, tracked(file1), tracked(file2)]
   - reference is taken from file1 if available, otherwise file2, otherwise tracked(file1)

Usage
-----
One file:
    python animate_trajectory_comparison.py file1.npz

Two files:
    python animate_trajectory_comparison.py file1.npz file2.npz

Options:
    --output-dir DIR
    --labels LABEL1 LABEL2 [LABEL3]
    --fps FPS
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimickit.util.visualize_smal_states import generate_smal_mesh


KIN2SMAL = {
    0: 1,  1: 2,  2: 3,  3: 4,  4: 5,  5: 6,  6: 7,  7: 8,  8: 9,  9: 10,
    10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
    17: 33, 18: 34, 19: 35,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24,
    27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32,
}


def kin2smal(body_exp_kin):
    body_exp_smal = np.zeros_like(body_exp_kin)
    for kin_idx in range(35):
        smal_j = KIN2SMAL[kin_idx] - 1
        body_exp_smal[:, smal_j, :] = body_exp_kin[:, kin_idx, :]
    return body_exp_smal


def mirror_translations(trans, axis=1):
    trans_m = np.array(trans, copy=True)
    trans_m[..., axis] *= -1
    return trans_m


def mirror_rotation_matrices(rotmats, axis=1):
    rotmats = np.asarray(rotmats)
    S = np.eye(3)
    S[axis, axis] = -1.0
    return S @ rotmats @ S


def mirror_rotvecs(rotvec, axis=1):
    rotvec = np.asarray(rotvec)
    orig_shape = rotvec.shape
    rot_flat = rotvec.reshape(-1, 3)

    rotmats = scipy.spatial.transform.Rotation.from_rotvec(rot_flat).as_matrix()
    rotmats_m = mirror_rotation_matrices(rotmats, axis=axis)
    rotvec_m = scipy.spatial.transform.Rotation.from_matrix(rotmats_m).as_rotvec()

    return rotvec_m.reshape(orig_shape)


def mirror_quaternions(quat, axis=1, wxyz=False):
    quat = np.asarray(quat)
    orig_shape = quat.shape
    quat_flat = quat.reshape(-1, 4)

    if wxyz:
        quat_xyzw = np.concatenate([quat_flat[:, 1:], quat_flat[:, :1]], axis=-1)
    else:
        quat_xyzw = quat_flat

    rotmats = scipy.spatial.transform.Rotation.from_quat(quat_xyzw).as_matrix()
    rotmats_m = mirror_rotation_matrices(rotmats, axis=axis)
    quat_m_xyzw = scipy.spatial.transform.Rotation.from_matrix(rotmats_m).as_quat()

    if wxyz:
        quat_m = np.concatenate([quat_m_xyzw[:, 3:], quat_m_xyzw[:, :3]], axis=-1)
    else:
        quat_m = quat_m_xyzw

    return quat_m.reshape(orig_shape)


def mirror_rotation_repr(x, axis=1, name="rotation"):
    x = np.asarray(x)

    if x.ndim >= 2 and x.shape[-2:] == (3, 3):
        print(f"Mirroring {name} as rotation matrices: {x.shape}")
        return mirror_rotation_matrices(x, axis=axis)

    if x.shape[-1] == 4:
        print(f"Mirroring {name} as quaternions: {x.shape}")
        return mirror_quaternions(x, axis=axis, wxyz=False)

    if x.shape[-1] == 3:
        print(f"Mirroring {name} as rotvecs: {x.shape}")
        return mirror_rotvecs(x, axis=axis)

    print(f"Warning: unsupported {name} shape {x.shape}; leaving unchanged")
    return np.array(x, copy=True)


def mirror_motion_data(data, axis=1):
    return {
        "trans": mirror_translations(data["trans"], axis=axis),
        "poses_root": mirror_rotation_repr(data["poses_root"], axis=axis, name="poses_root"),
        "poses_body": mirror_rotation_repr(data["poses_body"], axis=axis, name="poses_body"),
    }


def load_trajectory_data(npz_path):
    """
    Returns a dict with optional keys:
      - 'tracked'
      - 'reference'
    Each contains:
      - trans
      - poses_body
      - poses_root
    """
    data = np.load(npz_path, allow_pickle=True)
    result = {}

    if "poses_body" in data and "trans" in data and "poses_root" in data:
        result["tracked"] = {
            "poses_body": data["poses_body"],
            "trans": data["trans"],
            "poses_root": data["poses_root"],
        }

    if "ref_poses_body" in data and "ref_trans" in data and "ref_poses_root" in data:
        result["reference"] = {
            "poses_body": data["ref_poses_body"],
            "trans": data["ref_trans"],
            "poses_root": data["ref_poses_root"],
        }

    # fallback: if only tracked exists, clone it as reference
    if "reference" not in result and "tracked" in result:
        result["reference"] = {
            "poses_body": result["tracked"]["poses_body"].copy(),
            "trans": result["tracked"]["trans"].copy(),
            "poses_root": result["tracked"]["poses_root"].copy(),
        }

    return result


def maybe_convert_kin_to_smal(poses_body):
    """
    If poses_body looks like (T, 35, 3) kin expmap, map to SMAL order.
    Otherwise return as-is.
    """
    arr = np.asarray(poses_body)
    if arr.ndim == 3 and arr.shape[1:] == (35, 3):
        return kin2smal(arr)
    return arr


def build_tracks_from_inputs(data_list, user_labels=None):
    """
    data_list: list of results from load_trajectory_data
    Returns:
      tracks: list of dicts with keys {'label', 'data'}
    """
    if len(data_list) == 1:
        d0 = data_list[0]
        ref_data = d0["reference"]
        tracked_data = d0["tracked"]

        tracks = [
            {"label": "Reference", "data": ref_data},
            {"label": "Tracked", "data": tracked_data},
        ]

        if user_labels is not None:
            # one-file mode expects 2 labels
            for i in range(min(len(user_labels), 2)):
                tracks[i]["label"] = user_labels[i]

        return tracks

    elif len(data_list) == 2:
        d0, d1 = data_list
        ref_data = d0.get("reference", None)
        if ref_data is None:
            ref_data = d1.get("reference", None)
        if ref_data is None:
            ref_data = d0["tracked"]

        tracks = [
            {"label": "Reference", "data": ref_data},
            {"label": "Track1", "data": d0["tracked"]},
            {"label": "Track2", "data": d1["tracked"]},
        ]

        if user_labels is not None:
            for i in range(min(len(user_labels), 3)):
                tracks[i]["label"] = user_labels[i]

        return tracks

    else:
        raise ValueError("Only 1 or 2 npz files are supported.")


def trim_and_prepare_tracks(tracks, mirror_axis=1):
    """
    Trim all tracks to the same T, convert body ordering if needed, and mirror them.
    """
    T = min(track["data"]["trans"].shape[0] for track in tracks)

    prepared = []
    for track in tracks:
        d = track["data"]
        d_trim = {
            "trans": d["trans"][:T],
            "poses_body": kin2smal(d["poses_body"][:T]),
            "poses_root": d["poses_root"][:T],
        }
        d_trim = mirror_motion_data(d_trim, axis=mirror_axis)
        prepared.append({"label": track["label"], "data": d_trim})

    return prepared, T


def animate_trajectory_comparison(
    tracks,
    output_path,
    fps=30,
    smal_model_path=None,
    device="cpu",
):
    """
    tracks: list of {"label": str, "data": {...}}
    Supports 2 or 3 tracks.
    """
    tracks, T = trim_and_prepare_tracks(tracks, mirror_axis=1)

    colors = ["green", "blue", "red", "purple", "orange", "cyan"]
    edgecolors = {
        "green": "darkgreen",
        "blue": "darkblue",
        "red": "darkred",
        "purple": "indigo",
        "orange": "darkorange",
        "cyan": "teal",
    }

    print(f"Generating meshes for {T} frames across {len(tracks)} track(s)...")

    mesh_results = []
    use_mesh = True
    smal_model = None

    try:
        for i, track in enumerate(tracks):
            d = track["data"]
            verts, joints, smal_model = generate_smal_mesh(
                d["poses_body"],
                d["trans"],
                d["poses_root"],
                smal_model_path=smal_model_path,
                device=device,
            )
            mesh_results.append({"verts": verts, "joints": joints})
            print(f"Generated mesh for track {i}: {track['label']}")
    except Exception as e:
        print(f"Warning: mesh generation failed: {e}")
        print("Falling back to joints/root visualization.")
        use_mesh = False
        mesh_results = [{"verts": None, "joints": None} for _ in tracks]

    # collect bounds
    if use_mesh and mesh_results[0]["verts"] is not None:
        all_verts = np.concatenate([m["verts"] for m in mesh_results], axis=0)
        x_min, x_max = all_verts[:, :, 0].min(), all_verts[:, :, 0].max()
        y_min, y_max = all_verts[:, :, 1].min(), all_verts[:, :, 1].max()
        z_min, z_max = all_verts[:, :, 2].min(), all_verts[:, :, 2].max()
    else:
        all_trans = np.concatenate([t["data"]["trans"] for t in tracks], axis=0)
        x_min, x_max = all_trans[:, 0].min(), all_trans[:, 0].max()
        y_min, y_max = all_trans[:, 1].min(), all_trans[:, 1].max()
        z_min, z_max = all_trans[:, 2].min(), all_trans[:, 2].max()

    padding = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)

    all_trans = [t["data"]["trans"] for t in tracks]
    traj_x_min = min(arr[:, 0].min() for arr in all_trans)
    traj_x_max = max(arr[:, 0].max() for arr in all_trans)
    traj_y_min = min(arr[:, 1].min() for arr in all_trans)
    traj_y_max = max(arr[:, 1].max() for arr in all_trans)
    traj_padding = 10.0

    all_root_z = [arr[:, 2] for arr in all_trans]
    root_z_min = min(z.min() for z in all_root_z)
    root_z_max = max(z.max() for z in all_root_z)
    root_z_padding = 0.1 * max(root_z_max - root_z_min, 1e-6)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_trajectory = fig.add_subplot(gs[0, 1])
    ax_height = fig.add_subplot(gs[1, :])

    def draw_mesh_frame(ax, verts, color, alpha=0.8):
        sample_rate = max(1, verts.shape[0] // 1000)
        verts_sample = verts[::sample_rate]
        ax.scatter(
            verts_sample[:, 0],
            verts_sample[:, 1],
            verts_sample[:, 2],
            c=color,
            s=1,
            alpha=alpha,
        )

    def animate(frame):
        ax_3d.clear()
        ax_trajectory.clear()
        ax_height.clear()

        # 3D view
        ax_3d.set_xlim([x_min - padding, x_max + padding])
        ax_3d.set_ylim([y_min - padding, y_max + padding])
        ax_3d.set_zlim([z_min - padding, z_max + padding])
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        ax_3d.set_title("3D View")
        ax_3d.view_init(elev=50, azim=-45)

        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            label = track["label"]
            trans = track["data"]["trans"]

            if use_mesh and mesh_results[i]["verts"] is not None:
                alpha = 0.6 if i == 0 else 0.8
                draw_mesh_frame(ax_3d, mesh_results[i]["verts"][frame], color, alpha=alpha)
            else:
                ax_3d.scatter(
                    [trans[frame, 0]],
                    [trans[frame, 1]],
                    [trans[frame, 2]],
                    c=color,
                    s=100,
                    alpha=0.8,
                    label=label,
                )

        if frame == 0:
            handles = []
            for i, track in enumerate(tracks):
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[i % len(colors)],
                        markersize=8,
                        label=track["label"],
                    )
                )

        # top-down trajectory
        ax_trajectory.set_xlim([traj_x_min - traj_padding, traj_x_max + traj_padding])
        ax_trajectory.set_ylim([traj_y_min - traj_padding, traj_y_max + traj_padding])
        ax_trajectory.set_xlabel("X (m)")
        ax_trajectory.set_ylabel("Y (m)")
        ax_trajectory.set_title("Top-Down Trajectory View")
        ax_trajectory.set_aspect("equal", adjustable="datalim")
        ax_trajectory.grid(True, alpha=0.3)

        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            label = track["label"]
            traj = track["data"]["trans"][:frame + 1]

            if len(traj) > 1:
                ax_trajectory.plot(
                    traj[:, 0], traj[:, 1],
                    color=color, linewidth=2, alpha=0.7, label=label
                )

            ax_trajectory.scatter(
                [traj[-1, 0]], [traj[-1, 1]],
                c=color, s=100, marker="o",
                edgecolors=edgecolors.get(color, "black"),
                linewidths=2, zorder=5,
            )

        if frame == 0:
            ax_trajectory.legend(loc="best", fontsize=10)

        # root height
        ax_height.set_xlim([0, T - 1])
        ax_height.set_ylim([root_z_min - root_z_padding, root_z_max + root_z_padding])
        ax_height.set_xlabel("Frame")
        ax_height.set_ylabel("Root Height (m)")
        ax_height.set_title("Root Height Over Time")
        ax_height.grid(True, alpha=0.3)

        xs = np.arange(frame + 1)
        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            label = track["label"]
            root_z = track["data"]["trans"][:, 2]

            if len(xs) > 1:
                ax_height.plot(xs, root_z[:frame + 1], color=color, linewidth=2, alpha=0.7, label=label)

            ax_height.scatter([frame], [root_z[frame]], c=color, s=50, zorder=5)

        if frame == 0:
            ax_height.legend(loc="best", fontsize=10)

        return []

    print(f"Creating animation with {T} frames at {fps} fps...")
    anim = FuncAnimation(fig, animate, frames=T, interval=1000 / fps, blit=False, repeat=True)

    print(f"Saving animation to {output_path}...")

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(
        fps=fps,
        codec="mpeg4",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    anim.save(output_path, writer=writer)
    print(f"Animation saved to {output_path}")

    plt.close(fig)
    return anim


def main():
    parser = argparse.ArgumentParser(description="Animate trajectory comparison from one or two npz files")
    parser.add_argument(
        "npz_files",
        type=str,
        nargs="+",
        help="One or two .npz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/comparisons",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Custom labels.\n"
            "One-file mode: provide 2 labels (e.g. Reference Tracked)\n"
            "Two-file mode: provide 3 labels (e.g. Reference Track1 Track2)"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for animation",
    )
    parser.add_argument(
        "--smal-model-path",
        type=str,
        default=None,
        help="Optional SMAL model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for mesh generation",
    )

    args = parser.parse_args()

    if len(args.npz_files) not in [1, 2]:
        print("Error: please provide either 1 or 2 npz files.")
        sys.exit(1)

    for path in args.npz_files:
        if not os.path.exists(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    data_list = []
    for path in args.npz_files:
        print(f"Loading data from {path}...")
        data_list.append(load_trajectory_data(path))

    tracks = build_tracks_from_inputs(data_list, user_labels=args.labels)

    print("\nTracks to visualize:")
    for i, track in enumerate(tracks):
        n_frames = track["data"]["trans"].shape[0]
        print(f"  {i}: {track['label']} ({n_frames} frames)")

    base_names = [os.path.splitext(os.path.basename(p))[0] for p in args.npz_files]
    output_name = "animation_" + "_vs_".join(base_names) + ".mp4"
    output_path = os.path.join(args.output_dir, output_name)

    print("\nCreating animation...")
    animate_trajectory_comparison(
        tracks=tracks,
        output_path=output_path,
        fps=args.fps,
        smal_model_path=args.smal_model_path,
        device=args.device,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()