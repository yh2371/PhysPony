#!/usr/bin/env python3
"""
Analyze a kinematic motion sequence (.pkl) for:
1) Robust Kinematic Artifacts (Mid-swing reversals, leg swaps) - DEFAULT
2) Legacy Discontinuities (Large joint angle jumps) - OPTIONAL
3) MP4 video of skeleton motion highlighting bad frames
"""

import argparse
import os
import sys
import numpy as np
import pickle

# Add mimickit to path when running from MimicKit/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIMICKIT_DIR = os.path.join(SCRIPT_DIR, "mimickit")
if MIMICKIT_DIR not in sys.path:
    sys.path.insert(0, MIMICKIT_DIR)

# Optional: matplotlib for visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- LOADERS ---

def _load_kin_char(char_file):
    """Load kinematic character model from XML. Returns None if not found or parse error."""
    if not os.path.isfile(char_file):
        return None
    try:
        import anim.kin_char_model as kin_char_model
        import torch
        device = torch.device("cpu")
        model = kin_char_model.KinCharModel(device)
        model.load_char_file(char_file)
        return model
    except Exception as e:
        print(f"Could not load character from {char_file}: {e}")
        return None

def load_motion_pkl(path):
    """Load motion from .pkl. Returns dict with 'frames', 'fps', 'loop_mode'."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def extract_pose_data(frames):
    """frames: (N, D). Returns root_pos (N,3), root_rot (N,3 exp_map), joint_dof (N, dof_size)."""
    root_pos = frames[:, 0:3]
    root_rot = frames[:, 3:6]
    joint_dof = frames[:, 6:]
    return root_pos, root_rot, joint_dof.reshape(-1,105)

# --- ANALYSIS FUNCTIONS ---

def detect_discontinuities_legacy(joint_dof, fps, threshold_rad=0.5):
    """
    LEGACY: Detect frames where joint angles change abruptly.
    """
    N = joint_dof.shape[0]
    if N < 2:
        return np.zeros(N, dtype=bool)
    
    # Calculate simple frame-to-frame delta
    delta = np.abs(joint_dof[1:] - joint_dof[:-1])
    # Wrap angles for periodic dofs
    delta = np.minimum(delta, 2 * np.pi - delta)
    max_delta = np.max(delta, axis=1)
    
    # Pad to match size N (frame i is bad if jump from i-1 to i is large)
    has_discontinuity = np.zeros(N, dtype=bool)
    has_discontinuity[1:] = max_delta > threshold_rad
    return has_discontinuity

def detect_artifacts_robust(joint_rot_quat, body_pos, fps, accel_thresh=150.0, vel_thresh=20.0):
    """
    ROBUST FILTER: Detects physics violations using Quaternion math.
    
    Args:
        joint_rot_quat: (N, J, 4) numpy array or tensor of Joint Quaternions (xyzw)
        body_pos: (N, num_bodies, 3) numpy array of global body positions
        fps: Frames per second
        accel_thresh: Max allowed joint acceleration (rad/s^2)
        vel_thresh: Max allowed body part velocity (m/s)
        
    Returns:
        mask_inertial: (N,) bool
        mask_geometric: (N,) bool
    """
    # Convert to numpy if tensor
    if hasattr(joint_rot_quat, 'cpu'):
        joint_rot_quat = joint_rot_quat.detach().cpu().numpy()
        
    N, J, _ = joint_rot_quat.shape
    dt = 1.0 / fps

    # --- 1. Inertial Filter (Quaternion Acceleration) ---
    # We compute the geodesic distance (shortest angle) between frames.
    # theta = 2 * arccos( |q1 . q2| )
    
    q_t = joint_rot_quat[:-1]      # Frames 0 to N-2
    q_tp1 = joint_rot_quat[1:]     # Frames 1 to N-1
    
    # Dot product: sum(q_t * q_tp1, axis=-1)
    dot = np.sum(q_t * q_tp1, axis=-1)
    dot = np.abs(dot) # Shortest path ignores sign
    dot = np.clip(dot, -1.0, 1.0) # Numerical stability
    
    # Angular distance (magnitude of rotation) for every joint
    theta = 2.0 * np.arccos(dot) # Shape (N-1, J)
    
    # Angular Velocity (scalar magnitude)
    omega = theta / dt
    
    # Angular Acceleration (change in velocity magnitude)
    # This detects "snaps" where velocity suddenly spikes or stops
    alpha = (omega[1:] - omega[:-1]) / dt
    alpha_mag = np.abs(alpha)
    
    # Max acceleration across all joints
    max_acc = np.max(alpha_mag, axis=1) # Shape (N-2,)
    
    # Map back to N frames (pad 1 start, 1 end)
    mask_inertial = np.zeros(N, dtype=bool)
    mask_inertial[1:-1] = max_acc > accel_thresh
    
    # --- 2. Geometric Filter (Bone Implosion/Teleport) ---
    # Check velocity of all body parts
    body_vel = (body_pos[1:] - body_pos[:-1]) / dt
    body_vel_mag = np.linalg.norm(body_vel, axis=-1)
    
    # Max velocity across all body parts
    max_body_vel = np.max(body_vel_mag, axis=1)
    
    mask_geometric = np.zeros(N, dtype=bool)
    mask_geometric[1:] = max_body_vel > vel_thresh
    
    return mask_inertial, mask_geometric

# --- HELPER FUNCTIONS ---

def compute_body_positions_numpy(root_pos, root_rot_exp, joint_dof, kin_model):
    """Run FK using kin_model (torch). Returns body_pos (N, B, 3) and joint_quats (N, J, 4)."""
    import torch
    from util import torch_util
    device = torch.device("cpu")
    
    # Convert to torch
    rp = torch.tensor(root_pos, dtype=torch.float32, device=device)
    rr = torch_util.exp_map_to_quat(torch.tensor(root_rot_exp, dtype=torch.float32, device=device))
    jd = torch.tensor(joint_dof, dtype=torch.float32, device=device)
    
    # Convert DOF to Rotation (Quaternions)
    # kin_model.dof_to_rot returns [N, J, 4] typically for spherical joints
    jr = kin_model.dof_to_rot(jd) 
    jr = torch_util.quat_pos(jr) # Ensure positive w for consistency
    
    # Handle batch dimension if input was single frame
    if rp.ndim == 1:
        rp = rp.unsqueeze(0); rr = rr.unsqueeze(0); jr = jr.unsqueeze(0)
        
    body_pos, _ = kin_model.forward_kinematics(rp, rr, jr)
    
    # Return numpy body pos and raw tensor quaternions
    return body_pos.detach().cpu().numpy(), jr.detach().cpu()

def get_skeleton_edges(kin_model):
    """Return list of (body_i, body_j) for drawing skeleton lines (parent-child)."""
    edges = []
    num_bodies = kin_model.get_num_joints()
    for j in range(1, num_bodies):
        p = kin_model.get_parent_id(j)
        if hasattr(p, "item"): p = p.item()
        edges.append((int(p), j))
    return edges

def _get_limb_color(body_name):
    """
    Returns a matplotlib color string based on body name.
    """
    name = body_name.lower()
    if 'fl' in name or 'front_left' in name:
        return 'cyan'    # Front Left
    elif 'fr' in name or 'front_right' in name:
        return 'blue'    # Front Right
    elif 'bl' in name or 'back_left' in name:
        return 'magenta' # Back Left
    elif 'br' in name or 'back_right' in name:
        return 'red'     # Back Right
    else:
        return 'black'   # Spine/Head/Tail


def _get_limb_color_pair(body_name):
    """
    Returns a pair of colors (pkl_color, sim_color) for a body,
    so we can visualize raw PKL vs simulation reference with distinct hues.
    """
    name = body_name.lower()

    # PKL colors roughly match the single-skeleton scheme above.
    # Sim colors are deliberately different but still encode limb identity.
    if 'fl' in name or 'front_left' in name:
        return ('cyan', 'lime')          # Front Left: PKL vs Sim
    elif 'fr' in name or 'front_right' in name:
        return ('blue', 'navy')          # Front Right
    elif 'bl' in name or 'back_left' in name:
        return ('magenta', 'hotpink')    # Back Left
    elif 'br' in name or 'back_right' in name:
        return ('red', 'darkred')        # Back Right
    else:
        return ('black', 'gray')         # Spine/Head/Tail

def _equal_aspect_limits(positions, margin=0.1):
    pos = np.asarray(positions).reshape(-1, 3)
    mn = pos.min(axis=0)
    mx = pos.max(axis=0)
    r = (mx - mn) * 0.5
    half_span = max(r[0], r[1], r[2], 1e-6) * (1.0 + margin)
    center = (mn + mx) * 0.5
    xlim = (center[0] - half_span, center[0] + half_span)
    ylim = (center[1] - half_span, center[1] + half_span)
    zlim = (center[2] - half_span, center[2] + half_span)
    return xlim, ylim, zlim

# --- RENDERING ---

def render_skeleton_mp4(
    body_pos_all_frames,
    edges,
    mask_dict,
    output_path,
    kin_model,
    fps=30,
    equal_aspect=True,
    title="Skeleton motion",
):
    """
    Render skeleton motion as MP4 with LIMB COLORING.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping MP4.")
        return
    N, num_bodies, _ = body_pos_all_frames.shape
    xlim, ylim, zlim = _equal_aspect_limits(body_pos_all_frames, margin=0.15)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if equal_aspect:
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    
    # Pre-calculate colors for each edge based on body names
    edge_colors = []
    body_names = kin_model.get_body_names()
    for (parent, child) in edges:
        # Child defines the limb segment
        name = body_names[child]
        edge_colors.append(_get_limb_color(name))

    # Initial frame setup
    pos = body_pos_all_frames[0]
    line_artists = []
    
    for k, (i, j) in enumerate(edges):
        c = edge_colors[k]
        ln, = ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]], color=c, linewidth=2)
        line_artists.append(ln)
        
    pts, = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "o", color="black", markersize=3)
    frame_text = ax.text2D(0.02, 0.98, "Frame 0", transform=ax.transAxes, fontsize=10)
    status_text = ax.text2D(0.02, 0.94, "OK", transform=ax.transAxes, fontsize=10, color="green")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2, label='Front-L'),
        Line2D([0], [0], color='blue', lw=2, label='Front-R'),
        Line2D([0], [0], color='magenta', lw=2, label='Back-L'),
        Line2D([0], [0], color='red', lw=2, label='Back-R'),
        Line2D([0], [0], color='gold', lw=2, label='Inertial Fail'),
        Line2D([0], [0], color='crimson', lw=2, label='Geometric Fail'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    def init():
        return line_artists + [pts, frame_text, status_text]

    def update(f):
        pos = body_pos_all_frames[f]
        
        # Check masks
        is_geo = mask_dict.get('geometric', np.zeros(N, dtype=bool))[f]
        is_inertial = mask_dict.get('inertial', np.zeros(N, dtype=bool))[f]
        is_legacy = mask_dict.get('legacy', np.zeros(N, dtype=bool))[f]
        
        # Determine Color Override & Status
        override_color = None
        status = "OK"
        status_color = "green"
        
        if is_geo:
            override_color = "crimson"  # Bright Red for serious fail
            status = "FAIL: Geometric (Leg Swap)"
            status_color = "crimson"
        elif is_inertial:
            override_color = "gold" # Gold/Yellow for snap
            status = "FAIL: Inertial (Snap)"
            status_color = "gold"
        elif is_legacy:
            override_color = "magenta" 
            status = "FAIL: Legacy Jump"
            status_color = "magenta"

        # Update Lines
        for k, (i, j) in enumerate(edges):
            ln = line_artists[k]
            xs, ys, zs = [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]]
            
            if hasattr(ln, "set_data_3d"): ln.set_data_3d(xs, ys, zs)
            else: ln.set_data(xs, ys); ln.set_3d_properties(zs)
            
            # Apply color: Override if artifact, else use limb color
            if override_color:
                ln.set_color(override_color)
                ln.set_linewidth(3) # Make errors thicker
            else:
                ln.set_color(edge_colors[k])
                ln.set_linewidth(2)
            
        if hasattr(pts, "set_data_3d"): pts.set_data_3d(pos[:, 0], pos[:, 1], pos[:, 2])
        else: pts.set_data(pos[:, 0], pos[:, 1]); pts.set_3d_properties(pos[:, 2])
        
        frame_text.set_text(f"Frame {f}")
        status_text.set_text(status)
        status_text.set_color(status_color)
        
        return line_artists + [pts, frame_text, status_text]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=N, blit=False, interval=1000.0 / fps)
    
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist=""), bitrate=3000)
        anim.save(output_path, writer=writer)
        print(f"Saved skeleton motion MP4 to {output_path}")
    except Exception as e:
        print(f"FFMpegWriter failed: {e}. Trying GIF...")
        try:
            gif_path = output_path.rsplit(".", 1)[0] + ".gif"
            writer = animation.PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            print(f"Saved as GIF: {gif_path}")
        except Exception as e2:
            print(f"Could not save video: {e2}")
    plt.close()


def render_skeleton_comparison_mp4(
    body_pos_pkl,
    body_pos_sim,
    edges,
    output_path,
    kin_model,
    fps=30,
    equal_aspect=True,
    title="PKL (solid) vs SimRef (dashed)",
):
    """
    Render two skeletons in the same clip:
    - PKL reference (directly from motion .pkl), solid lines
    - Sim reference (through MotionLib, as used by the env), dashed lines
    Limbs are color‑coded and PKL/Sim use different hues for easier comparison.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping comparison MP4.")
        return

    body_pos_pkl = np.asarray(body_pos_pkl)
    body_pos_sim = np.asarray(body_pos_sim)

    N_pkl, num_bodies_pkl, _ = body_pos_pkl.shape
    N_sim, num_bodies_sim, _ = body_pos_sim.shape

    if num_bodies_pkl != num_bodies_sim:
        print(f"Body count mismatch for comparison: PKL={num_bodies_pkl}, Sim={num_bodies_sim}")
        return

    # Use the minimum common number of frames so arrays line up.
    N = min(N_pkl, N_sim)
    if N_pkl != N_sim:
        print(f"Frame count mismatch for comparison, truncating to {N} frames (PKL={N_pkl}, Sim={N_sim}).")
    body_pos_pkl = body_pos_pkl[:N]
    body_pos_sim = body_pos_sim[:N]

    # Shared axis limits across both skeletons.
    stacked = np.concatenate([body_pos_pkl, body_pos_sim], axis=0)
    xlim, ylim, zlim = _equal_aspect_limits(stacked, margin=0.15)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if equal_aspect:
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)

    body_names = kin_model.get_body_names()
    edge_colors_pkl = []
    edge_colors_sim = []
    for (parent, child) in edges:
        name = body_names[child]
        c_pkl, c_sim = _get_limb_color_pair(name)
        edge_colors_pkl.append(c_pkl)
        edge_colors_sim.append(c_sim)

    # Initial frame
    pos_pkl = body_pos_pkl[0]
    pos_sim = body_pos_sim[0]

    line_pkl = []
    line_sim = []
    for k, (i, j) in enumerate(edges):
        c_p = edge_colors_pkl[k]
        c_s = edge_colors_sim[k]

        ln_p, = ax.plot(
            [pos_pkl[i, 0], pos_pkl[j, 0]],
            [pos_pkl[i, 1], pos_pkl[j, 1]],
            [pos_pkl[i, 2], pos_pkl[j, 2]],
            color=c_p,
            linewidth=2.0,
        )
        ln_s, = ax.plot(
            [pos_sim[i, 0], pos_sim[j, 0]],
            [pos_sim[i, 1], pos_sim[j, 1]],
            [pos_sim[i, 2], pos_sim[j, 2]],
            color=c_s,
            linewidth=1.7,
            linestyle="--",
            alpha=0.9,
        )
        line_pkl.append(ln_p)
        line_sim.append(ln_s)

    pts_pkl, = ax.plot(pos_pkl[:, 0], pos_pkl[:, 1], pos_pkl[:, 2], "o", color="black", markersize=3)
    pts_sim, = ax.plot(pos_sim[:, 0], pos_sim[:, 1], pos_sim[:, 2], "o", color="gray", markersize=3, alpha=0.7)

    frame_text = ax.text2D(0.02, 0.98, "Frame 0", transform=ax.transAxes, fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2, label='PKL FL'),
        Line2D([0], [0], color='lime', lw=2, linestyle='--', label='Sim FL'),
        Line2D([0], [0], color='blue', lw=2, label='PKL FR'),
        Line2D([0], [0], color='navy', lw=2, linestyle='--', label='Sim FR'),
        Line2D([0], [0], color='magenta', lw=2, label='PKL BL'),
        Line2D([0], [0], color='hotpink', lw=2, linestyle='--', label='Sim BL'),
        Line2D([0], [0], color='red', lw=2, label='PKL BR'),
        Line2D([0], [0], color='darkred', lw=2, linestyle='--', label='Sim BR'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def init():
        return line_pkl + line_sim + [pts_pkl, pts_sim, frame_text]

    def update(f):
        pos_p = body_pos_pkl[f]
        pos_s = body_pos_sim[f]

        for k, (i, j) in enumerate(edges):
            ln_p = line_pkl[k]
            ln_s = line_sim[k]

            xs_p, ys_p, zs_p = [pos_p[i, 0], pos_p[j, 0]], [pos_p[i, 1], pos_p[j, 1]], [pos_p[i, 2], pos_p[j, 2]]
            xs_s, ys_s, zs_s = [pos_s[i, 0], pos_s[j, 0]], [pos_s[i, 1], pos_s[j, 1]], [pos_s[i, 2], pos_s[j, 2]]

            if hasattr(ln_p, "set_data_3d"):
                ln_p.set_data_3d(xs_p, ys_p, zs_p)
                ln_s.set_data_3d(xs_s, ys_s, zs_s)
            else:
                ln_p.set_data(xs_p, ys_p); ln_p.set_3d_properties(zs_p)
                ln_s.set_data(xs_s, ys_s); ln_s.set_3d_properties(zs_s)

        if hasattr(pts_pkl, "set_data_3d"):
            pts_pkl.set_data_3d(pos_p[:, 0], pos_p[:, 1], pos_p[:, 2])
            pts_sim.set_data_3d(pos_s[:, 0], pos_s[:, 1], pos_s[:, 2])
        else:
            pts_pkl.set_data(pos_p[:, 0], pos_p[:, 1]); pts_pkl.set_3d_properties(pos_p[:, 2])
            pts_sim.set_data(pos_s[:, 0], pos_s[:, 1]); pts_sim.set_3d_properties(pos_s[:, 2])

        frame_text.set_text(f"Frame {f}")
        return line_pkl + line_sim + [pts_pkl, pts_sim, frame_text]

    anim_obj = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N,
        blit=False,
        interval=1000.0 / fps,
    )

    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist=""), bitrate=3000)
        anim_obj.save(output_path, writer=writer)
        print(f"Saved comparison MP4 to {output_path}")
    except Exception as e:
        print(f"FFMpegWriter failed for comparison: {e}. Trying GIF...")
        try:
            gif_path = output_path.rsplit(".", 1)[0] + "_cmp.gif"
            writer = animation.PillowWriter(fps=fps)
            anim_obj.save(gif_path, writer=writer)
            print(f"Saved comparison as GIF: {gif_path}")
        except Exception as e2:
            print(f"Could not save comparison video: {e2}")
    plt.close()

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Analyze kinematic motion for robust artifacts.")
    parser.add_argument(
        "motion_pkl",
        type=str,
        nargs="?",
        default="data/motions/horse/horse_kin_long_scaled.pkl",
        help="Path to motion .pkl file",
    )
    parser.add_argument(
        "--char-file",
        type=str,
        default="data/assets/horse/test_full_box2_full.xml",
        help="Path to character XML (e.g. data/assets/horse/test_full.xml). If not set, will try default.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_results",
        help="Directory to save plots (default: same dir as motion file)",
    )

    # Filter Toggles
    parser.add_argument("--use_legacy", action="store_true", help="Enable legacy joint-angle discontinuity check")
    parser.add_argument("--legacy_thresh", type=float, default=0.5, help="Radians threshold for legacy check")
    
    # Robust Filter Params
    parser.add_argument("--accel_thresh", type=float, default=150.0, help="Inertial threshold (rad/s^2). Default 150.")
    parser.add_argument("--vel_thresh", type=float, default=20.0, help="Geometric threshold (m/s). Default 20.")
    
    parser.add_argument("--no_plot", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    motion_path = args.motion_pkl
    if not motion_path or not os.path.isfile(motion_path):
        print(f"Error: motion file not found: {motion_path}")
        return 1

    data = load_motion_pkl(motion_path)
    try:
        frames = data["frames"]
        fps = data.get("fps", 30)  
    except:
        frames = data.frames
        fps = data.fps
    if isinstance(frames, list): frames = np.array(frames)
    N, D = frames.shape
    print(f"Loaded motion: {N} frames, {D} dims, fps={fps}")
    root_pos, root_rot, joint_dof = extract_pose_data(frames)
    # Keep copies with the original root pose so we can compare
    # raw PKL FK vs the simulation's MotionLib reference.
    root_pos_orig = root_pos.copy()
    root_rot_orig = root_rot.copy()
    
    # Load Character (Required for Robust Filter)
    char_file = args.char_file
    if not char_file:
        # Try default paths
        candidates = [
            os.path.join(SCRIPT_DIR, "data/assets/horse/test_full_box2_full.xml"),
            os.path.join(SCRIPT_DIR, "data/assets/horse/test_full.xml")
        ]
        for c in candidates:
            if os.path.isfile(c): char_file = c; break
            
    kin_model = _load_kin_char(char_file) if char_file else None
    
    mask_dict = {}
    
    # --- 1. Robust Analysis (Default) ---
    if kin_model:
        print(f"Calculating Robust Artifacts (Inertial & Geometric)...")
        # Need Body Pos for Geometric check. For inertial / geometric thresholds
        # we zero the root so global drift does not affect velocities.
        root_pos[:,:] = 0
        root_rot[:,:] = 0
        
        # IMPORTANT: compute_body_positions_numpy now returns QUATERNIONS
        body_pos, joint_quats = compute_body_positions_numpy(root_pos, root_rot, joint_dof, kin_model)
        
        mask_inertial, mask_geometric = detect_artifacts_robust(
            joint_quats, body_pos, fps, 
            accel_thresh=args.accel_thresh, 
            vel_thresh=args.vel_thresh
        )
        mask_dict['inertial'] = mask_inertial
        mask_dict['geometric'] = mask_geometric
        
        n_inertial = np.sum(mask_inertial)
        n_geo = np.sum(mask_geometric)
        print(f"  > Inertial Violations (> {args.accel_thresh} rad/s^2): {n_inertial} frames ({100*n_inertial/N:.1f}%)")
        print(f"  > Geometric Violations (> {args.vel_thresh} m/s):      {n_geo} frames ({100*n_geo/N:.1f}%)")
        
        # Combined robust mask
        mask_robust = mask_inertial | mask_geometric
    else:
        print("Warning: No character XML loaded. Skipping Robust Filter (needs FK).")
        body_pos = None
        mask_robust = np.zeros(N, dtype=bool)

    # --- 2. Legacy Analysis (Optional) ---
    if args.use_legacy:
        print(f"Calculating Legacy Discontinuities...")
        mask_legacy = detect_discontinuities_legacy(joint_dof, fps, threshold_rad=args.legacy_thresh)
        mask_dict['legacy'] = mask_legacy
        n_legacy = np.sum(mask_legacy)
        print(f"  > Legacy Discontinuities (> {args.legacy_thresh} rad): {n_legacy} frames ({100*n_legacy/N:.1f}%)")
    
    # --- Summary ---
    total_bad = np.zeros(N, dtype=bool)
    for m in mask_dict.values(): total_bad |= m
    n_bad = np.sum(total_bad)
    
    print("\n--- Final Analysis Summary ---")
    print(f"Total Frames: {N}")
    print(f"Bad Frames:   {n_bad} ({100*n_bad/N:.2f}%)")
    print(f"Clean Frames: {N - n_bad} ({100*(N-n_bad)/N:.2f}%)")

    # --- Visualization ---
    if not args.no_plot and HAS_MATPLOTLIB and kin_model:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(motion_path))[0]
        out_mp4 = os.path.join(out_dir, f"{base}_robust_analysis.mp4")
        
        edges = get_skeleton_edges(kin_model)
        
        # Center root for visualization clarity (optional)
        body_pos_vis = body_pos.copy()
        body_pos_vis[:, :, 0] -= np.mean(body_pos_vis[:, 0, 0]) # Center X
        
        render_skeleton_mp4(
            body_pos_vis,
            edges,
            mask_dict,
            out_mp4,
            kin_model, # Pass model for color lookup
            fps=fps,
            equal_aspect=True,
            title=f"{base}: Inertial(Yel)/Geo(Red)"
        )

        # Optional: compare raw PKL FK vs MotionLib (simulation) reference.
        # This helps diagnose any convention / indexing mismatch between the
        # SMAL‑extracted frames and the env’s reference states.
        try:
            import torch
            import anim.motion_lib as motion_lib

            device = torch.device("cpu")

            # 1) FK directly from the PKL frames, using the original root pose.
            body_pos_pkl, _ = compute_body_positions_numpy(
                root_pos_orig,
                root_rot_orig,
                joint_dof,
                kin_model,
            )

            # 2) Reference state as seen by the environment, via MotionLib.
            motion_lib_obj = motion_lib.MotionLib(
                motion_file=motion_path,
                kin_char_model=kin_model,
                device=device,
                ground_preprocessing=False,
                ground_height=0.0,
            )

            num_motions = motion_lib_obj.get_num_motions()
            if num_motions != 1:
                print(f"Warning: MotionLib loaded {num_motions} motions from {motion_path}, comparison assumes 1.")

            # Use the first motion for comparison and sample one time per frame.
            N_frames = frames.shape[0]
            fps_sim = float(motion_lib_obj._motion_fps[0].item())

            motion_ids = torch.zeros(N_frames, dtype=torch.long, device=device)
            motion_times = torch.arange(N_frames, dtype=torch.float32, device=device) / fps_sim

            ref_root_pos, ref_root_rot, _, _, ref_joint_rot, _ = motion_lib_obj.calc_motion_frame(
                motion_ids,
                motion_times,
            )

            body_pos_sim, _ = kin_model.forward_kinematics(
                ref_root_pos,
                ref_root_rot,
                ref_joint_rot,
            )

            body_pos_pkl_np = body_pos_pkl
            body_pos_sim_np = body_pos_sim.detach().cpu().numpy()

            # Re‑center both along X for cleaner side‑by‑side comparison.
            body_pos_pkl_np = body_pos_pkl_np.copy()
            body_pos_sim_np = body_pos_sim_np.copy()
            body_pos_pkl_np[:, :, 0] -= np.mean(body_pos_pkl_np[:, 0, 0])
            body_pos_sim_np[:, :, 0] -= np.mean(body_pos_sim_np[:, 0, 0])

            out_cmp = os.path.join(out_dir, f"{base}_sim_vs_pkl.mp4")
            render_skeleton_comparison_mp4(
                body_pos_pkl_np,
                body_pos_sim_np,
                edges,
                out_cmp,
                kin_model,
                fps=fps,
                equal_aspect=True,
                title=f"{base}: PKL (solid) vs SimRef (dashed)",
            )
        except Exception as e:
            print(f"Could not generate PKL vs SimRef comparison: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())