"""
Visualization utilities for SMAL format states.
Supports:
1. 3D joint skeleton animation in matplotlib
2. OBJ file export for Blender
Uses SMAL model to generate actual mesh geometry.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys

import torch
from mimickit.SMAL.smal_torch.smal_torch import SMAL
SMAL_AVAILABLE = True

def load_smal_states(npz_path):
    """
    Load SMAL format states from .npz file.
    
    Supports both rotation matrices (3x3) and quaternions (4) for backward compatibility.
    
    Body rotations come back in the simulator's **KinChar** joint order exactly as
    stored; no reindexing happens here. Anything passing them to generate_smal_mesh()
    or forward_kinematics_smal() must call kin2smal() first.
    
    Returns:
        Dictionary with tracked and optionally reference states
    """
    raw = np.load(npz_path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        # Flat np.savez archive (TestStateCollector.save() format: poses_body,
        # trans, poses_root, body_pos, ref_* as separate top-level arrays).
        data = raw
    else:
        # Legacy format: a single dict pickled via np.save(...).
        data = raw.item()

    result = {
        'tracked': {
            'poses_body': data['poses_body'],
            'trans': data['trans'],
            'poses_root': data['poses_root']
        }
    }
    # Check if body positions are available (for foot identification)
    if 'body_pos' in data:
        result['tracked']['body_pos'] = data['body_pos']
    if 'ref_body_pos' in data:
        if 'reference' not in result:
            result['reference'] = {}
        result['reference']['body_pos'] = data['ref_body_pos']
    
    # Check if reference states are available
    if 'ref_poses_body' in data:
        if 'reference' not in result:
            result['reference'] = {}
        result['reference'].update({
            'poses_body': data['ref_poses_body'],
            'trans': data['ref_trans'],
            'poses_root': data['ref_poses_root']
        })
    
    return result


def quat_to_axis_angle(quat):
    """
    Convert quaternion (xyzw) to axis-angle representation.
    
    Args:
        quat: (..., 4) quaternion in xyzw format
    
    Returns:
        (..., 3) axis-angle vector
    """
    from scipy.spatial.transform import Rotation as R
    
    quat = quat.copy()
    # Normalize
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-8)
    
    # Convert xyzw to scipy format (wxyz)
    quat_scipy = np.zeros_like(quat)
    quat_scipy[..., 0] = quat[..., 3]  # w
    quat_scipy[..., 1] = quat[..., 0]  # x
    quat_scipy[..., 2] = quat[..., 1]  # y
    quat_scipy[..., 3] = quat[..., 2]  # z
    
    # Handle different input shapes
    if quat.ndim == 1:
        rot = R.from_quat(quat_scipy)
        axis_angle = rot.as_rotvec()
    else:
        rot = R.from_quat(quat_scipy.reshape(-1, 4))
        axis_angle = rot.as_rotvec().reshape(quat.shape[:-1] + (3,))
    
    return axis_angle


def quat_to_rot_matrix(quat):
    """
    Convert quaternion (xyzw) to rotation matrix.
    
    Args:
        quat: (..., 4) quaternion in xyzw format
    
    Returns:
        (..., 3, 3) rotation matrix
    """
    quat = quat.copy()
    # Normalize
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-8)
    
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Convert to rotation matrix
    R = np.zeros(quat.shape[:-1] + (3, 3))
    
    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x*y - z*w)
    R[..., 0, 2] = 2 * (x*z + y*w)
    R[..., 1, 0] = 2 * (x*y + z*w)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y*z - x*w)
    R[..., 2, 0] = 2 * (x*z - y*w)
    R[..., 2, 1] = 2 * (y*z + x*w)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R


def smal_states_to_theta(poses_root, poses_body):
    """
    Convert SMAL format states to theta (axis-angle) parameter for SMAL model.
    
    Args:
        poses_root: (T, 4) root quaternions (xyzw)
        poses_body: (T, J, 4) body joint quaternions (xyzw)
    
    Returns:
        theta: (T, 108) axis-angle parameters (36 joints * 3)
    """
    T = poses_root.shape[0]
    num_body_joints = poses_body.shape[1]
    
    # Convert root quaternion to axis-angle
    root_axis_angle = quat_to_axis_angle(poses_root)  # (T, 3)
    
    # Convert body joint quaternions to axis-angle
    body_axis_angle = quat_to_axis_angle(poses_body.reshape(-1, 4))  # (T*J, 3)
    body_axis_angle = body_axis_angle.reshape(T, num_body_joints, 3)  # (T, J, 3)
    
    # SMAL model expects 36 joints total (1 root + 35 body joints)
    # If we have fewer joints, pad with zeros
    num_smal_joints = 36
    theta = np.zeros((T, num_smal_joints, 3))
    
    # Root rotation (first 3 dims)
    theta[:, 0, :] = root_axis_angle
    
    # Body joint rotations
    if num_body_joints <= 35:
        theta[:, 1:1+num_body_joints, :] = body_axis_angle
    else:
        # If more joints, take first 35
        theta[:, 1:, :] = body_axis_angle[:, :35, :]
    
    # Flatten to (T, 108)
    theta = theta.reshape(T, -1)
    
    return theta


KIN2SMAL = {
    0: 1,  1: 2,  2: 3,  3: 4,  4: 5,  5: 6,  6: 7,  7: 8,  8: 9,  9: 10,
    10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
    17: 33, 18: 34, 19: 35,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24,
    27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 34: 32,
}


def kin2smal(poses_body):
    """Reorder (T, 35, ...) body rotations from the simulator's KinChar order to SMAL order.

    A states .npz records body rotations in the KinChar order the MimicKit character
    uses, and load_smal_states() returns them unchanged -- but generate_smal_mesh() and
    forward_kinematics_smal() index SMAL joints. The two orders agree on joints 0-16 and
    differ only on 17-34 (hind legs and tail), so skipping this step still yields a
    horse-shaped mesh, just one whose hind legs and tail are wrong, which is easy to miss
    by eye.

    This is the inverse of the gather in
    process_kinematic_results.ref_motion_to_mimickit_frames, which goes SMAL -> KinChar
    when building the motion files the simulator consumes.
    """
    poses_body = np.asarray(poses_body)
    assert poses_body.shape[1] == 35, f"expected 35 body joints, got {poses_body.shape[1]}"
    out = np.zeros_like(poses_body)
    for kin_idx in range(35):
        out[:, KIN2SMAL[kin_idx] - 1] = poses_body[:, kin_idx]
    return out


def generate_smal_mesh(poses_body, trans, poses_root, smal_model_path=None, device='cpu', beta=None):
    """
    Generate mesh vertices and joints using SMAL model.
    
    Args:
        poses_body: (T, J, 3, 3) or (T, J, 4) body joint rotations (matrices or quaternions),
                    in **SMAL** joint order. Rotations read straight from a states
                    .npz are in KinChar order -- pass them through kin2smal() first.
        trans: (T, 3) root translations
        poses_root: (T, 3, 3) or (T, 4) root rotation (matrix or quaternion)
        smal_model_path: Path to SMAL model .pkl file
        device: Device to run SMAL model on
        beta: (T, 9) shape parameters. If None, uses zeros.
    
    Returns:
        verts: (T, num_vertices, 3) mesh vertices
        joints: (T, num_joints, 3) joint positions
    """
    if not SMAL_AVAILABLE:
        raise ImportError("SMAL model not available. Cannot generate mesh.")
    
    T = poses_root.shape[0]
    
    # Default SMAL model path
    if smal_model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        smal_model_path = os.path.join(base_dir, 'SMAL', 'my_smpl_0000_horse_new_skeleton_horse.pkl')
    
    if not os.path.exists(smal_model_path):
        raise FileNotFoundError(f"SMAL model not found at: {smal_model_path}")
    
    # Initialize SMAL model
    smal_model = SMAL(smal_model_path, device=device, use_smal_betas=False)
    
    # Check if we have rotation matrices or quaternions
    if poses_root.ndim == 3 and poses_root.shape[-1] == 3:
        # Rotation matrices: (T, 3, 3)
        use_rot_matrices = True
    elif poses_root.ndim == 2 and poses_root.shape[-1] == 4:
        # Quaternions: (T, 4) - convert to rotation matrices
        use_rot_matrices = False
        from scipy.spatial.transform import Rotation as R
        poses_root = R.from_quat(poses_root[:, [1, 2, 3, 0]]).as_matrix()  # Convert xyzw to wxyz, then to matrix
        if poses_body.ndim == 3 and poses_body.shape[-1] == 4:
            poses_body_flat = poses_body.reshape(-1, 4)
            poses_body = R.from_quat(poses_body_flat[:, [1, 2, 3, 0]]).as_matrix().reshape(T, -1, 3, 3)
    else:
        raise ValueError(f"Unsupported rotation format: poses_root shape {poses_root.shape}")
    
    # SMAL expects (batch, 36, 3, 3) rotation matrices
    # Combine root and body rotations
    num_body_joints = poses_body.shape[1]
    num_smal_joints = 36
    
    # Create full rotation matrix array
    theta_rot = np.zeros((T, num_smal_joints, 3, 3))
    theta_rot[:, 0, :, :] = poses_root  # Root rotation
    
    # Body joint rotations
    if num_body_joints <= 35:
        theta_rot[:, 1:1+num_body_joints, :, :] = poses_body
    else:
        theta_rot[:, 1:, :, :] = poses_body[:, :35, :, :]
    
    # Convert to torch tensors
    theta_torch = torch.from_numpy(theta_rot).float().to(device)  # (T, 36, 3, 3)
    trans_torch = torch.from_numpy(trans).float().to(device)
    
    # Shape parameters (default to zeros if not provided)
    if beta is None:
        beta_torch = torch.zeros(T, 9, device=device, dtype=torch.float32)
    else:
        beta_torch = torch.from_numpy(beta).float().to(device)
    
    # Generate meshes for all frames
    all_verts = []
    all_joints = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    for i in range(0, T, batch_size):
        end_idx = min(i + batch_size, T)
        batch_theta = theta_torch[i:end_idx]  # (batch, 36, 3, 3)
        batch_trans = trans_torch[i:end_idx]
        batch_beta = beta_torch[i:end_idx]
        
        with torch.no_grad():
            # SMAL accepts rotation matrices directly when shape is (batch, 36, 3, 3)
            verts, joints, _ = smal_model(beta=batch_beta, theta=batch_theta, trans=batch_trans)
        
        all_verts.append(verts.cpu().numpy())
        all_joints.append(joints.cpu().numpy())
    
    verts = np.concatenate(all_verts, axis=0)
    joints = np.concatenate(all_joints, axis=0)
    
    return verts, joints, smal_model


def forward_kinematics_smal(poses_body, trans, poses_root, joint_parents, joint_offsets):
    """
    Compute joint positions from SMAL format using forward kinematics.
    
    Args:
        poses_body: (T, J, 3, 3) or (T, J, 4) body joint rotations (matrices or quaternions),
                    in **SMAL** joint order. Rotations read straight from a states
                    .npz are in KinChar order -- pass them through kin2smal() first.
        trans: (T, 3) root translations
        poses_root: (T, 3, 3) or (T, 4) root rotation (matrix or quaternion)
        joint_parents: (J+1,) parent indices (-1 for root)
        joint_offsets: (J+1, 3) local joint offsets
    
    Returns:
        joint_positions: (T, J+1, 3) joint positions in world space
    """
    T = poses_body.shape[0]
    num_joints = len(joint_parents)
    
    # Check if we have rotation matrices or quaternions
    if poses_root.ndim == 3 and poses_root.shape[-1] == 3:
        # Already rotation matrices
        root_rots = poses_root  # (T, 3, 3)
    elif poses_root.ndim == 2 and poses_root.shape[-1] == 4:
        # Quaternions - convert to rotation matrices
        root_rots = quat_to_rot_matrix(poses_root)  # (T, 3, 3)
    else:
        raise ValueError(f"Unsupported root rotation format: {poses_root.shape}")
    
    if poses_body.ndim == 4 and poses_body.shape[-1] == 3:
        # Already rotation matrices
        body_rots = poses_body  # (T, J, 3, 3)
    elif poses_body.ndim == 3 and poses_body.shape[-1] == 4:
        # Quaternions - convert to rotation matrices
        body_rots = quat_to_rot_matrix(poses_body.reshape(-1, 4)).reshape(T, -1, 3, 3)  # (T, J, 3, 3)
    else:
        raise ValueError(f"Unsupported body rotation format: {poses_body.shape}")
    
    # Combine root and body rotations
    all_rots = np.zeros((T, num_joints, 3, 3))
    all_rots[:, 0] = root_rots  # Root rotation
    all_rots[:, 1:] = body_rots  # Body joint rotations
    
    # Compute joint positions
    joint_positions = np.zeros((T, num_joints, 3))
    
    for t in range(T):
        for j in range(num_joints):
            parent_idx = joint_parents[j]
            
            if parent_idx == -1:
                # Root joint
                joint_positions[t, j] = trans[t]
            else:
                # Transform local offset by parent rotation
                parent_rot = all_rots[t, parent_idx]
                local_offset = joint_offsets[j]
                world_offset = parent_rot @ local_offset
                joint_positions[t, j] = joint_positions[t, parent_idx] + world_offset
    
    return joint_positions


def visualize_skeleton_3d(npz_path, output_path=None, joint_parents=None, joint_offsets=None, 
                          fps=30, show_reference=True, smal_model_path=None, device='cpu', use_mesh=True):
    """
    Visualize SMAL states as 3D skeleton/mesh animation in matplotlib.
    
    Args:
        npz_path: Path to .npz file with SMAL states
        output_path: Optional path to save animation (e.g., 'animation.gif')
        joint_parents: (J+1,) parent indices. If None and use_mesh=False, uses simple chain structure
        joint_offsets: (J+1, 3) joint offsets. If None and use_mesh=False, uses unit offsets
        fps: Frames per second for animation
        show_reference: Whether to show reference skeleton
        smal_model_path: Path to SMAL model .pkl file (default: auto-detect)
        device: Device to run SMAL model on ('cpu' or 'cuda')
        use_mesh: If True, use SMAL model to generate mesh. If False, use simple skeleton.
    """
    states = load_smal_states(npz_path)
    print(states.keys())
    tracked = states['tracked']
    print(tracked.keys())
    reference = states.get('reference', None)
    
    T = tracked['trans'].shape[0]
    
    # Try to use SMAL model if available and requested
    smal_model = None
    if use_mesh and SMAL_AVAILABLE:
        try:
            tracked_verts, tracked_joints, smal_model = generate_smal_mesh(
                kin2smal(tracked['poses_body']), tracked['trans'], tracked['poses_root'],
                smal_model_path=smal_model_path, device=device
            )
            
            if reference is not None and show_reference:
                ref_verts, ref_joints, _ = generate_smal_mesh(
                    kin2smal(reference['poses_body']), reference['trans'], reference['poses_root'],
                    smal_model_path=smal_model_path, device=device
                )
            else:
                ref_verts = None
                ref_joints = None
            
            # Use joints for skeleton visualization
            tracked_joints_vis = tracked_joints
            ref_joints_vis = ref_joints if ref_joints is not None else None
            use_mesh_vis = True
        except Exception as e:
            print(f"Warning: Failed to use SMAL model: {e}")
            print("Falling back to simple skeleton visualization.")
            use_mesh_vis = False
            tracked_verts = None
            ref_verts = None
            smal_model = None
    else:
        use_mesh_vis = Tr
        tracked_verts = None
        ref_verts = None
    
    # Fallback to simple skeleton if SMAL not available or failed
    if not use_mesh_vis:
        num_body_joints = tracked['poses_body'].shape[1]
        num_joints = num_body_joints + 1  # +1 for root
        
        # Default joint structure (simple chain if not provided)
        if joint_parents is None:
            joint_parents = np.array([-1] + list(range(num_body_joints)))  # Simple chain
        if joint_offsets is None:
            joint_offsets = np.zeros((num_joints, 3))
            joint_offsets[1:, 2] = -0.1  # Default offset along z-axis
        
        # Compute joint positions
        tracked_joints_vis = forward_kinematics_smal(
            kin2smal(tracked['poses_body']), tracked['trans'], tracked['poses_root'],
            joint_parents, joint_offsets
        )
        
        if reference is not None and show_reference:
            ref_joints_vis = forward_kinematics_smal(
                kin2smal(reference['poses_body']), reference['trans'], reference['poses_root'],
                joint_parents, joint_offsets
            )
        else:
            ref_joints_vis = None
    
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: 3D view (zoomed in)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Subplot 2: Top-down XY trajectory view
    ax_trajectory = fig.add_subplot(gs[0, 1])
    
    # Subplot 3: Z-elevation plot
    ax_elevation = fig.add_subplot(gs[1, :])
    
    # Compute bounds for 3D view (zoomed in - use smaller padding)
    # Note: use_mesh_vis is set earlier in the function
    if use_mesh_vis and tracked_verts is not None:
        # Use mesh vertices for bounds
        all_verts = tracked_verts
        if ref_verts is not None:
            all_verts = np.concatenate([tracked_verts, ref_verts], axis=0)
        x_min, x_max = all_verts[:, :, 0].min(), all_verts[:, :, 0].max()
        y_min, y_max = all_verts[:, :, 1].min(), all_verts[:, :, 1].max()
        z_min, z_max = all_verts[:, :, 2].min(), all_verts[:, :, 2].max()
    else:
        # Use joints for bounds
        all_joints = tracked_joints_vis
        if ref_joints_vis is not None:
            all_joints = np.concatenate([tracked_joints_vis, ref_joints_vis], axis=0)
        x_min, x_max = all_joints[:, :, 0].min(), all_joints[:, :, 0].max()
        y_min, y_max = all_joints[:, :, 1].min(), all_joints[:, :, 1].max()
        z_min, z_max = all_joints[:, :, 2].min(), all_joints[:, :, 2].max()
    
    # Zoomed in padding (smaller for better view)
    padding = 0.05 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Compute trajectory bounds for top-down view
    traj_x_min, traj_x_max = tracked['trans'][:, 0].min(), tracked['trans'][:, 0].max()
    traj_y_min, traj_y_max = tracked['trans'][:, 1].min(), tracked['trans'][:, 1].max()
    if reference is not None:
        traj_x_min = min(traj_x_min, reference['trans'][:, 0].min())
        traj_x_max = max(traj_x_max, reference['trans'][:, 0].max())
        traj_y_min = min(traj_y_min, reference['trans'][:, 1].min())
        traj_y_max = max(traj_y_max, reference['trans'][:, 1].max())
    traj_padding = 0.1 * max(traj_x_max - traj_x_min, traj_y_max - traj_y_min)
    
    # Identify foot joints/bodies (lowest z positions typically indicate feet)
    # Prefer body positions if available, otherwise use joints
    foot_indices = []
    foot_positions_tracked = None
    foot_positions_ref = None
    
    if 'body_pos' in tracked and tracked['body_pos'] is not None:
        # Use body positions (more accurate for feet)
        body_pos_tracked = tracked['body_pos']  # (T, num_bodies, 3)
        first_frame_bodies = body_pos_tracked[0]  # (num_bodies, 3)
        z_values = first_frame_bodies[:, 2]
        # Get indices of 4 lowest z values (assuming 4 feet)
        foot_indices = np.argsort(z_values)[:4]
        foot_positions_tracked = body_pos_tracked[:, foot_indices, :]  # (T, 4, 3)
        
        if reference is not None and 'body_pos' in reference and reference['body_pos'] is not None:
            body_pos_ref = reference['body_pos']
            foot_positions_ref = body_pos_ref[:, foot_indices, :]  # (T, 4, 3)
    elif tracked_joints_vis is not None:
        # Fallback to joints
        first_frame_joints = tracked_joints_vis[0]  # (num_joints, 3)
        z_values = first_frame_joints[:, 2]
        foot_indices = np.argsort(z_values)[:4]
        foot_positions_tracked = tracked_joints_vis[:, foot_indices, :]  # (T, 4, 3)
        
        if ref_joints_vis is not None:
            foot_positions_ref = ref_joints_vis[:, foot_indices, :]  # (T, 4, 3)
    
    # Initialize plot elements
    tracked_lines = []
    tracked_points = []
    ref_lines = []
    ref_points = []
    
    def draw_skeleton(ax, joints, lines_list, points_list, color, alpha=1.0, label=None):
        """Draw skeleton for a single frame."""
        # Clear previous lines and points
        for line in lines_list:
            line.remove()
        for point in points_list:
            point.remove()
        lines_list.clear()
        points_list.clear()
        
        num_joints = joints.shape[0]
        
        # Draw bones (lines between parent and child)
        # For skeleton visualization, connect joints based on SMAL parent structure
        if smal_model is not None and hasattr(smal_model, 'parents') and use_mesh_vis:
            parents = smal_model.parents
            for j in range(min(num_joints, len(parents))):
                parent_idx = parents[j]
                if parent_idx >= 0 and parent_idx < num_joints:
                    line, = ax.plot(
                        [joints[parent_idx, 0], joints[j, 0]],
                        [joints[parent_idx, 1], joints[j, 1]],
                        [joints[parent_idx, 2], joints[j, 2]],
                        color=color, linewidth=2, alpha=alpha, label=label if j == 0 else None
                    )
                    lines_list.append(line)
        else:
            # Simple chain structure
            for j in range(1, num_joints):
                line, = ax.plot(
                    [joints[j-1, 0], joints[j, 0]],
                    [joints[j-1, 1], joints[j, 1]],
                    [joints[j-1, 2], joints[j, 2]],
                    color=color, linewidth=2, alpha=alpha, label=label if j == 1 else None
                )
                lines_list.append(line)
        
        # Draw joints (points)
        for j in range(num_joints):
            point = ax.scatter(
                joints[j, 0], joints[j, 1], joints[j, 2],
                color=color, s=50, alpha=alpha
            )
            points_list.append(point)
    
    def draw_mesh(ax, verts, faces, color, alpha=1.0, label=None):
        """Draw mesh for a single frame."""
        # Sample vertices for visualization (every Nth vertex)
        sample_rate = max(1, verts.shape[0] // 1000)  # Limit to ~1000 vertices
        verts_sample = verts[::sample_rate]
        
        # Draw mesh as point cloud (faster than full mesh)
        ax.scatter(verts_sample[:, 0], verts_sample[:, 1], verts_sample[:, 2],
                  c=color, s=1, alpha=alpha, label=label)
    
    def animate(frame):
        """Animation function."""
        # Clear all subplots
        ax_3d.clear()
        ax_trajectory.clear()
        ax_elevation.clear()
        
        # ========== Subplot 1: 3D View (zoomed in) ==========
        ax_3d.set_xlim([x_min - padding, x_max + padding])
        ax_3d.set_ylim([y_min - padding, y_max + padding])
        ax_3d.set_zlim([z_min - padding, z_max + padding])
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        title_3d = '3D View (Zoomed)' if use_mesh_vis else '3D Skeleton (Zoomed)'
        ax_3d.set_title(f'{title_3d} - Frame {frame}/{T-1}')
        # Set view angle for better perspective
        ax_3d.view_init(elev=50, azim=-45)
        
        # Draw content and collect handles for legend
        legend_handles_3d = []
        
        if use_mesh_vis and tracked_verts is not None:
            # Draw mesh
            if smal_model is not None and hasattr(smal_model, 'faces'):
                faces = smal_model.faces.cpu().numpy()
                draw_mesh(ax_3d, tracked_verts[frame], faces, 'blue', 0.8)
            else:
                h1 = ax_3d.scatter(tracked_verts[frame][::10, 0], 
                             tracked_verts[frame][::10, 1], 
                             tracked_verts[frame][::10, 2],
                             c='blue', s=1, alpha=0.8, label='Tracked')
                if frame:
                    legend_handles_3d.append(h1)
            
            if tracked_joints_vis is not None:
                draw_skeleton(ax_3d, tracked_joints_vis[frame], tracked_lines, tracked_points, 'blue', 0.5)
            
            if ref_verts is not None and show_reference:
                if smal_model is not None and hasattr(smal_model, 'faces'):
                    faces = smal_model.faces.cpu().numpy()
                    draw_mesh(ax_3d, ref_verts[frame], faces, 'red', 0.5)
                else:
                    h2 = ax_3d.scatter(ref_verts[frame][::10, 0], 
                                 ref_verts[frame][::10, 1], 
                                 ref_verts[frame][::10, 2],
                                 c='red', s=1, alpha=0.5, label='Reference')
                    if frame:
                        legend_handles_3d.append(h2)
                
                if ref_joints_vis is not None:
                    draw_skeleton(ax_3d, ref_joints_vis[frame], ref_lines, ref_points, 'red', 0.3)
        else:
            # Draw skeleton only
            h1 = ax_3d.plot([], [], 'b-', linewidth=2, label='Tracked')[0] if frame else None
            draw_skeleton(ax_3d, tracked_joints_vis[frame], tracked_lines, tracked_points, 'blue', 1.0)
            if h1 is not None:
                legend_handles_3d.append(h1)
            
            if ref_joints_vis is not None and show_reference:
                h2 = ax_3d.plot([], [], 'r-', linewidth=2, label='Reference')[0] if frame else None
                draw_skeleton(ax_3d, ref_joints_vis[frame], ref_lines, ref_points, 'red', 0.6)
                if h2 is not None:
                    legend_handles_3d.append(h2)
        
        # Add legend using explicit handles
        if frame:
            # Create legend with explicit labels
            from matplotlib.lines import Line2D
            legend_elements = []
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label='Tracked'))
            if reference is not None and show_reference:
                legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Reference'))
            ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        # ========== Subplot 2: Top-down XY Trajectory View ==========
        ax_trajectory.set_xlim([traj_x_min - traj_padding, traj_x_max + traj_padding])
        ax_trajectory.set_ylim([traj_y_min - traj_padding, traj_y_max + traj_padding])
        ax_trajectory.set_xlabel('X (m)')
        ax_trajectory.set_ylabel('Y (m)')
        ax_trajectory.set_title('Top-Down Trajectory View')
        ax_trajectory.set_aspect('equal', adjustable='box')
        ax_trajectory.grid(True, alpha=0.3)
        
        # Plot full trajectory up to current frame
        tracked_traj = tracked['trans'][:frame+1]
        if len(tracked_traj) > 1:
            ax_trajectory.plot(tracked_traj[:, 0], tracked_traj[:, 1], 'b-', linewidth=2, alpha=0.5)
        ax_trajectory.scatter(tracked_traj[-1, 0], tracked_traj[-1, 1], c='blue', s=100, marker='o', 
                             edgecolors='darkblue', linewidths=2, zorder=5)
        
        if reference is not None and show_reference:
            ref_traj = reference['trans'][:frame+1]
            if len(ref_traj) > 1:
                ax_trajectory.plot(ref_traj[:, 0], ref_traj[:, 1], 'r-', linewidth=2, alpha=0.5)
            ax_trajectory.scatter(ref_traj[-1, 0], ref_traj[-1, 1], c='red', s=100, marker='s', 
                                 edgecolors='darkred', linewidths=2, zorder=5)
        
        # Add legend using explicit handles (only on first frame to avoid duplicates)
        if frame:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='Tracked Trajectory'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markeredgecolor='darkblue', markersize=10, markeredgewidth=2, label='Tracked (current)')
            ]
            if reference is not None and show_reference:
                legend_elements.extend([
                    Line2D([0], [0], color='red', lw=2, linestyle='--', alpha=0.5, label='Reference Trajectory'),
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                          markeredgecolor='darkred', markersize=10, markeredgewidth=2, label='Reference (current)')
                ])
            ax_trajectory.legend(handles=legend_elements, loc='upper right', fontsize=12, bbox_to_anchor=(2.05, 1.05),  # (x, y) in axes fraction coordinates
                                bbox_transform=ax_trajectory.transAxes)
        
        # ========== Subplot 3: Z-Elevation Plot ==========
        ax_elevation.set_xlim([0, T-1])
        ax_elevation.set_xlabel('Frame')
        ax_elevation.set_ylabel('Z Elevation (m)')
        ax_elevation.set_title('Z-Elevation: Root and Foot Joints (w.r.t. Ground Plane z=0)')
        ax_elevation.grid(True, alpha=0.3)
        
        ax_elevation.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Plot root elevation (full history)
        frames = np.arange(frame + 1)
        tracked_root_z = tracked['trans'][:frame+1, 2]
        if len(frames) > 1:
            ax_elevation.plot(frames, tracked_root_z, 'b-', linewidth=2, alpha=0.7)
        ax_elevation.scatter([frame], [tracked_root_z[-1]], c='blue', s=50, zorder=5)
        
        if reference is not None and show_reference:
            ref_root_z = reference['trans'][:frame+1, 2]
            if len(frames) > 1:
                ax_elevation.plot(frames, ref_root_z, 'r--', linewidth=2, alpha=0.7)
            ax_elevation.scatter([frame], [ref_root_z[-1]], c='red', s=50, marker='s', zorder=5)
        
        # Plot foot joint elevations (full history)
        if foot_positions_tracked is not None:
            for i in range(foot_positions_tracked.shape[1]):
                foot_z = foot_positions_tracked[:frame+1, i, 2]
                if len(frames) > 1:
                    ax_elevation.plot(frames, foot_z, 'b-', linewidth=1, alpha=0.4)
                ax_elevation.scatter([frame], [foot_z[-1]], c='blue', s=30, alpha=0.6, zorder=5)
            
            if foot_positions_ref is not None and show_reference:
                for i in range(foot_positions_ref.shape[1]):
                    foot_z = foot_positions_ref[:frame+1, i, 2]
                    if len(frames) > 1:
                        ax_elevation.plot(frames, foot_z, 'r--', linewidth=1, alpha=0.4)
                    ax_elevation.scatter([frame], [foot_z[-1]], c='red', s=30, marker='s', alpha=0.6, zorder=5)
        
        # Add legend using explicit handles (only on first frame)
        if frame:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label='Root (Tracked)'),
            ]
            if reference is not None and show_reference:
                legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label='Root (Reference)'))
            if foot_positions_tracked is not None:
                legend_elements.append(Line2D([0], [0], color='blue', lw=1, alpha=0.4, label='Feet (Tracked)'))
                if foot_positions_ref is not None and show_reference:
                    legend_elements.append(Line2D([0], [0], color='red', lw=1, linestyle='--', alpha=0.4, label='Feet (Reference)'))
            ax_elevation.legend(handles=legend_elements, loc='upper right', fontsize=12, ncol=2)
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=T, interval=1000/fps, blit=False, repeat=True)
    
    if output_path:
        # Save animation
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        elif output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {output_path}")
    else:
        plt.show()
    
    return anim


def load_uv_coordinates(uvmap_obj_path=None, num_vertices=None, template_vertices=None):
    """
    Load UV coordinates from a UV map OBJ file or generate default ones.
    
    Args:
        uvmap_obj_path: Path to UV map OBJ file. If None, generates default UVs.
        num_vertices: Number of vertices (required if uvmap_obj_path is None)
        template_vertices: (V, 3) template vertex positions for generating UVs if needed
    
    Returns:
        uv_coords: (V, 2) UV coordinates
    """
    if uvmap_obj_path and os.path.exists(uvmap_obj_path):
        # Load UV coordinates from OBJ file
        uv_coords = []
        with open(uvmap_obj_path, 'r') as f:
            for line in f:
                if line.startswith('vt '):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        u = float(parts[1])
                        v = float(parts[2])
                        uv_coords.append([u, v])
        if uv_coords:
            uv_array = np.array(uv_coords)
            # If the number of UV coordinates matches, return them
            if num_vertices is None or len(uv_array) == num_vertices:
                return uv_array
            else:
                print(f"Warning: UV map has {len(uv_array)} coordinates but mesh has {num_vertices} vertices. "
                      f"Using spherical mapping instead.")
    
    # Generate default UV coordinates (spherical mapping)
    if num_vertices is None:
        raise ValueError("num_vertices must be provided if uvmap_obj_path is None")
    
    uv_coords = np.zeros((num_vertices, 2))
    
    # Use template vertices if provided, otherwise will be generated per-frame
    if template_vertices is not None:
        for i, v in enumerate(template_vertices):
            # Normalize vertex position
            norm = np.linalg.norm(v)
            if norm > 0:
                v_norm = v / norm
                # Spherical coordinates to UV
                u = 0.5 + np.arctan2(v_norm[2], v_norm[0]) / (2 * np.pi)
                v_coord = 0.5 - np.arcsin(v_norm[1]) / np.pi
                uv_coords[i] = [u, v_coord]
    
    return uv_coords


def export_skeleton_obj(npz_path, output_dir, joint_parents=None, joint_offsets=None, 
                       prefix='frame', show_reference=True, smal_model_path=None, device='cpu', 
                       use_mesh=True, texture_dir=None, uvmap_obj_path=None):
    """
    Export SMAL states as OBJ files for Blender.
    Each frame is exported as a separate OBJ file with mesh or joint positions.
    
    Args:
        npz_path: Path to .npz file with SMAL states
        output_dir: Directory to save OBJ files
        joint_parents: (J+1,) parent indices. If None and use_mesh=False, uses simple chain structure
        joint_offsets: (J+1, 3) joint offsets. If None and use_mesh=False, uses unit offsets
        prefix: Prefix for output files (default: 'frame')
        show_reference: Whether to export reference skeleton as well
        smal_model_path: Path to SMAL model .pkl file (default: auto-detect)
        device: Device to run SMAL model on ('cpu' or 'cuda')
        use_mesh: If True, export full mesh. If False, export skeleton only.
        texture_dir: Directory containing texture PNG files. If provided, textures will be applied.
                     Texture files should be named like: {prefix}_tracked_{frame:04d}.png
        uvmap_obj_path: Path to UV map OBJ file for loading UV coordinates. If None, uses default mapping.
    """
    states = load_smal_states(npz_path)
    tracked = states['tracked']
    reference = states.get('reference', None)
    
    T = tracked['trans'].shape[0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to use SMAL model if available and requested
    smal_model = None
    if use_mesh and SMAL_AVAILABLE:
        try:
            tracked_verts, tracked_joints, smal_model = generate_smal_mesh(
                kin2smal(tracked['poses_body']), tracked['trans'], tracked['poses_root'],
                smal_model_path=smal_model_path, device=device
            )
            
            if reference is not None and show_reference:
                ref_verts, ref_joints, _ = generate_smal_mesh(
                    kin2smal(reference['poses_body']), reference['trans'], reference['poses_root'],
                    smal_model_path=smal_model_path, device=device
                )
            else:
                ref_verts = None
                ref_joints = None
            
            use_mesh_export = True
        except Exception as e:
            print(f"Warning: Failed to use SMAL model for export: {e}")
            print("Falling back to skeleton export.")
            use_mesh_export = False
            tracked_verts = None
            ref_verts = None
    else:
        use_mesh_export = False
        tracked_verts = None
        ref_verts = None
    
    # Fallback to skeleton export
    if not use_mesh_export:
        num_body_joints = tracked['poses_body'].shape[1]
        num_joints = num_body_joints + 1
        
        # Default joint structure
        if joint_parents is None:
            joint_parents = np.array([-1] + list(range(num_body_joints)))
        if joint_offsets is None:
            joint_offsets = np.zeros((num_joints, 3))
            joint_offsets[1:, 2] = -0.1
        
        # Compute joint positions
        tracked_joints = forward_kinematics_smal(
            kin2smal(tracked['poses_body']), tracked['trans'], tracked['poses_root'],
            joint_parents, joint_offsets
        )
        
        if reference is not None and show_reference:
            ref_joints = forward_kinematics_smal(
                kin2smal(reference['poses_body']), reference['trans'], reference['poses_root'],
                joint_parents, joint_offsets
            )
        else:
            ref_joints = None
    
    # Load UV coordinates if using mesh export
    uv_coords = None
    if use_mesh_export and tracked_verts is not None:
        num_verts = tracked_verts.shape[1]
        # Use first frame's vertices as template for UV generation
        template_verts = tracked_verts[0] if len(tracked_verts) > 0 else None
        uv_coords = load_uv_coordinates(uvmap_obj_path=uvmap_obj_path, 
                                       num_vertices=num_verts,
                                       template_vertices=template_verts)
    
    # Export each frame
    for t in range(T):
        if use_mesh_export and tracked_verts is not None:
            # Find texture file for this frame
            texture_file = None
            if texture_dir:
                # Try different naming patterns
                texture_patterns = [
                    os.path.join(texture_dir, f'001{t:04d}.png'),
                ]
                for pattern in texture_patterns:
                    if os.path.exists(pattern):
                        texture_file = pattern
                        break
            # Export tracked mesh
            tracked_file = os.path.join(output_dir, f'{prefix}_tracked_{t:04d}.obj')
            
            export_mesh_obj(tracked_verts[t], smal_model.faces.cpu().numpy(), tracked_file, 
                          color='blue', texture_file=texture_file, uv_coords=uv_coords)
            
            # Export reference mesh if available
            if ref_verts is not None and show_reference:
                # Find texture file for reference
                ref_texture_file = None
                if texture_dir:
                    ref_texture_patterns = [
                        os.path.join(texture_dir, f'001{t:04d}.png'),
                    ]
                    for pattern in ref_texture_patterns:
                        if os.path.exists(pattern):
                            ref_texture_file = pattern
                            break

                ref_file = os.path.join(output_dir, f'{prefix}_ref_{t:04d}.obj')
                export_mesh_obj(ref_verts[t], smal_model.faces.cpu().numpy(), ref_file,
                              color='red', texture_file=ref_texture_file, uv_coords=uv_coords)
        else:
            # Export tracked skeleton
            tracked_file = os.path.join(output_dir, f'{prefix}_tracked_{t:04d}.obj')
            export_frame_obj(tracked_joints[t], joint_parents, tracked_file, color='blue')
            
            # Export reference skeleton if available
            if reference is not None and show_reference and ref_joints is not None:
                ref_file = os.path.join(output_dir, f'{prefix}_ref_{t:04d}.obj')
                export_frame_obj(ref_joints[t], joint_parents, ref_file, color='red')
    
    print(f"Exported {T} frames to {output_dir}")


def export_mesh_obj(vertices, faces, output_file, color='blue', texture_file=None, uv_coords=None):
    """
    Export mesh vertices and faces as OBJ file with optional texture support.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices (0-indexed, will be converted to 1-indexed)
        output_file: Output file path
        color: Color name for material
        texture_file: Path to texture PNG file. If provided, texture will be applied.
        uv_coords: (V, 2) UV coordinates. If None and texture_file is provided, generates default UVs.
    """
    output_dir = os.path.dirname(output_file)
    output_basename = os.path.basename(output_file)
    obj_name = os.path.splitext(output_basename)[0]
    
    # Determine material file name
    mtl_name = f"{obj_name}.mtl"
    mtl_path = os.path.join(output_dir, mtl_name)
    
    # Generate UV coordinates if texture is provided but UVs are not
    has_texture = texture_file is not None and os.path.exists(texture_file)
    if has_texture and uv_coords is None:
        # Generate simple spherical UV mapping
        num_verts = len(vertices)
        uv_coords = np.zeros((num_verts, 2))
        # Simple spherical projection
        for i, v in enumerate(vertices):
            # Normalize vertex position
            norm = np.linalg.norm(v)
            if norm > 0:
                v_norm = v / norm
                # Spherical coordinates to UV
                u = 0.5 + np.arctan2(v_norm[2], v_norm[0]) / (2 * np.pi)
                v_coord = 0.5 - np.arcsin(v_norm[1]) / np.pi
                uv_coords[i] = [u, v_coord]
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# OBJ file exported from SMAL mesh\n")
        f.write(f"# Color: {color}\n")
        if has_texture:
            f.write(f"mtllib {mtl_name}\n")
            f.write(f"usemtl {obj_name}_material\n\n")
        else:
            f.write(f"mtllib {color}.mtl\n")
            f.write(f"usemtl {color}_material\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write texture coordinates if available
        if has_texture and uv_coords is not None:
            for uv in uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            f.write("\n")
        
        # Write faces (OBJ uses 1-indexed vertices)
        for face in faces:
            # Ensure indices are within valid range
            face_idx = face + 1  # Convert to 1-indexed
            if np.all(face_idx > 0) and np.all(face_idx <= len(vertices)):
                if has_texture and uv_coords is not None:
                    # Write faces with texture coordinates: f v1/vt1 v2/vt2 v3/vt3
                    f.write(f"f {face_idx[0]}/{face_idx[0]} {face_idx[1]}/{face_idx[1]} {face_idx[2]}/{face_idx[2]}\n")
                else:
                    f.write(f"f {face_idx[0]} {face_idx[1]} {face_idx[2]}\n")
    
    # Create MTL file
    if has_texture:
        # Copy texture file to output directory if it's not already there
        texture_basename = os.path.basename(texture_file)
        texture_dest = os.path.join(output_dir, texture_basename)
        if texture_file != texture_dest:
            import shutil
            shutil.copy2(texture_file, texture_dest)
        
        # Write MTL file
        with open(mtl_path, 'w') as f:
            f.write(f"# MTL file for {obj_name}\n")
            f.write(f"newmtl {obj_name}_material\n")
            f.write("Ns 96.078431\n")  # Specular exponent
            f.write("Ka 1.000000 1.000000 1.000000\n")  # Ambient color
            f.write("Kd 1.000000 1.000000 1.000000\n")  # Diffuse color
            f.write("Ks 0.500000 0.500000 0.500000\n")  # Specular color
            f.write("Ni 1.000000\n")  # Optical density
            f.write("d 1.000000\n")  # Dissolve (transparency)
            f.write("illum 2\n")  # Illumination model
            f.write(f"map_Kd {texture_basename}\n")  # Diffuse texture map
    else:
        # Create simple MTL file without texture
        with open(mtl_path, 'w') as f:
            f.write(f"# MTL file for {obj_name}\n")
            f.write(f"newmtl {color}_material\n")
            f.write("Ns 96.078431\n")
            if color == 'blue':
                f.write("Kd 0.000000 0.000000 1.000000\n")
            elif color == 'red':
                f.write("Kd 1.000000 0.000000 0.000000\n")
            else:
                f.write("Kd 0.800000 0.800000 0.800000\n")
            f.write("Ka 0.200000 0.200000 0.200000\n")
            f.write("Ks 0.500000 0.500000 0.500000\n")
            f.write("Ni 1.000000\n")
            f.write("d 1.000000\n")
            f.write("illum 2\n")


def export_frame_obj(joint_positions, joint_parents, output_file, color='blue'):
    """
    Export a single frame as OBJ file.
    
    Args:
        joint_positions: (J, 3) joint positions
        joint_parents: (J,) parent indices
        output_file: Output file path
        color: Color name for material
    """
    num_joints = len(joint_positions)
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# OBJ file exported from SMAL states\n")
        f.write(f"# Color: {color}\n")
        f.write(f"mtllib {color}.mtl\n")
        f.write(f"usemtl {color}_material\n\n")
        
        # Write vertices (joint positions)
        for j in range(num_joints):
            pos = joint_positions[j]
            f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        f.write("\n")
        
        # Write edges (bones) as lines
        # OBJ format doesn't directly support lines, so we use faces with very thin geometry
        # or we can use a custom format. For simplicity, we'll write comments for edges.
        f.write("# Edges (parent-child connections):\n")
        for j in range(num_joints):
            parent_idx = joint_parents[j]
            if parent_idx >= 0:
                # Write as a comment (Blender can import this with custom scripts)
                # Or we can create a simple line representation
                f.write(f"# Edge: {parent_idx+1} -> {j+1}\n")
        
        # Write faces for visualization (create small spheres at joints)
        # For each joint, create a small cube/sphere representation
        f.write("\n# Joint visualization (small cubes):\n")
        cube_size = 0.02
        for j in range(num_joints):
            pos = joint_positions[j]
            # Create a small cube around each joint
            v_base = j * 8 + 1  # Base vertex index for this cube
            
            # Cube vertices
            offsets = [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
            ]
            
            for offset in offsets:
                v_pos = pos + np.array(offset) * cube_size
                f.write(f"v {v_pos[0]:.6f} {v_pos[1]:.6f} {v_pos[2]:.6f}\n")
            
            # Cube faces
            f.write(f"f {v_base} {v_base+1} {v_base+2} {v_base+3}\n")  # bottom
            f.write(f"f {v_base+4} {v_base+7} {v_base+6} {v_base+5}\n")  # top
            f.write(f"f {v_base} {v_base+4} {v_base+5} {v_base+1}\n")  # front
            f.write(f"f {v_base+2} {v_base+6} {v_base+7} {v_base+3}\n")  # back
            f.write(f"f {v_base} {v_base+3} {v_base+7} {v_base+4}\n")  # left
            f.write(f"f {v_base+1} {v_base+5} {v_base+6} {v_base+2}\n")  # right


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_smal_states.py <npz_file> [output_dir]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Visualize
    print("Creating 3D animation...")
    anim = visualize_skeleton_3d(npz_path, output_path=output_dir + '.gif' if output_dir else None)
    
    # Export OBJ files
    if output_dir:
        print("Exporting OBJ files...")
        export_skeleton_obj(npz_path, output_dir + '_obj')

