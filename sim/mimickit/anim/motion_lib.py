import numpy as np
import os
import torch
import yaml

import anim.motion as motion
from util.logger import Logger
import util.torch_util as torch_util

def extract_pose_data(frame):
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof

def _variable_window_filter(x, window, reducer):
    """
    Apply `reducer` (e.g. np.median, np.mean) over a centered window of size
    `window`, SHRINKING the window near the two ends of the sequence rather
    than padding with repeated edge values. Edge-padding a centered filter
    means the window at the very first/last frame is dominated by copies of
    that frame's own value (e.g. window=5 at index 0 draws from
    [x0,x0,x0,x1,x2] with a plain edge pad), which makes the filter nearly a
    no-op exactly at the boundary - and the boundary (frame 0) is precisely
    where this filter needs to have an effect (a noisy/spiky reference
    velocity right at the reset frame). Using only the real, available
    neighbors on whichever side(s) exist avoids diluting the window with
    manufactured duplicate values.
    """
    n = x.shape[0]
    half = window // 2
    flat = x.reshape(n, -1)
    out_flat = np.zeros_like(flat, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out_flat[i, :] = reducer(flat[lo:hi, :], axis=0)
    return out_flat.reshape(x.shape).astype(x.dtype)


def _moving_average(x, window):
    return _variable_window_filter(x, window, np.mean)


def _median_filter_1d(x, window):
    return _variable_window_filter(x, window, np.median)


def denoise_reference_velocity(vel, method="median", window=5, clip_percentile=None):
    """
    Denoise a reference velocity array (root linear/angular or DOF velocity).
    Poses are never filtered directly.

    method:
      "median" (default) - centered median filter.
      "mean" - centered moving average.
      "clip" - clamp magnitudes to the clip's `clip_percentile` percentile.
    """
    if window is None or window <= 1:
        window = 5

    if method == "clip":
        pct = clip_percentile if clip_percentile is not None else 95.0
        mags = np.linalg.norm(vel.reshape(vel.shape[0], -1), axis=1)
        cap = np.percentile(mags, pct)
        scale = np.minimum(1.0, cap / np.maximum(mags, 1e-8))
        return (vel.reshape(vel.shape[0], -1) * scale[:, None]).reshape(vel.shape).astype(vel.dtype)
    elif method == "mean":
        return _moving_average(vel, window)
    else:  # "median", the default
        return _median_filter_1d(vel, window)

class MotionLib():
    def __init__(self, motion_file, kin_char_model, device, ground_preprocessing=False, ground_height=0.0,
                 motion_vel_denoise=None, motion_native_fps=None,
                 ground_contact_mode="first_frame", ground_contact_pad=0.1, ground_contact_smooth_window=1):
        self._device = device
        self._kin_char_model = kin_char_model
        self._ground_preprocessing = ground_preprocessing
        self._ground_height = ground_height
        # ground_contact_mode: "first_frame" shifts the whole clip by one z-offset from frame 0;
        # "per_frame" computes a median-smoothed offset per frame (ground_contact_smooth_window).
        # ground_contact_pad: extra clearance added after grounding (m).
        self._ground_contact_mode = ground_contact_mode
        self._ground_contact_pad = ground_contact_pad
        self._ground_contact_smooth_window = ground_contact_smooth_window
        # Optional reference-velocity denoising, e.g. {"method": "median", "window": 5}.
        self._motion_vel_denoise = motion_vel_denoise
        # True capture rate of motion files whose frames are denser than their declared fps
        # (e.g. 120 for PFERD); frames are decimated to the declared fps at load time.
        self._motion_native_fps = motion_native_fps
        self._load_motions(motion_file)
        return

    def get_num_motions(self):
        return self._motion_lengths.shape[0]

    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
    
    def get_motion_loop_mode(self, motion_ids):
        return self._motion_loop_modes[motion_ids]
    
    def calc_motion_phase(self, motion_ids, times):
        motion_len = self._motion_lengths[motion_ids]
        loop_mode = self._motion_loop_modes[motion_ids]
        phase = calc_phase(times=times, motion_len=motion_len, loop_mode=loop_mode)
        return phase

    def calc_motion_frame(self, motion_ids, motion_times):
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._frame_root_pos[frame_idx0]
        root_pos1 = self._frame_root_pos[frame_idx1]
        
        root_rot0 = self._frame_root_rot[frame_idx0]
        root_rot1 = self._frame_root_rot[frame_idx1]

        root_vel = self._frame_root_vel[frame_idx0]
        root_ang_vel = self._frame_root_ang_vel[frame_idx0]

        joint_rot0 = self._frame_joint_rot[frame_idx0]
        joint_rot1 = self._frame_joint_rot[frame_idx1]

        dof_vel = self._frame_dof_vel[frame_idx0]

        blend_unsq = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsq) * root_pos0 + blend_unsq * root_pos1
        root_rot = torch_util.slerp(root_rot0, root_rot1, blend)
        
        joint_rot = torch_util.slerp(joint_rot0, joint_rot1, blend_unsq)

        root_pos_offset = self._calc_loop_offset(motion_ids, motion_times)
        root_pos += root_pos_offset

        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel

    def joint_rot_to_dof(self, joint_rot):
        joint_dof = self._kin_char_model.rot_to_dof(joint_rot)
        return joint_dof

    def get_motion_lengths(self):
        return self._motion_lengths

    def get_motion_weights(self):
        return self._motion_weights
    
    def _preprocess_ground_contact(self, root_pos, root_rot, joint_rot):
        """
        Preprocess root positions to ensure proper ground contact.
        "first_frame" mode (legacy): a single z-offset computed from frame
        0's lowest body point, applied as a constant shift to the whole clip.
        "per_frame" mode: an independent, smoothed z-offset per frame - see
        __init__'s docstring comment for why (frame-0-only drifts into real
        ground penetration for a majority of kinematic-derived clips).
        """
        if not self._ground_preprocessing:
            return root_pos

        pad = self._ground_contact_pad

        if self._ground_contact_mode == "per_frame":
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            min_z = torch.min(body_pos[..., 2], dim=-1).values  # (num_frames,)

            window = self._ground_contact_smooth_window
            if window is not None and window > 1:
                min_z_np = min_z.detach().cpu().numpy()
                min_z_np = _median_filter_1d(min_z_np, window)
                min_z = torch.tensor(min_z_np, dtype=root_pos.dtype, device=root_pos.device)

            z_offset = self._ground_height - min_z + pad  # (num_frames,)
            adjusted_root_pos = root_pos.clone()
            adjusted_root_pos[..., 2] += z_offset
            return adjusted_root_pos

        # legacy "first_frame" behavior
        first_frame_root_pos = root_pos[0:1]  # First frame root position
        first_frame_joint_rot = joint_rot[0:1]  # First frame joint rotations
        first_frame_root_rot = root_rot[0:1]
        # Use forward kinematics to get all body positions for the first frame
        body_pos, _ = self._kin_char_model.forward_kinematics(
            first_frame_root_pos,
            first_frame_root_rot,
            first_frame_joint_rot
        )

        # Find the minimum z-coordinate (lowest point) of all body parts
        min_z = torch.min(body_pos[..., 2])

        # Calculate the offset needed to place the lowest point at ground height
        z_offset = self._ground_height - min_z

        # Apply the offset to all root positions
        adjusted_root_pos = root_pos.clone()
        adjusted_root_pos[..., 2] += z_offset + pad

        return adjusted_root_pos
    
    def _extract_frame_data(self, frame):
        root_pos, root_rot, joint_dof = extract_pose_data(frame)
        root_pos = torch.tensor(root_pos, dtype=torch.float32, device=self._device)
        root_rot = torch.tensor(root_rot, dtype=torch.float32, device=self._device)
        joint_dof = torch.tensor(joint_dof, dtype=torch.float32, device=self._device)

        root_rot_quat = torch_util.exp_map_to_quat(root_rot)

        joint_rot = self._kin_char_model.dof_to_rot(joint_dof)
        joint_rot = torch_util.quat_pos(joint_rot)

        # Apply ground preprocessing if enabled
        root_pos = self._preprocess_ground_contact(root_pos, root_rot_quat, joint_rot)

        return root_pos, root_rot_quat, joint_rot

    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        frame_start_idx = self._motion_start_idx[motion_ids]
        
        phase = self.calc_motion_phase(motion_ids, times)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0
        
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx

        return frame_idx0, frame_idx1, blend
    
    def _calc_loop_offset(self, motion_ids, times):
        loop_mode = self._motion_loop_modes[motion_ids]
        wrap_mask = (loop_mode == motion.LoopMode.WRAP.value)

        wrap_motion_ids = motion_ids[wrap_mask]
        times = times[wrap_mask]

        motion_len = self._motion_lengths[wrap_motion_ids]
        root_pos_deltas = self._motion_root_pos_delta[wrap_motion_ids]

        phase = times / motion_len
        phase = torch.floor(phase)
        phase = phase.unsqueeze(-1)
        
        root_pos_offset = torch.zeros((motion_ids.shape[0], 3), device=self._device)
        root_pos_offset[wrap_mask] = phase * root_pos_deltas

        return root_pos_offset
    

    def _load_motions(self, motion_file):
        self._load_motion_pkl(motion_file)
        
        num_motions = self.get_num_motions()
        total_len = self.get_total_length()
        Logger.print("Loaded motion file: {}".format(motion_file))
        Logger.print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))
        return

    def _load_motion_pkl(self, motion_file):
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_loop_modes = []
        self._motion_root_pos_delta = []
        self._motion_files = []
        
        self._motion_frames = []
        self._frame_root_pos = []
        self._frame_root_rot = []
        self._frame_root_vel = []
        self._frame_root_ang_vel = []
        self._frame_joint_rot = []
        self._frame_dof_vel = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            Logger.print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            
            curr_motion = motion.load_motion(curr_file)
            fps = curr_motion.fps
            loop_mode = curr_motion.loop_mode
            frames = curr_motion.frames
            curr_weight = motion_weights[f]
            if len(frames) < 1:
                print("Warning: Motion file {:s} has no frames".format(curr_file), len(frames))
                continue

            if self._motion_native_fps is not None and self._motion_native_fps > fps:
                stride = max(1, round(self._motion_native_fps / fps))
                frames = frames[::stride]
                Logger.print("  Resampled {:s}: native {:g}fps -> {:g}fps (stride {:d}), {:d} frames".format(
                    curr_file, self._motion_native_fps, fps, stride, frames.shape[0]))

            loop_mode = loop_mode.value
            dt = 1.0 / fps

            num_frames = frames.shape[0]
            curr_len = 1.0 / fps * (num_frames - 1)

            root_pos, root_rot, joint_rot = self._extract_frame_data(frames)
            root_pos_delta = root_pos[-1] - root_pos[0]
            root_pos_delta[..., -1] = 0.0

            root_vel = torch.zeros_like(root_pos)
            root_vel[..., :-1, :] = fps * (root_pos[..., 1:, :] - root_pos[..., :-1, :])
            root_vel[..., -1, :] = root_vel[..., -2, :]
                
            root_ang_vel = torch.zeros_like(root_pos)
            root_drot = torch_util.quat_diff(root_rot[..., :-1, :], root_rot[..., 1:, :])
            root_ang_vel[..., :-1, :] = fps * torch_util.quat_to_exp_map(root_drot)
            root_ang_vel[..., -1, :] = root_ang_vel[..., -2, :]

            dof_vel = self._kin_char_model.compute_frame_dof_vel(joint_rot, dt)

            if self._motion_vel_denoise is not None:
                opts = self._motion_vel_denoise
                method = opts.get("method", "median")
                window = opts.get("window", 5)
                clip_percentile = opts.get("clip_percentile", None)

                def _denoise(t):
                    arr = t.detach().cpu().numpy()
                    arr = denoise_reference_velocity(arr, method=method, window=window, clip_percentile=clip_percentile)
                    return torch.tensor(arr, dtype=t.dtype, device=t.device)

                root_vel = _denoise(root_vel)
                root_ang_vel = _denoise(root_ang_vel)
                dof_vel = _denoise(dof_vel)
                Logger.print("  Applied motion_vel_denoise ({:s}, window={}) to {:s}".format(method, window, curr_file))

            self._motion_weights.append(curr_weight)
            self._motion_fps.append(fps)
            self._motion_dt.append(dt)
            self._motion_num_frames.append(num_frames)
            self._motion_lengths.append(curr_len)
            self._motion_loop_modes.append(loop_mode)
            self._motion_root_pos_delta.append(root_pos_delta)
            self._motion_files.append(curr_file)
                
            self._motion_frames.append(frames)
            self._frame_root_pos.append(root_pos)
            self._frame_root_rot.append(root_rot)
            self._frame_root_vel.append(root_vel)
            self._frame_root_ang_vel.append(root_ang_vel)
            self._frame_joint_rot.append(joint_rot)
            self._frame_dof_vel.append(dof_vel)


        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_loop_modes = torch.tensor(self._motion_loop_modes, dtype=torch.int, device=self._device)
        
        self._motion_root_pos_delta = torch.stack(self._motion_root_pos_delta, dim=0)
        
        self._motion_frames = np.concatenate(self._motion_frames, axis=0)
        self._motion_frames = torch.tensor(self._motion_frames, dtype=torch.float32, device=self._device)
        
        self._frame_root_pos = torch.cat(self._frame_root_pos, dim=0)
        self._frame_root_rot = torch.cat(self._frame_root_rot, dim=0)
        self._frame_root_vel = torch.cat(self._frame_root_vel, dim=0)
        self._frame_root_ang_vel = torch.cat(self._frame_root_ang_vel, dim=0)
        self._frame_joint_rot = torch.cat(self._frame_joint_rot, dim=0)
        self._frame_dof_vel = torch.cat(self._frame_dof_vel, dim=0)
        
        num_motions = self.get_num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        
        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            motion_files = []
            motion_weights = []

            with open(motion_file, 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights


@torch.jit.script
def calc_phase(times, motion_len, loop_mode):
    phase = times / motion_len
        
    loop_wrap_mask = (loop_mode == motion.LoopMode.WRAP.value)
    phase_wrap = phase[loop_wrap_mask]
    phase_wrap = phase_wrap - torch.floor(phase_wrap)
    phase[loop_wrap_mask] = phase_wrap
        
    phase = torch.clip(phase, 0.0, 1.0)

    return phase
