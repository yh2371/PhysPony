import gym.spaces as spaces
import numpy as np
import torch

import anim.motion as motion
import anim.motion_lib as motion_lib
import envs.base_env as base_env
import envs.char_env as char_env
import envs.deepmimic_trajectory_env as deepmimic_trajectory_env
import envs.amp_env as amp_env
import util.circular_buffer as circular_buffer
import util.torch_util as torch_util


class DeepMimicTrajectoryAMPEnv(deepmimic_trajectory_env.DeepMimicTrajectoryEnv):
    """
    DeepMimic environment with soft trajectory following reward and AMP regularization.
    
    This environment extends DeepMimicTrajectoryEnv to add AMP discriminator observations
    that use a separate dataset of clean mocap motions as a prior, while the tracking
    reference uses a single (potentially noisy) sequence.
    
    Key features:
    - Trajectory following reward (from DeepMimicTrajectoryEnv)
    - AMP discriminator using clean mocap dataset (prior_motion_file)
    - Tracking reference using single sequence (tracking_motion_file)
    """
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        env_config = config["env"]

        # AMP discriminator parameters
        self._num_disc_obs_steps = env_config.get("num_disc_obs_steps", 1)
        
        # Motion file configuration
        # tracking_motion_file: single sequence for tracking reference (may be noisy)
        # prior_motion_file: dataset of clean mocap motions for AMP discriminator
        self._tracking_motion_file = env_config.get("tracking_motion_file", None)
        self._prior_motion_file = env_config.get("prior_motion_file", None)
        
        # If not specified, fall back to motion_file for backward compatibility
        if self._tracking_motion_file is None:
            self._tracking_motion_file = env_config.get("motion_file", None)
        if self._prior_motion_file is None:
            self._prior_motion_file = env_config.get("motion_file", None)
        
        # Initialize parent class (this will load tracking_motion_file as motion_file)
        # We'll override the motion loading to use tracking_motion_file
        super().__init__(config=config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)

        return
    
    def _load_motions(self, motion_file, config=None):
        """
        Load tracking motion file for reference tracking.
        The prior motion file for AMP will be loaded separately.
        """
        # Use tracking_motion_file if specified, otherwise use motion_file
        tracking_file = self._tracking_motion_file if self._tracking_motion_file else motion_file

        # Get ground preprocessing parameters from config
        ground_preprocessing = True
        ground_height = 0.0
        motion_vel_denoise = None
        motion_native_fps = None
        if config is not None:
            env_config = config.get("env", {})
            ground_preprocessing = env_config.get("ground_preprocessing", True)
            ground_height = env_config.get("ground_height", 0.0)
            motion_vel_denoise = env_config.get("motion_vel_denoise", None)
            motion_native_fps = env_config.get("motion_native_fps", None)
        elif hasattr(self, '_config'):
            env_config = self._config.get("env", {})
            ground_preprocessing = env_config.get("ground_preprocessing", True)
            ground_height = env_config.get("ground_height", 0.0)
            motion_vel_denoise = env_config.get("motion_vel_denoise", None)
            motion_native_fps = env_config.get("motion_native_fps", None)

        # Load tracking motion library (for reference tracking)
        self._motion_lib = motion_lib.MotionLib(
            motion_file=tracking_file,
            kin_char_model=self._kin_char_model,
            device=self._device,
            ground_preprocessing=ground_preprocessing,
            ground_height=ground_height,
            motion_vel_denoise=motion_vel_denoise,
            motion_native_fps=motion_native_fps
        )
        
        # Load prior motion library (for AMP discriminator)
        prior_file = self._prior_motion_file if self._prior_motion_file else tracking_file
        self._prior_motion_lib = motion_lib.MotionLib(
            motion_file=prior_file,
            kin_char_model=self._kin_char_model,
            device=self._device,
            ground_preprocessing=ground_preprocessing,
            ground_height=ground_height
        )
        
        return
    
    def get_disc_obs_space(self):
        """Get discriminator observation space (same as AMPEnv)."""
        disc_obs = self.fetch_disc_obs_demo(1)
        disc_obs_shape = list(disc_obs.shape[1:])
        disc_obs_dtype = torch_util.torch_dtype_to_numpy(disc_obs.dtype)
        disc_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=disc_obs_shape,
            dtype=disc_obs_dtype,
        )
        return disc_obs_space
    
    def fetch_disc_obs_demo(self, num_samples):
        """
        Fetch discriminator observations from the prior motion library (clean mocap).
        This uses the prior_motion_file dataset instead of the tracking motion.
        """
        motion_ids, motion_times0 = self._sample_prior_motion_times(num_samples)
        disc_obs = self._compute_disc_obs_demo(motion_ids, motion_times0)
        return disc_obs
    
    def _sample_prior_motion_times(self, n):
        """
        Sample motion times from the prior motion library (clean mocap dataset).
        """
        motion_ids = self._prior_motion_lib.sample_motions(n)
        
        if (self._rand_reset):
            motion_times = self._prior_motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float, device=self._device)
        
        return motion_ids, motion_times
    
    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        """
        Compute discriminator observations from prior motion library.
        Uses the same logic as AMPEnv but with prior_motion_lib.
        """
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = self._fetch_disc_demo_data(motion_ids, motion_times0)
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]

        disc_obs = amp_env.compute_disc_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=root_pos,
            root_rot=root_rot, 
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_obs=self._global_obs,
            root_height_obs=self._root_height_obs
        )
        return disc_obs
    
    def _fetch_disc_demo_data(self, motion_ids, motion_times0):
        """
        Fetch demonstration data from prior motion library for discriminator.
        Uses prior_motion_lib instead of motion_lib.
        """
        num_samples = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self._engine.get_timestep() * torch.arange(0, self._num_disc_obs_steps, device=self._device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        
        # Use prior_motion_lib instead of motion_lib
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._prior_motion_lib.calc_motion_frame(motion_ids, motion_times)

        if (self._has_key_bodies()):
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        root_pos = torch.reshape(root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]])
        root_rot = torch.reshape(root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]])
        root_vel = torch.reshape(root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]])
        key_pos = torch.reshape(key_pos, shape=[num_samples, self._num_disc_obs_steps, key_pos.shape[-2], key_pos.shape[-1]])
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos
    
    def _build_data_buffers(self):
        """Build data buffers including AMP discriminator observation buffers."""
        super()._build_data_buffers()
        self._build_disc_obs_buffers()
        self._info["disc_obs"] = self._disc_obs_buf
        return
    
    def _build_disc_obs_buffers(self):
        """Build circular buffers for discriminator observations (same as AMPEnv)."""
        num_envs = self.get_num_envs()
        n = self._num_disc_obs_steps
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_rot = self._engine.get_body_rot(char_id)

        self._disc_hist_root_pos = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n, 
            shape=root_pos.shape[1:],
            dtype=root_pos.dtype,
            device=self._device
        )

        self._disc_hist_root_rot = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_rot.shape[1:],
            dtype=root_rot.dtype,
            device=self._device
        )

        self._disc_hist_root_vel = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_vel.shape[1:],
            dtype=root_vel.dtype,
            device=self._device
        )
        
        self._disc_hist_root_ang_vel = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=root_ang_vel.shape[1:],
            dtype=root_ang_vel.dtype,
            device=self._device
        )
        
        self._disc_hist_joint_rot = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=body_rot[..., 1:, :].shape[1:],
            dtype=body_rot.dtype,
            device=self._device
        )

        self._disc_hist_dof_vel = circular_buffer.CircularBuffer(
            batch_size=num_envs,
            buffer_len=n,
            shape=dof_vel.shape[1:],
            dtype=dof_vel.dtype,
            device=self._device
        )
        
        if (self._has_key_bodies()):
            num_key_bodies = len(self._key_body_ids)
            self._disc_hist_key_pos = circular_buffer.CircularBuffer(
                batch_size=num_envs,
                buffer_len=n,
                shape=[num_key_bodies, 3],
                dtype=root_pos.dtype,
                device=self._device
            )
        
        disc_obs_space = self.get_disc_obs_space()
        disc_obs_dtype = torch_util.numpy_dtype_to_torch(disc_obs_space.dtype)
        self._disc_obs_buf = torch.zeros(
            [num_envs] + list(disc_obs_space.shape),
            device=self._device,
            dtype=disc_obs_dtype
        )

        return
    
    def _update_misc(self):
        """Update miscellaneous data including discriminator history."""
        super()._update_misc()
        self._update_disc_hist()
        return
    
    def _update_disc_hist(self):
        """Update discriminator observation history buffers."""
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        self._disc_hist_root_pos.push(root_pos)
        self._disc_hist_root_rot.push(root_rot)
        self._disc_hist_root_vel.push(root_vel)
        self._disc_hist_root_ang_vel.push(root_ang_vel)
        self._disc_hist_dof_vel.push(dof_vel)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        self._disc_hist_joint_rot.push(joint_rot)

        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
            self._disc_hist_key_pos.push(key_pos)

        return
    
    def _update_observations(self, env_ids=None):
        """Update observations including discriminator observations."""
        super()._update_observations(env_ids)

        if (env_ids is None or len(env_ids) > 0):
            self._update_disc_obs(env_ids)
        return
    
    def _update_disc_obs(self, env_ids=None):
        """Update discriminator observations from history buffers."""
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        
        if (self._has_key_bodies()):
            key_pos = self._disc_hist_key_pos.get_all()
        else:
            key_pos = torch.zeros([0], device=self._device)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            
            if (self._has_key_bodies()):
                key_pos = key_pos[env_ids]
        
        if (self._track_global_root()):
            ref_root_pos = torch.zeros_like(root_pos[..., -1, :])
            ref_root_rot = torch.zeros_like(root_rot[..., -1, :])
            ref_root_rot[..., -1] = 1
        else:
            ref_root_pos = root_pos[..., -1, :]
            ref_root_rot = root_rot[..., -1, :]

        disc_obs = amp_env.compute_disc_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=root_pos,
            root_rot=root_rot, 
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_obs=self._global_obs,
            root_height_obs=self._root_height_obs
        )

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

        return
    
    def _reset_envs(self, env_ids):
        """Reset environments including discriminator history."""
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_disc_hist(env_ids)
        return
    
    def _reset_disc_hist(self, env_ids):
        """Reset discriminator history buffers for reset environments."""
        # Sample from tracking motion (not prior) for initial state
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = self._get_motion_times(env_ids)
        
        # Fetch demo data from tracking motion for initial history
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos = self._fetch_disc_demo_data_from_tracking(motion_ids, motion_times0)
        
        self._disc_hist_root_pos.fill(env_ids, root_pos)
        self._disc_hist_root_rot.fill(env_ids, root_rot)
        self._disc_hist_root_vel.fill(env_ids, root_vel)
        self._disc_hist_root_ang_vel.fill(env_ids, root_ang_vel)
        self._disc_hist_joint_rot.fill(env_ids, joint_rot)
        self._disc_hist_dof_vel.fill(env_ids, dof_vel)

        if (self._has_key_bodies()):
            self._disc_hist_key_pos.fill(env_ids, key_pos)

        return
    
    def _fetch_disc_demo_data_from_tracking(self, motion_ids, motion_times0):
        """
        Fetch demonstration data from tracking motion library (for initial history).
        This is used during reset to initialize history from the tracking sequence.
        """
        num_samples = motion_ids.shape[0]

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_disc_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -self._engine.get_timestep() * torch.arange(0, self._num_disc_obs_steps, device=self._device)
        time_steps = torch.flip(time_steps, dims=[0])
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        
        # Use tracking motion_lib (not prior_motion_lib) for initial history
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        if (self._has_key_bodies()):
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        root_pos = torch.reshape(root_pos, shape=[num_samples, self._num_disc_obs_steps, root_pos.shape[-1]])
        root_rot = torch.reshape(root_rot, shape=[num_samples, self._num_disc_obs_steps, root_rot.shape[-1]])
        root_vel = torch.reshape(root_vel, shape=[num_samples, self._num_disc_obs_steps, root_vel.shape[-1]])
        root_ang_vel = torch.reshape(root_ang_vel, shape=[num_samples, self._num_disc_obs_steps, root_ang_vel.shape[-1]])
        joint_rot = torch.reshape(joint_rot, shape=[num_samples, self._num_disc_obs_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        dof_vel = torch.reshape(dof_vel, shape=[num_samples, self._num_disc_obs_steps, dof_vel.shape[-1]])
        key_pos = torch.reshape(key_pos, shape=[num_samples, self._num_disc_obs_steps, key_pos.shape[-2], key_pos.shape[-1]])
        
        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos

