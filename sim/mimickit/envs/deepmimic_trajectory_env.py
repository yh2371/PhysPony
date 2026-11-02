import numpy as np
import torch

import anim.motion as motion
import anim.motion_lib as motion_lib
import envs.base_env as base_env
import envs.char_env as char_env
import envs.deepmimic_env as deepmimic_env
import util.torch_util as torch_util


class DeepMimicTrajectoryEnv(deepmimic_env.DeepMimicEnv):
    """
    DeepMimic environment with soft trajectory following reward.
    
    This environment extends DeepMimicEnv to add rewards that encourage
    following the overall trajectory trend/curvature without requiring
    exact position matching. This is useful when the reference trajectory
    from SLAM may be noisy but has reasonable overall trends.
    """
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        env_config = config["env"]

        # Trajectory following parameters
        self._trajectory_reward_enabled = env_config.get("trajectory_reward_enabled", True)
        self._trajectory_reward_weight = env_config.get("trajectory_reward_weight", 0.1)
        self._trajectory_heading_weight = env_config.get("trajectory_heading_weight", 0.4)
        self._trajectory_speed_weight = env_config.get("trajectory_speed_weight", 0.3)
        self._trajectory_trend_weight = env_config.get("trajectory_trend_weight", 0.3)
        
        # Smoothing parameters for trajectory following
        self._trajectory_lookahead_time = env_config.get("trajectory_lookahead_time", 0.2)  # seconds
        self._trajectory_smoothing_alpha = env_config.get("trajectory_smoothing_alpha", 0.7)
        self._trajectory_position_tolerance = env_config.get("trajectory_position_tolerance", 0.5)
        self._trajectory_heading_scale = env_config.get("trajectory_heading_scale", 5.0)
        self._trajectory_speed_scale = env_config.get("trajectory_speed_scale", 2.0)
        self._trajectory_trend_scale = env_config.get("trajectory_trend_scale", 3.0)
        
        # Store previous positions and velocities for trajectory smoothing
        self._prev_ref_positions = None
        self._prev_curr_positions = None
        self._prev_ref_velocities = None
        self._smoothed_ref_velocity = None
        self._smoothed_curr_velocity = None
        
        # Initialize parent class
        super().__init__(config=config, num_envs=num_envs, device=device, visualize=visualize, record_video=record_video)

        return
    
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        num_envs = self.get_num_envs()
        
        # Initialize trajectory smoothing buffers
        self._prev_ref_positions = torch.zeros(num_envs, 3, device=self._device, dtype=torch.float32)
        self._prev_curr_positions = torch.zeros(num_envs, 3, device=self._device, dtype=torch.float32)
        self._prev_ref_velocities = torch.zeros(num_envs, 3, device=self._device, dtype=torch.float32)
        self._smoothed_ref_velocity = torch.zeros(num_envs, 3, device=self._device, dtype=torch.float32)
        self._smoothed_curr_velocity = torch.zeros(num_envs, 3, device=self._device, dtype=torch.float32)
        
        return
    
    def _reset_char(self, env_ids):
        super()._reset_char(env_ids)
        
        # Reset trajectory smoothing buffers for reset environments
        if len(env_ids) > 0:
            self._prev_ref_positions[env_ids] = self._ref_root_pos[env_ids].clone()
            self._prev_curr_positions[env_ids] = self._ref_root_pos[env_ids].clone()
            self._prev_ref_velocities[env_ids] = self._ref_root_vel[env_ids].clone()
            self._smoothed_ref_velocity[env_ids] = self._ref_root_vel[env_ids].clone()
            self._smoothed_curr_velocity[env_ids] = self._ref_root_vel[env_ids].clone()
        
        return
    
    def _update_reward(self):
        # Get base reward from parent class
        super()._update_reward()
        
        # Add trajectory following reward if enabled
        if self._trajectory_reward_enabled and self._time_buf.min() > 0:  # Skip first frame
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)
            root_vel = self._engine.get_root_vel(char_id)
            root_rot = self._engine.get_root_rot(char_id)
            
            ref_root_pos = self._ref_root_pos
            ref_root_vel = self._ref_root_vel
            ref_root_rot = self._ref_root_rot
            
            # Get future reference positions for trend following
            future_ref_pos, has_future = self._get_future_ref_positions()
            
            # Update smoothing buffers (exponential moving average)
            alpha = self._trajectory_smoothing_alpha
            self._smoothed_ref_velocity = alpha * self._smoothed_ref_velocity + (1 - alpha) * ref_root_vel
            self._smoothed_curr_velocity = alpha * self._smoothed_curr_velocity + (1 - alpha) * root_vel
            
            # Compute trajectory following reward
            trajectory_reward = compute_trajectory_reward(
                curr_pos=root_pos,
                curr_vel=root_vel,
                ref_pos=ref_root_pos,
                ref_vel=ref_root_vel,
                smoothed_ref_vel=self._smoothed_ref_velocity,
                smoothed_curr_vel=self._smoothed_curr_velocity,
                future_ref_pos=future_ref_pos,
                has_future=has_future,
                position_tolerance=self._trajectory_position_tolerance,
                heading_weight=self._trajectory_heading_weight,
                speed_weight=self._trajectory_speed_weight,
                trend_weight=self._trajectory_trend_weight,
                heading_scale=self._trajectory_heading_scale,
                speed_scale=self._trajectory_speed_scale,
                trend_scale=self._trajectory_trend_scale
            )
            
            # Update position buffers for next iteration
            self._prev_ref_positions = ref_root_pos.clone()
            self._prev_curr_positions = root_pos.clone()
            self._prev_ref_velocities = ref_root_vel.clone()
            
            # Add trajectory reward to total reward
            self._reward_buf[:] += self._trajectory_reward_weight * trajectory_reward
        
        return
    
    def _get_future_ref_positions(self):
        """Get future reference positions from motion library for trend following."""
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        lookahead_time = self._trajectory_lookahead_time
        
        # Compute future motion times
        future_motion_times = motion_times + lookahead_time
        
        # Get motion lengths and loop modes to determine validity
        motion_len = self._motion_lib.get_motion_length(motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(motion_ids)
        
        # Check which environments have valid future times
        # For WRAP mode (value=1), future times are always valid (wrapped)
        # For CLAMP mode (value=0), future times beyond motion length are invalid
        valid_mask = (future_motion_times < motion_len) | (motion_loop_mode == 1)
        
        # Get future positions from motion library
        # calc_motion_frame handles wrapping automatically for WRAP mode motions
        try:
            future_root_pos, _, _, _, _, _ = self._motion_lib.calc_motion_frame(motion_ids, future_motion_times)
            # For CLAMP mode motions beyond length, extrapolate from current position and velocity
            # The motion library will clamp to the last frame, so we extrapolate for better trend following
            future_root_pos = torch.where(
                valid_mask.unsqueeze(-1),
                future_root_pos,
                self._ref_root_pos + self._smoothed_ref_velocity * lookahead_time
            )
        except Exception as e:
            # If lookup fails for any reason, extrapolate from current velocity
            future_root_pos = self._ref_root_pos + self._smoothed_ref_velocity * lookahead_time
            valid_mask = torch.zeros_like(motion_ids, dtype=torch.bool)
        
        return future_root_pos, valid_mask


@torch.jit.script
def compute_trajectory_reward(curr_pos, curr_vel, ref_pos, ref_vel,
                              smoothed_ref_vel, smoothed_curr_vel, future_ref_pos, has_future,
                              position_tolerance: float, heading_weight: float, 
                              speed_weight: float, trend_weight: float,
                              heading_scale: float, speed_scale: float, trend_scale: float):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float) -> Tensor
    """
    Compute soft trajectory following reward.
    
    This reward encourages:
    1. Heading alignment: Matching the direction of travel (velocity direction)
    2. Speed matching: Matching the speed along the trajectory
    3. Trend following: Following the overall trajectory trend using lookahead positions
    
    All rewards use soft/exponential decay to tolerate noise in the reference trajectory.
    """
    num_envs = curr_pos.shape[0]
    
    # 1. Heading/Velocity Direction Alignment
    # Compute speed and normalized direction vectors
    curr_speed = torch.linalg.vector_norm(curr_vel, dim=-1)
    ref_speed = torch.linalg.vector_norm(ref_vel, dim=-1)
    
    # Normalize velocities to get direction vectors (handle zero velocity)
    curr_vel_normalized = curr_vel / (curr_speed.unsqueeze(-1) + 1e-6)
    ref_vel_normalized = ref_vel / (ref_speed.unsqueeze(-1) + 1e-6)
    
    # Compute cosine similarity between velocity directions
    heading_alignment = torch.sum(curr_vel_normalized * ref_vel_normalized, dim=-1)
    heading_alignment = torch.clamp(heading_alignment, -1.0, 1.0)
    # Convert to [0, 1] and apply exponential reward
    heading_reward = (heading_alignment + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
    heading_reward = torch.exp(-heading_scale * (1.0 - heading_reward))
    
    # Handle zero velocity cases
    zero_vel_mask = (curr_speed < 1e-3) & (ref_speed < 1e-3)
    heading_reward = torch.where(zero_vel_mask, torch.ones_like(heading_reward), heading_reward)
    
    # 2. Speed Matching (with tolerance for noise)
    # Use exponential decay for speed difference
    speed_diff = torch.abs(curr_speed - ref_speed)
    speed_reward = torch.exp(-speed_scale * speed_diff)
    
    # 3. Trend Following (using lookahead positions)
    # Compute desired direction from current position to future reference position
    # Direction from current to future reference position
    desired_direction = future_ref_pos - curr_pos
    desired_direction_norm = torch.linalg.vector_norm(desired_direction, dim=-1, keepdim=True)
    desired_direction = desired_direction / (desired_direction_norm + 1e-6)
    
    # Reward aligning current velocity with desired direction
    trend_alignment = torch.sum(curr_vel_normalized * desired_direction, dim=-1)
    trend_alignment = torch.clamp(trend_alignment, -1.0, 1.0)
    trend_reward_lookahead = (trend_alignment + 1.0) / 2.0  # Map to [0, 1]
    trend_reward_lookahead = torch.exp(-trend_scale * (1.0 - trend_reward_lookahead))
    
    # Fallback: use smoothed velocity direction matching for environments without valid future
    smoothed_ref_vel_norm = torch.linalg.vector_norm(smoothed_ref_vel, dim=-1, keepdim=True)
    smoothed_ref_vel_dir = smoothed_ref_vel / (smoothed_ref_vel_norm + 1e-6)
    trend_alignment_fallback = torch.sum(curr_vel_normalized * smoothed_ref_vel_dir, dim=-1)
    trend_alignment_fallback = torch.clamp(trend_alignment_fallback, -1.0, 1.0)
    trend_reward_fallback = (trend_alignment_fallback + 1.0) / 2.0
    trend_reward_fallback = torch.exp(-trend_scale * (1.0 - trend_reward_fallback))
    
    # Use lookahead reward where valid, fallback otherwise
    # Also handle cases where desired direction is undefined or velocity is zero
    invalid_trend_mask = (desired_direction_norm.squeeze(-1) < 1e-3) | (curr_speed < 1e-3)
    trend_reward_lookahead = torch.where(invalid_trend_mask, 
                                         torch.full_like(trend_reward_lookahead, 0.5),
                                         trend_reward_lookahead)
    trend_reward = torch.where(has_future,
                               trend_reward_lookahead,
                               trend_reward_fallback)
    
    # 4. Soft Position Matching (with tolerance)
    # Compute position error but use exponential decay with tolerance
    # This provides soft guidance without strict matching
    pos_diff = ref_pos - curr_pos
    pos_error = torch.linalg.vector_norm(pos_diff, dim=-1)
    # Soft reward: full reward if within tolerance, exponential decay beyond
    position_reward = torch.exp(-trend_scale * torch.clamp(pos_error - position_tolerance, min=0.0))
    
    # Combine rewards with normalized weights
    # Normalize weights to sum to 1.0
    total_weight = heading_weight + speed_weight + trend_weight
    if total_weight > 1e-6:
        heading_weight_norm = heading_weight / total_weight
        speed_weight_norm = speed_weight / total_weight
        trend_weight_norm = trend_weight / total_weight
    else:
        heading_weight_norm = 0.4
        speed_weight_norm = 0.3
        trend_weight_norm = 0.3
    
    # Combined trajectory reward
    trajectory_reward = (heading_weight_norm * heading_reward +
                        speed_weight_norm * speed_reward +
                        trend_weight_norm * trend_reward)
    
    # Add small position reward component (weighted lower since it's more about exact matching)
    trajectory_reward = 0.8 * trajectory_reward + 0.2 * position_reward
    
    return trajectory_reward

