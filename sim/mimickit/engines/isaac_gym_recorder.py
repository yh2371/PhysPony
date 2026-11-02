from __future__ import annotations

import isaacgym.gymapi as gymapi
import numpy as np

import engines.video_recorder as video_recorder
import engines.isaac_gym_engine as isaac_gym_engine


class IsaacGymVideoRecorder(video_recorder.VideoRecorder):
    def __init__(self, 
                 engine: isaac_gym_engine.IsaacGymEngine, 
                 resolution: tuple[int, int] = (854, 480), 
                 cam_pos: np.array = np.array([-3.5, -3.5, 2.0]),
                 cam_target: np.array = np.array([0.0, 0.0, 1.0]),
                 cam_config: dict = None):
        
        self._engine: isaac_gym_engine.IsaacGymEngine = engine
        self._env_id = 0
        self._obj_id = 0
        self._camera_ptr = None

        # Optional spherical placement, so a rollout can be recorded from the
        # same viewpoint as an offline render instead of the default fixed
        # diagonal offset. When follow_heading is set the azimuth is measured
        # from the direction the character is travelling, which keeps the view
        # on the character's flank as it turns.
        cfg = cam_config or {}
        self._cam_cfg = cfg
        self._fov = float(cfg.get("horizontal_fov", 60.0))
        self._use_spherical = "distance" in cfg
        self._azimuth = float(cfg.get("azimuth", 0.0))
        self._elevation = float(cfg.get("elevation", 10.0))
        self._distance = float(cfg.get("distance", 5.0))
        self._target_height = float(cfg.get("target_height", cam_target[2]))
        self._follow_heading = bool(cfg.get("follow_heading", False))
        self._heading_window = int(cfg.get("heading_window", 8))
        self._root_history = []
        self._last_heading = None
        if "resolution" in cfg:
            resolution = tuple(int(v) for v in cfg["resolution"])

        timestep = self._engine.get_timestep()
        fps = int(np.round(1.0 / timestep))
        super().__init__(fps, resolution, cam_pos, cam_target)
        return
    
    def _build_camera(self):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self._resolution[0]
        camera_props.height = self._resolution[1]
        camera_props.horizontal_fov = self._fov

        env_ptr = self._engine.get_env(self._env_id)
        gym = self._engine.get_gym()
        camera_ptr = gym.create_camera_sensor(env_ptr, camera_props)
        assert(camera_ptr != -1), "Unable to create video camera."
        
        self._camera_ptr = camera_ptr
        return
    
    def _record_frame(self):
        if (self._camera_ptr is None):
            self._build_camera()

        root_pos = self._engine.get_root_pos(self._obj_id)[self._env_id].cpu().numpy()
        if self._use_spherical:
            cam_pos, tar_pos = self._spherical_pose(root_pos)
        else:
            tar_pos = root_pos.copy()
            tar_pos[2] = self._cam_target[2]
            cam_delta = self._cam_pos - self._cam_target
            cam_pos = tar_pos + cam_delta
        self._set_camera_pose(cam_pos, tar_pos)

        gym = self._engine.get_gym()
        sim = self._engine.get_sim()
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.start_access_image_tensors(sim)

        env_ptr = self._engine.get_env(self._env_id)

        rgb_data = gym.get_camera_image(sim, env_ptr, self._camera_ptr, gymapi.IMAGE_COLOR)
        assert(rgb_data is not None), "Failed to render image."

        frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self._resolution[1], self._resolution[0], 4)
        frame = frame[:, :, :3].copy()  # drop alpha channel
        return frame
    
    def _heading_deg(self, root_pos):
        """Travel direction in degrees, from a short window of root positions."""
        self._root_history.append(np.array(root_pos[:2], dtype=np.float64))
        if len(self._root_history) > self._heading_window:
            self._root_history.pop(0)
        if len(self._root_history) >= 2:
            d = self._root_history[-1] - self._root_history[0]
            if np.linalg.norm(d) > 1e-4:
                self._last_heading = float(np.degrees(np.arctan2(d[1], d[0])))
        return self._last_heading if self._last_heading is not None else 0.0

    def _spherical_pose(self, root_pos):
        az = self._azimuth
        if self._follow_heading:
            az += self._heading_deg(root_pos)
        az_r = np.radians(az)
        el_r = np.radians(self._elevation)
        tar_pos = np.array([root_pos[0], root_pos[1], self._target_height], dtype=np.float64)
        offset = self._distance * np.array([
            np.cos(el_r) * np.cos(az_r),
            np.cos(el_r) * np.sin(az_r),
            np.sin(el_r)], dtype=np.float64)
        return tar_pos + offset, tar_pos

    def _set_camera_pose(self, cam_pos, cam_target):
        gym_cam_pos = gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2])
        gym_cam_target = gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2])

        gym = self._engine.get_gym()
        env_ptr = self._engine.get_env(self._env_id)
        gym.set_camera_location(self._camera_ptr, env_ptr, gym_cam_pos, gym_cam_target)
        return