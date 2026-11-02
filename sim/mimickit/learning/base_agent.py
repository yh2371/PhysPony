import abc
import enum
import gym.spaces as spaces
import numpy as np
import os
import time
import torch

import envs.base_env as base_env
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import learning.normalizer as normalizer
import learning.return_tracker as return_tracker
import learning.test_metrics_tracker as test_metrics_tracker
from util.logger import Logger
import util.mp_util as mp_util
import util.tb_logger as tb_logger
import util.torch_util as torch_util
import util.wandb_logger as wandb_logger

import learning.distribution_gaussian_diag as distribution_gaussian_diag

class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class BaseAgent(torch.nn.Module):
    NAME = "base"

    def __init__(self, config, env, device):
        super().__init__()

        self._env = env
        self._device = device
        self._iter = 0
        self._sample_count = 0
        self._config = config
        self._load_params(config)

        self._build_normalizers()
        self._build_model(config)
        self.to(self._device)

        self._build_optimizer(config)

        self._build_exp_buffer(config)
        self._build_return_tracker()
        self._build_test_metrics_tracker()
        
        self._mode = AgentMode.TRAIN
        self._curr_obs = None
        self._curr_info = None

        return

    def train_model(self, max_samples, out_model_file, int_output_dir, logger_type, log_file):
        start_time = time.time()

        self._curr_obs, self._curr_info = self._reset_envs()
        self._logger = self._build_logger(logger_type, log_file, self._config)
        self._init_train()

        while self._sample_count < max_samples:
            train_info = self._train_iter()
            
            self._sample_count = self._update_sample_count()
            output_iter = (self._iter % self._iters_per_output == 0) or (self._sample_count >= max_samples)

            if (output_iter):
                test_info = self.test_model(self._test_episodes)
            
            env_diag_info = self._env.get_diagnostics()
            self._log_train_info(train_info, test_info, env_diag_info, start_time) 
            self._logger.print_log()

            if (output_iter):
                self._logger.write_log()
                self._output_train_model(self._iter, out_model_file, int_output_dir)

                self._train_return_tracker.reset()
                self._curr_obs, self._curr_info = self._reset_envs()
            
            self._iter += 1

        return

    def test_model(self, num_episodes, save_states_path=None):
        self.eval()
        self.set_mode(AgentMode.TEST)
        
        num_procs = mp_util.get_num_procs()
        num_eps_proc = int(np.ceil(num_episodes / num_procs))

        # Initialize state collector if saving states
        state_collector = None
        if save_states_path is not None:
            from util.test_state_collector import TestStateCollector
            state_collector = TestStateCollector(device=self._device)

        with torch.no_grad():
            print("Rollout test starting")
            self._curr_obs, self._curr_info = self._reset_envs()
            test_info = self._rollout_test(num_eps_proc, state_collector=state_collector)
            print("Rollout test complete")
            
            # Save states if collector was used
            if state_collector is not None and save_states_path is not None:
                state_collector.save(save_states_path, tracked_only=False)
                print(f"Test states saved to {save_states_path}")

        return test_info
    
    def get_action_size(self):
        a_space = self._env.get_action_space()
        if (isinstance(a_space, spaces.Box)):
            a_size = np.prod(a_space.shape)
        elif (isinstance(a_space, spaces.Discrete)):
            a_size = 1
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return a_size
    
    def set_mode(self, mode):
        self._mode = mode
        if (self._mode == AgentMode.TRAIN):
            self._env.set_mode(base_env.EnvMode.TRAIN)
        elif (self._mode == AgentMode.TEST):
            self._env.set_mode(base_env.EnvMode.TEST)
        else:
            assert(False), "Unsupported agent mode: {}".format(mode)
        return

    def get_num_envs(self):
        return self._env.get_num_envs()

    def save(self, out_file):
        if (mp_util.is_root_proc()):
            state_dict = self.state_dict()
            torch.save(state_dict, out_file)
        return

    def load(self, in_file):
        state_dict = torch.load(in_file, map_location=self._device)
        self.load_state_dict(state_dict)
        self._sync_optimizer()
        Logger.print("Loaded model parameters from {:s}".format(in_file))
        return

    def calc_num_params(self):
        params = self.parameters()
        num_params = sum(p.numel() for p in params if p.requires_grad)
        return num_params

    def _load_params(self, config):
        self._discount = config["discount"]
        self._iters_per_output = config["iters_per_output"]
        self._normalizer_samples = config.get("normalizer_samples", np.inf)
        self._test_episodes = config["test_episodes"]
        
        self._steps_per_iter = config["steps_per_iter"]
        return

    @abc.abstractmethod
    def _build_model(self, config):
        return

    def _build_normalizers(self):
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_norm = normalizer.Normalizer(obs_space.shape, device=self._device, dtype=obs_dtype)
        self._a_norm = self._build_action_normalizer()
        return
    
    def _build_action_normalizer(self):
        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)

        if (isinstance(a_space, spaces.Box)):
            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device, dtype=a_dtype)
            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, dtype=a_dtype)
        elif (isinstance(a_space, spaces.Discrete)):
            a_mean = torch.tensor([0], device=self._device, dtype=a_dtype)
            a_std = torch.tensor([1], device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, min_std=0, dtype=a_dtype)
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return a_norm

    def _build_optimizer(self, config):
        opt_config = config["optimizer"]
        params = list(self.parameters())
        params = [p for p in params if p.requires_grad]
        self._optimizer = mp_optimizer.MPOptimizer(opt_config, params)
        return
    
    def _sync_optimizer(self):
        self._optimizer.sync()
        return

    def _build_exp_buffer(self, config):
        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()
        self._exp_buffer = experience_buffer.ExperienceBuffer(buffer_length=buffer_length, batch_size=batch_size,
                                                              device=self._device)
        
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        obs_buffer = torch.zeros([buffer_length, batch_size] + list(obs_space.shape), device=self._device, dtype=obs_dtype)
        self._exp_buffer.add_buffer("obs", obs_buffer)
        
        next_obs_buffer = torch.zeros_like(obs_buffer)
        self._exp_buffer.add_buffer("next_obs", next_obs_buffer)

        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)
        a_shape = list(a_space.shape)
        if (a_shape == []):
            a_shape = [1]

        action_buffer = torch.zeros([buffer_length, batch_size] + a_shape, device=self._device, dtype=a_dtype)
        self._exp_buffer.add_buffer("action", action_buffer)
        
        reward_buffer = torch.zeros([buffer_length, batch_size], device=self._device, dtype=torch.float)
        self._exp_buffer.add_buffer("reward", reward_buffer)
        
        done_buffer = torch.zeros([buffer_length, batch_size], device=self._device, dtype=torch.int)
        self._exp_buffer.add_buffer("done", done_buffer)

        return

    def _build_return_tracker(self):
        self._train_return_tracker = return_tracker.ReturnTracker(self.get_num_envs(), self._device)
        self._test_return_tracker = return_tracker.ReturnTracker(self.get_num_envs(), self._device)
        return

    def _build_test_metrics_tracker(self):
        self._test_metrics_tracker = test_metrics_tracker.TestMetricsTracker(self._device)
        return

    @abc.abstractmethod
    def _get_exp_buffer_length(self):
        return 0
    
    def _build_logger(self, logger_type, log_file, config):
        if (logger_type == "tb"):
            log = tb_logger.TBLogger()
        elif (logger_type == "wandb"):
            log = wandb_logger.WandbLogger("mimickit", config)
        else:
            assert(False), "Unsupported logger: {:s}".format(logger_type)

        log.set_step_key("Samples")
        if (mp_util.is_root_proc()):
            log.configure_output_file(log_file)
        return log

    def _update_sample_count(self):
        sample_count = self._exp_buffer.get_total_samples()
        sample_count = mp_util.reduce_sum(sample_count)
        return sample_count
    
    def _init_train(self):
        self._iter = 0
        self._sample_count = 0
        self._exp_buffer.clear()
        self._train_return_tracker.reset()
        self._test_return_tracker.reset()
        return

    def _train_iter(self):
        self._init_iter()
        
        self.eval()
        self.set_mode(AgentMode.TRAIN)

        with torch.no_grad():
            self._rollout_train(self._steps_per_iter)
        
        data_info = self._build_train_data()
        train_info = self._update_model()
        
        if (self._need_normalizer_update()):
            self._update_normalizers()

        info = {**train_info, **data_info}
        
        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["num_eps"] = self._train_return_tracker.get_episodes()
        
        return info

    def _init_iter(self):
        return

    def _rollout_train(self, num_steps):
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(r, done)
            self._record_data_post_step(next_obs, r, done, next_info)
            
            self._curr_obs, self._curr_info = self._reset_done_envs(done)
            self._exp_buffer.inc()

        return
    
    def _rollout_test(self, num_episodes, state_collector=None):
        self._test_return_tracker.reset()
        self._test_metrics_tracker.reset()

        if (num_episodes == 0):
            test_info = {
                "mean_return": 0.0,
                "mean_ep_len": 0.0,
                "num_eps": 0,
                "mpjpe": 0.0,
                "gmpjpe": 0.0,
                "accel_loss": 0.0,
                "vel_loss": 0.0,
                "success_rate": 0.0
            }
        else:
            num_envs = self.get_num_envs()
            # minimum number of episodes to collect per env
            # this is mitigate bias in the return estimate towards shorter episodes
            min_eps_per_env = int(np.ceil(num_episodes / num_envs))
            
            # Track which environment we're collecting from (use first env)
            collecting_env_idx = 0
            prev_done = torch.zeros(num_envs, dtype=torch.bool, device=self._device)

            while True:
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)

                next_obs, r, done, next_info = self._step_env(action)
                self._test_return_tracker.update(r, done)
                
                # Collect state if collector is provided
                if state_collector is not None:
                    # Reset collector when episode starts (after previous episode ended)
                    if prev_done[collecting_env_idx] and not done[collecting_env_idx]:
                        state_collector.reset()
                    
                    # Collect state if episode is active
                    if not done[collecting_env_idx]:
                        state_collector.collect_state(self._env, env_idx=collecting_env_idx)
                    
                    prev_done = done.clone()
                
                # Collect metrics for completed episodes
                self._collect_test_metrics(done, next_info)
            
                self._curr_obs, self._curr_info = self._reset_done_envs(done)
            
                eps_per_env = self._test_return_tracker.get_eps_per_env()
                if (torch.all(eps_per_env > min_eps_per_env - 1)):
                    break
        
            test_return = self._test_return_tracker.get_mean_return()
            test_ep_len = self._test_return_tracker.get_mean_ep_len()
            test_metrics = self._test_metrics_tracker.get_metrics()
            
            test_info = {
                "mean_return": test_return.item(),
                "mean_ep_len": test_ep_len.item(),
                "num_eps": self._test_return_tracker.get_episodes(),
                "mpjpe": test_metrics["mpjpe"],
                "gmpjpe": test_metrics["gmpjpe"],
                "accel_loss": test_metrics["accel_loss"],
                "vel_loss": test_metrics["vel_loss"],
                "success_rate": test_metrics["success_rate"]
            }
        return test_info

    def _collect_test_metrics(self, done, info):
        """Collect test metrics for completed episodes"""
        if torch.any(done):
            # Get environment-specific metrics if available
            if hasattr(self._env, 'get_test_metrics'):
                metrics = self._env.get_test_metrics(done)
                if metrics is not None:
                    # Extract metrics for completed episodes
                    done_indices = torch.where(done)[0]
                    for i, env_idx in enumerate(done_indices):
                        mpjpe = metrics.get('mpjpe', torch.tensor(0.0, device=self._device))
                        gmpjpe = metrics.get('gmpjpe', torch.tensor(0.0, device=self._device))
                        accel_loss = metrics.get('accel_loss', torch.tensor(0.0, device=self._device))
                        vel_loss = metrics.get('vel_loss', torch.tensor(0.0, device=self._device))
                        success_flag = metrics.get('success_flag', torch.tensor(1.0, device=self._device))
                        ep_len = metrics.get('ep_len', torch.tensor(0.0, device=self._device))
                        power = metrics.get('power', torch.tensor(0.0, device=self._device))
                        if power.dim() > 0:
                            power_val = power[env_idx] if power.shape[0] > env_idx else power[0]
                        else:
                            power_val = power
                            
                        # Handle tensor indexing
                        if mpjpe.dim() > 0:
                            mpjpe_val = mpjpe[env_idx] if mpjpe.shape[0] > env_idx else mpjpe[0]
                        else:
                            mpjpe_val = mpjpe
                            
                        if gmpjpe.dim() > 0:
                            gmpjpe_val = gmpjpe[env_idx] if gmpjpe.shape[0] > env_idx else gmpjpe[0]
                        else:
                            gmpjpe_val = gmpjpe
                            
                        if accel_loss.dim() > 0:
                            accel_val = accel_loss[env_idx] if accel_loss.shape[0] > env_idx else accel_loss[0]
                        else:
                            accel_val = accel_loss
                            
                        if vel_loss.dim() > 0:
                            vel_val = vel_loss[env_idx] if vel_loss.shape[0] > env_idx else vel_loss[0]
                        else:
                            vel_val = vel_loss
                            
                        if success_flag.dim() > 0:
                            success_val = success_flag[env_idx] if success_flag.shape[0] > env_idx else success_flag[0]
                        else:
                            success_val = success_flag
                            
                        if ep_len.dim() > 0:
                            ep_len_val = ep_len[env_idx] if ep_len.shape[0] > env_idx else ep_len[0]
                        else:
                            ep_len_val = ep_len
                        
                        self._test_metrics_tracker.update(
                            mpjpe_val, gmpjpe_val, accel_val, vel_val, success_val, ep_len_val,
                        )
        return

    @abc.abstractmethod
    def _decide_action(self, obs, info):
        a = None
        a_info = dict()
        return a, a_info

    def _step_env(self, action):
        obs, r, done, info = self._env.step(action)
        return obs, r, done, info

    def _record_data_pre_step(self, obs, info, action, action_info):
        self._exp_buffer.record("obs", obs)
        self._exp_buffer.record("action", action)
        
        if (self._need_normalizer_update()):
            self._obs_norm.record(obs)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        self._exp_buffer.record("next_obs", next_obs)
        self._exp_buffer.record("reward", r)
        self._exp_buffer.record("done", done)
        return

    def _reset_done_envs(self, done):
        done_indices = (done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False)
        env_ids = torch.flatten(done_indices)
        obs, info = self._reset_envs(env_ids)
        return obs, info

    def _reset_envs(self, env_ids=None):
        obs, info = self._env.reset(env_ids)
        return obs, info

    def _need_normalizer_update(self):
        return self._sample_count < self._normalizer_samples

    def _update_normalizers(self):
        self._obs_norm.update()
        return

    def _build_train_data(self):
        return dict()

    @abc.abstractmethod
    def _update_model(self):
        return

    def _compute_succ_val(self):
        r_succ = self._env.get_reward_succ()
        val_succ = r_succ / (1.0 - self._discount)
        return val_succ
    
    def _compute_fail_val(self):
        r_fail = self._env.get_reward_fail()
        val_fail = r_fail / (1.0 - self._discount)
        return val_fail

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        wall_time = (time.time() - start_time) / (60 * 60) # store time in hours
        self._logger.log("Iteration", self._iter, collection="1_Info")
        self._logger.log("Wall_Time", wall_time, collection="1_Info")
        self._logger.log("Samples", self._sample_count, collection="1_Info")
        
        test_return = test_info["mean_return"]
        test_ep_len = test_info["mean_ep_len"]
        test_eps = test_info["num_eps"]
        test_eps = mp_util.reduce_sum(test_eps)

        self._logger.log("Test_Return", test_return, collection="0_Main")
        self._logger.log("Test_Episode_Length", test_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Test_Episodes", test_eps, collection="1_Info", quiet=True)

        train_return = train_info.pop("mean_return")
        train_ep_len = train_info.pop("mean_ep_len")
        train_eps = train_info.pop("num_eps")
        train_eps = mp_util.reduce_sum(train_eps)

        self._logger.log("Train_Return", train_return, collection="0_Main")
        self._logger.log("Train_Episode_Length", train_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Train_Episodes", train_eps, collection="1_Info", quiet=True)

        for k, v in train_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v)

        for k, v in env_diag_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v, collection="2_Env", quiet=True)
            
        return
    
    def _compute_action_bound_loss(self, norm_a_dist):
        loss = None
        action_space = self._env.get_action_space()
        if (isinstance(action_space, spaces.Box)):
            a_low = action_space.low
            a_high = action_space.high
            valid_bounds = np.all(np.isfinite(a_low)) and np.all(np.isfinite(a_high))

            if (valid_bounds):
                assert(isinstance(norm_a_dist, distribution_gaussian_diag.DistributionGaussianDiag))
                # assume that actions have been normalized between [-1, 1]
                bound_min = -1
                bound_max = 1
                violation_min = torch.clamp_max(norm_a_dist.mode - bound_min, 0.0)
                violation_max = torch.clamp_min(norm_a_dist.mode - bound_max, 0)
                violation = torch.sum(torch.square(violation_min), dim=-1) \
                            + torch.sum(torch.square(violation_max), dim=-1)
                loss = violation

        return loss

    def _output_train_model(self, iter, out_model_file, int_output_dir):
        self.save(out_model_file)

        if (int_output_dir != ""):
            int_model_file = os.path.join(int_output_dir, "model_{:010d}.pt".format(iter))
            self.save(int_model_file)
        return
