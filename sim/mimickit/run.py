import numpy as np
import os
import shutil
import sys
import time

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize, motion_file=None, record_video=False):
    env_file = args.parse_string("env_config")
    env = env_builder.build_env(env_file, num_envs, device, visualize, motion_file, record_video=record_video)
    return env

def build_agent(agent_file, env, device):
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, out_model_file, int_output_dir, logger_type, log_file):
    agent.train_model(max_samples=max_samples, out_model_file=out_model_file, 
                      int_output_dir=int_output_dir, logger_type=logger_type,
                      log_file=log_file)
    return

def test(agent, test_episodes, save_states_path=None, env=None, video_path=None):
    result = agent.test_model(num_episodes=test_episodes, save_states_path=save_states_path)
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))

    # Print additional test metrics
    if "mpjpe" in result:
        Logger.print("MPJPE: {:.6f}".format(result["mpjpe"]))
    if "gmpjpe" in result:
        Logger.print("GMPJPE: {:.6f}".format(result["gmpjpe"]))
    if "accel_loss" in result:
        Logger.print("Acceleration Loss: {:.6f}".format(result["accel_loss"]))
    if "vel_loss" in result:
        Logger.print("Velocity Loss: {:.6f}".format(result["vel_loss"]))
    if "success_rate" in result:
        Logger.print("Success Rate: {:.4f}".format(result["success_rate"]))

    if (env is not None and env.enabled_record_video()):
        video = env.get_video()
        Logger.print("Captured {:d} video frames at {:.1f} fps".format(
            video.get_num_frames(), video.get_fps()))
        out_dir = os.path.dirname(video_path)
        if (out_dir != "" and not os.path.exists(out_dir)):
            os.makedirs(out_dir, exist_ok=True)
        video.save(video_path)
        Logger.print("Saved rollout video to {:s}".format(video_path))

    return result

def create_output_dirs(out_model_file, int_output_dir):
    if (mp_util.is_root_proc()):
        output_dir = os.path.dirname(out_model_file)
        if (output_dir != "" and (not os.path.exists(output_dir))):
            os.makedirs(output_dir, exist_ok=True)
        
        if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
            os.makedirs(int_output_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)
        
    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def run(rank, num_procs, device, master_port, args):
    mode = args.parse_string("mode", "train")
    if mode == "test":
        num_envs = 1
    else:
        num_envs = args.parse_int("num_envs", 1)
    visualize = False #False #args.parse_bool("visualize", False)
    logger_type = "wandb" #args.parse_string("logger", "tb")
    log_file = args.parse_string("log_file", "output/log.txt")
    out_model_file = args.parse_string("out_model_file", "output/model.pt")
    int_output_dir = args.parse_string("int_output_dir", "")
    model_file = args.parse_string("model_file", "")
    motion_file = args.parse_string("motion_file", None)
    record_video = args.parse_bool("video", False)
    video_path = args.parse_string("video_path", "output/rollout_video.mp4")
    print("DEVICE: ", device)
    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()

    create_output_dirs(out_model_file, int_output_dir)

    env = build_env(args, num_envs, device, visualize, motion_file, record_video=record_video)
    agent_file = args.parse_string("agent_config")
    print("Agent_device: ", device)
    agent = build_agent(agent_file, env, device)

    # Load pretrained checkpoint if specified (for AMP training from pretrained)
    pretrained_checkpoint = args.parse_string("pretrained_checkpoint", "")
    if (pretrained_checkpoint != ""):
        freeze_policy = args.parse_bool("freeze_policy", False)
        freeze_critic = args.parse_bool("freeze_critic", False)
        
        # Import here to avoid circular dependencies
        from util.load_pretrained_for_amp import load_pretrained_checkpoint
        
        Logger.print("Loading pretrained checkpoint for AMP training...")
        load_pretrained_checkpoint(
            agent,
            checkpoint_path=pretrained_checkpoint,
            freeze_policy=freeze_policy,
            freeze_critic=freeze_critic,
            strict=False,
            device=device
        )
    
    # Load regular model file if specified (for resuming training or testing)
    if (model_file != ""):
        agent.load(model_file)

    if (mode == "train"):
        env_file = args.parse_string("env_config")

        max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)
        train(agent=agent, max_samples=max_samples, out_model_file=out_model_file, 
              int_output_dir=int_output_dir, logger_type=logger_type, log_file=log_file)
    elif (mode == "test"):
        print("Testing...")
        test_episodes = args.parse_int("test_episodes", 1)
        save_states_path = args.parse_string("save_states_path", "")
        if save_states_path == "":
            save_states_path = None
        test(agent=agent, test_episodes=test_episodes, save_states_path=save_states_path,
             env=env, video_path=video_path)
    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def main(argv):
    root_rank = 0
    args = load_args(argv)
    master_port = args.parse_int("master_port", None)
    num_workers = args.parse_int("num_workers", 1)
    device = args.parse_string("device", "cuda:0")
    assert(num_workers > 0)
    
    # if master port is not specified, then pick a random one
    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for i in range(num_workers - 1):
        rank = i + 1
        if ("cuda" in device):
            curr_device = "cuda:" + str(rank)
        else:
            curr_device = device

        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)

    
    if (num_workers > 1 and "cuda" in device):
        curr_device = "cuda:" + str(root_rank)
    else:
        curr_device = device
    
    run(root_rank, num_workers, curr_device, master_port, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)
