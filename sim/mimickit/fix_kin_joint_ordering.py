import argparse
import os
import sys
import numpy as np
import pickle

def load_motion_pkl(path):
    """Load motion from .pkl. Returns dict with 'frames', 'fps', 'loop_mode'."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def extract_pose_data(frames):
    """frames: (N, D). Returns root_pos (N,3), root_rot (N,3 exp_map), joint_dof (N, dof_size)."""
    joint_dof = frames[:, 6:]
    kin2smal = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12, 12:13, 13:14, 14:15, 15:16, 16:17, 17:33, 18:34, 19:35, 
    20:18, 21:19, 22:20, 23:21, 24:22, 25:23, 26:24, 27:25, 28:26, 29:27, 30:28, 31:29, 32:30, 33:31, 34:32}

    joint_dof = joint_dof.reshape(-1, 35, 3)
    new_joint_dof = np.zeros_like(joint_dof)
    for i in range(joint_dof.shape[1]):
        new_joint_dof[:, i] = joint_dof[:, kin2smal[i]-1]
    frames[:, 6:] = new_joint_dof.reshape(-1,105)
    return frames



if __name__ == "__main__":
    import glob 
    for pkl in ["data/motions/horse/horse_kin_long_scaled.pkl"]: ##glob.glob("data/motions/horse_kin/*.pkl"):
        print(pkl)
        data = load_motion_pkl(pkl)
        try:
            frames = data["frames"]
            fps = data.get("fps", 30)
            if isinstance(frames, list): frames = np.array(frames)
            N, D = frames.shape
            print(f"Loaded motion: {N} frames, {D} dims, fps={fps}")
            frames = extract_pose_data(frames)
            data["frames"] = frames 
            with open(pkl, "wb") as f:
                pickle.dump(data, f)

        except:
            frames = data.frames
            fps = data.fps
            if isinstance(frames, list): frames = np.array(frames)
            N, D = frames.shape
            print(f"Loaded motion: {N} frames, {D} dims, fps={fps}")
            frames = extract_pose_data(frames)
            data.frames = frames 
            data.fps = 30
            with open(pkl, "wb") as f:
                pickle.dump(data, f)


        