import pickle 
import numpy as np
from phc_torch_utils import quat_to_exp_map
import torch
from scipy.spatial.transform import Rotation as R
from anim.motion import Motion

def load_ref(npz_path):
    """
    Loads and processes reference motion, adapting it to a model with reduced
    degrees of freedom (DoF).

    Args:
        npz_path (str): Path to the .npz file containing the motion data.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of (qpos, qvel) numpy arrays.
    """
    # --- Configuration: Define your new model's joint structure here ---
    # This makes the code readable and easy to modify.
    # List the 1-based indices of joints that should have a full 3 DoF (e.g., ball joints).
    DOF_3_JOINTS = [1,2,3,9,4,21,26,31,35,15,17] # e.g., hips, spine, neck, tail
    DOF_3_JOINTS = [1,2,3,9,10,4,5,21,22,26,27,31,35,14,15,17]
    qpos_2_smal_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:33, 19:34, 20:35, 21:18, 22:19, 23:20, 24:21, 25:22, 26:23, 27:24, 28:25, 29:26, 30:27, 31:28, 32:29, 33:30, 34:31, 35:32}
    qpos_2_smal_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:33, 19:34, 20:35, 21:23, 22:24, 23:25, 24:26, 25:27, 26:18, 27:19, 28:20, 29:21, 30:22, 31:28, 32:29, 33:30, 34:31, 35:32}
    
    TOTAL_JOINTS = 35 # The total number of joints in your original SMAL-like data

    # --- Load Data ---
    try:
        if not "lifted" in npz_path:
            ref_motion = np.load(npz_path, allow_pickle=True).item()
        else:
            ref_motion = np.load(npz_path)
            if 'qvel_final' in ref_motion:
                qvel = ref_motion['qvel_final']#[:10]
                qpos = ref_motion['qpos_lifted']#[:10]
                root_rot_wxyz = torch.from_numpy(qpos[:,3:7])
                root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
                qpos_root_exp_map = quat_to_exp_map(root_rot_xyzw).numpy()
                qpos = np.concatenate([qpos[:,:3],qpos_root_exp_map, qpos[:,7:]], axis=1)
                return qpos, qvel
            else:
                return ref_motion['qpos_lifted'], ref_motion['qvel_final']
        poses_body = torch.tensor(ref_motion['poses_body'], dtype=torch.float32)
        root_transl = torch.tensor(ref_motion['trans'], dtype=torch.float32)
        root_rot_xyzw = torch.tensor(ref_motion['poses_root'], dtype=torch.float32) # Assumes xyzw
    except (ValueError, TypeError, KeyError):
        # Fallback for a different .npz format
        ref_motion = np.load(npz_path)
        if True:
            qpos = ref_motion[:, :112]
            qvel = ref_motion[:, 112:]
            qpos_root = qpos[:, :7]
            qpos_dof = qpos[:, 7:]
            qvel_root = qvel[:, :6]
            qvel_dof = qvel[:, 6:]
            qpos_dof_list = []
            qvel_dof_list = []
            for i in range(35):
                if i+1 in DOF_3_JOINTS:
                    qpos_dof_list.append(qpos_dof[:, i*3:i*3+3])
                    qvel_dof_list.append(qvel_dof[:, i*3:i*3+3])
                else:
                    qpos_dof_list.append(qpos_dof[:, i*3+1:i*3+2])
                    qvel_dof_list.append(qvel_dof[:, i*3+1:i*3+2])

            qpos_dof = np.concatenate(qpos_dof_list, axis=1)
            qvel_dof = np.concatenate(qvel_dof_list, axis=1)
            qpos = np.concatenate([qpos_root, qpos_dof], axis=1)
            qvel = np.concatenate([qvel_root, qvel_dof], axis=1)
          
            return qpos, qvel
        return ref_motion[:, :112], ref_motion[:, 112:]

    ref_mot_fps = 30
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    N = poses_body.shape[0]
    
    qpos_dof_list = []
    qpos_exp_map_dof_list = []

    for i in range(TOTAL_JOINTS):

        joint_index_1_based = i+1         
        
        joint_quat_xyzw = poses_body[:, i, :]

        joint_exp_map = quat_to_exp_map(joint_quat_xyzw) # Shape: (N, 3)

        if joint_index_1_based in DOF_3_JOINTS: #i in DOF_3_JOINTS:
            # This is a 3-DoF joint, so we keep all components
            qpos_dof_list.append(joint_exp_map)
            qpos_exp_map_dof_list.append(joint_exp_map)
        else:
            rotations = R.from_quat(joint_quat_xyzw.numpy())
            euler_angles = rotations.as_euler('xyz', degrees=False) # Shape: (N, 3)
            y_angle = torch.tensor(euler_angles[:, 1:2], dtype=torch.float32) # Keep only y, shape (N, 1)
            exp_map_y_only = joint_exp_map.clone()
            exp_map_y_only[:, 0] = 0.0 # Zero out x
            exp_map_y_only[:, 2] = 0.0 # Zero out z
            qpos_dof_list.append(y_angle)
            qpos_exp_map_dof_list.append(y_angle) # Keep only the y-component scalar

    qpos_dofs = torch.cat(qpos_dof_list, dim=1)
    qpos_exp_map_dofs = torch.cat(qpos_exp_map_dof_list, dim=1)

    qpos_tensor = torch.cat([root_transl, root_rot_wxyz, qpos_dofs], dim=1)
    
    root_exp_map = quat_to_exp_map(root_rot_xyzw)
    
    qpos_exp_map_tensor = torch.cat([root_transl, root_exp_map, qpos_exp_map_dofs], dim=1)

    qvel_exp_map = (qpos_exp_map_tensor[1:] - qpos_exp_map_tensor[:-1]) * ref_mot_fps
    
    qvel_tensor = torch.cat([torch.zeros(1, qvel_exp_map.shape[1]), qvel_exp_map], dim=0)

    qpos = qpos_tensor.detach().numpy()
    qvel = qvel_tensor.detach().numpy()

    
    return qpos_exp_map_tensor.detach().numpy(), qvel#.detach().numpy()[start_frame:end_frame]

from glob import glob 
import os  
os.makedirs(f"data/motions/horse_chunked_ID4_partial_new", exist_ok=True)

for file in glob("mimickit/mocap_full/*.npy"):
    if not "ID_4" in file:
        continue
    horse = load_ref(file)
    filename = file.split("/")[-1][:-4]

    b = pickle.load(open("data/motions/g1/g1_walk.pkl", "rb"))
    window = 100
    for chunk in range(0,len(horse[0]), window):
        temp = Motion(b.loop_mode, b.fps, horse[0][chunk:chunk+window])
        with open(f"data/motions/horse_chunked_ID4_partial_new/{filename}_{chunk}.pkl", "wb") as f:
            pickle.dump(temp, f)
        print(temp.frames.shape)