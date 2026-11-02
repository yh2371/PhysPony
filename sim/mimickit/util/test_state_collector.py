"""
Utility for collecting and storing agent states during test mode in SMAL format.
Stores rotation matrices (3x3) directly instead of quaternions.
"""
import numpy as np
import torch
import util.torch_util as torch_util


class TestStateCollector:
    """Collects agent states during test mode and converts them to SMAL format."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset the collector for a new episode."""
        self.tracked_states = {
            'root_pos': [],
            'root_rot': [],
            'joint_rot': [],
            'body_pos': [],
            'body_rot': []
        }
        self.reference_states = {
            'root_pos': [],
            'root_rot': [],
            'joint_rot': [],
            'body_pos': [],
            'body_rot': []
        }
    
    def collect_state(self, env, env_idx=0):
        """
        Collect current state from environment.
        
        Args:
            env: The environment object (DeepMimicEnv or similar)
            env_idx: Index of environment to collect from (default: 0)
        """
        char_id = env._get_char_id()
        
        # Get tracked (simulated) state
        root_pos = env._engine.get_root_pos(char_id)[env_idx:env_idx+1]  # Keep batch dim
        root_rot = env._engine.get_root_rot(char_id)[env_idx:env_idx+1]
        dof_pos = env._engine.get_dof_pos(char_id)[env_idx:env_idx+1]
        joint_rot = env._kin_char_model.dof_to_rot(dof_pos)
        
        # Forward kinematics for body positions
        body_pos, body_rot = env._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        
        # Convert quaternions to rotation matrices
        # root_rot: (1, 4) -> (1, 3, 3)
        root_rot_mat = torch_util.quat_to_matrix(root_rot)  # (1, 3, 3)
        # body_rot: (1, num_bodies, 4) -> (1, num_bodies, 3, 3)
        body_rot_mat = torch_util.quat_to_matrix(body_rot)  # (1, num_bodies, 3, 3)
        # joint_rot: (1, num_joints, 4) -> (1, num_joints, 3, 3)
        joint_rot_mat = torch_util.quat_to_matrix(joint_rot)  # (1, num_joints, 3, 3)
        
        # Store tracked state (as rotation matrices)
        self.tracked_states['root_pos'].append(root_pos[0].detach().cpu().numpy())
        self.tracked_states['root_rot'].append(root_rot_mat[0].detach().cpu().numpy())  # (3, 3)
        self.tracked_states['joint_rot'].append(joint_rot_mat[0].detach().cpu().numpy())  # (J, 3, 3)
        self.tracked_states['body_pos'].append(body_pos[0].detach().cpu().numpy())
        self.tracked_states['body_rot'].append(body_rot_mat[0].detach().cpu().numpy())  # (num_bodies, 3, 3)
        
        # Get reference state if available
        if hasattr(env, '_ref_root_pos'):
            ref_root_pos = env._ref_root_pos[env_idx:env_idx+1]
            ref_root_rot = env._ref_root_rot[env_idx:env_idx+1]
            ref_joint_rot = env._ref_joint_rot[env_idx:env_idx+1]
            
            ref_body_pos, ref_body_rot = env._kin_char_model.forward_kinematics(
                ref_root_pos, ref_root_rot, ref_joint_rot
            )
            
            # Convert quaternions to rotation matrices
            ref_root_rot_mat = torch_util.quat_to_matrix(ref_root_rot)  # (1, 3, 3)
            ref_body_rot_mat = torch_util.quat_to_matrix(ref_body_rot)  # (1, num_bodies, 3, 3)
            ref_joint_rot_mat = torch_util.quat_to_matrix(ref_joint_rot)  # (1, num_joints, 3, 3)
            
            self.reference_states['root_pos'].append(ref_root_pos[0].detach().cpu().numpy())
            self.reference_states['root_rot'].append(ref_root_rot_mat[0].detach().cpu().numpy())  # (3, 3)
            self.reference_states['joint_rot'].append(ref_joint_rot_mat[0].detach().cpu().numpy())  # (J, 3, 3)
            self.reference_states['body_pos'].append(ref_body_pos[0].detach().cpu().numpy())
            self.reference_states['body_rot'].append(ref_body_rot_mat[0].detach().cpu().numpy())  # (num_bodies, 3, 3)
    
    def to_smal_format(self, states_dict, num_joints=None):
        """
        Convert collected states to SMAL format.
        
        SMAL format (updated to use rotation matrices):
        - poses_body: (T, J, 3, 3) rotation matrices for body joints (excluding root)
        - trans: (T, 3) root translation
        - poses_root: (T, 3, 3) root rotation matrix
        
        Args:
            states_dict: Dictionary with 'root_pos', 'root_rot', 'joint_rot' lists
            num_joints: Number of body joints (excluding root). If None, inferred from joint_rot
        
        Returns:
            Dictionary with 'poses_body', 'trans', 'poses_root' in numpy format
        """
        # Concatenate all frames
        root_pos = np.stack(states_dict['root_pos'], axis=0)  # (T, 3)
        root_rot = np.stack(states_dict['root_rot'], axis=0)  # (T, 3, 3) - rotation matrix
        joint_rot = np.stack(states_dict['joint_rot'], axis=0)  # (T, J, 3, 3) - rotation matrices
        
        T = root_pos.shape[0]
        
        # Root translation
        trans = root_pos  # (T, 3)
        
        # Root rotation (rotation matrix)
        poses_root = root_rot  # (T, 3, 3)
        
        # Body joint rotations (excluding root)
        # joint_rot is (T, J, 3, 3) where J is number of body joints (excluding root)
        poses_body = joint_rot  # (T, J, 3, 3)
        
        return {
            'poses_body': poses_body,
            'trans': trans,
            'poses_root': poses_root
        }
    
    def get_tracked_smal(self):
        """Get tracked states in SMAL format."""
        return self.to_smal_format(self.tracked_states)
    
    def get_reference_smal(self):
        """Get reference states in SMAL format."""
        return self.to_smal_format(self.reference_states)
    
    def save(self, filepath, tracked_only=False):
        """
        Save collected states to .npz file in SMAL format.
        
        Args:
            filepath: Path to save the .npz file
            tracked_only: If True, only save tracked states. Otherwise save both.
        """
        tracked_smal = self.get_tracked_smal()
        
        # Also save body positions for visualization
        tracked_body_pos = np.stack(self.tracked_states['body_pos'], axis=0)  # (T, num_bodies, 3)
        tracked_smal['body_pos'] = tracked_body_pos
        
        if tracked_only:
            np.savez(filepath, **tracked_smal)
        else:
            ref_smal = self.get_reference_smal()
            ref_body_pos = np.stack(self.reference_states['body_pos'], axis=0)  # (T, num_bodies, 3)
            ref_smal['body_pos'] = ref_body_pos
            
            np.savez(
                filepath,
                **tracked_smal,
                ref_poses_body=ref_smal['poses_body'],
                ref_trans=ref_smal['trans'],
                ref_poses_root=ref_smal['poses_root'],
                ref_body_pos=ref_smal['body_pos']
            )

