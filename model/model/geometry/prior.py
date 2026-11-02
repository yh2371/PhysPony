# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
import os
import sys
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SMAL.smal_torch.smal_torch import SMAL
import torch.nn as nn
###############################################################################
#  Geometry interface
###############################################################################

class SMALGeometry(torch.nn.Module):
    def __init__(self, smal_model_dir=None, device=None, **kwargs):
        super(SMALGeometry, self).__init__()
        self.register_buffer("_devref", torch.empty(0))  # no grad, moves with .to()

        self.setup_smal(smal_model_dir)
        vit_feat_dim = 384
        enc_feat_dim = 640
        self.netShape = nn.Sequential(
            nn.Conv2d(vit_feat_dim, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2), # Output size: [2, 256, 16, 16]
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2), # Output size: [2, 128, 8, 8]
            nn.Conv2d(128, enc_feat_dim, kernel_size=8)  #  # Output size: [2, 640, 1, 1]
        )
        self.fc_decshape1 = nn.Linear(enc_feat_dim + 9, 1024)
        self.fc_decshape2 = nn.Linear(1024, enc_feat_dim)
        self.decshape = nn.Linear(enc_feat_dim, 9)
        self.drop1 = nn.Dropout()
        init_shape = torch.from_numpy(np.zeros(9)).unsqueeze(0).type(torch.float32)
        self.register_buffer('init_shape', init_shape)


    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def forward_shape(self, x_shape, batch_size, num_frames):
        betas = self.init_shape.expand(batch_size*num_frames, -1)
        x = x_shape.view(batch_size*num_frames, -1)
        for i in range(3):
            xc = torch.cat([x, betas], 1)
            xc = self.fc_decshape1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decshape2(xc)
            xc = self.drop1(xc)
            betas = self.decshape(xc) + betas
        return betas

    def getMesh(self, material=None, total_iter=0, jitter_grid=True, feats=None, patch_out=None, patch_key=None):
        """
        Generate mesh from SMAL model.
        
        Args:
            smal_params: Dictionary containing SMAL parameters
                - beta: shape parameters (batch_size, 9)
                - theta: pose parameters (batch_size, 108) 
                - trans: translation (batch_size, 3)
                - camera_index: camera index for pose transformation (optional)
            material: Material for the mesh
        """
        x_shape = self.netShape(patch_key)
        dev, dtype = patch_key.device, patch_key.dtype

        batch_size = x_shape.shape[0]
        betas = self.init_shape.expand(batch_size, -1)
        x = x_shape.view(batch_size, -1)
        for i in range(3):
            xc = torch.cat([x, betas], 1)
            xc = self.fc_decshape1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decshape2(xc)
            xc = self.drop1(xc)
            betas = self.decshape(xc) + betas


        theta = torch.zeros(batch_size, self.default_theta_dim, device=dev, dtype=dtype)
        root_pose_yup = torch.tensor([-torch.pi / 2*0.0, 0.0, 0.0], device=dev, dtype=dtype)  
        theta[:, :3] = root_pose_yup
        trans = torch.zeros(batch_size, 3, device=dev, dtype=dtype)

        # Generate mesh from SMAL
        verts, _, _ = self.smal_model(
                beta=betas,
                theta=theta,
                trans=trans
            )

        # Store mesh vertices
        self.mesh_verts = verts

         # Apply texture if available
        if material is not None and hasattr(self, 'vt') and hasattr(self, 'ft'):
            batch_size = material.shape[0]
            textures = TexturesUV(
                maps=material,
                faces_uvs=self.ft.repeat(batch_size, 1, 1),
                verts_uvs=self.vt.repeat(batch_size, 1, 1), 
                sampling_mode="nearest"
            ).to(dev)
        else:
            textures = None
            
        # Create PyTorch3D mesh with texture\
        torch_mesh = Meshes(verts=verts, faces=self.faces.repeat(verts.shape[0], 1, 1), textures=textures).to(
            dev)
        
        return torch_mesh, betas, x_shape

    def to(self, *args, **kwargs):

        ret = super().to(*args, **kwargs)
        dev = self._devref.device
        # Push any tensor-like attribute inside the plain SMAL object to the same device
        for name, val in list(vars(self.smal_model).items()):
            if torch.is_tensor(val):
                setattr(self.smal_model, name, val.to(dev))
        if hasattr(self.smal_model, "device"):
            self.smal_model.device = dev
        self.faces = self.faces.to(dev)
        self.vt = self.vt.to(dev)
        self.ft = self.ft.to(dev)
        return ret

    def setup_smal(self, model_dir):
        """Setup SMAL model and related components."""
        self.smal_model = SMAL(os.path.join(model_dir, 'my_smpl_0000_horse_new_skeleton_horse.pkl'), device="cpu")
        self.faces = self.smal_model.faces.unsqueeze(0)

        if not hasattr(self, "ft") or not hasattr(self, "vt"):
            obj_filename = os.path.join(model_dir,
                                        "uvmap/uvmap_from_TextPaper/TexPaper_uv_mask.obj")
            smpl_texture_data = load_obj(obj_filename)
            self.vt = smpl_texture_data[-1].verts_uvs
            self.ft = smpl_texture_data[1].textures_idx

        # Set default parameters
        self.default_beta_dim = 9  # SMAL shape parameters
        self.default_theta_dim = 36*3  # SMAL pose parameters (35 joints * 3 + 3 for root)
        