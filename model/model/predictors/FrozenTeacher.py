"""Frozen copy of Dessie's camera/pose/shape encoders and decoders.

Provides per-frame teacher targets (encoder features and decoded parameters)
for distilling the motion transformer. Never trained or saved.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from model.paths import SMAL_DIR


class FrozenKinematicTeacher(nn.Module):
    def __init__(self, vit_feat_dim=384, dessie_dim=640, scale_min=0.3, scale_max=1.5):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

        def conv_stack(cout):
            return nn.Sequential(
                nn.Conv2d(vit_feat_dim, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, cout, kernel_size=8),
            )

        # Pose (articulation) encoder+decoder -- mirrors PosePredictorBase.
        self.netArticulation = conv_stack(dessie_dim)
        self.fc_decpose1 = nn.Linear(dessie_dim + 35 * 6, 1024)
        self.fc_decpose2 = nn.Linear(1024, dessie_dim)
        self.decpose = nn.Linear(dessie_dim, 35 * 6)

        # Camera encoder+decoder -- mirrors PosePredictorBase.
        ncam = 3
        self.netPose = conv_stack(dessie_dim)
        self.fc_deccam1 = nn.Linear(dessie_dim + ncam + 6, 1024)
        self.fc_deccam2 = nn.Linear(1024, dessie_dim)
        self.deccam = nn.Linear(dessie_dim, ncam)
        self.decroot = nn.Linear(dessie_dim, 1 * 6)

        # Shape encoder+decoder -- mirrors geometry/prior.py's SMALGeometry.
        self.netShape = conv_stack(dessie_dim)
        self.fc_decshape1 = nn.Linear(dessie_dim + 9, 1024)
        self.fc_decshape2 = nn.Linear(1024, dessie_dim)
        self.decshape = nn.Linear(dessie_dim, 9)

        self.drop1 = nn.Dropout()  # inert in eval() mode; kept for key/behavior parity with the source modules

        mean_params = np.load(os.path.join(SMAL_DIR, '6D_meanpose.npz'), allow_pickle=True)['mean_pose'].astype(np.float32)
        mean_params[0, :] = [-0.0517, 0.9979, -0.2873, -0.0517, -0.9564, -0.0384]
        init_pose = torch.reshape(torch.from_numpy(mean_params.copy()).unsqueeze(0), [1, -1])
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_cam', torch.from_numpy(np.array([0.6, 0.0, 0.0])).unsqueeze(0).float())
        self.register_buffer('init_shape', torch.zeros(1, 9))

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def load_dessie(self, dessie_ckpt):
        """Load Dessie's pretrained pose/cam/shape encoder+decoder weights.
        Key-mapping mirrors PonyModelBase.load_dessie exactly (same source
        checkpoint format), just targeting this standalone module's own
        (unprefixed) parameter names instead of netBase/netInstance."""
        dessie = torch.load(dessie_ckpt, map_location='cpu')
        sd = {}
        for key, val in dessie['state_dict'].items():
            if "model.key_encoder_pose" in key:
                sd[key.replace("model.key_encoder_pose", "netArticulation")] = val
            elif "model.key_encoder_cam" in key:
                sd[key.replace("model.key_encoder_cam", "netPose")] = val
            elif "model.key_encoder_shape" in key:
                sd[key.replace("model.key_encoder_shape", "netShape")] = val
            elif "decpose" in key or "decshape" in key or "deccam" in key or "decroot" in key:
                sd[key.replace("model.", "")] = val
        sd['init_cam'] = dessie['state_dict']['model.init_cam']
        sd['init_pose'] = dessie['state_dict']['model.init_pose']
        sd['init_shape'] = dessie['state_dict']['model.init_shape']
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[FrozenKinematicTeacher] loaded {len(sd)} tensors from {dessie_ckpt} "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def cam_bbox(self, crop_cam, r=256, focal_length=5000.):
        scale = crop_cam[:, 0].clamp(min=self.scale_min, max=self.scale_max)
        bs = r * scale + 1e-9
        tz = 2 * focal_length / bs
        return torch.stack([crop_cam[:, 1], crop_cam[:, 2], tz], dim=-1)

    @torch.no_grad()
    def encode(self, patch_feat):
        return self.netArticulation(patch_feat), self.netPose(patch_feat), self.netShape(patch_feat)

    @torch.no_grad()
    def forward_pose(self, xf_pose, batch_size, num_frames):
        x = xf_pose.view(batch_size * num_frames, -1)
        articulation_angles = self.init_pose[0, 6:].expand(batch_size * num_frames, -1)
        for _ in range(3):
            xc = torch.cat([x, articulation_angles], 1)
            xc = self.drop1(self.fc_decpose1(xc))
            xc = self.drop1(self.fc_decpose2(xc))
            articulation_angles = self.decpose(xc) + articulation_angles
        return articulation_angles

    @torch.no_grad()
    def forward_cam(self, xf_cam, batch_size, num_frames):
        x_cam = xf_cam.view(batch_size * num_frames, -1)
        pred_cam = self.init_cam.expand(batch_size * num_frames, -1)
        pred_root = self.init_pose[0, :6].expand(batch_size * num_frames, -1)
        for _ in range(3):
            xc = torch.cat([x_cam, pred_cam, pred_root], 1)
            xc = self.drop1(self.fc_deccam1(xc))
            xc = self.drop1(self.fc_deccam2(xc))
            pred_cam = self.deccam(xc) + pred_cam
            pred_root = self.decroot(xc) + pred_root
        return self.cam_bbox(pred_cam), pred_root

    @torch.no_grad()
    def forward_shape(self, xf_shape, batch_size, num_frames):
        betas = self.init_shape.expand(batch_size * num_frames, -1)
        x = xf_shape.view(batch_size * num_frames, -1)
        for _ in range(3):
            xc = torch.cat([x, betas], 1)
            xc = self.drop1(self.fc_decshape1(xc))
            xc = self.drop1(self.fc_decshape2(xc))
            betas = self.decshape(xc) + betas
        return betas
