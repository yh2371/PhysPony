from dataclasses import dataclass
import torch
from model.utils import misc
from ..predictors import PosePredictorBaseConfig, PosePredictorBase
from ..predictors.transformer import ThreeBranchModel
import torch.nn.functional as F


from pytorch3d.structures import Meshes
import pytorch3d


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


@dataclass
class MotionVAEConfig:
    latent_dim: int = 256
    z_token_num: int = 1
    transformer_layer_num: int = 4
    pe_dropout: float = 0.


@dataclass
class TemporalPosePredictorConfig(PosePredictorBaseConfig):
    enable_motion_vae: bool = True
    cfg_motion_vae: MotionVAEConfig = MotionVAEConfig()
    render_gt_mesh: bool = False

    # Randomly mask frames from the motion transformer's attention during training,
    # so it also works with few frames (down to single-image evaluation).
    enable_frame_masking_curriculum: bool = False
    frame_masking_min_frames: int = 1


class TemporalPosePredictor(PosePredictorBase):
    def __init__(self, cfg: TemporalPosePredictorConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, TemporalPosePredictorConfig)
        if self.enable_motion_vae:
            self.fuser = ThreeBranchModel(
                d_pose=640,
                d_cam=640,
                d_shape=640,
            )


    def forward_deformation(self, shape, feat=None, batch_size=None, num_frames=None):
        original_verts = shape.verts_packed()
        num_verts = original_verts.shape[1]
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat), 1, 1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3), multiply by 0.1 to minimize disruption when initially enabled
        if deformation.shape[0] > 1 and self.cfg_deform.force_avg_deform:
            assert batch_size is not None and num_frames is not None
            assert deformation.shape[0] == batch_size * num_frames
            deformation = deformation.view(batch_size, num_frames, *deformation.shape[1:])
            deformation = deformation.mean(dim=1, keepdim=True)
            deformation = deformation.repeat(1,num_frames,*[1]*(deformation.dim()-2))
            deformation = deformation.view(batch_size*num_frames, *deformation.shape[2:])
        shape.update_padded(original_verts + deformation)
        return shape, deformation

    def get_latents(self, patch_feat):
        xf_pose = self.netArticulation(patch_feat)
        xf_cam = self.netPose(patch_feat)
        return xf_pose, xf_cam

    def forward_pose(self, xf_pose, batch_size, num_frames):
        x = xf_pose.view(batch_size*num_frames, -1)
        articulation_angles = self.init_pose[0, 6:].expand(batch_size*num_frames, -1)
        for i in range(3):
            xc = torch.cat([x, articulation_angles], 1)
            xc = self.fc_decpose1(xc)
            xc = self.drop1(xc)
            xc = self.fc_decpose2(xc)
            xc = self.drop1(xc)
            articulation_angles = self.decpose(xc) + articulation_angles
    
        pose_6d = articulation_angles.view(batch_size*num_frames,35, 6)
        return pose_6d
        
    def forward_cam(self, xf_cam, batch_size, num_frames):
        x_cam = xf_cam.view(batch_size*num_frames, -1)
        pred_cam = self.init_cam.expand(batch_size*num_frames, -1)
        pred_root = self.init_pose[0, :6].expand(batch_size*num_frames, -1)
        
        for i in range(3):
            xc = torch.cat([x_cam, pred_cam, pred_root], 1)
            xc = self.fc_deccam1(xc)
            xc = self.drop1(xc)
            xc = self.fc_deccam2(xc)
            xc = self.drop1(xc)
            pred_cam = self.deccam(xc) + pred_cam
            pred_root = self.decroot(xc) + pred_root
         
        pred_cam = self.cam_bbox(pred_cam)
        return pred_cam, pred_root


    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter, betas=None, smal_model=None, pose_rot=None, pred_cam=None, xf_shape=None, netShape=None, **kwargs):
        """
        Forward propagation of articulation. For each bone, the network takes:
        1) the 3D location of the bone;
        2) the feature of the patch which the bone is projected to; and 3) an encoding of the bone's index to predict
            the bone's rotation (represented by an Euler angle).
        Args:
            shape: a Mesh object, whose v_pos has batch size BxF or 1.
            feat: the feature of the patches. Shape: (BxF, feat_dim, num_patches_per_axis, num_patches_per_axis)
            mvp: the model-view-projection matrix. Shape: (BxF, 4, 4)
        Returns:
            shape: a Mesh object, whose v_pos has batch size BxF (collapsed).
            articulation_angles: the predicted bone rotations. Shape: (B, F, num_bones, 3)
            aux: a dictionary containing auxiliary information.
        """
        if not self.enable_motion_vae:
            return super().forward_articulation(
                shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter, betas, smal_model=smal_model, pose_rot=pose_rot, pred_cam=pred_cam, xf_shape=xf_shape, netShape=netShape, **kwargs
            )
        
        verts = shape.verts_packed()
        if len(verts) == batch_size * num_frames:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])  # BxFxNx3
        else:
            verts = verts[None]  # 1x1xNx3


        # Per-frame encoders.
        xf_pose = self.netArticulation(patch_feat)
        xf_cam = self.netPose(patch_feat)
        xf_shape = netShape.netShape(patch_feat)

        frozen_teacher = getattr(self, 'frozen_teacher', None)
        if frozen_teacher is not None:
            # Targets from the frozen Dessie teacher.
            with torch.no_grad():
                xf_pose_t, xf_cam_t, xf_shape_t = frozen_teacher.encode(patch_feat)
                pose_t = frozen_teacher.forward_pose(xf_pose_t, batch_size, num_frames)
                gt_cam, gt_root = frozen_teacher.forward_cam(xf_cam_t, batch_size, num_frames)
                pose_t = torch.cat([gt_root.unsqueeze(1), pose_t.view(batch_size * num_frames, 35, 6)], 1)
                gt_shape = frozen_teacher.forward_shape(xf_shape_t, batch_size, num_frames)
            self.articulation_angles_gt = pose_t.detach().view(batch_size, num_frames, -1)
            self.gt_cam = gt_cam.detach().view(batch_size, num_frames, -1)
            self.gt_shape = gt_shape.detach().view(batch_size, num_frames, -1)
            # Encoder-level (feature) targets -- the "encoded" half of
            # teacher supervision, paired with pose/cam/shape_feat_loss_weight.
            self.xf_pose_student = xf_pose
            self.xf_cam_student = xf_cam
            self.xf_shape_student = xf_shape
            self.xf_pose_teacher = xf_pose_t.detach()
            self.xf_cam_teacher = xf_cam_t.detach()
            self.xf_shape_teacher = xf_shape_t.detach()
        else:
            # Without a frozen teacher, use the model's own detached per-frame predictions.
            with torch.no_grad():
                pose = self.forward_pose(xf_pose.detach(), batch_size, num_frames)
                gt_cam, gt_root = self.forward_cam(xf_cam.detach(), batch_size, num_frames)
                pose = torch.cat([gt_root.unsqueeze(1), pose], 1)
                gt_shape = netShape.forward_shape(xf_shape.detach(), batch_size, num_frames)
            self.articulation_angles_gt = pose.detach().view(batch_size, num_frames, -1)
            self.gt_cam = gt_cam.detach().view(batch_size, num_frames, -1)
            self.gt_shape = gt_shape.detach().view(batch_size, num_frames, -1)


        xf_pose = xf_pose.view(batch_size, num_frames, -1)
        xf_cam = xf_cam.view(batch_size, num_frames, -1)
        xf_shape = xf_shape.view(batch_size, num_frames, -1)
        pad_mask = None
        is_training = kwargs.get('is_training', True)
        if self.enable_frame_masking_curriculum and is_training and num_frames > 1:
            min_k = max(1, min(self.frame_masking_min_frames, num_frames))
            pad_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=xf_shape.device)
            for b in range(batch_size):
                k = torch.randint(min_k, num_frames + 1, (1,)).item()
                keep_idx = torch.randperm(num_frames, device=xf_shape.device)[:k]
                pad_mask[b, keep_idx] = False  # False = visible to the fuser, True = masked out

        z_pose, z_cam, z_shape = self.fuser(xf_pose, xf_cam, xf_shape, pad_mask=pad_mask)
        # Post-transformer features, used by *_fused_feat_loss.
        self.z_pose_fused = z_pose
        self.z_cam_fused = z_cam
        self.z_shape_fused = z_shape

        if num_frames == 1:
            # Single frame: decode shape directly from the per-frame feature.
            pred_shape = netShape.forward_shape(xf_shape, batch_size, num_frames)
        else:
            pred_shape = netShape.forward_shape(z_shape, batch_size, 1).unsqueeze(1).repeat(1,num_frames,1)

        # Per-frame shape prediction for shape_recon_perframe_loss.
        if num_frames > 1:
            self.pred_shape_perframe = netShape.forward_shape(
                xf_shape, batch_size, num_frames
            ).view(batch_size, num_frames, -1)


        pred_pose = self.forward_pose(z_pose, batch_size, num_frames)
        pred_cam, pred_root = self.forward_cam(z_cam, batch_size, num_frames)
        pred_pose = torch.cat([pred_root.unsqueeze(1), pred_pose], 1)

        self.articulation_angles_pred = pred_pose.view(batch_size, num_frames, -1) #pytorch3d.transforms.matrix_to_euler_angles(pred_pose, convention='XYZ').view(batch_size, num_frames, -1) 
        self.pred_cam = pred_cam.view(batch_size, num_frames, -1)
        self.pred_shape = pred_shape.view(batch_size, num_frames, -1)

        pred_pose_rot = rot6d_to_rotmat(pred_pose).view(batch_size*num_frames, -1, 3, 3)

        verts_articulated, joints, _ = smal_model(self.pred_shape.view(batch_size*num_frames, -1), pred_pose_rot, self.pred_cam.view(batch_size*num_frames, -1))

        verts_articulated = verts_articulated.view(batch_size*num_frames, -1, 3)  # (B*F)xNx3
        

        bones = self._joints_to_bones(joints)
        bones = bones.reshape(batch_size, num_frames, -1, 2, 3)

        aux = {}
        aux['bones_pred'] = bones
        aux['posed_bones'] = bones
        aux['betas'] = pred_shape.view(batch_size*num_frames, -1)
        aux['pose'] = pred_pose.view(batch_size*num_frames, -1)
        aux['cam'] = pred_cam
        aux['shape'] = pred_shape
        aux['trans'] = self.pred_cam.view(batch_size*num_frames, -1)
        articulated_shape = Meshes(verts=verts_articulated, faces=shape.faces_padded(), textures=shape.textures)
        articulation_angles = pytorch3d.transforms.matrix_to_euler_angles(pred_pose_rot, convention='XYZ').view(batch_size, num_frames, -1)
        return articulated_shape, articulation_angles, aux
