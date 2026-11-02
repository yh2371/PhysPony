import torch
import numpy as np
from einops import repeat
from dataclasses import dataclass
from itertools import chain
from .PonyModelBase import PonyModelBase, PonyModelBaseConfig
from ..geometry.skinning import euler_angles_to_matrix
from ..utils import misc
from ..predictors import (
    BasePredictorBase, TemporalPosePredictor, BasePredictorConfig, TemporalPosePredictorConfig,
    FrozenKinematicTeacher
)

@dataclass
class PonyKinematicsModelConfig(PonyModelBaseConfig):
    cfg_predictor_base: BasePredictorConfig = None
    cfg_predictor_instance: TemporalPosePredictorConfig = None
    # Use a frozen copy of Dessie's camera/pose/shape pathway (FrozenTeacher) as the
    # distillation target. Loaded from DESSIE_CKPT in run.py.
    use_frozen_teacher: bool = False
    # Freeze the texture/shape/DINO branches and train only the motion transformer.
    freeze_pretrained_branches: bool = True
    # With freeze_pretrained_branches=False, still keep the deformation branch frozen.
    freeze_deform_branch: bool = False


class PonyKinematicsModel(PonyModelBase):
    def __init__(self, cfg: PonyKinematicsModelConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, PonyKinematicsModelConfig)
        self.netBase = BasePredictorBase(self.cfg_predictor_base)
        self.netInstance = TemporalPosePredictor(self.cfg_predictor_instance)

        self.frozen_teacher = FrozenKinematicTeacher() if self.use_frozen_teacher else None
        # Attach without registering, so the teacher is never saved or optimised.
        object.__setattr__(self.netInstance, 'frozen_teacher', self.frozen_teacher)

        self.arti_recon_loss_fn = torch.nn.MSELoss()
        self.cam_recon_loss_fn = torch.nn.MSELoss()
        self.shape_recon_loss_fn = torch.nn.MSELoss()
        self.pose_feat_loss_fn = torch.nn.MSELoss()
        self.cam_feat_loss_fn = torch.nn.MSELoss()
        self.shape_feat_loss_fn = torch.nn.MSELoss()
        self.root_recon_loss_fn = torch.nn.MSELoss()
        self.get_default_pose()

    def get_default_pose(self):
        pose_canon = torch.concat([torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(self.device)
        mvp_canon, w2c_canon, campos_canon = self.get_predictor("netInstance").get_camera_extrinsics_from_pose(pose_canon, offset_extra=self.cfg_render.offset_extra)
        viewpoint_arti = torch.FloatTensor([0, -120, 0]) / 180 * np.pi
        mtx = torch.eye(4).to(self.device)
        mtx[:3, :3] = euler_angles_to_matrix(viewpoint_arti, "XYZ")
        w2c_arti = torch.matmul(w2c_canon, mtx[None])
        mvp_arti = torch.matmul(mvp_canon, mtx[None])
        campos_arti = campos_canon @ torch.linalg.inv(mtx[:3, :3]).T
        self.default_pose, self.default_mvp, self.default_w2c, self.default_campos = pose_canon, mvp_arti, w2c_arti, campos_arti

    def set_eval(self):
        super().set_eval()
        self.get_predictor("netInstance").forward_train = self.get_predictor("netInstance").forward

    def set_train(self):
        super().set_train()
        if self.freeze_pretrained_branches:
            for param in chain(self.netBase.parameters(), self.get_predictor("netInstance").parameters()):
                param.requires_grad = False
        elif self.freeze_deform_branch and hasattr(self.get_predictor("netInstance"), "netDeform"):
            self.freeze_module(self.get_predictor("netInstance").netDeform)
        for param in self.get_predictor("netInstance").fuser.parameters():
            param.requires_grad = True

    def load_model_state(self, cp):
        super().load_model_state(cp)
        if self.get_predictor("netInstance").enable_motion_vae and self.freeze_pretrained_branches:
            for param in self.netBase.parameters():
                param.requires_grad = False
            for param in self.get_predictor("netInstance").parameters():
                param.requires_grad = False
            for param in self.get_predictor("netInstance").fuser.parameters():
                param.requires_grad = True

    def compute_regularizers(self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None, prior_shape=None, betas=None, **kwargs):
        losses, aux = super().compute_regularizers(
            arti_params=arti_params, deformation=deformation, pose_raw=pose_raw, posed_bones=posed_bones,
            prior_shape=prior_shape, betas=betas, **kwargs
        )
        if self.cfg_loss.arti_recon_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "articulation_angles_gt"):
            assert not self.get_predictor("netInstance").articulation_angles_gt.requires_grad
            losses["arti_recon_loss"] = self.arti_recon_loss_fn(  # L_teacher
                self.get_predictor("netInstance").articulation_angles_pred, self.get_predictor("netInstance").articulation_angles_gt
            )
        if self.cfg_loss.cam_recon_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "gt_cam"):
            assert not self.get_predictor("netInstance").gt_cam.requires_grad
            pred_cam = self.get_predictor("netInstance").pred_cam
            gt_cam = self.get_predictor("netInstance").gt_cam
            # Per-component normalization (see LossConfig.cam_recon_scale):
            # default [1,1,1] reproduces the plain MSE exactly.
            scale = torch.as_tensor(
                list(self.cfg_loss.cam_recon_scale), dtype=pred_cam.dtype, device=pred_cam.device
            )
            losses["cam_recon_loss"] = (((pred_cam - gt_cam) / scale) ** 2).mean()  # L_teacher
        if self.cfg_loss.root_recon_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "articulation_angles_gt"):
            losses["root_recon_loss"] = self.root_recon_loss_fn(  # L_teacher
                self.get_predictor("netInstance").articulation_angles_pred[..., :6],
                self.get_predictor("netInstance").articulation_angles_gt[..., :6],
            )
        if self.cfg_loss.shape_recon_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "gt_shape"):
            assert not self.get_predictor("netInstance").gt_shape.requires_grad
            gt_shape = self.get_predictor("netInstance").gt_shape          # B x F x 9
            pred_shape = self.get_predictor("netInstance").pred_shape      # B x F x 9

            # The prediction is per sequence; optionally average the per-frame target to match.
            if self.cfg_loss.pool_shape_target:
                gt_shape = gt_shape.mean(dim=1, keepdim=True).expand_as(gt_shape)
            losses["shape_recon_loss"] = self.shape_recon_loss_fn(pred_shape, gt_shape)  # L_teacher

            # Per-frame shape distillation against the teacher's per-frame betas.
            if self.cfg_loss.shape_recon_perframe_loss_weight > 0 and hasattr(
                self.get_predictor("netInstance"), "pred_shape_perframe"
            ):
                losses["shape_recon_perframe_loss"] = self.shape_recon_loss_fn(
                    self.get_predictor("netInstance").pred_shape_perframe,
                    self.get_predictor("netInstance").gt_shape,
                )

            # Logged only: the teacher's within-sequence betas variance.
            with torch.no_grad():
                raw_gt = self.get_predictor("netInstance").gt_shape
                losses["shape_target_within_var"] = (
                    (raw_gt - raw_gt.mean(dim=1, keepdim=True)) ** 2
                ).mean()
        # Encoder-feature distillation (only with use_frozen_teacher=true).
        if self.cfg_loss.pose_feat_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "xf_pose_teacher"):
            assert not self.get_predictor("netInstance").xf_pose_teacher.requires_grad
            losses["pose_feat_loss"] = self.pose_feat_loss_fn(
                self.get_predictor("netInstance").xf_pose_student, self.get_predictor("netInstance").xf_pose_teacher
            )
        if self.cfg_loss.cam_feat_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "xf_cam_teacher"):
            assert not self.get_predictor("netInstance").xf_cam_teacher.requires_grad
            losses["cam_feat_loss"] = self.cam_feat_loss_fn(
                self.get_predictor("netInstance").xf_cam_student, self.get_predictor("netInstance").xf_cam_teacher
            )
        if self.cfg_loss.shape_feat_loss_weight > 0 and hasattr(self.get_predictor("netInstance"), "xf_shape_teacher"):
            assert not self.get_predictor("netInstance").xf_shape_teacher.requires_grad
            losses["shape_feat_loss"] = self.shape_feat_loss_fn(
                self.get_predictor("netInstance").xf_shape_student, self.get_predictor("netInstance").xf_shape_teacher
            )
        # Post-transformer feature anchoring to the teacher's per-frame features.
        net = self.get_predictor("netInstance")
        fused_specs = [
            ("pose_fused_feat_loss", self.cfg_loss.pose_fused_feat_loss_weight, "z_pose_fused", "xf_pose_teacher"),
            ("cam_fused_feat_loss", self.cfg_loss.cam_fused_feat_loss_weight, "z_cam_fused", "xf_cam_teacher"),
            ("shape_fused_feat_loss", self.cfg_loss.shape_fused_feat_loss_weight, "z_shape_fused", "xf_shape_teacher"),
        ]
        for name, weight, z_attr, t_attr in fused_specs:
            if weight <= 0 or not hasattr(net, z_attr) or not hasattr(net, t_attr):
                continue
            z = getattr(net, z_attr)
            t = getattr(net, t_attr)
            assert not t.requires_grad
            # Teacher features are (B*F, D, 1, 1); pool over frames for the per-sequence shape feature.
            t = t.view(z.shape[0], -1, t.shape[1]) if z.dim() == 3 else t.view(z.shape[0], -1, t.shape[1]).mean(dim=1)
            # Match feature norms only, leaving the direction free.
            losses[name] = ((z.norm(dim=-1) - t.norm(dim=-1)) ** 2).mean()
        return losses, aux

    def log_visuals(self, log, logger, **kwargs):
        log = super().log_visuals(log, logger)
        def log_video(name, frames, fps=10):
            logger.add_video(log.logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), log.total_iter, fps=fps)
        if log.num_frames > 1:
            log_video("sequence_image_gt", log.input_image[0])
            log_video("sequence_mask_gt", repeat(log.mask_gt[0], "f h w -> f c h w", c=3))
            if self.get_predictor("netInstance").enable_motion_vae and not log.is_training:
                suffix = "gen"
            else:
                suffix = "pred"
            log_video(f"sequence_image_{suffix}", log.image_pred[0])
            log_video(f"sequence_mask_{suffix}", repeat(log.mask_pred[0], "f h w -> f c h w", c=3))
            log_video(f"sequence_instance_geo_normal_{suffix}", log.geo_normal[0])
            if hasattr(log, "geo_normal_gt"):
                log_video(f"sequence_instance_geo_normal_gt", log.geo_normal_gt[0])
