from dataclasses import dataclass, field
from typing import List, Optional, Dict
from types import SimpleNamespace
import os
import numpy as np
import pytorch3d
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from torch.nn import functional as F
from scipy.spatial import cKDTree
from model.utils import misc
from model.utils.smooth_loss import SmoothLoss
from model.dataloaders import DataLoaderConfig
from model.render import util
from model.render import render_2
def assert_finite(name, x):
    finite_mask = torch.isfinite(x)
    if not finite_mask.all():
        bad = (~finite_mask).nonzero(as_tuple=False)[:8]
        # torch 1.10 (this environment's version) has neither Tensor.nanmin()
        # nor torch.nanmin() -- filter to the finite subset instead.
        finite_vals = x[finite_mask]
        if finite_vals.numel() > 0:
            stats = f"min={finite_vals.min().item():.3e}, max={finite_vals.max().item():.3e} (finite subset)"
        else:
            stats = "no finite values at all"
        raise RuntimeError(f"[NON-FINITE] {name} has NaN/Inf at {bad.tolist()} | stats: {stats}")

def lift_cam_to_world(p_cam, R_c2w, t_c2w):
    """
    p_cam:  (B,T,3) point in camera frame.
    R_c2w:  (B,T,3,3) camera->world rotation.
    t_c2w:  (B,T,3)   camera->world translation.
    Returns p_world: (B,T,3).
    """
    return torch.einsum("btij,btj->bti", R_c2w, p_cam) + t_c2w

def first_diff(x):
    return x[:, 1:] - x[:, :-1]

def second_diff(x):
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

def third_diff(x):
    # Jerk (3rd time derivative)
    return x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]

def hf_energy_loss(p, fps=30.0, cutoff_hz=5.0):
    """Penalize FFT energy above ``cutoff_hz``.

    Horse gaits live below ~5 Hz; anything above that is process noise.
    Diagnostics on ID_1_0001 showed 34% of root-trans FFT energy lives
    above 5 Hz — this loss directly attacks that.

    Args:
        p: (B, T, D) trajectory tensor.
        fps: frame rate.
        cutoff_hz: gait cut-off; energy above this is penalized.
    Returns:
        scalar mean(|FFT_hf|^2) normalized by sequence length.
    """
    B, T, D = p.shape
    # rFFT along time dim
    p_det = p - p.mean(dim=1, keepdim=True)
    F = torch.fft.rfft(p_det, dim=1)                   # (B, T//2+1, D)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(p.device)
    hf_mask = (freqs > cutoff_hz).float().view(1, -1, 1)
    return ((F.real ** 2 + F.imag ** 2) * hf_mask).mean()


# Losses gated by LossConfig.distill_loss_iter_range.
DISTILL_STAGE_LOSSES = (
    'arti_recon_loss', 'cam_recon_loss', 'shape_recon_loss', 'shape_recon_perframe_loss',
    'root_recon_loss', 'pose_feat_loss', 'cam_feat_loss', 'shape_feat_loss',
    'pose_fused_feat_loss', 'cam_fused_feat_loss', 'shape_fused_feat_loss',
    'campose_smooth_loss', 'camposevel_smooth_loss', 'predcam_smooth_loss',
    'predcamvel_smooth_loss', 'predcam_speed_hingeloss', 'kld_loss',
)


@dataclass
class OptimizerConfig:
    lr: float = 0.0001
    weight_decay: float = 0.0
    use_scheduler: bool = False
    scheduler_milestone: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    scheduler_gamma: float = 0.5


@dataclass
class RenderConfig:
    spatial_scale: float = 5.
    background_mode: str = 'none'  # none (black), white, checkerboard, background, input
    render_flow: bool = False
    cam_pos_z_offset: float = 10.0
    fov: float = 25.
    renderer_spp: int = 1
    offset_extra: float = 0.0
    render_default: bool = False
    renderer_type: str = 'pytorch3d'  # 'nvdiffrast' or 'pytorch3d'


@dataclass
class LossConfig:
    mask_loss_weight: float = 10.
    mask_dt_loss_weight: float = 0.
    mask_inv_dt_loss_weight: float = 100.
    rgb_loss_weight: float = 1.
    dino_feat_im_loss_weight: float = 10.
    sdf_bce_reg_loss_weight: float = 0.
    sdf_joints_reg_loss_weight: float = 0. #1.
    sdf_gradient_reg_loss_weight: float = 0.01
    logit_loss_weight: float = 1.
    logit_loss_target_weight: float = 0.
    logit_loss_dino_feat_im_loss_multiplier: float = 50.  # increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
    arti_reg_loss_iter_range: List[int] = field(default_factory=lambda: [60000, float("inf")])
    arti_reg_loss_weight: float = 0.1
    betas_reg_loss_weight: float = 0.1
    deform_reg_loss_weight: float = 10.
    arti_recon_loss_weight: float = 0
    cam_recon_loss_weight: float = 0
    shape_recon_loss_weight: float = 0

    # Average the teacher's per-frame betas over the sequence before shape_recon_loss.
    pool_shape_target: bool = False

    # Per-frame shape distillation: supervise betas decoded from each frame's
    # un-fused feature against the teacher's per-frame betas. Prevents shape collapse.
    shape_recon_perframe_loss_weight: float = 0

    # Logged only (teacher's within-sequence betas variance); keep at 0.
    shape_target_within_var_weight: float = 0
    # Encoder-feature distillation against the frozen teacher (needs use_frozen_teacher).
    pose_feat_loss_weight: float = 0
    cam_feat_loss_weight: float = 0
    shape_feat_loss_weight: float = 0
    # Per-axis scale for cam_recon_loss, ((pred - gt) / scale)^2 over [tx, ty, tz].
    # Set to the teacher's per-axis std so depth does not dominate the loss.
    cam_recon_scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Extra weight on the 6 root-orientation dims of the articulation.
    root_recon_loss_weight: float = 0
    # Anchor the fused (post-transformer) features to the teacher's per-frame
    # encoder features. Keep small: it fixes their scale, not their content.
    pose_fused_feat_loss_weight: float = 0
    cam_fused_feat_loss_weight: float = 0
    shape_fused_feat_loss_weight: float = 0
    prior_normal_reg_loss_weight: float = 0
    instance_normal_reg_loss_weight: float = 0
    depth_loss_weight: float = 1

    smooth_type: str = "dislocation"
    loss_type: str = "l2"
    arti_smooth_loss_weight: float = 0.
    deform_smooth_loss_weight: float = 0.
    campose_smooth_loss_weight: float = 0.
    camposevel_smooth_loss_weight: float = 0.
    predcam_smooth_loss_weight: float = 0.
    predcamvel_smooth_loss_weight: float = 0.
    artivel_smooth_loss_weight: float = 0.
    bone_smooth_loss_weight: float = 0.
    bonevel_smooth_loss_weight: float = 0.
    predcam_speed_hingeloss_weight: float = 0.

    # Camera-frame smoothness losses (raw frame differences, per-axis weights).
    pred_cam_pos_loss_weight:   float = 0.   # ||p_cam - p_cam.mean||
    pred_cam_vel_loss_weight:   float = 0.   # ||v_cam||  (frame-diff)
    pred_cam_accel_loss_weight: float = 0.   # ||a_cam||  (2nd diff) <-- jitter
    pred_cam_jerk_loss_weight:  float = 0.   # ||j_cam||  (3rd diff) <-- HF

    # Per-axis weights [tx, ty, tz].  In camera frame, tz = depth, which is
    # the noisy axis (0.23 raw vs 0.005-0.01 on xy).  Recommended: [1, 1, 3].
    cam_axis_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # High-frequency (> 5 Hz) penalty on the camera-frame translation; needs >= 8 frames.
    pred_cam_hf_loss_weight: float = 0.
    hf_cutoff_hz: float = 5.0
    fps: float = 30.0

    # Pull camera-frame translation toward an EMA of the model's own prediction,
    # starting at pred_cam_ema_start_iter.
    pred_cam_ema_loss_weight: float = 0.
    pred_cam_ema_decay: float = 0.99
    pred_cam_ema_start_iter: int = 5000

    # World-frame smoothness losses (off by default).
    pred_world_pos_loss_weight:   float = 0.
    pred_world_vel_loss_weight:   float = 0.
    pred_world_accel_loss_weight: float = 0.
    pred_world_jerk_loss_weight:  float = 0.
    world_axis_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    pred_world_hf_loss_weight: float = 0.
    pred_world_ema_loss_weight: float = 0.
    pred_world_ema_decay: float = 0.99
    pred_world_ema_start_iter: int = 5000

    arti_recon_loss_weight: float = 5.0
    kld_loss_weight: float = 0.001

    # Iteration range in which DISTILL_STAGE_LOSSES are active, e.g. [10000, inf]
    # for a rendering-only warmup. [-1, -1] = always on.
    distill_loss_iter_range: List[int] = field(default_factory=lambda: [-1, -1])

    mask_disc_loss_weight: float = 0.1
    mask_disc_loss_rv_weight: float = 0.0
    mask_disc_loss_iv_weight: float = 0.0

    logit_loss_dino_feat_im_loss_multiplier_dict: Dict[int, float] = field(default_factory=lambda: {0: 50., 300000: 500.})
    dino_feat_im_loss_weight_dict: Dict[int, float] =  field(default_factory=lambda: {0: 10., 300000: 1.})
    logit_loss_mask_multiplier: float = 0.05
    logit_loss_mask_inv_dt_multiplier: float = 0.05

@dataclass
class PonyModelBaseConfig:
    name: str
    dataset: DataLoaderConfig
    cfg_optim_base: OptimizerConfig
    cfg_optim_instance: OptimizerConfig
    cfg_render: RenderConfig
    cfg_loss: LossConfig
    cfg_predictor_base: None
    cfg_predictor_instance: None
    enable_render: bool = True
    export_mesh: bool = False  # opt-in: bake+save a textured OBJ/MTL/PNG per frame in save_results

    extra_renders: Optional[List] = None

    # Max gradient norm for clipping before each optimizer step (None = off).
    grad_clip_max_norm: Optional[float] = None

    # Freeze the Dessie-initialised per-frame encoders and decode heads.
    freeze_kinematic_core: bool = True

    # Freeze only the three per-frame encoders (netArticulation, netPose,
    # netShape.netShape); decode heads and the motion transformer still train.
    freeze_kinematic_encoders: bool = False

    # Load only the ViT and per-frame encoders from Dessie; decode heads start from scratch.
    dessie_encoders_only: bool = False

    # Freeze the shape decode head at Dessie's weights (loaded even with dessie_encoders_only).
    freeze_shape_decoder: bool = False


class PonyModelBase:
    def __init__(self, cfg: PonyModelBaseConfig):
        misc.load_cfg(self, cfg, PonyModelBaseConfig)

        if self.cfg_optim_base.use_scheduler:
            self.make_scheduler_base = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.cfg_optim_base.scheduler_milestone, gamma=self.cfg_optim_base.scheduler_gamma)
        if self.cfg_optim_instance.use_scheduler:
            self.make_scheduler_instance = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.cfg_optim_instance.scheduler_milestone, gamma=self.cfg_optim_instance.scheduler_gamma)

        self.device = None
        self.accelerator = None
        self.mixed_precision = False
        self.total_loss = 0.

        self.smooth_loss_fn = SmoothLoss(
            frame_dim=1, smooth_type=self.cfg_loss.smooth_type, loss_type=self.cfg_loss.loss_type
        )
        print("Loss weights:")
        for key, value in vars(self.cfg_loss).items():
            print(f"\t{key}={value}")

    def get_predictor(self, name):
        predictor = getattr(self, name)
        if isinstance(predictor, torch.nn.parallel.DistributedDataParallel):
            return predictor.module
        return predictor

    def load_dessie(self, dessie_ckpt):
        dessie = torch.load(dessie_ckpt)
        new_state_dict = {}
        new_state_dict_base = {}
        for key in dessie['state_dict'].keys():
            if "model.encoder.model" in key:
                print("Loaded", key)
                new_state_dict[key.replace("model.encoder.model", "netEncoder.ViT")] = dessie['state_dict'][key]
            if "model.key_encoder_pose" in key:
                new_state_dict[key.replace("model.key_encoder_pose", "netArticulation")] = dessie['state_dict'][key]
                print("Loaded", key)
            if "decpose" in key:
                new_state_dict[key.replace("model.", "")] = dessie['state_dict'][key]
                print("Loaded", key)
        for key in dessie['state_dict'].keys():
            if "model.key_encoder_shape" in key:
                new_state_dict_base[key.replace("model.key_encoder_shape", "netShape.netShape")] = dessie['state_dict'][key]
                print("Loaded", key)
            if "decshape" in key:
                new_state_dict_base[key.replace("model.", "netShape.")] = dessie['state_dict'][key]
                print("Loaded", key)
        for key in dessie['state_dict'].keys():
            if "model.key_encoder_cam" in key:
                new_state_dict[key.replace("model.key_encoder_cam", "netPose")] = dessie['state_dict'][key]
                print("Loaded", key)
            if "deccam" in key:
                new_state_dict[key.replace("model.", "")] = dessie['state_dict'][key]
                print("Loaded", key)
            if "decroot" in key:
                new_state_dict[key.replace("model.", "")] = dessie['state_dict'][key]
                print("Loaded", key)
            
        new_state_dict['init_cam'] = dessie['state_dict']['model.init_cam']
        new_state_dict['init_pose'] = dessie['state_dict']['model.init_pose']
        new_state_dict_base['netShape.init_shape'] = dessie['state_dict']['model.init_shape']
        new_state_dict['init_pose'] = dessie['state_dict']['model.init_pose']

        if getattr(self, "dessie_encoders_only", False):
            # Encoder-only init: keep the ViT and per-frame encoders, drop the decode heads.
            def _is_encoder(k):
                return k.startswith("netEncoder.") or k.startswith("netArticulation.") \
                    or k.startswith("netPose.") or k.startswith("netShape.netShape.")

            # A frozen shape head must keep Dessie's weights.
            def _is_kept_shape_decoder(k):
                return getattr(self, "freeze_shape_decoder", False) and (
                    k.startswith("netShape.fc_decshape1.")
                    or k.startswith("netShape.fc_decshape2.")
                    or k.startswith("netShape.decshape.")
                    or k == "netShape.init_shape"
                )

            def _keep(k):
                return _is_encoder(k) or _is_kept_shape_decoder(k)

            dropped = [k for k in list(new_state_dict) + list(new_state_dict_base) if not _keep(k)]
            kept_shape_dec = [k for k in new_state_dict_base if _is_kept_shape_decoder(k)]
            new_state_dict = {k: v for k, v in new_state_dict.items() if _keep(k)}
            new_state_dict_base = {k: v for k, v in new_state_dict_base.items() if _keep(k)}
            print(f"[load_dessie] encoders_only: kept {len(new_state_dict)}+{len(new_state_dict_base)} "
                  f"encoder tensors, dropped {len(dropped)} decode-head/init tensors")
            if getattr(self, "freeze_shape_decoder", False):
                assert kept_shape_dec, (
                    "freeze_shape_decoder=True but load_dessie found no decshape tensors to keep; "
                    "the head would be frozen at random init. Check the Dessie checkpoint."
                )
                print(f"[load_dessie] freeze_shape_decoder: additionally kept "
                      f"{len(kept_shape_dec)} shape decode-head tensors: {sorted(kept_shape_dec)}")

        self.get_predictor("netBase").load_state_dict(new_state_dict_base, strict=False)
        self.get_predictor("netInstance").load_state_dict(new_state_dict, strict=False)
        

    def load_model_state(self, cp):
        base_missing, base_unexpected = self.get_predictor("netBase").load_state_dict(cp["netBase"], strict=False)
        instance_missing, instance_unexpected = self.get_predictor("netInstance").load_state_dict(cp["netInstance"], strict=False)
        if base_missing: print(f"Missing keys in netBase:\n{base_missing}")
        if base_unexpected: print(f"Unexpected keys in netBase:\n{base_unexpected}")
        if instance_missing: print(f"Missing keys in netInstance:\n{instance_missing}")
        if instance_unexpected: print(f"Unexpected keys in netInstance:\n{instance_unexpected}")

    def load_optimizer_state(self, cp):
        self.optimizerBase.load_state_dict(cp["optimizerBase"])
        self.optimizerInstance.load_state_dict(cp["optimizerInstance"])
        if self.cfg_optim_base.use_scheduler and 'schedulerBase' in cp:
            self.schedulerBase.load_state_dict(cp["schedulerBase"])
        if self.cfg_optim_instance.use_scheduler and 'schedulerInstance' in cp:
            self.schedulerInstance.load_state_dict(cp["schedulerInstance"])

    def get_model_state(self):
        state = {
            "netBase": self.accelerator.unwrap_model(self.netBase).state_dict(),
            "netInstance": self.accelerator.unwrap_model(self.netInstance).state_dict()
        }
        return state

    def get_optimizer_state(self):
        state = {"optimizerBase": self.optimizerBase.state_dict(),
                 "optimizerInstance": self.optimizerInstance.state_dict()}
        if self.cfg_optim_base.use_scheduler:
            state["schedulerBase"] = self.schedulerBase.state_dict()
        if self.cfg_optim_instance.use_scheduler:
            state["schedulerInstance"] = self.schedulerInstance.state_dict()
        return state

    def to(self, device):
        self.device = device
        self.get_predictor("netBase").to(device)
        self.get_predictor("netInstance").to(device)
        if getattr(self, "frozen_teacher", None) is not None:
            self.frozen_teacher.to(device)
        try:
            self.get_predictor("netBase").netShape.verts = self.get_predictor("netBase").netShape.verts.to(device)
            self.get_predictor("netBase").netShape.indices = self.get_predictor("netBase").netShape.indices.to(device)
        except:
            pass

    def set_train(self):
        self.get_predictor("netBase").train()
        self.get_predictor("netInstance").train()
        if self.freeze_kinematic_core:
            self.freeze_module(self.netInstance.netArticulation)
            self.freeze_module(self.netBase.netShape.netShape)
            self.freeze_module(self.netInstance.fc_decpose1)
            self.freeze_module(self.netInstance.fc_decpose2)
            self.freeze_module(self.netInstance.decpose)
            self.freeze_module(self.netBase.netShape.fc_decshape1)
            self.freeze_module(self.netBase.netShape.fc_decshape2)
            self.freeze_module(self.netBase.netShape.decshape)
            self.freeze_module(self.netInstance.netPose)
            self.freeze_module(self.netInstance.fc_deccam1)
            self.freeze_module(self.netInstance.fc_deccam2)
            self.freeze_module(self.netInstance.deccam)
            self.freeze_module(self.netInstance.decroot)
        elif self.freeze_kinematic_encoders:
            # Encoders only -- decode heads and the fuser stay trainable.
            self.freeze_module(self.netInstance.netArticulation)
            self.freeze_module(self.netInstance.netPose)
            self.freeze_module(self.netBase.netShape.netShape)

        if self.freeze_shape_decoder and not self.freeze_kinematic_core:
            self.freeze_module(self.netBase.netShape.fc_decshape1)
            self.freeze_module(self.netBase.netShape.fc_decshape2)
            self.freeze_module(self.netBase.netShape.decshape)
            print("[set_train] freeze_shape_decoder: pinned fc_decshape1/2 + decshape "
                  "at Dessie; fuse_shape left trainable")

    def freeze_module(self, m):
        print("Freezing module...")
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    def set_train_post_load(self):
        # perform any post-load operations, i.e copy weights between networks
        pass

    def set_eval(self):
        self.get_predictor("netBase").eval()
        self.get_predictor("netInstance").eval()

    def reset_optimizers(self):
        print("Resetting optimizers...")
        self.optimizerBase = get_optimizer(self.get_predictor("netBase"), lr=self.cfg_optim_base.lr, weight_decay=self.cfg_optim_base.weight_decay)
        self.optimizerInstance = get_optimizer(self.get_predictor("netInstance"), self.cfg_optim_instance.lr, weight_decay=self.cfg_optim_instance.weight_decay)
        if self.cfg_optim_base.use_scheduler:
            self.schedulerBase = self.make_scheduler_base(self.optimizerBase)
        if self.cfg_optim_instance.use_scheduler:
            self.schedulerInstance = self.make_scheduler_instance(self.optimizerInstance)

    def backward(self):
        self.optimizerInstance.zero_grad()
        self.optimizerBase.zero_grad()
        if self.mixed_precision and self.scaler:
            self.scaler.scale(self.total_loss)
            self.accelerator.backward(self.total_loss)
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.get_predictor("netInstance").parameters(), max_norm=self.grad_clip_max_norm)
                torch.nn.utils.clip_grad_norm_(self.get_predictor("netBase").parameters(), max_norm=self.grad_clip_max_norm)
            # Step optimizer if it is used, Unused optimizers will raise assertion error when unscaling
            try:
                self.scaler.step(self.optimizerInstance)
            except AssertionError:
                pass
            try:
                self.scaler.step(self.optimizerBase)
            except AssertionError:
                pass
            self.scaler.update()
        else:
            self.accelerator.backward(self.total_loss)
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.get_predictor("netInstance").parameters(), max_norm=self.grad_clip_max_norm)
                torch.nn.utils.clip_grad_norm_(self.get_predictor("netBase").parameters(), max_norm=self.grad_clip_max_norm)
            self.optimizerInstance.step()
            self.optimizerBase.step()
        self.total_loss = 0.

    def scheduler_step(self):
        if self.cfg_optim_base.use_scheduler:
            self.schedulerBase.step()
        if self.cfg_optim_instance.use_scheduler:
            self.schedulerInstance.step()

    def render(self, render_modes, shape, texture, mvp, w2c, campos, resolution, background=None, im_features=None, light=None, prior_shape=None, dino_net=None, bsdf='diffuse', two_sided_shading=True, num_frames=None, spp=None):
        assert self.cfg_render.renderer_type == 'pytorch3d', \
            f"Only the pytorch3d renderer is supported, got renderer_type={self.cfg_render.renderer_type!r}"
        mvp, w2c, campos, im_features = map(to_float, [mvp, w2c, campos, im_features])
        h, w = resolution
        N = len(mvp)
        if background is None:
            background = self.cfg_render.background_mode
        if spp is None:
            spp = self.cfg_render.renderer_spp

        if background in ['none', 'black']:
            bg_image = torch.zeros((N, h, w, 3), device=mvp.device)
        elif background == 'white':
            bg_image = torch.ones((N, h, w, 3), device=mvp.device)
        elif background == 'checkerboard':
            bg_image = torch.FloatTensor(util.checkerboard((h, w), 8), device=mvp.device).repeat(N, 1, 1, 1)  # NxHxWxC
        else:
            raise NotImplementedError

        rendered, vertex_2d = render_2.render_mesh(
            ctx=None,  # Not used in PyTorch3D
            mesh=shape,
            mtx_in=mvp,
            w2c=w2c,
            view_pos=campos,
            material=texture,
            lgt=light,
            resolution=resolution,
            spp=spp,
            num_layers=1,
            msaa=True,
            background=bg_image,
            bsdf=bsdf,
            feat=im_features,
            render_modes=render_modes,
            prior_mesh=prior_shape,
            two_sided_shading=two_sided_shading,
            dino_net=dino_net,
            num_frames=num_frames,
            )
        return rendered, vertex_2d

    def compute_reconstruction_losses(self, image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode='none', reduce=False, betas=None):
        losses = {}
        batch_size, num_frames, _, h, w = image_pred.shape  # BxFxCxHxW

        ## mask L2 loss
        mask_pred_valid = mask_pred * mask_valid
        mask_loss = (mask_pred_valid - mask_gt) ** 2
        losses['mask_loss'] = mask_loss.reshape(batch_size, num_frames, -1).mean(2)
        losses['mask_dt_loss'] = (mask_pred * mask_dt[:,:,1]).reshape(batch_size, num_frames, -1).mean(2)
        losses['mask_inv_dt_loss'] = ((1-mask_pred) * mask_dt[:,:,0]).reshape(batch_size, num_frames, -1).mean(2)

        mask_pred_binary = (mask_pred_valid > 0.).float().detach()
        mask_both_binary = collapseBF(mask_pred_binary * mask_gt)  # BFxHxW
        mask_both_binary = (nn.functional.avg_pool2d(mask_both_binary.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float().detach()  # erode by 1 pixel
        mask_both_binary = expandBF(mask_both_binary, b=batch_size, f=num_frames)  # BxFxHxW

        ## RGB L1 loss
        rgb_loss = (image_pred - image_gt).abs()
        if background_mode in ['background', 'input']:
            pass
        else:
            rgb_loss = rgb_loss * mask_both_binary.unsqueeze(2)
        losses['rgb_loss'] = rgb_loss.reshape(batch_size, num_frames, -1).mean(2)

        ## flow loss between consecutive frames
        if flow_pred is not None:
            flow_loss = (flow_pred - flow_gt) ** 2.
            flow_loss_mask = mask_both_binary[:,:-1].unsqueeze(2).expand_as(flow_gt)

            ## ignore frames where GT flow is too large (likely inaccurate)
            large_flow = (flow_gt.abs() > 0.5).float() * flow_loss_mask
            large_flow = (large_flow.view(batch_size, num_frames-1, -1).sum(2) > 0).float()
            self.large_flow = large_flow

            flow_loss = flow_loss * flow_loss_mask * (1 - large_flow[:,:,None,None,None])
            num_mask_pixels = flow_loss_mask.reshape(batch_size, num_frames-1, -1).sum(2).clamp(min=1)
            losses['flow_loss'] = (flow_loss.reshape(batch_size, num_frames-1, -1).sum(2) / num_mask_pixels)

        ## DINO feature loss
        if dino_feat_im_pred is not None and dino_feat_im_gt is not None:
            dino_feat_loss = (dino_feat_im_pred - dino_feat_im_gt) ** 2
            dino_feat_loss = dino_feat_loss * mask_both_binary.unsqueeze(2)
            losses['dino_feat_im_loss'] = dino_feat_loss.reshape(batch_size, num_frames, -1).mean(2)


        if reduce:
            for k, v in losses.item():
                losses[k] = v.mean()
        return losses

    def compute_regularizers(self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None, prior_shape=None, bones_pred=None, betas=None, pred_cam=None, R_cw=None, t_cw=None, total_iter=None, batch_size=None, num_frames=None, **kwargs):
        """
        Args:
            R_cw, t_cw: (B, T, 3, 3) / (B, T, 3) metric camera transforms, used
                to lift ``pred_cam`` to world space for the world-frame losses.
            total_iter: current iteration (schedules the EMA loss).
            batch_size, num_frames: batch dimensions.
        """
        losses = {}
        aux = {}
        if betas is not None:
            losses['betas_reg_loss'] = (betas ** 2).mean()  # R_betas
        if arti_params is not None:
            losses['arti_reg_loss'] = (arti_params ** 2).mean()  # R_art
        if deformation is not None:
            losses['deform_reg_loss'] = (deformation ** 2).mean()  # R_def

        # Motion regularizers on sequence data
        if "sequence" in self.dataset.data_type and self.dataset.num_frames > 1:
            if batch_size is not None and num_frames is not None:
                b, f = batch_size, num_frames
            elif arti_params is not None:
                b, f = arti_params.shape[:2]
            else:
                b, f = None, None
            if self.cfg_loss.deform_smooth_loss_weight > 0 and deformation is not None:
                losses["deform_smooth_loss"] = self.smooth_loss_fn(expandBF(deformation, b, f))
            if arti_params is not None:
                if self.cfg_loss.arti_smooth_loss_weight > 0:
                    losses["arti_smooth_loss"] = self.smooth_loss_fn(arti_params)
                if self.cfg_loss.artivel_smooth_loss_weight > 0:
                    artivel = arti_params[:, 1:, ...] - arti_params[:, :(f-1), ...]
                    losses["artivel_smooth_loss"] = self.smooth_loss_fn(artivel)  # R_temp
            if pose_raw is not None:
                campose = expandBF(pose_raw, b, f)
                if self.cfg_loss.campose_smooth_loss_weight > 0:
                    losses["campose_smooth_loss"] = self.smooth_loss_fn(campose)
                if self.cfg_loss.camposevel_smooth_loss_weight > 0:
                    camposevel = campose[:, 1:, ...] - campose[:, :(f-1), ...]
                    losses["camposevel_smooth_loss"] = self.smooth_loss_fn(camposevel)
            if pred_cam is not None:
                pred_cam = pred_cam.view(b, f, -1)

                # Camera-space smoothness on the full translation vector.
                if self.cfg_loss.predcam_smooth_loss_weight > 0:
                    losses["predcam_smooth_loss"] = self.smooth_loss_fn(pred_cam)
                if self.cfg_loss.predcamvel_smooth_loss_weight > 0:
                    predcamvel = pred_cam[:, 1:, ...] - pred_cam[:, :(f-1), ...]
                    losses["predcamvel_smooth_loss"] = self.smooth_loss_fn(predcamvel)

                # Camera-frame smoothness with per-axis weights.
                caw = torch.tensor(
                    self.cfg_loss.cam_axis_weights,
                    device=pred_cam.device, dtype=pred_cam.dtype
                ).view(1, 1, 3)

                # Velocity (frame-to-frame difference).
                v_cam = first_diff(pred_cam)                 # (B, T-1, 3)
                # Acceleration (second difference) -- the closest analytic
                # proxy to the diagnostic's per-frame `acc_std` metric.
                a_cam = second_diff(pred_cam) if f >= 3 else None  # (B, T-2, 3)
                # Jerk (third difference).
                j_cam = third_diff(pred_cam)  if f >= 4 else None  # (B, T-3, 3)

                if self.cfg_loss.pred_cam_pos_loss_weight > 0:
                    losses["pred_cam_pos_loss"] = (
                        (pred_cam - pred_cam.mean(1, keepdim=True)) ** 2 * caw
                    ).mean()
                if self.cfg_loss.pred_cam_vel_loss_weight > 0:
                    losses["pred_cam_vel_loss"] = (v_cam ** 2 * caw).mean()
                if self.cfg_loss.pred_cam_accel_loss_weight > 0 and a_cam is not None:
                    losses["pred_cam_accel_loss"] = (a_cam ** 2 * caw).mean()
                if self.cfg_loss.pred_cam_jerk_loss_weight > 0 and j_cam is not None:
                    losses["pred_cam_jerk_loss"] = (j_cam ** 2 * caw).mean()

                # FFT band penalty (>5 Hz = process noise) in cam frame.
                nyquist = self.cfg_loss.fps / 2.0
                if (
                    self.cfg_loss.pred_cam_hf_loss_weight > 0
                    and f >= 8
                    and self.cfg_loss.hf_cutoff_hz < nyquist
                ):
                    # Subtract per-axis mean so the DC component (the
                    # large tz offset ~30-100m) does not dominate the FFT.
                    losses["pred_cam_hf_loss"] = hf_energy_loss(
                        pred_cam * caw.sqrt(),
                        fps=self.cfg_loss.fps,
                        cutoff_hz=self.cfg_loss.hf_cutoff_hz,
                    )

                # Hinge on per-frame camera translation jumps larger than 5 units.
                if self.cfg_loss.predcam_speed_hingeloss_weight > 0:
                    v_norm = v_cam.norm(dim=-1)
                    losses["predcam_speed_hingeloss"] = torch.clamp(
                        v_norm - 5.0, min=0.0
                    ).mean()

                # EMA self-distillation in CAMERA frame.
                if (
                    self.cfg_loss.pred_cam_ema_loss_weight > 0
                    and total_iter is not None
                    and total_iter >= self.cfg_loss.pred_cam_ema_start_iter
                ):
                    if (
                        not hasattr(self, "_pred_cam_ema")
                        or self._pred_cam_ema is None
                        or self._pred_cam_ema.shape != pred_cam.shape
                    ):
                        self._pred_cam_ema = pred_cam.detach()
                    decay = float(self.cfg_loss.pred_cam_ema_decay)
                    self._pred_cam_ema = (
                        decay * self._pred_cam_ema + (1 - decay) * pred_cam.detach()
                    )
                    losses["pred_cam_ema_loss"] = (
                        (pred_cam - self._pred_cam_ema) ** 2 * caw
                    ).mean()

                # World-frame smoothness (opt-in).
                any_world_w = (
                    self.cfg_loss.pred_world_pos_loss_weight
                    + self.cfg_loss.pred_world_vel_loss_weight
                    + self.cfg_loss.pred_world_accel_loss_weight
                    + self.cfg_loss.pred_world_jerk_loss_weight
                    + self.cfg_loss.pred_world_hf_loss_weight
                    + self.cfg_loss.pred_world_ema_loss_weight
                ) > 0

                if any_world_w and R_cw is not None and t_cw is not None:
                    # lift_cam_to_world now uses the CORRECT c2w formula:
                    # p_world = R_c2w @ p_cam + t_c2w.
                    p_world = lift_cam_to_world(pred_cam, R_cw, t_cw)
                    aw = torch.tensor(
                        self.cfg_loss.world_axis_weights,
                        device=p_world.device, dtype=p_world.dtype
                    ).view(1, 1, 3)

                    # Use raw frame differences (no /dt) to keep
                    # magnitudes stable.
                    v_world = first_diff(p_world)
                    a_world = second_diff(p_world) if f >= 3 else None
                    j_world = third_diff(p_world)  if f >= 4 else None

                    if self.cfg_loss.pred_world_pos_loss_weight > 0:
                        losses["pred_world_pos_loss"] = (
                            (p_world - p_world.mean(1, keepdim=True)) ** 2 * aw
                        ).mean()
                    if self.cfg_loss.pred_world_vel_loss_weight > 0:
                        losses["pred_world_vel_loss"] = (v_world ** 2 * aw).mean()
                    if self.cfg_loss.pred_world_accel_loss_weight > 0 and a_world is not None:
                        losses["pred_world_accel_loss"] = (a_world ** 2 * aw).mean()
                    if self.cfg_loss.pred_world_jerk_loss_weight > 0 and j_world is not None:
                        losses["pred_world_jerk_loss"] = (j_world ** 2 * aw).mean()
                    if (
                        self.cfg_loss.pred_world_hf_loss_weight > 0
                        and f >= 8
                        and self.cfg_loss.hf_cutoff_hz < nyquist
                    ):
                        losses["pred_world_hf_loss"] = hf_energy_loss(
                            p_world * aw.sqrt(),
                            fps=self.cfg_loss.fps,
                            cutoff_hz=self.cfg_loss.hf_cutoff_hz,
                        )
                    if (
                        self.cfg_loss.pred_world_ema_loss_weight > 0
                        and total_iter is not None
                        and total_iter >= self.cfg_loss.pred_world_ema_start_iter
                    ):
                        if (
                            not hasattr(self, "_p_world_ema")
                            or self._p_world_ema is None
                            or self._p_world_ema.shape != p_world.shape
                        ):
                            self._p_world_ema = p_world.detach()
                        d = float(self.cfg_loss.pred_world_ema_decay)
                        self._p_world_ema = d * self._p_world_ema + (1 - d) * p_world.detach()
                        losses["pred_world_ema_loss"] = (
                            (p_world - self._p_world_ema) ** 2 * aw
                        ).mean()

            if posed_bones is not None:
                if self.cfg_loss.bone_smooth_loss_weight > 0:
                    losses["bone_smooth_loss"] = self.smooth_loss_fn(posed_bones)
                if self.cfg_loss.bonevel_smooth_loss_weight > 0:
                    bonevel = posed_bones[:, 1:, ...] - posed_bones[:, :(f-1), ...]
                    losses["bonevel_smooth_loss"] = self.smooth_loss_fn(bonevel)
        return losses, aux

    def forward(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, seq_idx, frame_idx, depth, transforms, store_path = batch
        global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.dataset.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.dataset.out_image_size
        aux_viz = {}

        if transforms is not None:
            R_cw = transforms[:, :, :3, :3]
            t_cw = transforms[:, :, :3, 3]
        else:
            R_cw = None
            t_cw = None

        dino_feat_im_gt = None if dino_feat_im is None else expandBF(torch.nn.functional.interpolate(collapseBF(dino_feat_im), size=[h, w], mode="bilinear"), batch_size, num_frames)[:, :, :self.cfg_predictor_base.cfg_dino.feature_dim]
        dino_cluster_im_gt = None if dino_cluster_im is None else expandBF(torch.nn.functional.interpolate(collapseBF(dino_cluster_im), size=[h, w], mode="nearest"), batch_size, num_frames)

        ## GT image
        image_gt = input_image
        if self.dataset.out_image_size != self.dataset.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'), batch_size, num_frames)
            if flow_gt is not None:
                flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"), batch_size, num_frames-1)


        batch_size, num_frames = input_image.shape[:2]

        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                feat_out, feat_key, patch_out, patch_key = self.netInstance(images=input_image, image_only=True)  # first two dimensions are collapsed N=(B*F)

                prior_shape, betas, dino_net, smal_model, xf_shape, netShape = self.netBase(total_iter=total_iter, is_training=is_training, patch_key=patch_key)
        else:
            feat_out, feat_key, patch_out, patch_key = self.netInstance(images=input_image, image_only=True)  # first two dimensions are collapsed N=(B*F)

            prior_shape, betas, dino_net, smal_model, xf_shape, netShape = self.netBase(total_iter=total_iter, is_training=is_training, patch_key=patch_key)
        
        prior_shape.update_padded(prior_shape.verts_padded()+sum([p.sum() * 0 for p in dino_net.parameters()]))  # Dummy operation for ddp

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training, betas=betas, feat_out=feat_out, feat_key=feat_key, patch_out=patch_out, patch_key=patch_key, smal_model=smal_model, xf_shape=xf_shape, netShape=netShape)
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training, betas=betas, feat_out=feat_out, feat_key=feat_key, patch_out=patch_out, patch_key=patch_key, smal_model=smal_model, xf_shape=xf_shape, netShape=netShape)  # first two dim dimensions already collapsed N=(B*F)

        aux_viz.update(forward_aux)

        final_losses = {}

        ## render images
        if self.enable_render or not is_training:  # Force render for val and test
            render_flow = self.cfg_render.render_flow and num_frames > 1
            render_modes = ['shaded', 'dino_pred']
            if render_flow:
                render_modes += ['flow']
            render_mvp, render_w2c, render_campos = (mvp, w2c, campos) if not self.cfg_render.render_default else (self.default_mvp.to(self.device), self.default_w2c.to(self.device), self.default_campos.to(self.device))
            
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    renders, vertex_2d = self.render(
                        render_modes, shape, texture, render_mvp, render_w2c, render_campos, (h, w), im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                    )
            else:
                renders, vertex_2d = self.render(
                    render_modes, shape, texture, render_mvp, render_w2c, render_campos, (h, w), im_features=im_features, light=light,
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                )
            if batch_size * num_frames != renders[0].shape[0]:
                batch_size = int(renders[0].shape[0]/num_frames)
            renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
            if render_flow:
                shaded, dino_feat_im_pred, flow_pred = renders
                flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW
            else:
                shaded, dino_feat_im_pred = renders
                flow_pred = None
            assert_finite("image_pred", shaded[:, :, :3]) 
            assert_finite("mask_pred", shaded[:, :, 3])
            image_pred =  shaded[:, :, :3]
            mask_pred = shaded[:, :, 3]

            ## compute reconstruction losses and depth loss
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    losses = self.compute_reconstruction_losses(
                        image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                        dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False, betas=betas
                    )
            else:
                losses = self.compute_reconstruction_losses(
                    image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                    dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False, betas=betas
                )
            if losses is not None:
                for name, loss in losses.items():
                    final_losses[name] = loss.mean()

        # regularizers
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                regularizers, aux = self.compute_regularizers(
                    arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                    posed_bones=forward_aux.get("posed_bones"), prior_shape=prior_shape, bones_pred=forward_aux.get("bones_pred"), betas=betas, pred_cam=forward_aux.get("trans"),
                    R_cw=R_cw, t_cw=t_cw, total_iter=total_iter, batch_size=batch_size, num_frames=num_frames
                )
        else:
            regularizers, aux = self.compute_regularizers(
                arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                posed_bones=forward_aux.get("posed_bones"), prior_shape=prior_shape, bones_pred=forward_aux.get("bones_pred"), betas=betas, pred_cam=forward_aux.get("trans"),
                R_cw=R_cw, t_cw=t_cw, total_iter=total_iter, batch_size=batch_size, num_frames=num_frames
            )
        final_losses.update(regularizers)
        aux_viz.update(aux)

        ## compute final losses
        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = getattr(self.cfg_loss, f"{name}_weight")
            if loss_weight <= 0:
                continue
            if not in_range(total_iter, self.cfg_predictor_instance.cfg_texture.texture_iter_range) and (name in ['rgb_loss']):
                continue
            if not in_range(total_iter, self.cfg_loss.arti_reg_loss_iter_range) and (name in ['arti_reg_loss']):
                continue
            if not in_range(total_iter, self.cfg_loss.distill_loss_iter_range) and (name in DISTILL_STAGE_LOSSES):
                continue
            if name in ["logit_loss_target"]:
                continue
            total_loss += loss * loss_weight
        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import pdb; pdb.set_trace()

        distillation_loss = sum(
            final_losses[name] * getattr(self.cfg_loss, f"{name}_weight")
            for name in ('arti_recon_loss', 'cam_recon_loss', 'shape_recon_loss')
            if name in final_losses
        )
        metrics = {'loss': total_loss, 'distillation_loss': distillation_loss, **final_losses}

        log = SimpleNamespace(**locals())
        if self.accelerator.is_main_process and logger is not None and (self.enable_render or not is_training):
            self.log_visuals(log, logger)
        if self.accelerator.is_main_process and save_results:
            self.save_results(log)
        return metrics

    def log_visuals(self, log, logger, sdf_feats=None, text=None):
        print("LOGGIN")
        b0 = max(min(log.batch_size, 16//log.num_frames), 1)
        def log_image(name, image):
            logger.add_image(log.logger_prefix+'image/'+name, misc.image_grid(collapseBF(image[:b0,:]).detach().cpu().clamp(0,1)), log.total_iter)
        def log_video(name, frames, fps=2):
            logger.add_video(log.logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), log.total_iter, fps=fps)

        log_image('image_gt', log.input_image)
        log_image('image_pred', log.image_pred)
        log_image('mask_gt', log.mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
        log_image('mask_pred', log.mask_pred.unsqueeze(2).repeat(1,1,3,1,1))

        if log.dino_feat_im_gt is not None:
            log_image('dino_feat_im_gt', log.dino_feat_im_gt[:,:,:3])
        if log.dino_feat_im_pred is not None:
            log_image('dino_feat_im_pred', log.dino_feat_im_pred[:,:,:3])
        if log.dino_cluster_im_gt is not None:
            log_image('dino_cluster_im_gt', log.dino_cluster_im_gt)

        if self.cfg_render.render_flow and log.flow_gt is not None:
            flow_gt_viz = torch.nn.functional.pad(log.flow_gt, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
            flow_gt_viz = flow_gt_viz + 0.5  # -0.5~1.5
            flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization

            ## draw marker on large flow frames
            large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
            large_flow_marker_mask[:,:,:,:8,:8] = 1.
            large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1)
            large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
            red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(flow_gt_viz.device)
            flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz
            log_image('flow_gt', flow_gt_viz)

        if self.cfg_render.render_flow and log.flow_pred is not None:
            flow_pred_viz = torch.nn.functional.pad(log.flow_pred, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
            flow_pred_viz = flow_pred_viz + 0.5  # -0.5~1.5
            flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization
            log_image('flow_pred', flow_pred_viz)

        if log.deformation is not None:
            logger.add_histogram(log.logger_prefix+'deformation', log.deformation, log.total_iter)

        rot_rep = self.cfg_predictor_instance.cfg_pose.rot_rep
        if rot_rep == 'euler_angle':
            for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose[...,i], log.total_iter)
        elif rot_rep == 'quaternion':
            for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose[...,i], log.total_iter)
            rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(log.pose.detach().cpu()[...,:4]), convention='XYZ')
            for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, rot_euler[...,i], log.total_iter)
        elif rot_rep in ['lookat', 'quadlookat', 'octlookat']:
            for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose_raw[...,i], log.total_iter)
            for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose_raw[...,-3+i], log.total_iter)

        logger.add_histogram(log.logger_prefix+'coordinates', log.shape.verts_padded(), log.total_iter)

        render_modes = ['geo_normal', 'kd', 'shading']
        rendered, _ = self.render(render_modes, log.shape, log.texture, log.mvp, log.w2c, log.campos, (log.h, log.w), im_features=log.im_features, light=log.light, prior_shape=log.prior_shape)
        geo_normal, albedo, shading = map(lambda x: expandBF(x, log.batch_size, log.num_frames), rendered)
        if hasattr(self.get_predictor("netInstance"), "articulated_shape_gt"):
            rendered_gt,_ = self.render(render_modes, self.get_predictor("netInstance").articulated_shape_gt, log.texture, log.mvp, log.w2c, log.campos, (log.h, log.w), im_features=log.im_features, light=log.light, prior_shape=log.prior_shape)
            geo_normal_gt, albedo_gt, shading_gt = map(lambda x: expandBF(x, log.batch_size, log.num_frames), rendered_gt)
            del self.get_predictor("netInstance").articulated_shape_gt

        if log.light is not None:
            param_names = ['dir_x', 'dir_y', 'dir_z', 'int_ambient', 'int_diffuse']
            log_image('albedo', albedo)
            log_image('shading', shading.repeat(1,1,3,1,1) /2.)

        ## add bone visualizations
        if 'posed_bones' in log.aux_viz:
            
            rendered_bone_image = self.render_bones(log.mvp, log.aux_viz['posed_bones'], (log.h, log.w))
            rendered_bone_image_mask = (rendered_bone_image < 1).float()
            
            geo_normal = rendered_bone_image_mask*0.8 * rendered_bone_image + (1-rendered_bone_image_mask*0.8) * albedo #geo_normal
            if log.aux_viz.get("posed_bones_gt") is not None:
                rendered_bone_image = self.render_bones(log.mvp, log.aux_viz['posed_bones_gt'], (log.h, log.w))
                rendered_bone_image_mask = (rendered_bone_image < 1).float()
                geo_normal_gt = rendered_bone_image_mask * 0.8 * rendered_bone_image + (
                            1 - rendered_bone_image_mask * 0.8) * geo_normal_gt
                log_image('instance_geo_normal_gt', geo_normal_gt)

        ## draw marker on images with randomly sampled pose
        # oh so this was what the red corner thing was

        log_image('instance_geo_normal', geo_normal)

        rot_frames = self.render_rotation_frames('geo_normal', log.shape, log.texture, log.light, (log.h, log.w), im_features=log.im_features, prior_shape=log.prior_shape, num_frames=15, b=1, text=text)
        log_video('instance_normal_rotation', rot_frames)
        print("????rendered")

        rot_frames = self.render_rotation_frames('shaded', log.prior_shape, log.texture, log.light, (log.h, log.w), im_features=log.im_features, num_frames=15, b=1, text=text, log=log)
        log_video('prior_image_rotation', rot_frames)
        print("????rendered222")

        rot_frames = self.render_rotation_frames('geo_normal', log.prior_shape, log.texture, log.light, (log.h, log.w), im_features=log.im_features, num_frames=15, b=1, text=text)
        log_video('prior_normal_rotation', rot_frames)

        log.__dict__.update({k: v for k, v in locals().items() if k != "log"})
        return log

    def save_results(self, log):
        b0 = log.batch_size * log.num_frames
 
        fnames = [f'{log.total_iter:07d}_{fid:10d}' for fid in collapseBF(log.global_frame_id.int())][:b0]
        print(log.store_path)
        fnames = ["_".join((f[0] if isinstance(f, (list, tuple)) else f).split("/")[-2:])[:-2] for f in log.store_path]
        print(fnames)
        

        def save_image(name, image):
            misc.save_images(log.save_dir, collapseBF(image)[:b0].clamp(0,1).detach().cpu().numpy(), suffix=name, fnames=fnames)

        save_image('image_gt', log.image_gt)
        save_image('image_pred', log.image_pred)
        save_image('mask_gt', log.mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
        save_image('mask_pred', log.mask_pred.unsqueeze(2).repeat(1,1,3,1,1))

        if self.cfg_render.render_flow and log.flow_gt is not None:
            flow_gt_viz = torch.cat([log.flow_gt, torch.zeros_like(log.flow_gt[:,:,:1])], 2) + 0.5  # -0.5~1.5
            flow_gt_viz = flow_gt_viz.view(-1, *flow_gt_viz.shape[2:])
            save_image('flow_gt', flow_gt_viz)
        if log.flow_pred is not None:
            flow_pred_viz = torch.cat([log.flow_pred, torch.zeros_like(log.flow_pred[:,:,:1])], 2) + 0.5  # -0.5~1.5
            flow_pred_viz = flow_pred_viz.view(-1, *flow_pred_viz.shape[2:])
            save_image('flow_pred', flow_pred_viz)

        feat = log.im_features[:b0] if log.im_features is not None else None
        betas = log.forward_aux['betas'][:b0] if 'betas' in log.forward_aux else None
        pose = log.forward_aux['pose'][:b0] if 'pose' in log.forward_aux else None
        trans = log.forward_aux['trans'][:b0] if 'trans' in log.forward_aux else None
        try:
            vertex_vis = render_2.verts_visibility(log.shape, log.mvp, log.w2c, log.campos, (log.h, log.w))
        except Exception as e:
            print(f"Warning: failed to compute vertex_vis for save_results: {e}")
            vertex_vis = None
        pkl_data = {
            'rotation': log.transforms.view(b0, -1, 4, 4)[:b0], # if log.transforms else None,
            'betas': betas,
            'pose': pose,
            'trans': trans,
            'vertex_2d' : log.vertex_2d,
            'vertex_vis' : vertex_vis,
        }
        # Save as pickle files
        
        misc.save_pkl_batch(log.save_dir, pkl_data, suffix='params', fnames=fnames)

        if self.export_mesh:
            self.export_textured_mesh_batch(log.shape[:b0], log.prior_shape[:b0], log.texture, feat, log.save_dir, fnames)

        misc.save_txt(log.save_dir, log.pose[:b0].detach().cpu().numpy(), suffix='pose', fnames=fnames)
        misc.save_txt(log.save_dir, rearrange(log.arti_params, "b f c -> (b f) c").cpu().numpy(), suffix='arti_params', fnames=fnames, delim=' ')

    def export_textured_mesh_batch(self, posed_mesh, canonical_mesh, material, feat, save_dir, fnames):
        """Bake a UV texture atlas per frame and write a textured OBJ+MTL+PNG.

        The texture field is defined over the *canonical* (T-pose) mesh, so kd
        is sampled there; geometry is exported from the *posed* mesh so the
        saved mesh matches the frame's predicted articulation. Uses the
        pytorch3d UV-bake helpers in model.render.render_2, keyed to the same
        UVMAP_OBJ atlas used by the renderer.
        """
        kd_batched = render_2._sample_kd_per_vertex(canonical_mesh, material=material, feat=feat)  # (B,V,3)
        for i, fname in enumerate(fnames):
            kd_img = render_2._bake_kd_atlas_single(
                canonical_mesh[i:i + 1], kd_batched[i], render_2.VT, render_2.FT,
                device=str(posed_mesh.device),
            )
            obj_path = os.path.join(save_dir, f"{fname}_mesh.obj")
            render_2.save_textured_obj(obj_path, posed_mesh[i:i + 1], render_2.VT, render_2.FT, kd_img)
            tex_path = os.path.join(save_dir, f"{fname}_kd.png")
            render_2._save_kd_image_single(kd_img, tex_path)

    def render_rotation_frames(self, render_mode, mesh, texture, light, resolution, background=None, im_features=None, prior_shape=None, num_frames=36, b=None, text=None, log=None):
        if b is None:
            b = len(mesh)
        else:
            mesh = mesh[:b]
            if prior_shape is not None:
                prior_shape = prior_shape[:b]
            feat = im_features[:b] if im_features is not None else None
            

        delta_angle = np.pi / num_frames * 2
        delta_rot_matrix = torch.FloatTensor([
            [np.cos(delta_angle),  0, np.sin(delta_angle), 0],
            [0,                    1, 0,                   0],
            [-np.sin(delta_angle), 0, np.cos(delta_angle), 0],
            [0,                    0, 0,                   1],
        ]).to(self.accelerator.device).repeat(b, 1, 1)

        w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
        w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cfg_render.cam_pos_z_offset *1.1])
        w2c = w2c.repeat(b, 1, 1).to(self.accelerator.device)
        proj = util.perspective(self.cfg_render.fov / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.accelerator.device)
        mvp = torch.bmm(proj, w2c)
        campos = -w2c[:, :3, 3]

        def rotate_pose(mvp, campos):
            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
            return mvp, campos

        frames = []
        import ipdb
        for _ in range(num_frames):
            rendered, _ = self.render([render_mode], mesh, texture, mvp, w2c, campos, resolution, background=background, im_features=feat, light=light, prior_shape=prior_shape)
            shaded = rendered[0]
            frames += [misc.image_grid(shaded[:, :3])]
            mvp, campos = rotate_pose(mvp, campos)
        
        if text is not None:
            frames = [torch.Tensor(misc.add_text_to_image(f, text)).permute(2, 0, 1) for f in frames]
        return torch.stack(frames, dim=0)  # Shape: (T, C, H, W)

    def render_bones(self, mvp, bones_pred, size=(256, 256), show_legend=False, overlay_img=None):
        bone_world4 = torch.concat([bones_pred, torch.ones_like(bones_pred[..., :1]).to(bones_pred.device)], dim=-1)
        b, f, num_bones = bone_world4.shape[:3]
        #print("???",xx mvp.shape, mvp.transpose(-1, -2).shape)
        bones_clip4 = (bone_world4.view(b, f, num_bones*2, 1, 4) @ (mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4))).view(b, f, num_bones, 2, 4)
        bones_uv = bones_clip4[..., :2] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 2
        dpi = 32
        fx, fy = size[1] // dpi, size[0] // dpi

        rendered = []
        for b_idx in range(b):
            for f_idx in range(f):
                frame_bones_uv = bones_uv[b_idx, f_idx].cpu().numpy()
                fig = plt.figure(figsize=(fx, fy), dpi=dpi, frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                if overlay_img is not None:
                    bg_img = overlay_img[b_idx, f_idx].permute(1, 2, 0).cpu().numpy()  # (h, w, 3)
                    bg_img = np.flipud(bg_img)
                    ax.imshow(bg_img, extent=(-1, 1, -1, 1), alpha=0.5)  # Adjust extent to match plot coordinates
                colors = plt.cm.tab20(np.linspace(0, 1, frame_bones_uv.shape[0]))
                for bone_idx, bone in enumerate(frame_bones_uv):
                    ax.plot(bone[:, 0], bone[:, 1], marker='o', linewidth=8, markersize=20, color=colors[bone_idx], label=str(bone_idx))
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.invert_yaxis()
                if show_legend:
                    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
                # Convert to image
                fig.canvas.draw_idle()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image.resize(h, w, 3)
                rendered += [image / 255.]
                plt.close(fig)

        rendered = expandBF(torch.FloatTensor(np.stack(rendered, 0)).permute(0, 3, 1, 2).to(bones_pred.device), b, f)
        return rendered


## utility functions
def in_range(x, range):
    return misc.in_range(x, range, default_indicator=-1)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')


def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def get_optimizer(model, lr=0.0001, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

def to_float(x):
    try:
        return x.float()
    except AttributeError:
        return x


def find_nearest_mask_coords(coords, mask):
    """Given a set of coordinates and a binary mask, find the nearest unmasked coordinates for each input coordinate"""
    b, f, p, _, _ = coords.shape  # (b, f, p, n, 2)
    device = coords.device
    coords = rearrange(coords, 'b f p n c -> (b f) (p n) c').cpu().numpy()
    mask = rearrange(mask, 'b f h w -> (b f) h w').cpu().numpy()
    nearest_coords_list = []
    for i in range(b):
        mask_np = mask[i]
        mask_y, mask_x = np.where(mask_np == 1)
        mask_points = np.stack((mask_x, mask_y), axis=-1)
        query_points = coords[i]
        if mask_points.size == 0:
            nearest_coords_list.append(torch.from_numpy(query_points).to(device))
            continue
        tree = cKDTree(mask_points)
        distances, indices = tree.query(query_points, k=1)
        nearest_mask_coords = mask_points[indices]
        nearest_coords_list.append(torch.from_numpy(nearest_mask_coords).to(device))
    nearest_coords = torch.stack(nearest_coords_list, dim=0)
    nearest_coords = rearrange(nearest_coords, '(b f) (p n) c -> b f p n c', b=b, f=f, p=p, n=2)
    return nearest_coords


def get_distance_threshold_mask(coord_pairs_xy, threshold=20):
    """Mask pairs of coordinates where the distance between the points is smaller than the threshold"""
    assert coord_pairs_xy.shape[-2] == 2, "Each pair should contain exactly two points."
    point1 = coord_pairs_xy[..., 0, :]  # Shape: (b, f, p, 2)
    point2 = coord_pairs_xy[..., 1, :]  # Shape: (b, f, p, 2)
    diff = point2 - point1  # Shape: (b, f, p, 2)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # Shape: (b, f, p)
    mask = dist >= threshold  # Shape: (b, f, p), dtype: torch.bool
    return mask


def disable_articulation_loss(articulation_gt_flag, w2c):
    if articulation_gt_flag is None:
        return None
    b = len(articulation_gt_flag)
    object_front_normal = repeat(torch.tensor([1., 0., 0.], device=w2c.device), "c -> b c 1", b=b)
    R_world_to_cam = w2c[:, :3, :3]
    cam_forward_in_world = R_world_to_cam.transpose(1, 2) @ repeat(torch.tensor([0., 0., 1.], device=w2c.device), "c -> b c 1", b=b)
    similarity = F.cosine_similarity(cam_forward_in_world, object_front_normal).abs()
    articulation_gt_flag = articulation_gt_flag * (similarity > 0.25).squeeze()
    return articulation_gt_flag


def disable_keypoint_loss(w2c, b):
    object_front_normal = repeat(torch.tensor([1., 0., 0.], device=w2c.device), "c -> b c 1", b=b)
    R_world_to_cam = w2c[:, :3, :3]
    cam_forward_in_world = R_world_to_cam.transpose(1, 2) @ repeat(torch.tensor([0., 0., 1.], device=w2c.device), "c -> b c 1", b=b)
    similarity = F.cosine_similarity(cam_forward_in_world, object_front_normal).abs()
    keypoint_gt_flag = (similarity > 0.25).squeeze()
    return keypoint_gt_flag


def draw_keypoints(image_t, keypoint, gt_flag, circle_color=(0, 0, 255), text_color=(0, 255, 255), radius=3):
    b, f = image_t.shape[:2]
    img_size = image_t.shape[-1]
    image_t = rearrange(image_t, "b f ... -> (b f) ...")
    keypoint = rearrange((keypoint + 1) / 2 * img_size, "b f ... -> (b f) ...")
    if gt_flag is None:
        gt_flag = torch.ones(b*f, device=image_t.device, dtype=torch.bool)
    elif gt_flag.dim() == 2:
        gt_flag = rearrange(gt_flag, "b f -> (b f)")
    out_list = []
    for img, k, flag in zip(image_t, keypoint, gt_flag):
        if flag:
            img_np = (img.clone() * 255).permute(1, 2, 0).clamp(0, 255).contiguous().cpu().numpy().astype(np.uint8)
            for i, (x, y) in enumerate(k[:, :2]):
                x, y = int(x.item()), int(y.item())
                cv2.circle(
                    img_np,
                    center=(x, y),
                    radius=radius,
                    color=circle_color,
                    thickness=-1  # -1 => filled circle
                )
                cv2.putText(
                    img_np,
                    text=str(i),
                    org=(x + radius + 1, y - radius - 1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
            out_list.append(torch.from_numpy(img_np).permute(2, 0, 1) / 255)
        else:
            out_list.append(img.cpu())
    out_t = rearrange(torch.stack(out_list), "(b f) ... -> b f ...", b=b, f=f)
    return out_t
