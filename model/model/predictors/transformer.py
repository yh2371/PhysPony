import torch, torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # moves with .to()
    def forward(self, x):  # [B,T,D]
        return x + self.pe[:x.size(1)]

class TemporalFuser(nn.Module):
    """Generic per-branch fuser. If use_cls=True, returns a single [B,1,D] token."""
    def __init__(self, d_in, d_model=256, nlayer=3, nhead=4, use_cls=False):
        super().__init__()
        self.use_cls = use_cls
        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, activation='gelu', batch_first=True)
        self.tr = TransformerEncoder(enc_layer, nlayer)
        self.pe = PosEnc(d_model)
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.out = nn.Linear(d_model, d_in)  # residual back to latent dim
    def forward(self, z, key_padding_mask=None):  # z: [B,T,Din]
        h = self.in_proj(z)
        h = self.pe(h)
        if self.use_cls:
            cls = self.cls.expand(h.size(0), -1, -1)   # [B,1,D]
            h = torch.cat([cls, h], dim=1)             # [B,1+T,D]
            # pad mask: prepend False for cls
            if key_padding_mask is not None:
                pad = torch.zeros(h.size(0),1, dtype=torch.bool, device=h.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)
        h = self.tr(h, src_key_padding_mask=key_padding_mask)
        if self.use_cls:
            h_cls = h[:,0:1,:]
            return self.out(h_cls)                      # [B,1,Din]
        return self.out(h)                              # [B,T,Din]

class ThreeBranchModel(nn.Module):
    def __init__(self, d_pose, d_cam, d_shape):
        super().__init__()
        self.fuse_pose  = TemporalFuser(d_pose,  use_cls=False)
        self.fuse_cam   = TemporalFuser(d_cam,   use_cls=False)
        self.fuse_shape = TemporalFuser(d_shape, use_cls=True)
        # reuse pretrained heads

    def forward(self, z_pose, z_cam, z_shape, pad_mask=None):
        # z_*: [B,T,D*], pad_mask: [B,T]=True for padding
        z_pose_f  = self.fuse_pose(z_pose,  key_padding_mask=pad_mask)
        z_cam_f   = self.fuse_cam(z_cam,    key_padding_mask=pad_mask)
        z_shape_f = self.fuse_shape(z_shape,key_padding_mask=pad_mask)  # [B,1,D]
        # decode
        return z_pose_f, z_cam_f, z_shape_f[:,0]
