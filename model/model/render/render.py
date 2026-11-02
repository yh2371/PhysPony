# Adapted from NVDiffRec (https://github.com/NVlabs/nvdiffrec)
# Modified by Shangzhe Wu for MagicPony in 2023
#
# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
try:
    import nvdiffrast.torch as dr
except:
    pass
try:
    from pytorch3d.structures import Meshes
except:
    pass
from . import util
from . import renderutils as ru
from . import light
try:
    from pytorch3d.io import load_obj
except:
    pass
from ..paths import UVMAP_OBJ

obj_filename = UVMAP_OBJ
try:
    smpl_texture_data = load_obj(obj_filename)
    VT = smpl_texture_data[-1].verts_uvs.to(device='cuda')
    FT = smpl_texture_data[1].textures_idx.to(device='cuda')
except:
    pass

# ==============================================================================================
#  Helper functions for PyTorch3D Meshes compatibility
# ==============================================================================================
def get_mesh_vertices(mesh):
    """Extract vertices from PyTorch3D Meshes object"""
    return mesh.verts_padded()

def get_mesh_faces(mesh):
    """Extract faces from PyTorch3D Meshes object"""
    return mesh.faces_padded()

def get_mesh_normals(mesh):
    """Extract vertex normals from PyTorch3D Meshes object"""
    if hasattr(mesh, 'faces_normals_padded') and mesh.faces_normals_padded() is not None:
        return mesh.faces_normals_padded()
    else:
        raise ValueError("Mesh does not have normals")

def get_mesh_tangents(mesh):
    """Extract vertex tangents from PyTorch3D Meshes object"""
    if hasattr(mesh, 'verts_tangents_padded') and mesh.verts_tangents_padded() is not None:
        return mesh.verts_tangents_padded()
    else:
        # Return zeros if not available
        return compute_tangents(mesh)    

def get_mesh_textures(mesh):
    """Extract texture coordinates from PyTorch3D Meshes object"""
    batch_size = mesh.verts_padded().shape[0]
    return VT.repeat(batch_size, 1, 1)
   
def get_mesh_texture_faces(mesh):
    """Extract texture face indices from PyTorch3D Meshes object"""
    batch_size = mesh.faces_padded().shape[0]
    return FT.repeat(batch_size, 1, 1)

def compute_vertex_normals(mesh):
    """Compute vertex normals for PyTorch3D Meshes object"""
    verts = mesh.verts_padded()
    faces = mesh.faces_padded()
    
    # Compute face normals
    v0 = verts[:, faces[0, :, 0], :]
    v1 = verts[:, faces[0, :, 1], :]
    v2 = verts[:, faces[0, :, 2], :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    
    # Initialize vertex normals
    batch_size, num_verts = verts.shape[:2]
    v_nrm = torch.zeros_like(verts)
    
    # Accumulate face normals to vertices
    for i in range(3):
        v_nrm.scatter_add_(1, faces[0, :, i:i+1].expand(-1, 3), face_normals)
    
    # Normalize
    v_nrm = util.safe_normalize(v_nrm)
    return v_nrm

def compute_tangents(imesh):
    batch_size = get_mesh_vertices(imesh).shape[0]

    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = get_mesh_vertices(imesh)[:, get_mesh_faces(imesh)[0, :, i]]
        tex[i] = get_mesh_textures(imesh)[:, get_mesh_texture_faces(imesh)[0, :, i]]
        vn_idx[i] = get_mesh_faces(imesh)[0, :, i:i+1]

    v_nrm = get_mesh_normals(imesh)
    tangents = torch.zeros_like(v_nrm)    
    tansum   = torch.zeros_like(v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]  # Shape: (B, F, 2)
    uve2 = tex[2] - tex[0]  # Shape: (B, F, 2)
    pe1  = pos[1] - pos[0]  # Shape: (B, F, 3)
    pe2  = pos[2] - pos[0]  # Shape: (B, F, 3)
    
    nom   = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]  # Shape: (B, F, 3)
    denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]  # Shape: (B, F, 1)
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))  # Shape: (B, F, 3)
    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i].repeat(batch_size, 1, 3)  # Shape: (B, F, 3)
        tangents.scatter_add_(1, idx, tang)       # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(1, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, v_nrm) * v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))
    return tangents


def is_mesh_empty(mesh):
    """Check if PyTorch3D Meshes object is empty"""
    return mesh.isempty()

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
#  render_modes: shaded, kd, ks, normal, geo_normal, tangent, shading, flow, dino_pred
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_tex_pos,
        w2c,
        view_pos,
        lgt,
        material,
        bsdf,
        feat=None,
        render_modes=None,
        two_sided_shading=True,
        delta_xy_interp=None,
        dino_net=None,
        class_vector=None
    ):

    ################################################################################
    # Texture lookups
    ################################################################################
    # Combined texture, used for MLPs because lookups are expensive
    if material is not None:
        all_tex = material.sample(gb_tex_pos, feat=feat)
    else:
        all_tex = torch.ones(*gb_pos.shape[:-1], 9, device=gb_pos.device)
    kd, ks, perturbed_nrm = all_tex[..., :3], all_tex[..., 3:6], all_tex[..., 6:9]

    # Sample from optimized DINO feature field
    if dino_net is not None:
        dino_pred = dino_net.sample(gb_tex_pos, feat=class_vector)
    else:
        dino_pred = None

    # Default alpha is 1
    alpha = torch.ones_like(kd[..., 0:1])

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    perturbed_nrm = None
    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=two_sided_shading, opengl=True, use_python=True)

    b, h, w, _ = gb_normal.shape
    cam_normal = util.safe_normalize(torch.matmul(gb_normal.view(b, -1, 3), w2c[:,:3,:3].transpose(2,1))).view(b, h, w, 3)

    ################################################################################
    # Evaluate BSDF
    ################################################################################
    assert bsdf is not None or material.bsdf is not None, "Material must specify a BSDF type"
    bsdf = bsdf if bsdf is not None else material.bsdf
    shading = None
    if bsdf == 'pbr':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'diffuse':
        if lgt is None:
            shaded_col = kd
        elif isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=False)
        else:
            shaded_col, shading = lgt.shade(feat, kd, cam_normal)
    else:
        assert False, "Invalid BSDF '%s'" % bsdf

    ################################################################################
    # Compute Depth
    ################################################################################
    depth = None
    if 'depth' in render_modes:
        gb_pos_hom = torch.cat([gb_pos, torch.ones_like(gb_pos[..., :1])], dim=-1)  # Shape: (b, h, w, 4)
        gb_pos_cam = torch.matmul(gb_pos_hom.view(b, -1, 4), w2c.transpose(-1, -2)).view(b, h, w, 4)
        depth = gb_pos_cam[..., 2]  # Shape: (b, h, w)
        depth_min = depth.amin(dim=(1, 2), keepdim=True)
        depth_max = depth.amax(dim=(1, 2), keepdim=True)
        depth = ((depth - depth_min) / (depth_max - depth_min)).unsqueeze(-1)

    buffers = {
        'shaded' : shaded_col,
        'kd'     : kd,
        'ks'     : ks,
        'normal' : (gb_normal + 1.0) * 0.5,
        'geo_normal' : (gb_geometric_normal + 1.0) * 0.5,
        'tangent' : (gb_tangent + 1.0) * 0.5,
    }
    if shading is not None:
        buffers['shading'] = shading
    if delta_xy_interp is not None:
        buffers['flow'] = delta_xy_interp
    if dino_pred is not None:
        buffers['dino_pred'] = dino_pred
    if depth is not None:
        buffers['depth'] = depth

    if render_modes is not None:
        buffers = {mode: torch.cat((buffers[mode], alpha), dim=-1) for mode in render_modes}
    else:
        buffers = {'shaded': torch.cat((shaded_col, alpha), dim=-1)}

    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        w2c,
        view_pos,
        material,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        feat,
        render_modes=None,
        prior_mesh=None,
        two_sided_shading=True,
        delta_xy=None,
        dino_net=None,
        class_vector=None,

    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    if prior_mesh is None:
        prior_mesh = mesh


    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(get_mesh_vertices(mesh), rast_out_s, get_mesh_faces(mesh)[0].int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = get_mesh_vertices(mesh)[:, get_mesh_faces(mesh)[0, :, 0], :]
    v1 = get_mesh_vertices(mesh)[:, get_mesh_faces(mesh)[0, :, 1], :]
    v2 = get_mesh_vertices(mesh)[:, get_mesh_faces(mesh)[0, :, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))
    num_faces = face_normals.shape[1]
    face_normal_indices = (torch.arange(0, num_faces, dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals, rast_out_s, face_normal_indices.int())

    # Compute tangent space
    gb_normal, _ = interpolate(get_mesh_normals(mesh), rast_out_s, get_mesh_faces(mesh)[0].int())
    gb_tangent, _ = interpolate(get_mesh_tangents(mesh), rast_out_s, get_mesh_faces(mesh)[0].int()) # Interpolate tangents

    # 2D flow
    if 'flow' in render_modes:
        delta_xy_interp, _ = interpolate(delta_xy, rast_out_s, get_mesh_faces(mesh)[0].int())
    else:
        delta_xy_interp = None

    ################################################################################
    # Shade
    ################################################################################

    # Sample texture from canonical mesh
    gb_tex_pos, _ = interpolate(get_mesh_vertices(prior_mesh), rast_out_s, get_mesh_faces(mesh)[0].int())
    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_tex_pos, w2c, view_pos, lgt, material, bsdf, feat=feat, render_modes=render_modes, two_sided_shading=two_sided_shading, delta_xy_interp=delta_xy_interp, dino_net=dino_net, class_vector=class_vector)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        w2c,
        view_pos,
        material,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None,
        bsdf        = None,
        feat        = None,
        render_modes = None,
        prior_mesh = None,
        two_sided_shading = True,
        dino_net = None,
        num_frames = None,
        class_vector = None,
    ):

    assert not is_mesh_empty(mesh) #mesh.faces_padded().shape[1] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(
                    accum.contiguous().float(), rast.float(), v_pos_clip.float(),
                    get_mesh_faces(mesh)[0].int()
                )
        return accum

    # Render higher samples per pixel (SPP) for multisample anti-aliasing (MSAA)
    full_res = [resolution[0] * spp, resolution[1] * spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)  # Shape: (B, 1, 1, 3)

    # Clip space transform
    v_pos_clip = ru.xfm_points(get_mesh_vertices(mesh), mtx_in, use_python=True)

    # Render flow
    if 'flow' in render_modes:
        v_pos_clip2 = v_pos_clip[..., :2] / v_pos_clip[..., -1:]
        v_pos_clip2 = v_pos_clip2.view(-1, num_frames, *v_pos_clip2.shape[1:])
        delta_xy = v_pos_clip2[:, 1:] - v_pos_clip2[:, :-1]
        delta_xy = torch.cat([delta_xy, torch.zeros_like(delta_xy[:, :1])], dim=1)
        delta_xy = delta_xy.view(-1, *delta_xy.shape[2:])
    else:
        delta_xy = None

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip.float(), get_mesh_faces(mesh)[0].int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            rendered = render_layer(rast, db, mesh, w2c, view_pos, material, lgt, resolution, spp, msaa, bsdf, feat=feat, render_modes=render_modes, prior_mesh=prior_mesh, two_sided_shading=two_sided_shading, delta_xy=delta_xy, dino_net=dino_net, class_vector=class_vector)
            layers += [(rendered, rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    out_buffers = []
    for key in render_modes:
        if key not in layers[0][0].keys():
            out_buffers.append(None)
        else:
            antialias = key in ['shaded', 'flow', 'dino_pred', 'depth', 'shading']
            bg = background if key in ['shaded', 'geo_normal', 'shading'] else torch.zeros_like(layers[0][0][key])
            if key == "shading" and background is not None:
                bg = bg[:, :, :, 2:]
            accum = composite_buffer(key, layers, bg, antialias)

            # Downscale to framebuffer resolution. Use avg pooling 
            out_buffer = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

            if key == 'shaded':
                pass  # RGBA channels
            elif key in ['kd', 'ks', 'normal', 'geo_normal']:
                out_buffer = out_buffer[..., :3]
            elif key == 'shading':
                out_buffer = out_buffer[..., :1]
            elif key == 'flow':
                out_buffer = out_buffer[..., :2]
            elif key == 'dino_pred':
                out_buffer = out_buffer[..., :-1]
            elif key == 'depth':
                out_buffer = out_buffer[..., :1]

            # NHWC -> NCHW
            out_buffer = out_buffer.permute(0, 3, 1, 2)
            out_buffers.append(out_buffer)

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture, feat=None):

    # Clip space transform 
    uv_clip = get_mesh_textures(mesh) * 2.0 - 1.0

    # Pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # Rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, get_mesh_texture_faces(mesh)[0].int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(get_mesh_vertices(mesh), rast, get_mesh_faces(mesh)[0].int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos, feat=feat)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)