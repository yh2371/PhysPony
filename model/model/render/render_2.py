# PyTorch3D-based renderer for better numerical stability
# Compatible with the existing render_mesh interface from render.py

import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.ops import interpolate_face_attributes
import numpy as np

from . import util
from . import light
from pytorch3d.io import load_obj
from ..paths import UVMAP_OBJ

# Load SMAL texture data
obj_filename = UVMAP_OBJ
smpl_texture_data = load_obj(obj_filename)
VT = smpl_texture_data[-1].verts_uvs.to(device='cuda')
FT = smpl_texture_data[1].textures_idx.to(device='cuda')


# ==============================================================================================
#  Helper functions for Mesh Texture Baking
# ==============================================================================================
import os
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.io import save_obj
from pytorch3d.ops import interpolate_face_attributes

@torch.no_grad()
def _sample_kd_per_vertex(mesh, material=None, feat=None):
    """
    Returns kd per-vertex in [0,1], shape (B,V,3).
    """
    B, V, _ = mesh.verts_padded().shape
    verts = mesh.verts_padded()
    tex_full = material.sample(verts, feat=feat)  # (B*V, C>=3)
    kd = tex_full[..., :3]
    return kd

@torch.no_grad()
def _bake_kd_atlas_single(mesh, kd_verts, VT, FT, tex_res=1024, device="cuda"):
    """
    Bake a kd atlas in UV space.
    - mesh: PyTorch3D Meshes (single batch) for GEOMETRY faces.
    - kd_verts: (V,3) per-geometry-vertex kd in [0,1]
    - VT: (Vt,2) UV vertices in [0,1]
    - FT: (F,3) UV faces (textures_idx) corresponding 1:1 with geometry faces
    """
    # 1) Build a UV-space mesh: vertices are VT mapped to NDC
    uv_verts_ndc = torch.stack(
        [2 * VT[:, 0] - 1,
         2 * VT[:, 1] - 1,
         torch.full_like(VT[:, 0], 1.0)],  # inside default ortho slab
        dim=-1
    )
    uv_mesh = Meshes(verts=[uv_verts_ndc], faces=[FT.to(device)])

    # 2) Rasterize in UV space
    cams = OrthographicCameras(device=device)
    rast = MeshRasterizer(
        cameras=cams,
        raster_settings=RasterizationSettings(
            image_size=tex_res, blur_radius=1e-4, faces_per_pixel=1,
            perspective_correct=False, cull_backfaces=False
        )
    )
    fragments = rast(uv_mesh)  # (1,H,W,1)

    # 3) Interpolate kd using GEOMETRY faces, not FT
    faces_geom = mesh.faces_packed().to(device)  # (F,3) geometry vertex indices
    FT_dev = FT.to(device)
    # Sanity: face counts should match
    assert faces_geom.shape[0] == FT_dev.shape[0], \
        f"Face count mismatch: geom {faces_geom.shape[0]} vs uv {FT_dev.shape[0]}"

    # Gather per-face-corner kd from geometry vertices
    kd_verts = kd_verts.to(device)               # (V,3)
    face_kd = kd_verts[faces_geom]               # (F,3,3)

    # 4) Interpolate onto the UV image via the UV raster fragments
    kd_img = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_kd
    )[:, :, :, 0, :]                              # (1,H,W,3)
    kd_img = kd_img[0]                            # (H,W,3)
    kd_img = torch.flip(kd_img.clamp(0, 1), dims=[1])
    return kd_img

@torch.no_grad()
def save_textured_obj(out_obj_path, mesh, VT, FT, kd_img):
    """
    Save a single posed mesh as a textured OBJ+MTL+PNG (loadable in Blender).

    mesh: PyTorch3D Meshes with a single batch element, geometry only (no
        material needed - texture comes from kd_img via the shared VT/FT UV
        atlas). kd_img: (H,W,3) float in [0,1]. VT (Vt,2), FT (F,3).
    """
    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    verts = mesh.verts_padded()[0].detach().cpu()
    faces = mesh.faces_packed().detach().cpu()
    verts_uvs = VT.detach().cpu()
    faces_uvs = FT.detach().cpu()
    tex_map = kd_img.detach().cpu().contiguous()
    save_obj(
        f=out_obj_path,
        verts=verts,
        faces=faces,
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        texture_map=tex_map,
        decimal_places=6
    )
    return out_obj_path

import imageio.v2 as imageio

@torch.no_grad()
def _save_kd_image_single(kd_img, out_path):
    """
    Save kd_img (H,W,3) tensor in [0,1] to PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = (kd_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(out_path, img)
    return out_path

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
        return compute_vertex_normals(mesh)

def get_mesh_tangents(mesh):
    """Extract vertex tangents from PyTorch3D Meshes object"""
    if hasattr(mesh, 'verts_tangents_padded') and mesh.verts_tangents_padded() is not None:
        return mesh.verts_tangents_padded()
    else:
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
#  Custom Shaders for PyTorch3D
# ==============================================================================================
class MaterialShader(ShaderBase):
    """Custom shader that uses the original material system with PyTorch3D"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def debug_material_shader(self, fragments, meshes, material, prior_mesh, feat):
        """Debug what's happening in the material shader"""
        
        B, H, W, K = fragments.pix_to_face.shape
        device = meshes.device

        # Check if we have valid fragments
        valid_fragments = (fragments.pix_to_face >= 0).sum()
        print(f"Valid fragments: {valid_fragments}/{fragments.pix_to_face.numel()}")
        
        # Check mesh data
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        print(f"Mesh vertices: {verts.shape}, faces: {faces.shape}")
        print(f"Vertex bounds: {verts.min():.3f} to {verts.max():.3f}")
        
        # Check normals
        normals = meshes.verts_normals_packed()
        print(f"Normals shape: {normals.shape}")
        print(f"Normal bounds: {normals.min():.3f} to {normals.max():.3f}")
        
        # Check material sampling
        if material is not None and prior_mesh is not None:
            prior_verts = prior_mesh.verts_packed()
            print(f"Prior mesh vertices: {prior_verts.shape}")
            print(f"Prior vertex bounds: {prior_verts.min():.3f} to {prior_verts.max():.3f}")
            
            # Test material sampling at a few points
            test_pos = prior_verts[:10]  # Sample first 10 vertices
            try:
                test_tex = material.sample(test_pos, feat=feat)
                print(f"Material sample shape: {test_tex.shape}")
                print(f"Material sample bounds: {test_tex.min():.3f} to {test_tex.max():.3f}")
            except Exception as e:
                print(f"Material sampling failed: {e}")
        
        return valid_fragments > 0
    
    def forward(self, fragments, meshes, material=None, lgt=None, bsdf='diffuse', 
                feat=None, dino_net=None, class_vector=None, prior_mesh=None, **kwargs):
        """
        Custom forward pass that mimics the original renderer's shading using PyTorch3D
        """
        B, H, W, K = fragments.pix_to_face.shape
        device = meshes.device

        # Vertex and face info
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        pix_to_face = fragments.pix_to_face
        bary = fragments.bary_coords

        # Interpolate normals for valid fragments
        normals = meshes.verts_normals_packed()
        face_normals = normals[faces]
        pix_normals = interpolate_face_attributes(
            pix_to_face,  # (B, H, W, K)
            bary,  # (B, H, W, K, 3)
            face_normals         # (F, 3, C)
        ) # (B, H, W, K, C)
        pix_normals = pix_normals[..., 0, :]
        pix_normals = util.safe_normalize(pix_normals)

        # Sample material for valid fragments
        if material is not None and prior_mesh is not None:
            prior_verts = prior_mesh.verts_packed()
            face_prior_verts = prior_verts[faces]
            tex_pos_k = interpolate_face_attributes(
                pix_to_face,  # (B, H, W, K)
                bary,  # (B, H, W, K, 3)
                face_prior_verts         # (F, 3, C)
            ) # (B, H, W, K, C)

            tex_pos = tex_pos_k[..., 0, :]  # (B, H, W, 3)
            print("????check", tex_pos_k.shape, tex_pos.shape, feat.shape)

            all_tex_full = material.sample(tex_pos, feat=feat) # Shape: (B, H, W, 9)
            # Extract values for valid fragments
            kd = all_tex_full[..., :3]
            ks = all_tex_full[..., 3:6]
            perturbed_nrm = all_tex_full[..., 6:9]
            
        else:
            # Default material
            kd = torch.ones(B, H, W, 3, device=device) * 0.5
            ks = torch.ones(B, H, W, 3, device=device) * 0.1
            perturbed_nrm = torch.zeros_like(pix_normals)

        # Apply lighting for valid fragments
        if bsdf == 'diffuse':
            if lgt is None:
                shaded_col = kd
            elif isinstance(lgt, light.EnvironmentLight):
                shaded_col = lgt.shade(tex_pos, pix_normals, kd, ks, None, specular=False)
            else:
                # Use the same lighting as original renderer
                try:
                    shaded_col, shading = lgt.shade(feat, kd, pix_normals)
                except:
                    # Fallback to simple lighting
                    shaded_col = kd * 0.7 + 0.3
        else:
            shaded_col = kd

        # Scatter results back to full resolution

        # Create output buffers
        buffers = {
            'shaded': shaded_col,
            'kd': kd,
            'ks': ks,
            'normal': (pix_normals + 1.0) * 0.5,
            'geo_normal': (pix_normals + 1.0) * 0.5,
        }
        if shading is not None:
            buffers['shading'] = shading
        # Handle DINO features
        if dino_net is not None and prior_mesh is not None:
            # Create full texture coordinate map for DINO
            dino_pred_full = dino_net.sample(tex_pos, feat=class_vector)  # Shape: (B, H, W, 16)
            buffers['dino_pred'] = dino_pred_full

        return buffers

class InterpolateShader(ShaderBase):
    """Shader for interpolating vertex features"""
    def forward(self, fragments, meshes, **kwargs):
        # Get per-vertex features (B, V, C)
        verts_features = kwargs["vertex_features"].reshape(-1, 16)  # passed manually

        # Get face indices (B, F, 3)
        faces = meshes.faces_packed()
        face_features = verts_features[faces]  # (B, F, 3, C)
        
        # Interpolate using barycentric coords
        interpolated = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_features
        )  # (B, H, W, K, C)

        # Use closest face (K=1)
        return interpolated[:, :, :, 0, :]  # (B, H, W, C)

# ==============================================================================================
#  PyTorch3D Renderer Classes
# ==============================================================================================
class PyTorch3DRenderer:
    """PyTorch3D-based renderer for better numerical stability"""
    
    def __init__(self, resolution, device='cuda'):
        self.resolution = resolution
        self.device = device
        self.h, self.w = resolution
        
        # Initialize renderers for different modes
        self._init_renderers()
    
    def _init_renderers(self):
        """Initialize different renderers for various modes"""
        
        # Silhouette renderer for masks
        blend_params_silhouette = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings_silhouette = RasterizationSettings(
            image_size=self.resolution[0],
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params_silhouette.sigma,
            faces_per_pixel=100,
            perspective_correct=False,
            cull_backfaces=False,
        )
        
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params_silhouette)
        )
        
        # Material renderer for complex shading
        raster_settings_material = RasterizationSettings(
            image_size=self.resolution[0],
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,
        )
        
        self.material_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_material
            ),
            shader=MaterialShader(device=self.device)
        )
        
        # DINO renderer for feature interpolation
        raster_settings_dino = RasterizationSettings(
            image_size=self.resolution[0],
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,
        )
        
        self.dino_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings_dino), 
            shader=InterpolateShader()
        )
        
    
    # Add this debug code to see if mesh is in camera view
    def debug_mesh_visibility(self, mesh, mvp, w2c, campos):
        verts = mesh.verts_padded()
        
        # Transform vertices to clip space
        verts_homo = torch.cat([verts, torch.ones_like(verts[..., :1])], dim=-1)
        verts_clip = (verts_homo @ mvp.transpose(-1, -2))
        
        # Check if vertices are in view frustum
        verts_ndc = verts_clip[..., :3] / (verts_clip[..., 3:] + 1e-6)
        
        in_frustum = (verts_ndc[..., 0] >= -1) & (verts_ndc[..., 0] <= 1) & \
                    (verts_ndc[..., 1] >= -1) & (verts_ndc[..., 1] <= 1) & \
                    (verts_ndc[..., 2] >= -1) & (verts_ndc[..., 2] <= 1)
        
        print(f"Vertices in frustum: {in_frustum.sum()}/{in_frustum.numel()}")
        print(f"NDC bounds: {verts_ndc.min(dim=1)[0]} to {verts_ndc.max(dim=1)[0]}")
        
        return in_frustum

    def _create_camera_from_mvp(self, mvp, w2c, campos):
        """Create PyTorch3D camera from MVP matrix and camera parameters"""
        batch_size = mvp.shape[0]

        R_gl = w2c[:, :3, :3]
        T_gl = w2c[:, :3, 3]
        wrc_inv = torch.inverse(w2c)
        proj = torch.bmm(mvp, wrc_inv)

        device = R_gl.device
        B = R_gl.shape[0]

        # --- 1) Convert OpenGL extrinsics to PyTorch3D extrinsics (+Z in) ---
        
        S = torch.diag(torch.tensor([-1., 1., -1.], device=device)).expand(B, 3, 3).float()
        off = axis_angle_to_matrix(torch.tensor([0.0 , torch.pi/2, 0.0], device=device))#[:,:2].reshape(1, 6)
        R_p3d = off @ S @ R_gl
        T_p3d = (S @ T_gl[..., None]).squeeze(-1)  # (B,3)

        # --- 2) Extract FoV, aspect, znear, zfar from OpenGL proj ---
        fy = proj[:, 1, 1]                    # = 1/tan(fov_y/2)
        fov_y = 2.0 * torch.atan(1.0 / fy)    # radians
        fov_deg = -1 * fov_y * (180.0 / torch.pi)

        aspect = -1 *proj[:, 1, 1] / proj[:, 0, 0]

        a = proj[:, 2, 2]
        b = proj[:, 2, 3]
        znear = b * (a + 1.0) / (a * a - 1.0)
        zfar  = b * (a - 1.0) / (a * a - 1.0)

        # (Optional) clamp for safety
        znear = torch.clamp(znear, min=1e-6)

        # --- 3) Create PyTorch3D FoV camera ---
        self.R = torch.tensor([[-1, 0, 0],
                               [0, -1, 0],
                               [0, 0, 1]]).repeat(1, 1, 1).to(device)
        self.T = torch.zeros(3).repeat(1, 1).to(device)
        fov = 2 * np.arctan(256/ (5000 * 2)) * 180 / torch.pi
        cameras = OpenGLPerspectiveCameras(zfar=350, fov=fov, R = self.R, T = self.T, device=self.device)


        R = w2c[:, :3, :3]
        T = w2c[:, :3, 3]
        
        # Create transform
        transform = Transform3d(device=self.device)
        transform = transform.rotate(R)
        transform = transform.translate(T)
        
        return cameras, transform
    
    def _composite_buffer(self, key, rendered, fragments, background, mesh):
        """Composite buffer with proper alpha handling using silhouette renderer"""
        # Get the rendered buffer
        buffer = rendered[key]  # (B, H, W, C) without alpha
        
        # Use silhouette renderer to get proper alpha mask (like render_p3d.py)
        # The silhouette renderer gives us the proper coverage mask
        silhouette = self.silhouette_renderer(mesh)  # This should return (B, H, W, 4) with alpha in last channel
        alpha = silhouette[..., -1].unsqueeze(-1)  # Extract alpha channel from silhouette (B, H, W)

        # Add alpha to buffer
        buffer_with_alpha = torch.cat([buffer, alpha], dim=-1)  # (B, H, W, C+1)
        
        # Composite with background (like original renderer's composite_buffer)
        # Use alpha blending: result = alpha * foreground + (1 - alpha) * background
        if background is not None:
            # Ensure background has same shape
            if background.shape[-1] == buffer_with_alpha.shape[-1] - 1:
                # Background doesn't have alpha, add it
                bg_alpha = torch.zeros_like(background[..., :1])
                background = torch.cat([background, bg_alpha], dim=-1)
            # Composite
            result = torch.lerp(background, buffer_with_alpha, alpha)
        else:
            result = buffer_with_alpha
        return result
    
    def render_mesh(
        self,
        ctx,  # Ignored for PyTorch3D
        mesh,
        mtx_in,
        w2c,
        view_pos,
        material,
        lgt,
        resolution,
        spp=1,
        num_layers=1,
        msaa=False,
        background=None,
        bsdf=None,
        feat=None,
        render_modes=None,
        prior_mesh=None,
        two_sided_shading=True,
        dino_net=None,
        num_frames=None,
        class_vector=None,
    ):
        """
        PyTorch3D-based render_mesh function with same interface as original
        """
        if render_modes is None:
            render_modes = ['shaded']
        # Ensure mesh is not empty
        if is_mesh_empty(mesh):
            # Return empty buffers
            out_buffers = []
            for mode in render_modes:
                if mode == 'shaded':
                    buffer = torch.zeros(1, 4, resolution[0], resolution[1], device=mesh.device)
                elif mode in ['kd', 'ks', 'normal', 'geo_normal']:
                    buffer = torch.zeros(1, 3, resolution[0], resolution[1], device=mesh.device)
                elif mode in ['shading', 'depth']:
                    buffer = torch.zeros(1, 1, resolution[0], resolution[1], device=mesh.device)
                elif mode == 'flow':
                    buffer = torch.zeros(1, 2, resolution[0], resolution[1], device=mesh.device)
                elif mode == 'dino_pred':
                    buffer = torch.zeros(1, 3, resolution[0], resolution[1], device=mesh.device)
                else:
                    buffer = None
                out_buffers.append(buffer)
            return out_buffers
        
        batch_size = mesh.verts_padded().shape[0]
        
        # Create camera
        cameras, transform = self._create_camera_from_mvp(mtx_in, w2c, view_pos)
        
        # Update renderer cameras
        self.silhouette_renderer.rasterizer.cameras = cameras
        self.material_renderer.rasterizer.cameras = cameras
        self.dino_renderer.rasterizer.cameras = cameras
        
        out_buffers = []

        transformed_mesh =  mesh.clone()
        fragments = self.material_renderer.rasterizer(transformed_mesh)
        rendered = self.material_renderer(
            transformed_mesh,
            material=material,
            lgt=lgt,
            bsdf=bsdf,
            feat=feat,
            dino_net=dino_net,
            class_vector=class_vector,
            prior_mesh=prior_mesh
        )
        
        points = transformed_mesh.verts_padded()
        proj_points = cameras.transform_points_screen(points, image_size=(256,256))[:, :, :2]

        
        for mode in render_modes:
            bg = background if mode in ['shaded', 'geo_normal'] else torch.zeros_like(rendered[mode])
            out = self._composite_buffer(mode, rendered, fragments, bg, transformed_mesh)
            if mode == 'shaded':
                pass  # RGBA channels
            elif mode in ['kd', 'ks', 'normal', 'geo_normal']:
                out = out[..., :3]
            elif mode == 'shading':
                out = out[..., :1]
            elif mode == 'flow':
                out = out[..., :2]
            elif mode == 'dino_pred':
                out = out[..., :-1]
            elif mode == 'depth':
                out = out[..., :1]
           
            out = out.permute(0, 3, 1, 2)  # NHWC -> NCHW
            out_buffers.append(out)

        return out_buffers, proj_points

    def get_verts_vis(self, mesh, mtx_in, w2c, view_pos):
        """(B,V) float mask: 1 where the vertex belongs to at least one rasterized
        (visible) face, 0 otherwise. Used by the KT-PCK@0.1 eval (model/eval/) to
        match Dessie's evalkp_utils.lassierender.Renderer.get_verts_vis, but driven
        by this model's own predicted camera rather than a fixed LASSIE camera."""
        cameras, _ = self._create_camera_from_mvp(mtx_in, w2c, view_pos)
        self.material_renderer.rasterizer.cameras = cameras
        fragments = self.material_renderer.rasterizer(mesh)
        packed_faces = mesh.faces_packed()  # (sum_F, 3), packed vertex indices
        pix_to_face = fragments.pix_to_face  # (B, H, W, K)
        visible_faces = pix_to_face[pix_to_face >= 0].unique()
        visible_verts = packed_faces[visible_faces].unique()
        batch_size, num_verts = mesh.verts_padded().shape[:2]
        out = torch.zeros(batch_size * num_verts, device=mesh.device)
        out[visible_verts] = 1.0
        return out.view(batch_size, num_verts)

# ==============================================================================================
#  Main render_mesh function (compatible interface)
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
    spp=1,
    num_layers=1,
    msaa=False,
    background=None,
    bsdf=None,
    feat=None,
    render_modes=None,
    prior_mesh=None,
    two_sided_shading=True,
    dino_net=None,
    num_frames=None,
    class_vector=None,
):
    """
    PyTorch3D-based render_mesh function with same interface as original render.py
    """

    # Create renderer instance
    renderer = PyTorch3DRenderer(resolution, device=mesh.device)

    # Call the renderer
    return renderer.render_mesh(
        ctx, mesh, mtx_in, w2c, view_pos, material, lgt, resolution,
        spp, num_layers, msaa, background, bsdf, feat, render_modes,
        prior_mesh, two_sided_shading, dino_net, num_frames, class_vector
    )


def verts_visibility(mesh, mtx_in, w2c, view_pos, resolution):
    """Module-level wrapper matching the render_mesh(...) pattern above."""
    renderer = PyTorch3DRenderer(resolution, device=mesh.device)
    return renderer.get_verts_vis(mesh, mtx_in, w2c, view_pos)

# ==============================================================================================
#  Render UVs (compatible interface)
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture, feat=None):
    """
    PyTorch3D-based render_uv function with same interface as original
    """
    # Simplified UV rendering - returns basic texture sampling
    vertices = get_mesh_vertices(mesh)
    
    # Sample texture from MLP
    all_tex = mlp_texture.sample(vertices, feat=feat)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    
    perturbed_nrm = all_tex[..., -3:]
    kd = all_tex[..., :3]
    ks = all_tex[..., 3:6]
    
    # Create simple mask (all pixels are valid)
    mask = torch.ones(vertices.shape[0], 1, device=vertices.device)
    
    return mask, kd, ks, util.safe_normalize(perturbed_nrm) 