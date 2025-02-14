import os
import cv2
import numpy as np
import torch

# geometry
import pytorch3d
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

# Colors (source: Zhang et al. 2020)
BLUE = [0.65098039, 0.74117647, 0.85882353]
RED = [251 / 255.0, 128 / 255.0, 114 / 255.0]


## copy this without hands
def render_hoi(
    image: np.array,
    cameras: pytorch3d.renderer.cameras.CamerasBase,
    # hand_verts: torch.Tensor,
    # hand_faces: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_faces: torch.Tensor
):
    """
    Render hand and object meshes given pytorch3d camera.

    Args:
        image (np.array): Input image.
        cameras (CamerasBase): pytorch3d Camera object.
        hand_verts (Tensor): Hand vertices of shape (V_h, 3).
        hand_faces (Tensor): Hand faces of shape (F_h, 3).
        obj_verts (Tensor): Obj vertices of shape (V_o, 3).
        obj_faces (Tensor): Obj faces of shape (F_o, 3).
    """
    imh, imw = image.shape[:2]
    raster_settings = RasterizationSettings(
        image_size=(imh, imw), 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )

    # Render hand and obj
    # hand_colors = np.array([(*BLUE, 1.0)] * hand_verts.shape[0])
    obj_colors = np.array([(*RED, 1.0)] * obj_verts.shape[0])
    # verts_rgb = [torch.tensor(c, dtype=torch.float32).to(device)[..., :3] for c in [hand_colors, obj_colors]] 
    verts_rgb = [torch.tensor(c, dtype=torch.float32).to(obj_verts.device)[..., :3] for c in [obj_colors]]
    textures = TexturesVertex(verts_features=verts_rgb)
    # all_verts = [hand_verts, obj_verts]
    # all_verts = [obj_verts]
    # breakpoint()
    # Create Meshes for hands and object together
    obj_mesh = Meshes(
        # verts=all_verts,
        verts = obj_verts.unsqueeze(0),
        # faces=[hand_faces, obj_faces],
        faces = obj_faces.verts_idx.unsqueeze(0),
        textures=textures
    )
    obj_mesh = pytorch3d.structures.meshes.join_meshes_as_scene(obj_mesh)

    # We can add a point light in front of the object. 
    # lights = PointLights(device=obj_verts.device, location=light_location)
    cameras = cameras.to(obj_verts.device)
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        # shader=HardPhongShader(device=obj_verts.device, cameras=cameras, lights=lights)
        shader=HardPhongShader(device=obj_verts.device, cameras=cameras)
    )
    render = phong_renderer(meshes_world=obj_mesh)
    combined_mask = render[0, ..., 3].detach().cpu().numpy()
    combined_mask = (combined_mask > 0).astype(np.uint8)*255

    render = render[0].detach().squeeze().cpu().numpy()
    render = (render*255).astype(np.uint8)
    mask = (render[..., [3]] > 0).astype(np.uint8)
    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
    if type(image)==torch.Tensor:
        image = image.cpu().numpy()
    overlay_image = render[..., :3]*mask + image*(1-mask)
    return overlay_image

