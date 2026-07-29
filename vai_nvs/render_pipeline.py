"""End-to-end view synthesis pipeline shared by training-eval, test rendering
and sanity checks — guarantees the exact same math everywhere:

    gsplat pinhole render (supersampled)
      -> per-image appearance (optional)
      -> redistort to the real SIMPLE_RADIAL camera (maps from cameras.bin)
      -> area downsample to target size
      -> uint8

The redistortion resample is the ONLY resample: maps are built at the output
supersample so the supersampled render is warped 1:1 (INTER_LINEAR) and then
area-averaged down, which both applies the lens distortion and anti-aliases.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from . import cameras as camlib
from .gs_render import render_gs, apply_appearance


class RedistortCache:
    """Cache of (map_x, map_y) keyed by the full geometric signature."""

    def __init__(self):
        self._maps = {}

    def get(self, render_K, dist_cam, width, height, supersample):
        key = (tuple(np.round(render_K, 8)), dist_cam.model,
               tuple(np.round(dist_cam.params, 10)), width, height, float(supersample))
        if key not in self._maps:
            self._maps[key] = camlib.build_redistort_maps(
                render_K, dist_cam, width, height,
                out_supersample=supersample, render_supersample=supersample)
        return self._maps[key]


@torch.no_grad()
def render_view_to_distorted(splats: dict, qvec, tvec, render_K, dist_cam,
                             width: int, height: int, supersample: float,
                             sh_degree: int, antialiased: bool = True,
                             app_emb6: torch.Tensor | None = None,
                             cache: RedistortCache | None = None) -> np.ndarray:
    """Render one target view and return the distorted uint8 RGB image.

    render_K: (fx, fy, cx, cy) pinhole intrinsics of the target (CSV values
              for test poses, cameras.bin pinhole for train/val views).
    dist_cam: COLMAP Camera carrying the distortion geometry (cameras.bin).
    width/height: output size (CSV width/height).
    """
    device = splats["means"].device
    s = float(supersample)
    rw, rh = int(round(width * s)), int(round(height * s))
    fx, fy, cx, cy = render_K
    
    # S13: Supersample Principal Point Alignment
    if s == 2.0:
        cx_s = 2.0 * cx
        cy_s = 2.0 * cy
    else:
        cx_s = cx * s
        cy_s = cy * s
        
    K = torch.tensor([[fx * s, 0.0, cx_s],
                      [0.0, fy * s, cy_s],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
                      
    viewmat = torch.from_numpy(camlib.w2c_matrix(qvec, tvec)).float()
    render, _, _ = render_gs(splats, viewmat, K, rw, rh, sh_degree,
                             render_mode="RGB", antialiased=antialiased)
    
    # S13: Fast GPU PyTorch Redistortion
    rgb_gpu = torch.clamp(apply_appearance(render, app_emb6), 0.0, 1.0)
    rgb_gpu = rgb_gpu.permute(2, 0, 1).unsqueeze(0)  # (1, 3, rh, rw)

    if cache is None:
        cache = RedistortCache()
    map_x, map_y = cache.get(render_K, dist_cam, width, height, s)
    
    grid_x = torch.from_numpy(map_x).to(device)
    grid_y = torch.from_numpy(map_y).to(device)
    
    # Normalize to [-1, 1] for grid_sample
    grid_x_norm = (grid_x / (rw - 1)) * 2.0 - 1.0
    grid_y_norm = (grid_y / (rh - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1).unsqueeze(0)  # (1, rh, rw, 2)
    
    # Reflection padding matching BORDER_REPLICATE
    warped_gpu = F.grid_sample(rgb_gpu, grid, mode='bilinear' if s != 1.0 else 'bicubic', 
                               padding_mode='reflection', align_corners=True)
                               
    if s != 1.0:
        warped_gpu = F.adaptive_avg_pool2d(warped_gpu, (height, width))
        
    warped = warped_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
    warped = np.clip(warped, 0.0, 1.0)
    return (warped * 255.0 + 0.5).astype(np.uint8)
