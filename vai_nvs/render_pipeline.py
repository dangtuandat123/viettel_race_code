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
    K = torch.tensor([[fx * s, 0.0, cx * s],
                      [0.0, fy * s, cy * s],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    viewmat = torch.from_numpy(camlib.w2c_matrix(qvec, tvec)).float()
    render, _, _ = render_gs(splats, viewmat, K, rw, rh, sh_degree,
                             render_mode="RGB", antialiased=antialiased)
    rgb = apply_appearance(render, app_emb6)
    rgb = torch.clamp(rgb, 0.0, 1.0).cpu().numpy().astype(np.float32)

    if cache is None:
        cache = RedistortCache()
    map_x, map_y = cache.get(render_K, dist_cam, width, height, s)
    interp = cv2.INTER_LANCZOS4 if s == 1.0 else cv2.INTER_LINEAR
    warped = cv2.remap(rgb, map_x, map_y, interpolation=interp,
                       borderMode=cv2.BORDER_REPLICATE)
    if s != 1.0:
        warped = cv2.resize(warped, (width, height), interpolation=cv2.INTER_AREA)
    warped = np.clip(warped, 0.0, 1.0)
    return (warped * 255.0 + 0.5).astype(np.uint8)


def camera_from_meta(cam_meta: dict):
    """Rebuild a colmap_io.Camera from meta.json camera entry."""
    from .colmap_io import Camera
    return Camera(0, cam_meta["model"], int(cam_meta["width"]), int(cam_meta["height"]),
                  np.array(cam_meta["params"], dtype=np.float64))
