"""Shared gsplat rendering + appearance helpers (lazy gsplat import).

All rendering goes through render_gs() so that every stage (training eval,
test rendering, warp fusion) uses identical conventions:
  - viewmat: 4x4 COLMAP world-to-camera (OpenCV axes) — gsplat native.
  - K: COLMAP-convention intrinsics (cx = W/2 style, +0.5 pixel centers) —
    matches gsplat's rasterizer sample points.
"""

from __future__ import annotations

import numpy as np
import torch


def render_gs(splats: dict, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int,
              sh_degree: int, render_mode: str = "RGB", antialiased: bool = True,
              absgrad: bool = False, background: torch.Tensor | None = None):
    """Render one camera. Returns (render [H,W,C], alpha [H,W,1], info).

    render_mode "RGB" -> C=3; "RGB+ED" -> C=4 with expected z-depth in ch 3.
    """
    from gsplat import rasterization

    device = splats["means"].device
    if background is None:
        background = torch.zeros(1, 3, device=device)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    renders, alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"].squeeze(-1) if splats["opacities"].ndim == 2 else splats["opacities"]),
        colors=colors,
        viewmats=viewmat[None].to(device),
        Ks=K[None].to(device),
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=absgrad,
        rasterize_mode="antialiased" if antialiased else "classic",
        render_mode=render_mode,
        backgrounds=background if render_mode == "RGB" else None,
    )
    return renders[0], alphas[0], info


def apply_appearance(rgb: torch.Tensor, emb6: torch.Tensor | None) -> torch.Tensor:
    """Per-image affine color: out = rgb * (1 + gain) + bias, emb6 = [gain|bias]."""
    if emb6 is None:
        return rgb
    gain = emb6[:3].view(1, 1, 3)
    bias = emb6[3:].view(1, 1, 3)
    return rgb * (1.0 + gain) + bias


def interp_appearance(target_frame_idx, train_frame_idxs, emb_weight: torch.Tensor, k=4):
    """Distance-weighted average of the k temporally nearest train embeddings.

    target_frame_idx: int or None; train_frame_idxs: list[int|None] aligned
    with emb_weight rows. Returns emb6 tensor or None (identity).
    """
    if emb_weight is None:
        return None
    valid = [(i, fi) for i, fi in enumerate(train_frame_idxs) if fi is not None]
    if target_frame_idx is None or not valid:
        return torch.zeros(6, device=emb_weight.device, dtype=emb_weight.dtype)
    dists = sorted(valid, key=lambda t: abs(t[1] - target_frame_idx))[:k]
    ws = np.array([1.0 / (1.0 + abs(fi - target_frame_idx)) for _, fi in dists])
    ws = ws / ws.sum()
    idxs = torch.tensor([i for i, _ in dists], device=emb_weight.device)
    w = torch.tensor(ws, device=emb_weight.device, dtype=emb_weight.dtype)
    return (emb_weight[idxs] * w[:, None]).sum(dim=0)


# ------------------------------ checkpoint IO -------------------------------

def save_checkpoint(path, step, splats, app_emb, app_image_names, config: dict, extra=None):
    payload = {
        "step": step,
        "splats": {k: v.detach().cpu() for k, v in splats.items()},
        "app_emb": (app_emb.detach().cpu() if app_emb is not None else None),
        "app_image_names": app_image_names,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    app = ckpt.get("app_emb")
    if app is not None:
        app = app.to(device)
    return splats, app, ckpt
