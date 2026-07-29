"""Evaluation metrics: PSNR, SSIM (Wang et al. gaussian 11x11 sigma 1.5),
LPIPS (alex/vgg), and the official competition score.

Score = 0.4*(1 - LPIPS) + 0.3*SSIM + 0.3*clamp(PSNR/PSNR_max, 0, 1).
PSNR_max is unknown; report several candidates (default headline 40).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_WINDOW_CACHE: dict = {}
_LPIPS_CACHE: dict = {}


def _gaussian_window(channels, device, dtype, win_size=11, sigma=1.5):
    key = (channels, str(device), dtype, win_size, sigma)
    if key not in _WINDOW_CACHE:
        coords = torch.arange(win_size, dtype=torch.float64) - (win_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g = (g / g.sum()).to(dtype)
        k2d = torch.outer(g, g)
        _WINDOW_CACHE[key] = k2d.expand(channels, 1, win_size, win_size).contiguous().to(device)
    return _WINDOW_CACHE[key]


def ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range=1.0, win_size=11, sigma=1.5):
    """SSIM with gaussian window, 'valid' convolution (classic Wang et al.).

    x, y: [B, C, H, W] float tensors in [0, data_range]. Differentiable.
    """
    c = x.shape[1]
    w = _gaussian_window(c, x.device, x.dtype, win_size, sigma)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    pad = win_size // 2
    # Hotfix #2: Apply reflection padding to prevent border zeroing floaters
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    y = F.pad(y, (pad, pad, pad, pad), mode='reflect')
    
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, w, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, groups=c) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return ssim_map.mean()


def psnr_torch(x: torch.Tensor, y: torch.Tensor, data_range=1.0):
    mse = torch.mean((x - y) ** 2)
    return 10.0 * torch.log10(data_range ** 2 / torch.clamp(mse, min=1e-12))


def get_lpips(net="vgg", device="cuda"):
    key = (net, str(device))
    if key not in _LPIPS_CACHE:
        import lpips  # lazy: heavy import
        model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


@torch.no_grad()
def lpips_value(x: torch.Tensor, y: torch.Tensor, net="vgg", device="cuda"):
    """x, y: [B,3,H,W] in [0,1]."""
    model = get_lpips(net, device)
    return float(model(x.to(device) * 2 - 1, y.to(device) * 2 - 1).mean())


def official_score(lpips_val: float, ssim_val: float, psnr_val: float, psnr_max: float = 50.0):
    psnr_norm = float(np.clip(psnr_val / psnr_max, 0.0, 1.0))
    return 0.4 * (1.0 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm


@torch.no_grad()
def compare_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray, device="cuda",
                  lpips_nets=("alex",)) -> dict:
    """Full metric set between two HxWx3 uint8 RGB images (as an evaluator
    reading files from disk would see them)."""
    assert pred_u8.shape == gt_u8.shape, f"shape mismatch {pred_u8.shape} vs {gt_u8.shape}"
    x = torch.from_numpy(pred_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    y = torch.from_numpy(gt_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    out = {
        "psnr": float(psnr_torch(x, y)),
        "ssim": float(ssim_torch(x, y)),
    }
    for net in lpips_nets:
        lp = lpips_value(x, y, net=net, device=device)
        out[f"lpips_{net}"] = lp
        for pm in (30.0, 40.0, 50.0):
            out[f"score@{int(pm)}_{net}"] = official_score(lp, out["ssim"], out["psnr"], pm)
    # Leaderboard round 1 decoded (2026-07-11): PSNR_max=50, LPIPS backbone=VGG.
    # "score_lb" is the best-known estimate of the official score.
    if "score@50_vgg" in out:
        out["score_lb"] = out["score@50_vgg"]
    # legacy unsuffixed keys (alex-based) kept for old eval.jsonl compatibility
    if "lpips_alex" in out:
        for pm in (30.0, 40.0, 50.0):
            out[f"score@{int(pm)}"] = out[f"score@{int(pm)}_alex"]
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean of each numeric key across per-image metric dicts."""
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
