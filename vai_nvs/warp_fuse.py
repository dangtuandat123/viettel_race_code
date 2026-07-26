"""Depth-guided neighbor warp fusion, v2 (round 2: video, stride 5/10).

Per target pose:
  1. GS base render exactly like the submission pipeline (supersample x2 ->
     area-downsample), plus native-res expected depth (RGB+ED) and alpha.
  2. Sources: the single nearest train frame on each temporal side (+-1
     stride; arbitrary gaps handled — double/triple-gap targets just get
     farther neighbors and lower confidence).
  3. Backward-warp each source into the target via GS depth (bicubic).
  4. Per-source robust gain/bias (IRLS) alignment to the GS base — kills
     video auto-exposure drift without GT.
  5. Residual optical-flow refinement (DIS, magnitude-clamped) of each warp
     against the GS base — absorbs the 1-3 px misalignment caused by
     imperfect GS depth.
  6. Soft confidence = occlusion-consistency x photo-consistency (low-pass)
     x cross-neighbor agreement x depth-edge guard x alpha gate.
  7. fused = M * conf-weighted warp blend + (1-M) * GS base;  M feathered,
     capped. Degrades gracefully to the plain GS render everywhere.

v1 -> v2 changes (v1 lost 3-4 points on fold-0):
  - 1 neighbor/side instead of 2+2 pose-NN (2nd-order neighbors were 2x
    misaligned; correlated errors defeated the median agreement gate)
  - explicit exposure alignment (v1 had none with appearance disabled)
  - soft confidence blend instead of hard median replacement
  - residual flow refinement (v1: none)
  - GS base kept at supersample x2 (v1 dropped to x1 and lost quality even
    where it fell back to GS)

GATING: run --mode val on the fold-0 checkpoint first; writes
fusion_decision.json (enable iff mean delta >= margin AND p10 >= -0.005).
--mode test refuses to write unless enabled (or --force).

Usage:
  python -m vai_nvs.warp_fuse --work W --scene S --run mcmc_f0_s60k  --mode val --fold 0
  python -m vai_nvs.warp_fuse --work W --scene S --run mcmc_f-1_s60k --mode test
  python -m vai_nvs.render_test --work W --scene S --run mcmc_f-1_s60k \
      --fused-dir <work>/<scene>/fused_test --out <work>/<scene>/pred_test_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import colmap_io, dataset as ds, metrics
from . import cameras as camlib
from .gs_render import render_gs, apply_appearance, interp_appearance, load_checkpoint


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", choices=["best", "last"], default="last")
    ap.add_argument("--mode", choices=["val", "test"], required=True)
    ap.add_argument("--fold", type=int, default=0, help="fold for --mode val")
    ap.add_argument("--supersample", type=float, default=2.0, help="GS base render supersample")
    ap.add_argument("--n-temporal", type=int, default=1, help="neighbors per side")
    ap.add_argument("--depth-rel-tol", type=float, default=0.015, help="soft occlusion sigma (rel depth)")
    ap.add_argument("--sigma-photo", type=float, default=0.12, help="photo-consistency sigma (low-pass)")
    ap.add_argument("--sigma-pair", type=float, default=0.08, help="cross-neighbor sigma (low-pass)")
    ap.add_argument("--one-side-conf", type=float, default=0.65, help="pair conf when only one side")
    ap.add_argument("--align-mode", choices=["pair", "gs", "none"], default="pair",
                    help="exposure alignment target: 'pair' = temporal interpolation "
                         "between the two neighbors' own exposures (unbiased by GS "
                         "photometry; falls back to 'gs' when one-sided), 'gs' = GS base")
    ap.add_argument("--edge-tol", type=float, default=0.02, help="rel depth gradient edge threshold")
    ap.add_argument("--edge-dilate", type=int, default=3, help="edge dilation iterations (3x3)")
    ap.add_argument("--edge-factor", type=float, default=0.15, help="confidence multiplier at edges")
    ap.add_argument("--flow-radius", type=float, default=6.0, help="max residual flow px (0 = off)")
    ap.add_argument("--gain-clamp", type=float, default=0.35, help="|gain-1| clamp for exposure fit")
    ap.add_argument("--bias-clamp", type=float, default=0.15, help="|bias| clamp for exposure fit")
    ap.add_argument("--m-cap", type=float, default=0.95, help="max warp weight in the final blend")
    ap.add_argument("--feather", type=float, default=2.0, help="gaussian sigma for mask feathering")
    ap.add_argument("--min-alpha", type=float, default=0.5)
    ap.add_argument("--score-margin", type=float, default=0.003,
                    help="mean fold-0 delta needed to enable fusion")
    ap.add_argument("--p10-floor", type=float, default=-0.005,
                    help="10th-percentile per-image delta must stay above this")
    ap.add_argument("--force", action="store_true", help="test mode: ignore decision file")
    ap.add_argument("--save-debug", type=int, default=0, help="val: dump debug PNGs for first N images")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ------------------------------- geometry -----------------------------------

def unproject_project(depth_t: torch.Tensor, K_t, w2c_t: torch.Tensor,
                      w2c_s: torch.Tensor, K_s):
    """Map every target pixel to source pixel coords via target z-depth.

    depth_t: [H,W] z-depth at the target. Returns (us, vs [H,W] COLMAP coords
    in the source image, z_s [H,W] source-camera z of the same points).
    """
    device = depth_t.device
    H, W = depth_t.shape
    fx_t, fy_t, cx_t, cy_t = K_t
    fx_s, fy_s, cx_s, cy_s = K_s
    js = torch.arange(W, device=device, dtype=torch.float32)
    is_ = torch.arange(H, device=device, dtype=torch.float32)
    v, u = torch.meshgrid(is_ + 0.5, js + 0.5, indexing="ij")
    x = (u - cx_t) / fx_t
    y = (v - cy_t) / fy_t
    d = depth_t
    xc = torch.stack([x * d, y * d, d, torch.ones_like(d)], dim=-1)  # [H,W,4]
    T = (w2c_s @ torch.linalg.inv(w2c_t)).to(device=device, dtype=torch.float32)
    xs = xc @ T.T
    z_s = xs[..., 2]
    z_safe = torch.where(z_s > 1e-6, z_s, torch.ones_like(z_s))
    us = fx_s * (xs[..., 0] / z_safe) + cx_s
    vs = fy_s * (xs[..., 1] / z_safe) + cy_s
    return us, vs, z_s


def sample_at(img_chw: torch.Tensor, us: torch.Tensor, vs: torch.Tensor,
              W: int, H: int, mode: str = "bilinear") -> torch.Tensor:
    """grid_sample at COLMAP coords (align_corners=False: gx = 2u/W - 1)."""
    gx = 2.0 * us / W - 1.0
    gy = 2.0 * vs / H - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(img_chw[None], grid, mode=mode,
                        padding_mode="zeros", align_corners=False)[0]
    return out


# ------------------------------ photometric ---------------------------------

def robust_gain_bias(src: torch.Tensor, ref: torch.Tensor, w: torch.Tensor,
                     iters: int = 3, gain_clamp: float = 0.35,
                     bias_clamp: float = 0.15):
    """Per-channel robust affine fit ref ~= g*src + b on weighted pixels.

    src, ref: [N,3]; w: [N] initial weights (>=0). IRLS with hard trimming of
    the worst 20% residuals each round. Returns (g[3], b[3]) tensors.
    """
    g = torch.ones(3, device=src.device)
    b = torch.zeros(3, device=src.device)
    for c in range(3):
        x, y = src[:, c], ref[:, c]
        wc = w.clone()
        for _ in range(iters):
            sw = wc.sum().clamp(min=1e-6)
            mx = (wc * x).sum() / sw
            my = (wc * y).sum() / sw
            vx = (wc * (x - mx) ** 2).sum() / sw
            cxy = (wc * (x - mx) * (y - my)).sum() / sw
            gc = cxy / vx.clamp(min=1e-8) if float(vx) > 1e-8 else torch.tensor(1.0, device=x.device)
            gc = gc.clamp(1.0 - gain_clamp, 1.0 + gain_clamp)
            bc = (my - gc * mx).clamp(-bias_clamp, bias_clamp)
            r = (gc * x + bc - y).abs()
            if r.numel() > 100:
                thr = torch.quantile(r[wc > 0], 0.8) if (wc > 0).sum() > 100 else r.max()
                wc = w * (r <= thr).float()
            g[c], b[c] = gc, bc
    return g, b


def lowpass(img_hwc: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return cv2.GaussianBlur(img_hwc, (0, 0), sigma)


def to_u8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def gray_u8(img01: np.ndarray) -> np.ndarray:
    return to_u8(cv2.cvtColor(img01, cv2.COLOR_RGB2GRAY))


def clamp_flow(flow: np.ndarray, radius: float) -> np.ndarray:
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    scale = np.minimum(1.0, radius / np.maximum(mag, 1e-6)).astype(np.float32)
    return flow * scale[..., None]


# --------------------------------- fuser -------------------------------------

class Fuser:
    def __init__(self, args, meta, work_scene: Path, splats, app_emb, ckpt, device):
        self.args, self.meta, self.work_scene = args, meta, work_scene
        self.splats, self.app_emb, self.device = splats, app_emb, device
        self.sh_degree = ckpt["config"]["sh_degree"]
        self.antialiased = not ckpt["config"].get("no_antialiased", False)
        self.name2im = {im["name"]: im for im in meta["images"]}
        self.app_names = ckpt.get("app_image_names") or []
        self.app_frame_idxs = [self.name2im[n]["frame_idx"] if n in self.name2im else None
                               for n in self.app_names]
        self.depth_cache: dict = {}

    def cam_K(self, camera_id):
        cm = self.meta["cameras"][str(camera_id)]
        cam = colmap_io.Camera(0, cm["model"], cm["width"], cm["height"], np.array(cm["params"]))
        return camlib.camera_pinhole(cam)

    def emb_for(self, frame_idx):
        if self.app_emb is None:
            return None
        return interp_appearance(frame_idx, self.app_frame_idxs, self.app_emb)

    @torch.no_grad()
    def _render(self, qvec, tvec, K, W, H, mode):
        viewmat = torch.from_numpy(camlib.w2c_matrix(qvec, tvec)).float()
        Kt = torch.tensor([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], dtype=torch.float32)
        out, alpha, _ = render_gs(self.splats, viewmat, Kt, W, H, self.sh_degree,
                                  render_mode=mode, antialiased=self.antialiased)
        return out, alpha[..., 0], viewmat

    @torch.no_grad()
    def render_target(self, qvec, tvec, K, W, H, frame_idx):
        """GS base at supersample (pipeline-identical), depth+alpha at native."""
        s = float(self.args.supersample)
        rw, rh = int(round(W * s)), int(round(H * s))
        Ks = (K[0] * s, K[1] * s, K[2] * s, K[3] * s)
        hi, _, _ = self._render(qvec, tvec, Ks, rw, rh, "RGB")
        emb6 = self.emb_for(frame_idx)
        hi = apply_appearance(hi[..., :3], emb6).clamp(0, 1)
        base = F.interpolate(hi.permute(2, 0, 1)[None], size=(H, W), mode="area")[0]
        base = base.permute(1, 2, 0).contiguous()
        out, alpha_t, viewmat = self._render(qvec, tvec, K, W, H, "RGB+ED")
        depth_t = out[..., 3]
        return base, depth_t, alpha_t, viewmat

    @torch.no_grad()
    def source_depth(self, name):
        """Cached native-res (depth, alpha, viewmat, K, W, H) at a train pose."""
        if name not in self.depth_cache:
            im = self.name2im[name]
            K = self.cam_K(im["camera_id"])
            cm = self.meta["cameras"][str(im["camera_id"])]
            out, alpha, viewmat = self._render(np.array(im["qvec"]), np.array(im["tvec"]),
                                               K, cm["width"], cm["height"], "RGB+ED")
            if len(self.depth_cache) > 150:
                self.depth_cache.clear()
            self.depth_cache[name] = (out[..., 3].half().cpu(), alpha.half().cpu(),
                                      viewmat, K, cm["width"], cm["height"])
        return self.depth_cache[name]

    def load_source_image(self, name) -> torch.Tensor:
        arr = np.asarray(Image.open(self.work_scene / "images_ud" / (name + ".png"))
                         .convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).to(self.device)

    def source_frame_idx(self, name):
        im = self.name2im.get(name)
        return None if im is None else im.get("frame_idx")

    def pick_neighbors(self, frame_idx, pool):
        """Nearest train frame(s) on each temporal side, by frame index."""
        a = self.args
        idxed = sorted((self.name2im[n]["frame_idx"], n) for n in pool
                       if n in self.name2im and self.name2im[n]["frame_idx"] is not None)
        if frame_idx is None or not idxed:
            return []
        before = [(i, n) for i, n in idxed if i < frame_idx]
        after = [(i, n) for i, n in idxed if i > frame_idx]
        picks = [n for _, n in reversed(before[-a.n_temporal:])] \
              + [n for _, n in after[:a.n_temporal]]
        return picks

    @torch.no_grad()
    def fuse_target(self, qvec, tvec, K_t, W, H, frame_idx, pool):
        """Returns (fused float32 np [H,W,3] in [0,1], gs_base np, stats dict)."""
        a, dev = self.args, self.device
        gs_base_t, depth_t, alpha_t, w2c_t = self.render_target(qvec, tvec, K_t, W, H, frame_idx)
        gs_base = gs_base_t.cpu().numpy().astype(np.float32)
        stats = {"n_sources": 0, "coverage": 0.0, "pair_err": None}

        names = self.pick_neighbors(frame_idx, pool)
        if not names:
            return gs_base, gs_base, stats

        # depth-edge guard (native depth)
        z = depth_t.clamp(min=1e-6)
        gx = torch.zeros_like(z); gy = torch.zeros_like(z)
        gx[:, 1:] = (z[:, 1:] - z[:, :-1]).abs()
        gy[1:, :] = (z[1:, :] - z[:-1, :]).abs()
        edge = ((gx + gy) / z > a.edge_tol).float().cpu().numpy().astype(np.uint8)
        if a.edge_dilate > 0:
            edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=a.edge_dilate)
        edge_w = np.where(edge > 0, a.edge_factor, 1.0).astype(np.float32)

        alpha_ok = (alpha_t > a.min_alpha).float().cpu().numpy().astype(np.float32)
        gs_lp = lowpass(gs_base)
        gs_g8 = gray_u8(gs_base)

        # ---- phase A: geometric warp + residual flow refinement per source ----
        warps, cgeos, fidxs = [], [], []
        for nm in names:
            d_s, a_s, w2c_s, K_s, Ws, Hs = self.source_depth(nm)
            us, vs, z_s = unproject_project(depth_t, K_t, w2c_t, w2c_s, K_s)
            img_s = self.load_source_image(nm)
            col = sample_at(img_s.permute(2, 0, 1), us, vs, Ws, Hs, "bicubic")
            col = col.permute(1, 2, 0).clamp(0, 1)
            d_smp = sample_at(d_s.float().to(dev)[None], us, vs, Ws, Hs)[0]
            a_smp = sample_at(a_s.float().to(dev)[None], us, vs, Ws, Hs)[0]
            inb = ((us >= 0.5) & (us <= Ws - 0.5) & (vs >= 0.5) & (vs <= Hs - 0.5)
                   & (z_s > 1e-4)).float()
            dz = (z_s - d_smp).clamp(min=0.0)
            conf_occ = torch.exp(-(dz / (a.depth_rel_tol * z_s + 1e-6)) ** 2)
            conf_occ = conf_occ * torch.where(a_smp > 0.5, torch.ones_like(a_smp),
                                              torch.full_like(a_smp, 0.3))
            warp = col.cpu().numpy().astype(np.float32)
            cgeo = (inb * conf_occ).cpu().numpy().astype(np.float32)

            if a.flow_radius > 0:
                # exposure-invariant DIS inputs: standardize both grays
                def _norm(g8):
                    g = g8.astype(np.float32)
                    return np.clip((g - g.mean()) / (g.std() + 1e-6) * 48 + 128,
                                   0, 255).astype(np.uint8)
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = dis.calc(_norm(gs_g8), _norm(gray_u8(warp)), None)
                flow = clamp_flow(flow, a.flow_radius)
                hgrid, wgrid = np.meshgrid(np.arange(H, dtype=np.float32),
                                           np.arange(W, dtype=np.float32), indexing="ij")
                mx = wgrid + flow[..., 0]
                my = hgrid + flow[..., 1]
                warp = cv2.remap(warp, mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
                cgeo = cv2.remap(cgeo, mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            warps.append(warp)
            cgeos.append(cgeo)
            fidxs.append(self.source_frame_idx(nm))

        # ---- phase B: exposure alignment ----
        def _fit_apply(i, ref_img, mask):
            """Fit lowpass(warp_i) -> ref_img (affine, robust) and apply to warp_i."""
            if mask.sum() <= 5000:
                return
            idx = np.flatnonzero(mask.ravel())
            if idx.size > 200_000:
                idx = idx[np.random.RandomState(0).choice(idx.size, 200_000, replace=False)]
            src_lp = lowpass(warps[i])
            src = torch.from_numpy(src_lp.reshape(-1, 3)[idx]).to(dev)
            ref = torch.from_numpy(ref_img.reshape(-1, 3)[idx]).to(dev)
            g, b = robust_gain_bias(src, ref, torch.ones(len(idx), device=dev),
                                    gain_clamp=a.gain_clamp, bias_clamp=a.bias_clamp)
            warps[i] = np.clip(warps[i] * g.cpu().numpy() + b.cpu().numpy(), 0, 1)

        # 'pair': align both warps to the temporal interpolation of their own
        # exposures at the target time — unbiased even if GS photometry is off.
        pair_ok = (a.align_mode == "pair" and len(warps) == 2
                   and frame_idx is not None and None not in fidxs
                   and fidxs[0] != fidxs[1])
        if pair_ok:
            t = float(np.clip((frame_idx - fidxs[0]) / (fidxs[1] - fidxs[0]), 0.0, 1.0))
            joint = (cgeos[0] > 0.5) & (cgeos[1] > 0.5) & (edge == 0)
            if joint.sum() > 5000:
                ref = (1.0 - t) * lowpass(warps[0]) + t * lowpass(warps[1])
                for i in range(2):
                    _fit_apply(i, ref, joint)
            else:
                pair_ok = False
        if not pair_ok and a.align_mode != "none":
            for i in range(len(warps)):
                _fit_apply(i, gs_lp, (cgeos[i] > 0.6) & (edge == 0))

        # ---- phase C: photo consistency vs GS (gross-error guard, low-pass) ----
        confs = []
        for warp, cgeo in zip(warps, cgeos):
            diff = np.abs(lowpass(warp) - gs_lp).mean(axis=2)
            conf_photo = np.exp(-((diff / a.sigma_photo) ** 2)).astype(np.float32)
            confs.append(cgeo * conf_photo)

        # cross-neighbor agreement
        if len(warps) >= 2:
            pd = np.abs(lowpass(warps[0]) - lowpass(warps[1])).mean(axis=2)
            conf_pair = np.exp(-((pd / a.sigma_pair) ** 2)).astype(np.float32)
            stats["pair_err"] = float(pd.mean())
        else:
            conf_pair = np.full((H, W), a.one_side_conf, dtype=np.float32)

        total = []
        for c in confs:
            total.append(c * conf_pair * edge_w * alpha_ok)
        wsum = np.sum(total, axis=0)
        blend = np.zeros_like(gs_base)
        for c, wp in zip(total, warps):
            blend += c[..., None] * wp
        safe = wsum > 1e-4
        blend[safe] = blend[safe] / wsum[safe][..., None]
        blend[~safe] = gs_base[~safe]

        M = 1.0 - np.prod([1.0 - c for c in total], axis=0)
        M = np.minimum(M, a.m_cap).astype(np.float32)
        M = cv2.erode(M, np.ones((3, 3), np.uint8))
        if a.feather > 0:
            M = cv2.GaussianBlur(M, (0, 0), a.feather)
        fused = M[..., None] * blend + (1.0 - M[..., None]) * gs_base
        fused = np.clip(fused, 0.0, 1.0)

        stats["n_sources"] = len(warps)
        stats["coverage"] = float(M.mean())
        stats["align"] = ("pair" if pair_ok else
                          ("none" if a.align_mode == "none" else "gs"))
        return fused, gs_base, stats


# ----------------------------------- main ------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    work_scene = Path(args.work) / args.scene
    meta = ds.load_meta(work_scene)
    scene_dir = Path(meta["scene_dir"])
    ckpt_path = work_scene / "runs" / args.run / f"ckpt_{args.which}.pt"
    splats, app_emb, ckpt = load_checkpoint(ckpt_path, device)
    fuser = Fuser(args, meta, work_scene, splats, app_emb, ckpt, device)

    if args.mode == "val":
        folds = ds.load_folds(work_scene)
        fold = folds[str(args.fold)]
        pool = list(fold["train"])
        ckpt_train = set(ckpt.get("app_image_names") or [])
        if ckpt_train and not ckpt_train.issubset(set(pool)):
            print("WARNING: checkpoint saw images outside this fold's train set — "
                  "gating is NOT out-of-fold. Use the fold checkpoint.")
        dbg_dir = work_scene / "fusion_debug"
        rows_gs, rows_fu, per_image = [], [], []
        for i, nm in enumerate(fold["val"]):
            im = fuser.name2im[nm]
            K = fuser.cam_K(im["camera_id"])
            cm = meta["cameras"][str(im["camera_id"])]
            fused, gs_base, st = fuser.fuse_target(
                np.array(im["qvec"]), np.array(im["tvec"]), K,
                cm["width"], cm["height"], im["frame_idx"], pool)
            gt = np.asarray(Image.open(scene_dir / "train" / "images" / nm).convert("RGB"))
            m_gs = metrics.compare_uint8(to_u8(gs_base), gt, device=device, lpips_nets=("vgg",))
            m_fu = metrics.compare_uint8(to_u8(fused), gt, device=device, lpips_nets=("vgg",))
            rows_gs.append(m_gs); rows_fu.append(m_fu)
            d = m_fu["score_lb"] - m_gs["score_lb"]
            per_image.append({"name": nm, "delta": d, "coverage": st["coverage"],
                              "pair_err": st["pair_err"],
                              "gs": m_gs["score_lb"], "fused": m_fu["score_lb"]})
            print(f"  {nm}: cov={st['coverage']:.3f} gs={m_gs['score_lb']:.5f} "
                  f"fused={m_fu['score_lb']:.5f} d={d:+.5f}")
            if i < args.save_debug:
                dbg_dir.mkdir(exist_ok=True)
                Image.fromarray(to_u8(gs_base)).save(dbg_dir / f"{nm}.gs.png")
                Image.fromarray(to_u8(fused)).save(dbg_dir / f"{nm}.fused.png")
        agg_gs, agg_fu = metrics.aggregate(rows_gs), metrics.aggregate(rows_fu)
        deltas = np.array([r["delta"] for r in per_image])
        enable = (float(deltas.mean()) >= args.score_margin
                  and float(np.percentile(deltas, 10)) >= args.p10_floor)
        decision = {
            "enable": bool(enable),
            "mean_delta": float(deltas.mean()),
            "p10_delta": float(np.percentile(deltas, 10)),
            "gs": {k: agg_gs[k] for k in ("psnr", "ssim", "lpips_vgg", "score_lb")},
            "fused": {k: agg_fu[k] for k in ("psnr", "ssim", "lpips_vgg", "score_lb")},
            "run": args.run, "fold": args.fold,
            "params": {k: getattr(args, k) for k in (
                "supersample", "n_temporal", "depth_rel_tol", "sigma_photo",
                "sigma_pair", "one_side_conf", "edge_tol", "edge_dilate",
                "edge_factor", "flow_radius", "gain_clamp", "bias_clamp",
                "m_cap", "feather", "min_alpha")},
            "per_image": per_image,
        }
        with open(work_scene / "fusion_decision.json", "w") as f:
            json.dump(decision, f, indent=1)
        print(f"\n[{args.scene}] GS   psnr={agg_gs['psnr']:.2f} ssim={agg_gs['ssim']:.4f} "
              f"lpips={agg_gs['lpips_vgg']:.4f} score={agg_gs['score_lb']:.5f}")
        print(f"[{args.scene}] FUSE psnr={agg_fu['psnr']:.2f} ssim={agg_fu['ssim']:.4f} "
              f"lpips={agg_fu['lpips_vgg']:.4f} score={agg_fu['score_lb']:.5f}")
        print(f"[{args.scene}] mean_delta={deltas.mean():+.5f} p10={np.percentile(deltas, 10):+.5f} "
              f"-> enable={enable} (fusion_decision.json saved)")

    else:  # test
        dec_path = work_scene / "fusion_decision.json"
        if not args.force:
            if not dec_path.exists():
                raise SystemExit(f"[{args.scene}] no fusion_decision.json — run --mode val "
                                 f"first (or pass --force).")
            dec = json.load(open(dec_path))
            if not dec.get("enable", False):
                print(f"[{args.scene}] fusion disabled by fold-0 gate "
                      f"(mean_delta={dec.get('mean_delta'):+.5f}) — writing nothing; "
                      f"render_test will use plain GS renders.")
                return
        pool = [im["name"] for im in meta["images"]]
        out_dir = work_scene / "fused_test"
        out_dir.mkdir(exist_ok=True)
        manifest = []
        for tp in meta["test_poses"]:
            K = (tp["fx"], tp["fy"], tp["cx"], tp["cy"])
            fused, _, st = fuser.fuse_target(
                np.array(tp["qvec"]), np.array(tp["tvec"]), K,
                tp["width"], tp["height"], tp["frame_idx"], pool)
            np.save(out_dir / (tp["image_name"] + ".npy"), fused.astype(np.float16))
            manifest.append({"name": tp["image_name"], **st})
            print(f"  {tp['image_name']}: cov={st['coverage']:.3f} sources={st['n_sources']}")
        with open(out_dir / "_fused_manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"[{args.scene}] fused pinhole images -> {out_dir}\n"
              f"  next: python -m vai_nvs.render_test --work {args.work} --scene {args.scene} "
              f"--run <full_run> --fused-dir {out_dir} --out {work_scene / 'pred_test_final'}")


if __name__ == "__main__":
    main()
