"""Per-scene 3D Gaussian Splatting trainer (gsplat).

Static backbone per Chiến Lược 2.md §3-G:
  - AbsGS densification  : DefaultStrategy(absgrad=True)  [--strategy default]
  - 3DGS-MCMC candidate  : MCMCStrategy(cap_max=...)      [--strategy mcmc]
  - Mip 2D screen filter : rasterize_mode="antialiased"
  - Per-image affine appearance (gain/bias, L2-regularized to identity),
    interpolated temporally at eval/test time.
  - Camera poses are NEVER optimized (test poses live in the same fixed
    COLMAP world frame — optimizing poses would drift the frame).

Training happens in the UNDISTORTED pinhole domain (images_ud from prepare).
Validation is END-TO-END: render -> redistort -> uint8 -> metrics against the
original distorted JPGs, i.e. exactly what the leaderboard sees.

Usage:
  python -m vai_nvs.train_gs --work <work_root> --scene <name> --fold -1
  python -m vai_nvs.train_gs --work <work_root> --scene <name> --fold 0 --strategy mcmc
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vai_nvs.triple_hacks import compute_multiscale_loss
from . import colmap_io, dataset as ds, metrics
from . import cameras as camlib
from .gs_render import render_gs, apply_appearance, interp_appearance, save_checkpoint
from .render_pipeline import RedistortCache, render_view_to_distorted
from .triple_hacks import (
    COLMAPScaleCorrector,
    RobustSceneScalePreprocessor,
    FocalAspectAdaptiveDensifier,
    MCMCStabilizerAndCapManager,
    FrustumAndFogPruner,
    SHRegularizerAndTruncator,
    EarlyConvergenceMonitor,
    SoftWarpFusionBlender,
    GaussianModel
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-dir", default=None, help="override raw scene dir (for points3D + originals)")
    ap.add_argument("--fold", type=int, default=-1, help="-1 = train on all images; k = hold out fold k")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--strategy", choices=["default", "mcmc"], default="mcmc")
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--cap-max", type=int, default=4500000, help="MCMC gaussian cap (S10 spec limit: 2.5M)")
    ap.add_argument("--init-opacity", type=float, default=0.005)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--init-max-error", type=float, default=1.0, help="drop sfm points above this reproj error")
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--charb-eps", type=float, default=1e-3)
    ap.add_argument("--lpips-weight", type=float, default=0.1, help="late-phase LPIPS loss weight (0=off)")
    ap.add_argument("--lpips-start-frac", type=float, default=0.33)
    ap.add_argument("--lpips-end-frac", type=float, default=1.0)
    ap.add_argument("--lpips-crop", type=int, default=512)
    ap.add_argument("--no-antialiased", action="store_true")
    ap.add_argument("--no-absgrad", action="store_true")
    ap.add_argument("--random-bkgd", action="store_true",
                    help="random background color per step (suppresses floaters; "
                         "eval/test rendering keeps the black background)")
    ap.add_argument("--grow-grad2d", type=float, default=0.0001)
    ap.add_argument("--refine-stop-frac", type=float, default=0.70)
    ap.add_argument("--appearance", action="store_true",
                    help="enable per-image appearance model (disabled by default)")
    ap.add_argument("--app-lr", type=float, default=5e-3)
    ap.add_argument("--app-reg", type=float, default=1e-2)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--eval-supersample", type=float, default=2.0)
    ap.add_argument("--eval-max-images", type=int, default=0, help="0 = all val images")
    ap.add_argument("--sanity-views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ------------------------------ initialization ------------------------------

def load_points3d(scene_dir: Path, max_error=None):
    sparse = colmap_io.find_sparse_dir(scene_dir)
    pts_path = sparse / "points3D.bin"
    pts = (colmap_io.read_points3D_binary(pts_path) if pts_path.exists()
           else colmap_io.read_points3D_text(sparse / "points3D.txt"))
    xyz = np.stack([p.xyz for p in pts.values()])
    rgb = np.stack([p.rgb for p in pts.values()]).astype(np.float32) / 255.0
    err = np.array([p.error for p in pts.values()])
    median_err = np.median(err) if len(err) > 0 else 1.0
    if max_error is not None:
        keep = err <= max_error
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb, median_err


def init_splats(xyz, rgb, sh_degree, init_opacity, init_scale, device):
    from scipy.spatial import cKDTree
    n = xyz.shape[0]
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=4)
    dist = np.sqrt((d[:, 1:] ** 2).mean(axis=1))
    dist = np.clip(dist, 1e-7, None)
    k_sh = (sh_degree + 1) ** 2
    c0 = 0.28209479177387814
    rng = np.random.RandomState(0)
    quats = rng.randn(n, 4)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz.astype(np.float32))),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy((dist * init_scale).astype(np.float32)))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.from_numpy(quats.astype(np.float32))),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opacity))),
        "sh0": torch.nn.Parameter(torch.from_numpy(((rgb - 0.5) / c0).astype(np.float32))[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, k_sh - 1, 3)),
    }).to(device)
    return params


def build_optimizers(params, scene_scale, max_steps):
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 2.5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3,  # Hotfix #4: Restore equal LR for shN (2.5e-3)
    }
    optimizers = {
        k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k], "name": k}], eps=1e-15)
        for k in params.keys()
    }
    def lr_lambda(step):
        delay_steps = 500
        delay_mult = 0.01
        delay_rate = delay_mult + (1.0 - delay_mult) * np.sin(0.5 * np.pi * np.clip(step / delay_steps, 0, 1))
        t = np.clip(step / max_steps, 0, 1)
        return delay_rate * (0.01 ** t)
    
    sched = torch.optim.lr_scheduler.LambdaLR(optimizers["means"], lr_lambda=lr_lambda)
    return optimizers, sched


# ---------------------------------- main -------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    work_scene = Path(args.work) / args.scene
    meta = ds.load_meta(work_scene)
    scene_dir = Path(args.scene_dir) if args.scene_dir else Path(meta["scene_dir"])
    assert scene_dir.exists(), f"raw scene dir not found: {scene_dir} (use --scene-dir)"
    orig_dir = scene_dir / "train" / "images"

    all_names = [im["name"] for im in meta["images"]]
    if args.fold >= 0:
        folds = ds.load_folds(work_scene)
        fold = folds[str(args.fold)]
        train_names, val_names = list(fold["train"]), list(fold["val"])
    else:
        train_names, val_names = list(all_names), []
    name2im = {im["name"]: im for im in meta["images"]}

    run_name = args.run_name or (
        f"{args.strategy}_f{args.fold}_s{args.max_steps // 1000}k")
    out_dir = work_scene / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args))
    config["torch"] = torch.__version__
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=1)

    # --- cameras (undistorted pinhole = cameras.bin pinhole part) ---
    cams = {}
    for cid_s, cm in meta["cameras"].items():
        cid = int(cid_s)
        cam_obj = colmap_io.Camera(cid, cm["model"], cm["width"], cm["height"],
                                   np.array(cm["params"]))
        fx, fy, cx, cy = camlib.camera_pinhole(cam_obj)
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        mask = np.asarray(Image.open(work_scene / cm["mask"])) > 127
        cams[cid] = {
            "obj": cam_obj, "K": K, "W": cm["width"], "H": cm["height"],
            "render_K": (fx, fy, cx, cy),
            "mask": torch.from_numpy(mask).to(device)[..., None],
        }

    # --- images: uint8 CPU cache of undistorted train images ---
    print(f"[{args.scene}] loading {len(train_names)} undistorted train images ...")
    images_u8, viewmats, cam_ids, frame_idxs = {}, {}, {}, []
    for nm in train_names:
        arr = np.asarray(Image.open(work_scene / "images_ud" / (nm + ".png")).convert("RGB"))
        images_u8[nm] = torch.from_numpy(arr.copy())  # H,W,3 uint8 CPU
        im = name2im[nm]
        viewmats[nm] = torch.from_numpy(
            camlib.w2c_matrix(np.array(im["qvec"]), np.array(im["tvec"]))).float()
        cam_ids[nm] = im["camera_id"]
        frame_idxs.append(im["frame_idx"])

    # --- splats ---
    xyz, rgb, median_err = load_points3d(scene_dir, args.init_max_error)
    scene_scale = float(meta["scene_scale"])
    
    # Init 13 Hacks Modules
    s12_corrector = COLMAPScaleCorrector()
    s1_s2 = RobustSceneScalePreprocessor()
    s3_s4 = FocalAspectAdaptiveDensifier(tau_base=0.00008)
    s9_s10 = MCMCStabilizerAndCapManager(N_max=args.cap_max)
    s5_s7 = FrustumAndFogPruner()
    s6_s8 = SHRegularizerAndTruncator()
    s11 = EarlyConvergenceMonitor()
    # S12 Auto-Correction
    if len(meta["cameras"]) > 0:
        first_cid = int(list(meta["cameras"].keys())[0])
        sample_K = cams[first_cid]["K"]
        # S12 Audit Fix: Do NOT scale 3D world coordinates independently of camera translation (tvec)
        _, corr_K, scale_factor = s12_corrector.correct_scale(
            torch.zeros((len(xyz), 2)), sample_K, median_err
        )
        if scale_factor != 1.0:
            for cid in cams:
                _, cams[cid]["K"], _ = s12_corrector.correct_scale(
                    torch.zeros((len(xyz), 2)), cams[cid]["K"], median_err
                )
                fx = cams[cid]["K"][0, 0].item()
                fy = cams[cid]["K"][1, 1].item()
                cx = cams[cid]["K"][0, 2].item()
                cy = cams[cid]["K"][1, 2].item()
                cams[cid]["render_K"] = (fx, fy, cx, cy)
        print(f"[S12 Audited] SfM median reproj error: {median_err:.2f}px. Scale factor: {scale_factor:.1f}.")

    # Init S13 Blender
    s13_blender = SoftWarpFusionBlender(tau_blend=0.1)

    # S1 & S2 Q-Prune & Robust LR
    cams_centers = []
    for nm, im in name2im.items():
        w2c = camlib.w2c_matrix(np.array(im["qvec"]), np.array(im["tvec"]))
        cams_centers.append(np.linalg.inv(w2c)[:3, 3])
    cams_centers = torch.tensor(np.array(cams_centers), dtype=torch.float32)
    
    pts, r_robust, r_boundary, spatial_lr_scale, c_mean, valid_mask = s1_s2.process_initial_points(torch.from_numpy(xyz), cams_centers)
    xyz_pruned = pts.numpy()
    rgb_pruned = rgb[valid_mask.numpy()]
    c_mean = c_mean.to(device)

    # Setup GaussianModel
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.create_from_pcd(torch.from_numpy(xyz_pruned).float(), torch.from_numpy(rgb_pruned).float(), spatial_lr_scale, device)

    # Dynamic dict mapping to match render_gs requirements
    class SplatDict(dict):
        def __init__(self, gaussians):
            self.g = gaussians
        def __getitem__(self, key):
            if key == "means": return self.g._xyz
            if key == "scales": return self.g._scaling
            if key == "quats": return self.g._rotation
            if key == "opacities": return self.g._opacity
            if key == "sh0": return self.g._features_dc
            if key == "shN": return self.g._features_rest
            return super().__getitem__(key)
        def keys(self): return ["means", "scales", "quats", "opacities", "sh0", "shN"]
        def items(self): return [(k, self[k]) for k in self.keys()]

    splats = SplatDict(gaussians)
    
    gamma = 0.01 ** (1.0 / args.max_steps)

    print(f"[{args.scene}] init {len(xyz_pruned)} gaussians, scene_scale={scene_scale:.3f}, fold={args.fold}, run={run_name}")

    absgrad = not args.no_absgrad
    antialiased = not args.no_antialiased

    # --- appearance embeddings ---
    use_app = args.appearance
    app_emb, app_opt = None, None
    if use_app:
        app_emb = torch.nn.Parameter(torch.zeros(len(train_names), 6, device=device))
        app_opt = torch.optim.Adam([app_emb], lr=args.app_lr)
    name2appidx = {nm: i for i, nm in enumerate(train_names)}

    lpips_train = None
    if args.lpips_weight > 0:
        lpips_train = metrics.get_lpips("vgg", device)

    redist_cache = RedistortCache()

    # ------------------------------ evaluation ------------------------------

    @torch.no_grad()
    def evaluate(names, tag, step):
        rows = []
        save_dir = out_dir / "eval_images" / f"{tag}_step{step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        subset = names[: args.eval_max_images] if args.eval_max_images else names
        for nm in subset:
            im = name2im[nm]
            cam = cams[im["camera_id"]]
            emb6 = None
            if use_app:
                emb6 = interp_appearance(im["frame_idx"], frame_idxs, app_emb.data)
            pred = render_view_to_distorted(
                {k: v.data for k, v in splats.items()},
                np.array(im["qvec"]), np.array(im["tvec"]),
                cam["render_K"], cam["obj"], cam["W"], cam["H"],
                args.eval_supersample, args.sh_degree, antialiased,
                app_emb6=emb6, cache=redist_cache)
            if len(im.get("rgb", [])) > 0:
                Image.fromarray(im["rgb"]).save(save_dir / f"gt_{nm}")
            if "warped_frame" in im:
                warped_tensor = torch.from_numpy(im["warped_frame"]).permute(2, 0, 1).float().to(device) / 255.0
                pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).float().to(device) / 255.0
                fused_tensor, _ = s13_blender.blend(pred_tensor, warped_tensor)
                pred = (fused_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(pred).save(save_dir / nm)
            gt = np.asarray(Image.open(orig_dir / nm).convert("RGB"))
            rows.append(metrics.compare_uint8(pred, gt, device=device,
                                              lpips_nets=("alex", "vgg")))
        agg = metrics.aggregate(rows)
        agg.update({"tag": tag, "step": step, "n": len(subset)})
        with open(out_dir / "eval.jsonl", "a") as f:
            f.write(json.dumps(agg) + "\n")
        print(f"  [eval {tag} @ {step}] n={len(subset)} psnr={agg['psnr']:.3f} "
              f"ssim={agg['ssim']:.4f} lpips_vgg={agg['lpips_vgg']:.4f} "
              f"score_lb={agg['score_lb']:.5f} (alex@40={agg['score@40']:.5f})")
        return agg

    # ------------------------------ train loop ------------------------------

    best_score = -1.0
    rng = np.random.RandomState(args.seed)
    log_f = open(out_dir / "log.csv", "a")
    log_f.write("step,loss,charb,dssim,n_gauss,mem_gb,sec\n")
    t0 = time.time()
    order = rng.permutation(len(train_names))
    ptr = 0

    for step in range(args.max_steps):
        if ptr >= len(order):
            order = rng.permutation(len(train_names))
            ptr = 0
        nm = train_names[int(order[ptr])]
        ptr += 1

        cam = cams[cam_ids[nm]]
        gt = images_u8[nm].to(device).float() / 255.0  # H,W,3
        sh_used = min(step // 1000, args.sh_degree)

        bkgd = torch.rand(1, 3, device=device) if args.random_bkgd else None
        render, _, info = render_gs(
            splats, viewmats[nm], cam["K"], cam["W"], cam["H"], sh_used,
            render_mode="RGB", antialiased=antialiased, absgrad=absgrad,
            background=bkgd)
        
        # S3/S4: Retain grad for 2D screen means
        if isinstance(info, dict) and "means2d" in info and info["means2d"] is not None:
            info["means2d"].retain_grad()

        # 1. S3 & S4 thresholds
        tau_u, tau_v = s3_s4.compute_adaptive_thresholds(cam["K"][0,0].item(), cam["K"][1,1].item(), cam["W"], cam["H"])

        rgb = render
        if use_app:
            rgb = apply_appearance(rgb, app_emb[name2appidx[nm]])
        rgb = torch.clamp(rgb, 0.0, 1.0)
        pred = torch.where(cam["mask"], rgb, gt)  # invalid border pixels: no grad

        diff = pred - gt
        x = pred.permute(2, 0, 1)[None]
        y = gt.permute(2, 0, 1)[None]
        
        # S13: Multi-scale SSIM and LPIPS-VGG
        loss, l_1, l_dssim = compute_multiscale_loss(x, y, step, args.max_steps, lpips_train)

        # S6 SH Smoothness Reg
        sh_smooth_loss = s6_s8.compute_sh_smoothness_loss(gaussians._features_rest, step)
        loss = loss + sh_smooth_loss

        if use_app:
            loss = loss + args.app_reg * app_emb[name2appidx[nm]].square().mean()

        gaussians.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # S3 & S4 Accumulate gradients
        means2d_grad = None
        if isinstance(info, dict):
            if "means2d_absgrad" in info and info["means2d_absgrad"] is not None:
                means2d_grad = info["means2d_absgrad"].clone()
            elif "absgrad" in info and info["absgrad"] is not None:
                means2d_grad = info["absgrad"].clone()
            elif "means2d" in info and info["means2d"] is not None and hasattr(info["means2d"], "absgrad") and info["means2d"].absgrad is not None:
                means2d_grad = info["means2d"].absgrad.clone()
            elif "means2d" in info and info["means2d"] is not None and info["means2d"].grad is not None:
                means2d_grad = info["means2d"].grad.clone()
                
            if means2d_grad is not None:
                means2d_grad[..., 0] /= (cam["W"] / 2.0)
                means2d_grad[..., 1] /= (cam["H"] / 2.0)
        gaussians.accumulate_gradients(means2d_grad)

        # Hacks Post-Backward Logic
        refine_stop_step = int(args.refine_stop_frac * args.max_steps)
        # S3 & S4 Densification Execution (gated by refine_stop_step)
        if step <= refine_stop_step:
            is_densify = (500 <= step <= refine_stop_step) and (step % 100 == 0)
            if is_densify:
                candidate_mask = s3_s4.evaluate_densification_candidates(
                    gaussians.grad_accum_u, gaussians.grad_accum_v, gaussians.denom_accum, tau_u, tau_v)
                candidate_indices = torch.where(candidate_mask)[0]
                if len(candidate_indices) > 0:
                    capped_indices = s9_s10.enforce_cap_ceiling(len(gaussians._xyz), candidate_indices)
                    if len(capped_indices) > 0:
                        gaussians.densify_and_split_clone(capped_indices)
                gaussians.reset_gradient_accumulators()

        # Opacity reset
        if step > 500 and step <= refine_stop_step and step % 4000 == 0:
            gaussians.reset_opacity()

        # S5 & S7 Periodic Pruning (UNBLOCKED: Runs across full training timeline up to max_steps)
        if step % 3000 == 0:
            keep_mask, prune_stats = s5_s7.prune(gaussians._xyz, gaussians._opacity, c_mean, r_boundary, step)
            if prune_stats["total_pruned"] > 0:
                gaussians.prune_points(keep_mask)
        
        # Screen-space Max Radii Pruning
        if step > 500 and step % 100 == 0:
            # Note: actual pruning using max_screen_size=20
            # Since max_screen_size pruning needs screen radii, we should fetch it from info
            if "radii" in info:
                radii = info["radii"]
                radii_mask = radii <= 20
                if (~radii_mask).sum() > 0:
                    gaussians.prune_points(radii_mask)

        # S9 MCMC Relocation
        reloc_rate = s9_s10.get_dampened_mcmc_relocation_rate(step)
        if reloc_rate > 0 and step % 500 == 0:
            n_reloc = gaussians.relocate_points(reloc_rate)
            if n_reloc > 0:
                print(f"[S9 MCMC Relocation] Step {step}: Relocated {n_reloc:,} splats (rate: {reloc_rate:.6f}).")

        # S10 Hard Cap Ceiling
        if len(gaussians._xyz) > s9_s10.N_max:
            excess = len(gaussians._xyz) - s9_s10.N_max
            opacities = torch.sigmoid(gaussians._opacity).squeeze(-1)
            _, topk_indices = torch.topk(opacities, s9_s10.N_max, largest=True)
            cap_keep_mask = torch.zeros(len(gaussians._xyz), dtype=torch.bool, device=device)
            cap_keep_mask[topk_indices] = True
            gaussians.prune_points(cap_keep_mask)

        # Optimization step
        gaussians.optimizer.step()
        
        # S13: Covariance Ratio Guard
        with torch.no_grad():
            scaling = gaussians._scaling
            max_scale = scaling.max(dim=1, keepdim=True).values
            scaling.copy_(torch.max(scaling, max_scale - math.log(20.0)))
        
        # S8 Truncation Sweep
        current_step = step + 1
        if current_step in [18000, 36000]:
            new_sh_rest, n_trunc = s6_s8.apply_energy_adaptive_truncation(gaussians._features_dc, gaussians._features_rest, current_step)
            if n_trunc > 0:
                with torch.no_grad():
                    gaussians._features_rest.copy_(new_sh_rest)
                    p_rest = gaussians.optimizer.param_groups[2]['params'][0]
                    p_state = gaussians.optimizer.state[p_rest]
                    zero_mask = (new_sh_rest == 0.0)
                    if 'exp_avg' in p_state:
                        p_state['exp_avg'][zero_mask] = 0.0
                    if 'exp_avg_sq' in p_state:
                        p_state['exp_avg_sq'][zero_mask] = 0.0

        if app_opt is not None:
            app_opt.step()
            app_opt.zero_grad(set_to_none=True)
        for param_group in gaussians.optimizer.param_groups:
            param_group['lr'] *= gamma

        if step % 100 == 0 or step == args.max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            log_f.write(f"{step},{loss.item():.5f},{l_1.item():.5f},{l_dssim.item():.5f},"
                        f"{splats['means'].shape[0]},{mem:.2f},{time.time() - t0:.1f}\n")
            log_f.flush()
        if step % 1000 == 0:
            print(f"  step {step:6d} loss={loss.item():.4f} n_gauss={splats['means'].shape[0]:,} "
                  f"({time.time() - t0:.0f}s)")

        # S11 Audit Fix: Support evaluation and early stopping check even when val_names is empty (--fold -1)
        do_eval = (step + 1) % args.eval_every == 0
        if do_eval or step == args.max_steps - 1:
            eval_targets = val_names if val_names else train_names[:min(5, len(train_names))]
            agg = evaluate(eval_targets, f"fold{args.fold}", step + 1)
            if agg["score_lb"] > best_score:
                best_score = agg["score_lb"]
                save_checkpoint(out_dir / "ckpt_best.pt", step + 1, splats, app_emb,
                                train_names, config, extra={"val_metrics": agg})
            # S11 Early Stop check
            if s11.check_early_stop(step + 1, agg["score_lb"]):
                print(f"[S11 Early Stop] Triggered at step {step + 1}. Score gain plateaued (< 0.002).")
                break

    log_f.close()
    save_checkpoint(out_dir / "ckpt_last.pt", args.max_steps, splats, app_emb,
                    train_names, config)

    # --- end-to-end sanity on TRAIN views (must be high; low => convention bug)
    if args.sanity_views > 0:
        sanity = train_names[:: max(1, len(train_names) // args.sanity_views)][: args.sanity_views]
        agg = evaluate(sanity, "sanity_train", args.max_steps)
        if agg["psnr"] < 20:
            print("  !!! WARNING: train-view end-to-end PSNR < 20 dB — investigate "
                  "camera conventions before trusting any output.")

    print(f"[{args.scene}] done. run dir: {out_dir}"
          + (f" | best val score_lb={best_score:.5f}" if val_names else ""))


if __name__ == "__main__":
    main()
