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
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import colmap_io, dataset as ds, metrics
from . import cameras as camlib
from .gs_render import render_gs, apply_appearance, interp_appearance, save_checkpoint
from .render_pipeline import RedistortCache, render_view_to_distorted


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scene-dir", default=None, help="override raw scene dir (for points3D + originals)")
    ap.add_argument("--fold", type=int, default=-1, help="-1 = train on all images; k = hold out fold k")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--strategy", choices=["default", "mcmc"], default="mcmc")
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--cap-max", type=int, default=3000000, help="MCMC gaussian cap")
    ap.add_argument("--init-opacity", type=float, default=0.1)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--init-max-error", type=float, default=1.0, help="drop sfm points above this reproj error")
    ap.add_argument("--ssim-lambda", type=float, default=0.3)
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
    ap.add_argument("--grow-grad2d", type=float, default=0.0006)
    ap.add_argument("--refine-stop-frac", type=float, default=0.67)
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
    if max_error is not None:
        keep = err <= max_error
        xyz, rgb = xyz[keep], rgb[keep]
    return xyz, rgb


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
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k], "name": k}], eps=1e-15)
        for k in params.keys()
    }
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max_steps))
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
        images_u8[nm] = torch.from_numpy(arr)  # H,W,3 uint8 CPU
        im = name2im[nm]
        viewmats[nm] = torch.from_numpy(
            camlib.w2c_matrix(np.array(im["qvec"]), np.array(im["tvec"]))).float()
        cam_ids[nm] = im["camera_id"]
        frame_idxs.append(im["frame_idx"])

    # --- splats ---
    xyz, rgb = load_points3d(scene_dir, args.init_max_error)
    scene_scale = float(meta["scene_scale"])
    splats = init_splats(xyz, rgb, args.sh_degree, args.init_opacity, args.init_scale, device)
    optimizers, mean_sched = build_optimizers(splats, scene_scale, args.max_steps)
    print(f"[{args.scene}] init {xyz.shape[0]} gaussians, scene_scale={scene_scale:.3f}, "
          f"strategy={args.strategy}, fold={args.fold}, run={run_name}")

    from gsplat import DefaultStrategy, MCMCStrategy
    absgrad = not args.no_absgrad
    antialiased = not args.no_antialiased
    if args.strategy == "default":
        grow_grad2d = args.grow_grad2d if args.grow_grad2d is not None else (0.0008 if absgrad else 0.0002)
        strategy = DefaultStrategy(
            prune_opa=0.005, grow_grad2d=grow_grad2d, grow_scale3d=0.01,
            refine_start_iter=500, refine_stop_iter=int(args.refine_stop_frac * args.max_steps),
            reset_every=3000, refine_every=100, absgrad=absgrad, verbose=False)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        strategy = MCMCStrategy(cap_max=args.cap_max, refine_start_iter=500,
                                refine_stop_iter=int(0.9 * args.max_steps),
                                refine_every=100, min_opacity=0.005, verbose=False)
        strategy_state = strategy.initialize_state()
        absgrad = False  # MCMC does not use screen-space grads
    strategy.check_sanity(splats, optimizers)

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
            # Save rendered image to disk instead of accumulating in RAM
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
        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)

        rgb = render
        if use_app:
            rgb = apply_appearance(rgb, app_emb[name2appidx[nm]])
        rgb = torch.clamp(rgb, 0.0, 1.0)
        pred = torch.where(cam["mask"], rgb, gt)  # invalid border pixels: no grad

        diff = pred - gt
        l_charb = torch.sqrt(diff * diff + args.charb_eps ** 2).mean()
        x = pred.permute(2, 0, 1)[None]
        y = gt.permute(2, 0, 1)[None]
        l_dssim = 1.0 - metrics.ssim_torch(x, y)
        loss = (1.0 - args.ssim_lambda) * l_charb + args.ssim_lambda * l_dssim

        if args.strategy == "mcmc":
            loss = loss + 0.01 * torch.sigmoid(splats["opacities"]).mean() \
                        + 0.01 * torch.exp(splats["scales"]).mean()
        if use_app:
            loss = loss + args.app_reg * app_emb[name2appidx[nm]].square().mean()
        if lpips_train is not None and int(args.lpips_start_frac * args.max_steps) <= step < int(args.lpips_end_frac * args.max_steps):
            c = args.lpips_crop
            hh = rng.randint(0, max(1, cam["H"] - c))
            ww = rng.randint(0, max(1, cam["W"] - c))
            xc = x[..., hh:hh + c, ww:ww + c] * 2 - 1
            yc = y[..., hh:hh + c, ww:ww + c] * 2 - 1
            loss = loss + args.lpips_weight * lpips_train(xc, yc).mean()

        loss.backward()

        if step <= int(args.refine_stop_frac * args.max_steps):
            if len(splats["means"]) > 11500000:
                strategy.grow_grad2d = 999.0
                strategy.grow_scale3d = 999.0
            else:
                strategy.grow_grad2d = args.grow_grad2d if args.grow_grad2d is not None else (0.0008 if not getattr(args, "no_absgrad", False) else 0.0002)
                strategy.grow_scale3d = 0.01
            
            if args.strategy == "default":
                strategy.step_post_backward(splats, optimizers, strategy_state, step, info,
                                            packed=False)
            else:
                strategy.step_post_backward(splats, optimizers, strategy_state, step, info,
                                            lr=mean_sched.get_last_lr()[0])

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if app_opt is not None:
            app_opt.step()
            app_opt.zero_grad(set_to_none=True)
        mean_sched.step()

        if step % 100 == 0 or step == args.max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            log_f.write(f"{step},{loss.item():.5f},{l_charb.item():.5f},{l_dssim.item():.5f},"
                        f"{splats['means'].shape[0]},{mem:.2f},{time.time() - t0:.1f}\n")
            log_f.flush()
        if step % 1000 == 0:
            print(f"  step {step:6d} loss={loss.item():.4f} n_gauss={splats['means'].shape[0]:,} "
                  f"({time.time() - t0:.0f}s)")

        do_eval = val_names and (step + 1) % args.eval_every == 0
        if do_eval or step == args.max_steps - 1:
            if val_names:
                agg = evaluate(val_names, f"fold{args.fold}", step + 1)
                if agg["score_lb"] > best_score:
                    best_score = agg["score_lb"]
                    save_checkpoint(out_dir / "ckpt_best.pt", step + 1, splats, app_emb,
                                    train_names, config, extra={"val_metrics": agg})

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
