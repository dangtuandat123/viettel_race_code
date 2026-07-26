"""Render all test poses of a scene from a trained checkpoint.

Geometry: CSV (fx,fy,cx,cy,width,height) defines the pinhole projection;
cameras.bin supplies the lens distortion that the ground-truth photos carry.
Output files use the EXACT `image_name` string from test_poses.csv
(JPEG saved at --jpeg-quality with 4:4:4 subsampling; PNG if .png).

Also runs an end-to-end sanity check on train views first: if the pipeline
had a convention bug (w2c, k1, principal point...), train-view PSNR collapses
and we abort before writing garbage.

Usage:
  python -m vai_nvs.render_test --work <root> --scene <name> --run default_f-1_s30k
  python -m vai_nvs.render_test ... --fused-dir <dir>   # use warp-fused pinhole inputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

from . import colmap_io, dataset as ds, metrics
from . import cameras as camlib
from .gs_render import load_checkpoint, interp_appearance
from .render_pipeline import RedistortCache, render_view_to_distorted


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", default=None, help="run name under <work>/<scene>/runs/")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --run)")
    ap.add_argument("--which", choices=["best", "last"], default="last")
    ap.add_argument("--out", default=None, help="default <work>/<scene>/pred_test")
    ap.add_argument("--supersample", type=float, default=2.0)
    ap.add_argument("--appearance", choices=["interp", "identity"], default="interp")
    ap.add_argument("--jpeg-quality", type=int, default=100)
    ap.add_argument("--sanity", type=int, default=4, help="train views for sanity check (0=skip)")
    ap.add_argument("--sanity-min-psnr", type=float, default=18.0)
    ap.add_argument("--fused-dir", default=None,
                    help="dir of <image_name>.npy fused pinhole images (from warp_fuse)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def pick_dist_camera(meta, width, height, fx):
    """Choose the cameras.bin camera describing the distortion of this target."""
    best, best_d = None, np.inf
    for cm in meta["cameras"].values():
        if cm["width"] == width and cm["height"] == height:
            cam = colmap_io.Camera(0, cm["model"], cm["width"], cm["height"],
                                   np.array(cm["params"]))
            d = abs(camlib.camera_pinhole(cam)[0] - fx)
            if d < best_d:
                best, best_d = cam, d
    if best is None:
        raise RuntimeError(f"No cameras.bin camera matches {width}x{height}")
    return best


def save_exact(img_u8: np.ndarray, path: Path, jpeg_quality: int):
    ext = path.suffix.lower()
    im = Image.fromarray(img_u8)
    if ext in (".jpg", ".jpeg"):
        im.save(path, format="JPEG", quality=jpeg_quality, subsampling=0)
    elif ext == ".png":
        im.save(path, format="PNG")
    else:
        raise ValueError(f"Unexpected extension in image_name: {path.name}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    work_scene = Path(args.work) / args.scene
    meta = ds.load_meta(work_scene)
    scene_dir = Path(meta["scene_dir"])

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        assert args.run, "provide --run or --ckpt"
        ckpt_path = work_scene / "runs" / args.run / f"ckpt_{args.which}.pt"
    splats, app_emb, ckpt = load_checkpoint(ckpt_path, device)
    sh_degree = ckpt["config"]["sh_degree"]
    antialiased = not ckpt["config"].get("no_antialiased", False)
    app_names = ckpt.get("app_image_names") or []
    name2im = {im["name"]: im for im in meta["images"]}
    app_frame_idxs = [name2im[n]["frame_idx"] if n in name2im else None for n in app_names]
    use_app = (args.appearance == "interp") and (app_emb is not None)

    out_dir = Path(args.out) if args.out else work_scene / "pred_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = RedistortCache()

    def emb_for(frame_idx):
        if not use_app:
            return None
        return interp_appearance(frame_idx, app_frame_idxs, app_emb)

    # ------------------------- sanity on train views -------------------------
    if args.sanity > 0:
        train_names = app_names or [im["name"] for im in meta["images"]]
        picks = train_names[:: max(1, len(train_names) // args.sanity)][: args.sanity]
        psnrs = []
        for nm in picks:
            im = name2im[nm]
            cm = meta["cameras"][str(im["camera_id"])]
            cam = colmap_io.Camera(0, cm["model"], cm["width"], cm["height"], np.array(cm["params"]))
            pred = render_view_to_distorted(
                splats, np.array(im["qvec"]), np.array(im["tvec"]),
                camlib.camera_pinhole(cam), cam, cm["width"], cm["height"],
                args.supersample, sh_degree, antialiased,
                app_emb6=emb_for(im["frame_idx"]), cache=cache)
            gt = np.asarray(Image.open(scene_dir / "train" / "images" / nm).convert("RGB"))
            m = metrics.compare_uint8(pred, gt, device=device, lpips_nets=())
            psnrs.append(m["psnr"])
        med = float(np.median(psnrs))
        print(f"[{args.scene}] sanity train-view PSNR: {['%.2f' % p for p in psnrs]} (median {med:.2f} dB)")
        if med < args.sanity_min_psnr:
            raise SystemExit(f"SANITY FAILED: median train PSNR {med:.2f} < "
                             f"{args.sanity_min_psnr} dB — convention bug? Aborting.")

    # ----------------------------- test renders ------------------------------
    manifest = {"ckpt": str(ckpt_path), "supersample": args.supersample,
                "appearance": args.appearance, "jpeg_quality": args.jpeg_quality,
                "fused_dir": args.fused_dir, "images": []}
    for tp in meta["test_poses"]:
        w, h = tp["width"], tp["height"]
        render_K = (tp["fx"], tp["fy"], tp["cx"], tp["cy"])
        dist_cam = pick_dist_camera(meta, w, h, tp["fx"])
        fused_path = Path(args.fused_dir) / (tp["image_name"] + ".npy") if args.fused_dir else None
        if fused_path and fused_path.exists():
            pin = np.load(fused_path).astype(np.float32)  # H,W,3 in [0,1], pinhole native
            assert pin.shape[:2] == (h, w), f"fused size mismatch for {tp['image_name']}"
            mx, my = cache.get(render_K, dist_cam, w, h, 1.0)
            out = cv2.remap(pin, mx, my, interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)
            img_u8 = (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            src = "fused"
        else:
            img_u8 = render_view_to_distorted(
                splats, np.array(tp["qvec"]), np.array(tp["tvec"]),
                render_K, dist_cam, w, h, args.supersample, sh_degree, antialiased,
                app_emb6=emb_for(tp["frame_idx"]), cache=cache)
            src = "gs"
        save_exact(img_u8, out_dir / tp["image_name"], args.jpeg_quality)
        manifest["images"].append({"name": tp["image_name"], "source": src})
    with open(out_dir / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    n_fused = sum(1 for x in manifest["images"] if x["source"] == "fused")
    print(f"[{args.scene}] rendered {len(manifest['images'])} test images -> {out_dir} "
          f"({n_fused} fused)")


if __name__ == "__main__":
    main()
