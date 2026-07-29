"""Prepare a scene workspace: undistort train images, write metadata,
pseudo-validation folds, and the temporal/pose neighbor graph.

Outputs under <work>/<scene>/:
  images_ud/<orig_name>.png   undistorted (pinhole) train images
  mask_cam<id>.png            valid-pixel mask of the undistortion (255=valid)
  meta.json                   cameras, per-image poses, test poses, scene scale
  folds.json                  pseudo-validation folds (missing-block style)
  graph.json                  per-test-target temporal + pose-based neighbors

Usage:
  python -m vai_nvs.prepare --data <set_dir_or_scene> --work <work_root>
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import cameras as camlib
from . import dataset as ds

cv2.setNumThreads(0)  # threads managed by the pool below


def undistort_scene_images(scene: ds.Scene, out_dir: Path, workers=4) -> dict:
    """Undistort every usable train image to PNG. Returns {camera_id: mask_path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = {}
    mask_paths = {}
    for cid, cam in scene.cameras.items():
        mx, my, valid = camlib.build_undistort_maps(cam)
        maps[cid] = (mx, my)
        mask_path = out_dir.parent / f"mask_cam{cid}.png"
        Image.fromarray((valid * 255).astype(np.uint8)).save(mask_path)
        mask_paths[cid] = mask_path.name

    def work(name):
        dst = out_dir / (name + ".png")
        if dst.exists():
            return
        rec = scene.images[name]
        mx, my = maps[rec.camera_id]
        with Image.open(scene.train_images_dir / name) as im:
            arr = np.asarray(im.convert("RGB"))
        # Hotfix #5: Disable Bicubic Pre-Blurring by skipping cv2.remap
        und = arr
        Image.fromarray(und).save(dst, compress_level=3)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, scene.usable_train))
    return mask_paths


def compute_scene_scale(scene: ds.Scene) -> float:
    centers = np.stack([camlib.camera_center(scene.images[n].qvec, scene.images[n].tvec)
                        for n in scene.usable_train])
    center = centers.mean(axis=0)
    return float(np.max(np.linalg.norm(centers - center, axis=1)) * 1.1)


def build_graph(scene: ds.Scene, k_pose=8) -> dict:
    """Per test target: temporal prev/next train frames + pose-nearest train
    cameras (angle-filtered). Uses only filenames and poses — no test tracks."""
    train = scene.usable_train
    t_idx = {n: ds.parse_dji_name(n)[1] for n in train}
    seq = [n for n in train if t_idx[n] is not None]
    seq.sort(key=lambda n: t_idx[n])
    centers = {n: camlib.camera_center(scene.images[n].qvec, scene.images[n].tvec) for n in train}
    dirs = {n: camlib.view_dir_world(scene.images[n].qvec) for n in train}

    graph = {}
    for tp in scene.test_poses:
        entry = {"frame_idx": tp.frame_idx, "prev": [], "next": [], "pose_nn": []}
        if tp.frame_idx is not None and seq:
            before = [n for n in seq if t_idx[n] < tp.frame_idx]
            after = [n for n in seq if t_idx[n] > tp.frame_idx]
            entry["prev"] = before[-2:][::-1]  # nearest first
            entry["next"] = after[:2]
        c_t = camlib.camera_center(tp.qvec, tp.tvec)
        d_t = camlib.view_dir_world(tp.qvec)
        scored = []
        for n in train:
            cosang = float(np.dot(dirs[n], d_t))
            if cosang < 0.5:  # >60 deg apart: useless source
                continue
            scored.append((float(np.linalg.norm(centers[n] - c_t)), cosang, n))
        scored.sort(key=lambda t: t[0])
        entry["pose_nn"] = [{"name": n, "dist": round(d, 4), "cos": round(c, 4)}
                            for d, c, n in scored[:k_pose]]
        graph[tp.image_name] = entry
    return graph


def prepare_scene(scene_dir: Path, work_root: Path, n_folds: int, seed: int, workers: int):
    print(f"\n=== Prepare: {scene_dir.name} ===")
    scene = ds.load_scene(scene_dir, load_points3D=True)
    for cid, cam in scene.cameras.items():
        if cam.model not in camlib.SUPPORTED_MODELS:
            raise SystemExit(f"{scene.name}: unsupported camera model {cam.model}")

    work = work_root / scene.name
    work.mkdir(parents=True, exist_ok=True)

    print(f"  undistorting {len(scene.usable_train)} images -> images_ud/ ...")
    mask_paths = undistort_scene_images(scene, work / "images_ud", workers=workers)

    scene_scale = compute_scene_scale(scene)
    meta = {
        "scene": scene.name,
        "scene_dir": str(scene_dir.resolve()),
        "scene_scale": scene_scale,
        "cameras": {
            str(cid): {
                "model": cam.model, "width": cam.width, "height": cam.height,
                "params": cam.params.tolist(), "mask": mask_paths[cid],
            } for cid, cam in scene.cameras.items()
        },
        "images": [
            {
                "name": n,
                "qvec": scene.images[n].qvec.tolist(),
                "tvec": scene.images[n].tvec.tolist(),
                "camera_id": scene.images[n].camera_id,
                "frame_idx": ds.parse_dji_name(n)[1],
                "timestamp": ds.parse_dji_name(n)[0],
            } for n in scene.usable_train
        ],
        "test_poses": [
            {
                "image_name": tp.image_name,
                "qvec": tp.qvec.tolist(), "tvec": tp.tvec.tolist(),
                "fx": tp.fx, "fy": tp.fy, "cx": tp.cx, "cy": tp.cy,
                "width": tp.width, "height": tp.height,
                "frame_idx": tp.frame_idx, "timestamp": tp.timestamp,
            } for tp in scene.test_poses
        ],
    }
    with open(work / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    folds = ds.make_folds(scene, n_folds=n_folds, seed=seed)
    with open(work / "folds.json", "w") as f:
        json.dump(folds, f, indent=1)
    for k, v in folds.items():
        print(f"  fold {k}: train={len(v['train'])} val={len(v['val'])}")

    graph = build_graph(scene)
    with open(work / "graph.json", "w") as f:
        json.dump(graph, f, indent=1)
    n_with_temporal = sum(1 for g in graph.values() if g["prev"] or g["next"])
    print(f"  graph: {n_with_temporal}/{len(graph)} targets have temporal neighbors; "
          f"scene_scale={scene_scale:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scenes", default=None, help="comma-separated scene names filter")
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    scene_dirs = ds.discover_scenes(args.data)
    if args.scenes:
        keep = {s.strip() for s in args.scenes.split(",")}
        scene_dirs = [sd for sd in scene_dirs if sd.name in keep]
    if not scene_dirs:
        raise SystemExit("No scenes to prepare")
    for sd in scene_dirs:
        prepare_scene(sd, Path(args.work), args.n_folds, args.seed, args.workers)
    print("\nPrepare done.")


if __name__ == "__main__":
    main()
