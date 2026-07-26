"""Scene audit (Chiến Lược 2.md — Giai đoạn 1). Run BEFORE spending GPU time.

Per scene, verifies every assumption that can silently destroy the score:
  - camera model supported; CSV width/height/intrinsics vs cameras.bin
  - train files vs images.bin registrations (never assume they match)
  - test names present in CSV, disjoint from train files, presence in images.bin
  - points2D coordinate scale (images.bin keypoints may be at original 4x res)
  - reprojection sanity of the sparse model (median error should be < 1 px)
  - temporal structure: frame runs of test targets, nearest-train gaps

Usage:
  python -m vai_nvs.audit --data <phase1_dir_or_scene> [--out report.json]
Exit code 1 if any FAIL was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import cameras as camlib
from . import dataset as ds

# Windows consoles may default to cp1252 — never let printing kill an audit.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _check(report, level, ok, msg):
    tag = "OK" if ok else level
    report["checks"].append({"level": tag, "msg": msg})
    print(f"    [{tag:4s}] {msg}")
    return ok


def detect_points2d_scale(scene: ds.Scene) -> float:
    """Compare keypoint coordinate extent to camera size; test scales 1/2/4/8
    by reprojection error and return the best."""
    cams = scene.cameras
    # collect candidate images with enough 3D-linked keypoints
    linked = [(n, r) for n, r in scene.images.items()
              if n in set(scene.usable_train) and (r.point3D_ids >= 0).sum() >= 50]
    if not linked or scene.points3D is None:
        return 1.0, float("nan")
    linked = linked[: 5]
    best_scale, best_err = 1.0, np.inf
    for scale in (1.0, 2.0, 4.0, 8.0):
        errs = []
        for name, rec in linked:
            cam = cams[rec.camera_id]
            sel = rec.point3D_ids >= 0
            pairs = [(scene.points3D[p].xyz, xy)
                     for p, xy in zip(rec.point3D_ids[sel], rec.xys[sel])
                     if p in scene.points3D]
            if not pairs:
                continue
            xyz = np.stack([a for a, _ in pairs])
            obs = np.stack([b for _, b in pairs]) / scale
            uv, in_front = camlib.project_points(xyz, rec.qvec, rec.tvec, cam)
            e = np.linalg.norm(uv - obs, axis=1)[in_front]
            if len(e):
                errs.append(np.median(e))
        if errs and np.median(errs) < best_err:
            best_err, best_scale = float(np.median(errs)), scale
    return best_scale, best_err


def audit_scene(scene_dir: Path) -> dict:
    print(f"\n=== Scene: {scene_dir.name} ===")
    report = {"scene": scene_dir.name, "path": str(scene_dir), "checks": []}
    readme = scene_dir / "README.txt"
    if readme.exists():
        head = readme.read_text(errors="replace")[:400].strip()
        report["readme_head"] = head
        safe = head.encode("ascii", "replace").decode().replace("\n", " | ")
        print("    README: " + safe[:200])

    scene = ds.load_scene(scene_dir, load_points3D=True)
    report["n_train_files"] = len(scene.train_files)
    report["n_registered"] = len(scene.images)
    report["n_usable_train"] = len(scene.usable_train)
    report["n_test_poses"] = len(scene.test_poses)
    report["n_points3D"] = len(scene.points3D)
    report["has_test_gt"] = scene.has_test_gt

    # --- cameras ---
    cam_ids = sorted(scene.cameras.keys())
    report["cameras"] = {
        str(cid): {"model": scene.cameras[cid].model,
                   "width": scene.cameras[cid].width,
                   "height": scene.cameras[cid].height,
                   "params": scene.cameras[cid].params.tolist()}
        for cid in cam_ids
    }
    for cid in cam_ids:
        cam = scene.cameras[cid]
        _check(report, "FAIL", cam.model in camlib.SUPPORTED_MODELS,
               f"camera {cid} model {cam.model} supported (params={np.round(cam.params, 6).tolist()})")

    # --- counts / intersections ---
    train_set, reg_set = set(scene.train_files), set(scene.images.keys())
    test_names = [tp.image_name for tp in scene.test_poses]
    _check(report, "WARN", len(scene.usable_train) == len(scene.train_files),
           f"train files={len(scene.train_files)}, registered={len(scene.images)}, "
           f"usable(train AND registered)={len(scene.usable_train)}")
    orphan_files = sorted(train_set - reg_set)
    if orphan_files:
        _check(report, "WARN", False, f"{len(orphan_files)} train files NOT in images.bin "
               f"(will be skipped): {orphan_files[:3]}...")
    _check(report, "FAIL", train_set.isdisjoint(set(test_names)),
           "test image names are disjoint from train files")
    n_test_in_bin = sum(1 for n in test_names if n in reg_set)
    report["n_test_in_images_bin"] = n_test_in_bin
    _check(report, "WARN", True,
           f"{n_test_in_bin}/{len(test_names)} test poses also registered in images.bin")
    if n_test_in_bin:
        # verify CSV pose values match images.bin (should be ~identical)
        diffs = []
        for tp in scene.test_poses:
            if tp.image_name in scene.images:
                rec = scene.images[tp.image_name]
                diffs.append(max(np.abs(rec.qvec - tp.qvec).max(),
                                 np.abs(rec.tvec - tp.tvec).max()))
        report["max_csv_vs_bin_pose_diff"] = float(max(diffs))
        _check(report, "WARN", max(diffs) < 1e-6,
               f"CSV poses match images.bin poses (max diff {max(diffs):.2e})")

    # --- CSV intrinsics vs cameras.bin ---
    whs = {(tp.width, tp.height) for tp in scene.test_poses}
    ks = {(tp.fx, tp.fy, tp.cx, tp.cy) for tp in scene.test_poses}
    report["csv_whs"] = sorted(whs)
    report["csv_Ks"] = [list(k) for k in sorted(ks)]
    cam0 = scene.cameras[cam_ids[0]]
    fx0, fy0, cx0, cy0 = camlib.camera_pinhole(cam0)
    _check(report, "FAIL", all(w == cam0.width and h == cam0.height for w, h in whs),
           f"CSV width/height {sorted(whs)} == cameras.bin ({cam0.width},{cam0.height})")
    kdiff = max(abs(v[0] - fx0) + abs(v[1] - fy0) + abs(v[2] - cx0) + abs(v[3] - cy0) for v in ks)
    report["csv_vs_bin_K_absdiff"] = float(kdiff)
    _check(report, "WARN", kdiff < 1.0,
           f"CSV intrinsics vs cameras.bin pinhole: sum|diff|={kdiff:.4f} "
           f"(csv={sorted(ks)[0]}, bin=({fx0:.3f},{fy0:.3f},{cx0:.1f},{cy0:.1f}))")

    # --- image files really match camera size (sample 3) ---
    from PIL import Image
    for name in scene.usable_train[:: max(1, len(scene.usable_train) // 3)][:3]:
        with Image.open(scene.train_images_dir / name) as im:
            w, h = im.size
        cam = scene.cameras[scene.images[name].camera_id]
        _check(report, "FAIL", (w, h) == (cam.width, cam.height),
               f"{name}: file size {w}x{h} == camera {cam.width}x{cam.height}")

    # --- points2D scale + reprojection sanity ---
    scale, med_err = detect_points2d_scale(scene)
    report["points2d_scale"] = scale
    report["median_reproj_err_px"] = med_err
    _check(report, "WARN", True, f"points2D coordinate scale detected: /{scale:g} "
           f"(median reprojection error {med_err:.3f} px at that scale)")
    _check(report, "FAIL", med_err < 2.0,
           f"sparse model reprojection sanity: median {med_err:.3f} px < 2.0 px")

    # --- temporal structure ---
    runs = ds.test_run_lengths(scene)
    hist = {}
    for r in runs:
        hist[r] = hist.get(r, 0) + 1
    report["test_run_length_hist"] = hist
    train_idx = [ds.parse_dji_name(n)[1] for n in scene.usable_train]
    train_idx = sorted(i for i in train_idx if i is not None)
    gaps = []
    for tp in scene.test_poses:
        if tp.frame_idx is None or not train_idx:
            continue
        gaps.append(int(min(abs(tp.frame_idx - i) for i in train_idx)))
    report["nearest_train_gap"] = {
        "median": float(np.median(gaps)) if gaps else None,
        "max": int(max(gaps)) if gaps else None,
    }
    _check(report, "WARN", True,
           f"test run-length hist {hist}; nearest-train frame gap median="
           f"{report['nearest_train_gap']['median']}, max={report['nearest_train_gap']['max']}")

    # --- test GT (public scenes) ---
    if scene.has_test_gt:
        gt_files = set(ds.list_image_files(scene.test_gt_dir))
        missing = [n for n in test_names if n not in gt_files]
        _check(report, "WARN", not missing,
               f"public GT: {len(gt_files)} files, {len(missing)} CSV rows missing GT")

    n_fail = sum(1 for c in report["checks"] if c["level"] == "FAIL")
    n_warn = sum(1 for c in report["checks"] if c["level"] == "WARN")
    report["n_fail"], report["n_warn"] = n_fail, n_warn
    print(f"    => {n_fail} FAIL, {n_warn} WARN")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="phase1 dir, set dir, or single scene dir")
    ap.add_argument("--out", default=None, help="output JSON report path")
    args = ap.parse_args()

    scene_dirs = ds.discover_scenes(args.data)
    if not scene_dirs:
        raise SystemExit(f"No scenes found under {args.data}")
    print(f"Found {len(scene_dirs)} scene(s)")
    reports = [audit_scene(sd) for sd in scene_dirs]

    total_fail = sum(r["n_fail"] for r in reports)
    print("\n================ SUMMARY ================")
    for r in reports:
        print(f"  {r['scene']:12s} train={r['n_usable_train']:4d} test={r['n_test_poses']:3d} "
              f"pts3D={r['n_points3D']:7d} p2Dscale=/{r['points2d_scale']:g} "
              f"reproj={r['median_reproj_err_px']:.3f}px testInBin={r['n_test_in_images_bin']:3d} "
              f"FAIL={r['n_fail']} WARN={r['n_warn']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"Report saved to {args.out}")
    if total_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
