"""Package and validate the submission ZIP.

Hard rules enforced (missing anything voids the whole leaderboard entry):
  - one folder per scene, named exactly like the data scene folder
    (or per --rename-json mapping), at the ZIP root
  - every image_name from every test_poses.csv present, exact string match
  - every image has the exact CSV width x height
  - no extra image files

Usage:
  python -m vai_nvs.make_submission --data <private_set1> --work <work_root> \
      --out submission_round1.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from PIL import Image

from . import dataset as ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="private set root (scene folders)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--pred-dirname", default="pred_test")
    ap.add_argument("--out", default="submission_round1.zip")
    ap.add_argument("--rename-json", default=None,
                    help='optional {"data_scene_name": "zip_scene_name"} mapping')
    args = ap.parse_args()

    rename = {}
    if args.rename_json:
        with open(args.rename_json) as f:
            rename = json.load(f)

    scene_dirs = ds.discover_scenes(args.data)
    if not scene_dirs:
        raise SystemExit(f"No scenes found under {args.data}")

    problems = []
    plan = []  # (src_path, arcname)
    total = 0
    for sd in scene_dirs:
        zip_scene = rename.get(sd.name, sd.name)
        rows = ds.read_test_poses_csv(sd / "test" / "test_poses.csv")
        pred_dir = Path(args.work) / sd.name / args.pred_dirname
        names_expected = set()
        for tp in rows:
            names_expected.add(tp.image_name)
            src = pred_dir / tp.image_name
            if not src.exists():
                problems.append(f"{sd.name}: MISSING {tp.image_name}")
                continue
            with Image.open(src) as im:
                if im.size != (tp.width, tp.height):
                    problems.append(f"{sd.name}/{tp.image_name}: size {im.size} != "
                                    f"({tp.width},{tp.height})")
                    continue
                if im.mode != "RGB":
                    problems.append(f"{sd.name}/{tp.image_name}: mode {im.mode} != RGB")
                    continue
            plan.append((src, f"{zip_scene}/{tp.image_name}"))
        extra = [p.name for p in pred_dir.glob("*")
                 if p.suffix.lower() in ds.IMAGE_EXTS and p.name not in names_expected]
        if extra:
            print(f"[{sd.name}] note: {len(extra)} extra files in pred dir are NOT packaged")
        total += len(rows)
        print(f"[{sd.name}] {len(rows)} poses, "
              f"{sum(1 for s, a in plan if a.startswith(zip_scene + '/'))} files ready")

    if problems:
        print("\n!!! SUBMISSION BLOCKED — fix these first:")
        for p in problems[:50]:
            print("   " + p)
        raise SystemExit(1)

    out = Path(args.out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in plan:
            zf.write(src, arc)
    # re-open and verify
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    assert bad is None, f"corrupt member: {bad}"
    assert len(names) == len(plan) == total, (len(names), len(plan), total)
    print(f"\nOK: {out} — {len(names)} images, {len(scene_dirs)} scenes, "
          f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
