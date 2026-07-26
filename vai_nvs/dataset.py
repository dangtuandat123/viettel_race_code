"""Scene discovery, test-pose CSV parsing, DJI filename parsing, fold IO."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import colmap_io

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# e.g. DJI_20241229095618_0222_V.JPG -> ts=20241229095618, frame=0222
_DJI_RE = re.compile(r"_(\d{14})_(\d{3,6})")


def parse_dji_name(name: str):
    """Returns (timestamp_str or None, frame_idx int or None)."""
    m = _DJI_RE.search(name)
    if m:
        return m.group(1), int(m.group(2))
    m2 = re.search(r"(\d{3,6})\D*$", Path(name).stem)
    return None, (int(m2.group(1)) if m2 else None)


def list_image_files(images_dir) -> list[str]:
    """Sorted image file names, excluding hidden/AppleDouble junk."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    out = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("._"):
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            out.append(p.name)
    return sorted(out)


@dataclass
class TestPose:
    image_name: str
    qvec: np.ndarray
    tvec: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame_idx: int | None = None
    timestamp: str | None = None


def read_test_poses_csv(path) -> list[TestPose]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [fn.strip() for fn in reader.fieldnames]
        for raw in reader:
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            if not r.get("image_name"):
                continue
            ts, fi = parse_dji_name(r["image_name"])
            rows.append(TestPose(
                image_name=r["image_name"],
                qvec=np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                tvec=np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                fx=float(r["fx"]), fy=float(r["fy"]),
                cx=float(r["cx"]), cy=float(r["cy"]),
                width=int(float(r["width"])), height=int(float(r["height"])),
                frame_idx=fi, timestamp=ts,
            ))
    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    return rows


@dataclass
class Scene:
    name: str
    scene_dir: Path
    sparse_dir: Path
    cameras: dict          # id -> Camera
    images: dict           # name -> ImageRecord (all registered, incl. any test)
    points3D: dict | None  # id -> Point3D (None if skipped)
    train_files: list[str]     # files physically present in train/images
    usable_train: list[str]    # train_files ∩ images.bin names (sorted temporally)
    test_poses: list[TestPose]
    has_test_gt: bool          # test/images exists (public scenes)

    @property
    def test_gt_dir(self) -> Path:
        return self.scene_dir / "test" / "images"

    @property
    def train_images_dir(self) -> Path:
        return self.scene_dir / "train" / "images"


def temporal_sort_key(name: str):
    ts, fi = parse_dji_name(name)
    return (ts or "", fi if fi is not None else -1, name)


def load_scene(scene_dir, load_points3D=True) -> Scene:
    scene_dir = Path(scene_dir)
    sparse_dir = colmap_io.find_sparse_dir(scene_dir)
    cameras, images, points = (None, None, None)
    cams, imgs, pts = colmap_io.read_model(sparse_dir) if load_points3D else (None, None, None)
    if not load_points3D:
        cams = colmap_io.read_cameras_binary(sparse_dir / "cameras.bin") \
            if (sparse_dir / "cameras.bin").exists() else colmap_io.read_cameras_text(sparse_dir / "cameras.txt")
        imgs = colmap_io.read_images_binary(sparse_dir / "images.bin") \
            if (sparse_dir / "images.bin").exists() else colmap_io.read_images_text(sparse_dir / "images.txt")
        pts = None
    train_files = list_image_files(scene_dir / "train" / "images")
    usable = sorted(set(train_files) & set(imgs.keys()), key=temporal_sort_key)
    csv_path = scene_dir / "test" / "test_poses.csv"
    if not csv_path.exists():
        # tolerate the singular/plural naming in the problem statement
        alt = scene_dir / "test" / "test_pose.csv"
        csv_path = alt if alt.exists() else csv_path
    test_poses = read_test_poses_csv(csv_path)
    has_gt = (scene_dir / "test" / "images").exists()
    return Scene(scene_dir.name, scene_dir, sparse_dir, cams, imgs, pts,
                 train_files, usable, test_poses, has_gt)


def discover_scenes(data_root) -> list[Path]:
    """Find scene dirs (containing train/sparse) under a root, recursively."""
    data_root = Path(data_root)
    if (data_root / "train").exists():
        return [data_root]
    out = []
    for p in sorted(data_root.rglob("train")):
        if "__MACOSX" in p.parts:
            continue
        if p.is_dir() and (p / "sparse").exists():
            out.append(p.parent)
    return out


# --------------------------------- folds ------------------------------------

def test_run_lengths(scene: Scene) -> list[int]:
    """Lengths of consecutive-frame runs in the test set (by frame index)."""
    idxs = sorted(tp.frame_idx for tp in scene.test_poses if tp.frame_idx is not None)
    if not idxs:
        return [1] * len(scene.test_poses)
    runs, cur = [], 1
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def make_folds(scene: Scene, n_folds=2, seed=0, min_gap=2) -> dict:
    """Pseudo-validation folds that mimic the test set's missing-block
    structure (Chiến Lược 2.md §Giai đoạn 3). Blocks are consecutive in the
    temporally sorted usable-train sequence."""
    seq = scene.usable_train
    n = len(seq)
    lengths = test_run_lengths(scene)
    # cap total val fraction at 35% to keep folds trainable
    max_val = int(0.35 * n)
    if sum(lengths) > max_val and sum(lengths) > 0:
        scale = max_val / sum(lengths)
        lengths = [max(1, int(round(l * scale))) for l in lengths]
    folds = {}
    for k in range(n_folds):
        rng = np.random.RandomState(seed * 1000 + k)
        order = sorted(lengths, reverse=True)
        occupied = np.zeros(n, dtype=bool)
        blocks = []
        for L in order:
            placed = False
            for _ in range(2000):
                s = int(rng.randint(min_gap, max(min_gap + 1, n - L - min_gap)))
                lo, hi = max(0, s - min_gap), min(n, s + L + min_gap)
                if not occupied[lo:hi].any():
                    occupied[s:s + L] = True
                    blocks.append((s, L))
                    placed = True
                    break
            if not placed:  # fall back: shrink block
                for s in range(min_gap, n - 1 - min_gap):
                    if not occupied[max(0, s - min_gap):min(n, s + 1 + min_gap)].any():
                        occupied[s] = True
                        blocks.append((s, 1))
                        break
        val = sorted({seq[s + i] for s, L in blocks for i in range(L)},
                     key=temporal_sort_key)
        train = [x for x in seq if x not in set(val)]
        folds[str(k)] = {"val": val, "train": train}
    return folds


def load_folds(work_scene_dir) -> dict:
    with open(Path(work_scene_dir) / "folds.json", "r") as f:
        return json.load(f)


def load_meta(work_scene_dir) -> dict:
    with open(Path(work_scene_dir) / "meta.json", "r") as f:
        return json.load(f)
