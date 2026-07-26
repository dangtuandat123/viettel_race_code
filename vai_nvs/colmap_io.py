"""COLMAP sparse model readers (binary + text), numpy-accelerated.

Binary layout follows COLMAP's src/colmap/scene/reconstruction_io.cc.
All values are little-endian. Coordinate conventions:
  - images.bin stores WORLD-TO-CAMERA rotation (qvec, scalar-first) and
    translation (tvec). Camera center C = -R(q)^T @ t.
  - 2D point coordinates use COLMAP's continuous pixel convention where the
    center of the top-left pixel is (0.5, 0.5).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# model_id -> (name, num_params). From COLMAP camera_models.h.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),   # f, cx, cy
    1: ("PINHOLE", 4),          # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),    # f, cx, cy, k1
    3: ("RADIAL", 5),           # f, cx, cy, k1, k2
    4: ("OPENCV", 8),           # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # float64


@dataclass
class ImageRecord:
    id: int
    qvec: np.ndarray        # (4,) w2c quaternion, scalar first
    tvec: np.ndarray        # (3,) w2c translation
    camera_id: int
    name: str
    xys: np.ndarray         # (N, 2) keypoint coords (COLMAP convention)
    point3D_ids: np.ndarray  # (N,) int64, -1 if not triangulated


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


def _read_bytes(fid, num_bytes, fmt):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_cameras_binary(path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as fid:
        num = _read_bytes(fid, 8, "Q")[0]
        for _ in range(num):
            cam_id, model_id, width, height = _read_bytes(fid, 24, "iiQQ")
            name, num_params = CAMERA_MODELS[model_id]
            params = np.array(_read_bytes(fid, 8 * num_params, "d" * num_params))
            cameras[cam_id] = Camera(cam_id, name, int(width), int(height), params)
    return cameras


def read_images_binary(path) -> dict[str, ImageRecord]:
    """Returns dict keyed by image NAME (unique in COLMAP models)."""
    images = {}
    pt_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
    with open(path, "rb") as fid:
        num = _read_bytes(fid, 8, "Q")[0]
        for _ in range(num):
            image_id = _read_bytes(fid, 4, "i")[0]
            qvec = np.array(_read_bytes(fid, 32, "dddd"))
            tvec = np.array(_read_bytes(fid, 24, "ddd"))
            camera_id = _read_bytes(fid, 4, "i")[0]
            name_bytes = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_pts = _read_bytes(fid, 8, "Q")[0]
            buf = fid.read(24 * num_pts)
            pts = np.frombuffer(buf, dtype=pt_dtype)
            xys = np.stack([pts["x"], pts["y"]], axis=1) if num_pts else np.zeros((0, 2))
            p3d = pts["id"].copy() if num_pts else np.zeros((0,), dtype=np.int64)
            images[name] = ImageRecord(image_id, qvec, tvec, camera_id, name, xys, p3d)
    return images


def read_points3D_binary(path) -> dict[int, Point3D]:
    points = {}
    trk_dtype = np.dtype([("image_id", "<i4"), ("p2d_idx", "<i4")])
    with open(path, "rb") as fid:
        num = _read_bytes(fid, 8, "Q")[0]
        for _ in range(num):
            pid = _read_bytes(fid, 8, "Q")[0]
            xyz = np.array(_read_bytes(fid, 24, "ddd"))
            rgb = np.array(_read_bytes(fid, 3, "BBB"), dtype=np.uint8)
            error = _read_bytes(fid, 8, "d")[0]
            track_len = _read_bytes(fid, 8, "Q")[0]
            buf = fid.read(8 * track_len)
            trk = np.frombuffer(buf, dtype=trk_dtype)
            points[pid] = Point3D(pid, xyz, rgb, float(error),
                                  trk["image_id"].copy(), trk["p2d_idx"].copy())
    return points


# ----------------------------- text fallbacks ------------------------------

def read_cameras_text(path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            cam_id, model = int(elems[0]), elems[1]
            width, height = int(elems[2]), int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return cameras


def read_images_text(path) -> dict[str, ImageRecord]:
    images = {}
    with open(path, "r") as fid:
        lines = [ln.strip() for ln in fid if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        image_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        camera_id = int(elems[8])
        name = elems[9]
        pts = lines[i + 1].split()
        n = len(pts) // 3
        xys = np.array([[float(pts[3 * j]), float(pts[3 * j + 1])] for j in range(n)])
        p3d = np.array([int(pts[3 * j + 2]) for j in range(n)], dtype=np.int64)
        if n == 0:
            xys = np.zeros((0, 2))
        images[name] = ImageRecord(image_id, qvec, tvec, camera_id, name, xys, p3d)
    return images


def read_points3D_text(path) -> dict[int, Point3D]:
    points = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            pid = int(e[0])
            xyz = np.array(list(map(float, e[1:4])))
            rgb = np.array(list(map(int, e[4:7])), dtype=np.uint8)
            error = float(e[7])
            trk = list(map(int, e[8:]))
            image_ids = np.array(trk[0::2], dtype=np.int32)
            p2d = np.array(trk[1::2], dtype=np.int32)
            points[pid] = Point3D(pid, xyz, rgb, error, image_ids, p2d)
    return points


# ------------------------------- entry point --------------------------------

def _pick(sparse_dir: Path, stem: str) -> tuple[Path, str]:
    b, t = sparse_dir / f"{stem}.bin", sparse_dir / f"{stem}.txt"
    if b.exists():
        return b, "bin"
    if t.exists():
        return t, "txt"
    raise FileNotFoundError(f"Neither {b} nor {t} exists")


def read_model(sparse_dir):
    """Read a COLMAP sparse model directory (bin preferred, txt fallback).

    Returns (cameras: {id: Camera}, images: {name: ImageRecord},
             points3D: {id: Point3D}).
    """
    sparse_dir = Path(sparse_dir)
    cam_path, cam_kind = _pick(sparse_dir, "cameras")
    img_path, img_kind = _pick(sparse_dir, "images")
    pts_path, pts_kind = _pick(sparse_dir, "points3D")
    cameras = read_cameras_binary(cam_path) if cam_kind == "bin" else read_cameras_text(cam_path)
    images = read_images_binary(img_path) if img_kind == "bin" else read_images_text(img_path)
    points = read_points3D_binary(pts_path) if pts_kind == "bin" else read_points3D_text(pts_path)
    return cameras, images, points


def find_sparse_dir(scene_dir) -> Path:
    """Locate the sparse model dir: train/sparse/0 preferred, else train/sparse."""
    scene_dir = Path(scene_dir)
    for cand in [scene_dir / "train" / "sparse" / "0", scene_dir / "train" / "sparse"]:
        if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
            return cand
    raise FileNotFoundError(f"No COLMAP sparse model under {scene_dir}/train/sparse")
