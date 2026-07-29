"""Camera pose + lens distortion math.

Conventions (critical — see Chiến Lược 2.md §1.1-1.2):
  - COLMAP/CSV poses are WORLD-TO-CAMERA: x_cam = R(qvec) @ x_world + tvec.
    Camera center C = -R^T t. (tx,ty,tz) is NOT the camera position.
  - Pixel coordinates: COLMAP continuous convention, center of top-left pixel
    is (0.5, 0.5). gsplat rasterizes pixel (i,j) at (j+0.5, i+0.5), i.e. the
    SAME convention, so cameras.bin / CSV intrinsics can be fed to gsplat
    unchanged. OpenCV's remap() instead treats integer coordinates as pixel
    centers, so COLMAP coords are converted with a -0.5 offset exactly once,
    inside the map builders here.
  - Normalized distortion: SIMPLE_RADIAL applies x_d = x * (1 + k1 * r^2).
"""

from __future__ import annotations

import numpy as np

PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}
DISTORTED_MODELS = {"SIMPLE_RADIAL", "RADIAL", "OPENCV"}
SUPPORTED_MODELS = PINHOLE_MODELS | DISTORTED_MODELS


# ------------------------------ pose helpers -------------------------------

def qvec2rotmat(qvec):
    """COLMAP scalar-first quaternion -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def w2c_matrix(qvec, tvec) -> np.ndarray:
    """4x4 world-to-camera matrix from COLMAP qvec/tvec."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = qvec2rotmat(qvec)
    m[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return m


def camera_center(qvec, tvec) -> np.ndarray:
    R = qvec2rotmat(qvec)
    return -R.T @ np.asarray(tvec, dtype=np.float64)


def view_dir_world(qvec) -> np.ndarray:
    """Camera optical axis (+Z of camera) expressed in world coords."""
    R = qvec2rotmat(qvec)
    return R[2, :]  # row 2 of w2c R == R^T @ [0,0,1]


# --------------------------- intrinsics helpers -----------------------------

def camera_pinhole(cam) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a COLMAP Camera, ignoring distortion params."""
    p = cam.params
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx, fy = float(p[0]), float(p[0])
    elif cam.model in ("PINHOLE", "OPENCV"):
        fx, fy = float(p[0]), float(p[1])
    else:
        raise ValueError(f"Unsupported camera model: {cam.model}")
    # Hotfix #1: Force principal point to center to match train_gs_original's 
    # intentional projection matrix shift.
    cx = cam.width / 2.0
    cy = cam.height / 2.0
    return fx, fy, cx, cy


def distortion_params(cam) -> np.ndarray:
    """Distortion coefficient vector [k1, k2, p1, p2] (zeros if pinhole)."""
    p = cam.params
    if cam.model in PINHOLE_MODELS:
        return np.zeros(4)
    if cam.model == "SIMPLE_RADIAL":
        return np.array([p[3], 0.0, 0.0, 0.0])
    if cam.model == "RADIAL":
        return np.array([p[3], p[4], 0.0, 0.0])
    if cam.model == "OPENCV":
        return np.array([p[4], p[5], p[6], p[7]])
    raise ValueError(f"Unsupported camera model: {cam.model}")


# ---------------------------- distortion math -------------------------------

def apply_distortion(dist, x, y):
    """Normalized undistorted -> distorted. dist = [k1, k2, p1, p2]."""
    k1, k2, p1, p2 = dist
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + k2 * r2)
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort_points(dist, xd, yd, num_iters=20):
    """Normalized distorted -> undistorted via fixed-point iteration.

    Converges quickly for the small distortions in this dataset (|k1|~0.008).
    Raises if the round-trip residual is not negligible.
    """
    k1, k2, p1, p2 = dist
    x, y = np.array(xd, dtype=np.float64, copy=True), np.array(yd, dtype=np.float64, copy=True)
    for _ in range(num_iters):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + k2 * r2)
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    xchk, ychk = apply_distortion(dist, x, y)
    err = np.max(np.abs(np.stack([xchk - xd, ychk - yd])))
    if err > 1e-6:
        raise RuntimeError(f"Undistortion fixed-point did not converge (max err {err:.3e})")
    return x, y


# ------------------------------ remap builders ------------------------------

def _pixel_grid(width, height, supersample=1.0):
    """COLMAP-convention pixel-center coordinates of an output grid whose
    pixels are (1/supersample) original pixels wide."""
    w, h = int(round(width * supersample)), int(round(height * supersample))
    js, is_ = np.meshgrid(np.arange(w), np.arange(h))
    u = (js + 0.5) / supersample
    v = (is_ + 0.5) / supersample
    return u, v


def build_undistort_maps(cam):
    """Maps to create an UNDISTORTED image (same size, same pinhole K as cam)
    by sampling the original distorted image with cv2.remap.

    Returns (map_x, map_y, valid_mask) — float32 maps in OpenCV coords,
    valid_mask True where the source sample lies fully inside the image.
    """
    fx, fy, cx, cy = camera_pinhole(cam)
    dist = distortion_params(cam)
    u, v = _pixel_grid(cam.width, cam.height)
    x = (u - cx) / fx
    y = (v - cy) / fy
    xd, yd = apply_distortion(dist, x, y)
    us = fx * xd + cx
    vs = fy * yd + cy
    map_x = (us - 0.5).astype(np.float32)
    map_y = (vs - 0.5).astype(np.float32)
    valid = (map_x >= 0) & (map_x <= cam.width - 1) & (map_y >= 0) & (map_y <= cam.height - 1)
    return map_x, map_y, valid


def build_redistort_maps(render_K, dist_cam, out_width, out_height, out_supersample=1.0,
                         render_supersample=1.0):
    """Maps to synthesize the DISTORTED target image from a pinhole render.

    render_K:  (fx, fy, cx, cy) of the pinhole render at supersample 1
               (the actual render is render_supersample x that size).
    dist_cam:  COLMAP Camera describing the distortion geometry (its own
               pinhole params normalize the distortion; may differ from
               render_K, e.g. CSV intrinsics vs cameras.bin).
    out_*:     distorted output size at supersample 1; maps are built for an
               output grid of out_supersample x that size (downsample after).
    """
    dfx, dfy, dcx, dcy = camera_pinhole(dist_cam)
    dist = distortion_params(dist_cam)
    rfx, rfy, rcx, rcy = render_K
    u, v = _pixel_grid(out_width, out_height, out_supersample)
    xd = (u - dcx) / dfx
    yd = (v - dcy) / dfy
    if np.any(dist != 0):
        x, y = undistort_points(dist, xd, yd)
    else:
        x, y = xd, yd
    ur = (rfx * x + rcx) * render_supersample
    vr = (rfy * y + rcy) * render_supersample
    map_x = (ur - 0.5).astype(np.float32)
    map_y = (vr - 0.5).astype(np.float32)
    return map_x, map_y


# --------------------------- projection (audit) -----------------------------

def project_points(xyz_world, qvec, tvec, cam):
    """Project world points into a distorted COLMAP camera.

    Returns (uv [N,2] COLMAP pixel coords, in_front [N] bool).
    """
    R = qvec2rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    xc = xyz_world @ R.T + t
    in_front = xc[:, 2] > 1e-6
    z = np.where(in_front, xc[:, 2], 1.0)
    x = xc[:, 0] / z
    y = xc[:, 1] / z
    xd, yd = apply_distortion(distortion_params(cam), x, y)
    fx, fy, cx, cy = camera_pinhole(cam)
    u = fx * xd + cx
    v = fy * yd + cy
    return np.stack([u, v], axis=1), in_front
