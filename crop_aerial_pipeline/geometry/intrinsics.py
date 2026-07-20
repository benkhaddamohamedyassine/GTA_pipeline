"""Pinhole camera intrinsics: the source (phone) camera used only for
back-projection, and a completely separate virtual (rendering) camera whose
focal length is auto-solved to frame the crop interior.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def estimate_source_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole intrinsics from image size + an assumed horizontal FOV. This is
    the *source* camera -- used only to back-project the (possibly padded)
    source image into 3D, never for rendering.
    """
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx  # square pixels -- true for essentially all phone sensors
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def shift_principal_point(K: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Updates a source intrinsics matrix's principal point after padding the
    image it describes -- e.g. ``shift_principal_point(K, pad_x, pad_y)``.
    """
    K_shifted = K.copy()
    K_shifted[0, 2] += dx
    K_shifted[1, 2] += dy
    return K_shifted


def solve_virtual_intrinsics(
    interior_points_cam: np.ndarray,
    out_w: int,
    out_h: int,
    frame_fill: float,
    min_hfov_deg: float,
    max_hfov_deg: float,
    robust_quantile: float = 0.98,
) -> Tuple[np.ndarray, float]:
    """Auto-zoom: solves the virtual focal length so the crop interior's
    *robust* (``robust_quantile``, not true min/max) angular extent fills
    ``frame_fill`` of the output frame, clamped to
    ``[min_hfov_deg, max_hfov_deg]``.

    ``interior_points_cam`` must already be transformed into the *virtual*
    camera's coordinate frame (via its extrinsic) -- X right, Y down, Z
    forward/depth.
    """
    if interior_points_cam.shape[0] == 0:
        raise ValueError("solve_virtual_intrinsics received zero interior points")

    z = np.clip(interior_points_cam[:, 2], 1e-3, None)
    ratio_x = np.abs(interior_points_cam[:, 0] / z)
    ratio_y = np.abs(interior_points_cam[:, 1] / z)
    ext_x = max(float(np.quantile(ratio_x, robust_quantile)), 1e-3)
    ext_y = max(float(np.quantile(ratio_y, robust_quantile)), 1e-3)

    fx_candidate = (frame_fill * out_w / 2.0) / ext_x
    fy_candidate = (frame_fill * out_h / 2.0) / ext_y
    f = min(fx_candidate, fy_candidate)  # more restrictive axis wins -- keeps both <= target fill

    hfov_deg = math.degrees(2 * math.atan((out_w / 2.0) / f))
    hfov_clamped = float(np.clip(hfov_deg, min_hfov_deg, max_hfov_deg))
    f_final = (out_w / 2.0) / math.tan(math.radians(hfov_clamped) / 2.0)

    K_virtual = np.array(
        [[f_final, 0, out_w / 2.0], [0, f_final, out_h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    return K_virtual, hfov_clamped
