"""Back-projects an (extended) RGB+depth source image into a 3D point cloud,
carrying along per-point provenance (real source vs. synthesized extension)
and crop-mask membership -- fully vectorized, no per-pixel Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

NEAR_M = 0.4
FAR_M = 18.0


def depth_norm_to_metric(depth_norm: np.ndarray, near_m: float = NEAR_M, far_m: float = FAR_M) -> np.ndarray:
    """Maps a relative depth map's [0,1] range (1.0 = nearest to camera) onto a
    plausible metric range. Monocular relative depth carries no true scale, so
    this is a deliberate approximation -- see ``DepthEstimator`` docs -- good
    enough to preserve relative 3D structure for reprojection, not a
    measurement.
    """
    return near_m + (1.0 - depth_norm) * (far_m - near_m)


@dataclass
class BackprojectionResult:
    points: np.ndarray  # (N, 3) float64, +X right, +Y up, +Z forward
    colors: np.ndarray  # (N, 3) float64 in [0, 1]
    pixel_coords: np.ndarray  # (N, 2) int64 [u, v] in the (padded) source image
    is_original: np.ndarray  # (N,) bool -- True for real/super-resolved pixels, False for synthesized extension
    in_crop_mask: np.ndarray  # (N,) bool -- True if the source pixel is inside the (possibly extended) crop mask


def backproject(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    stride: int,
    origin_mask: Optional[np.ndarray] = None,
    crop_mask: Optional[np.ndarray] = None,
) -> BackprojectionResult:
    """Vectorized back-projection. Camera convention: origin at the source
    camera, +X right, +Y up, +Z forward (increasing with depth).

    ``origin_mask``: boolean HxW, True where the source pixel is real (or
    super-resolved-but-real) content, False where it was synthesized by Stage
    5's source extension. Defaults to all-True (nothing synthesized).

    ``crop_mask``: boolean HxW crop/vegetation mask (already extended into any
    synthesized region by Stage 5, if applicable). Defaults to all-True.
    """
    h, w = depth_m.shape
    if origin_mask is None:
        origin_mask = np.ones((h, w), dtype=bool)
    if crop_mask is None:
        crop_mask = np.ones((h, w), dtype=bool)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))

    d = depth_m[vs, us]
    x = (us - cx) * d / fx
    y = -(vs - cy) * d / fy
    z = d

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)
    colors = (rgb[vs, us].reshape(-1, 3).astype(np.float64)) / 255.0
    pixel_coords = np.stack([us, vs], axis=-1).reshape(-1, 2).astype(np.int64)
    is_original = origin_mask[vs, us].reshape(-1)
    in_crop_mask = crop_mask[vs, us].reshape(-1)

    return BackprojectionResult(
        points=points,
        colors=colors,
        pixel_coords=pixel_coords,
        is_original=is_original,
        in_crop_mask=in_crop_mask,
    )


def sample_mask_at_stride(mask: np.ndarray, h: int, w: int, stride: int) -> np.ndarray:
    """Samples a boolean HxW mask at the same pixel grid ``backproject`` used,
    without re-running the (relatively costly) back-projection -- useful when
    testing the same point cloud against a different mask (e.g. a fallback
    selector).
    """
    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    return mask[vs, us].reshape(-1)
