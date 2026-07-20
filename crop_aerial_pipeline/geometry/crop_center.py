"""Finds the crop *interior* (not its visible boundary) in 2D, and the robust
3D center derived from it -- this is the fix for the camera being pulled
toward the field edge. Padded/synthesized points are excluded by construction
(callers pass only ``is_original & in_crop_interior`` points in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def compute_interior_selector(mask: np.ndarray, quantile: float) -> np.ndarray:
    """Keeps only the crop-mask pixels farthest from the mask's own boundary --
    the top ``quantile`` fraction by distance-from-boundary. The boundary
    itself must never contribute to where the camera is centered.
    """
    mask_u8 = mask.astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_values = dist[mask]
    if dist_values.size == 0:
        return mask.copy()
    threshold = np.quantile(dist_values, 1.0 - quantile)
    return mask & (dist >= threshold)


@dataclass
class CropCenterResult:
    center_x: float
    center_z: float
    canopy_top_y: float
    interior_point_count: int
    fallback_used: str  # "interior" | "full_crop_mask" | "original_region"


def select_interior_points(
    points: np.ndarray,
    is_original: np.ndarray,
    in_crop_interior: np.ndarray,
    in_crop_mask: Optional[np.ndarray] = None,
    min_points_interior: int = 50,
    min_points_fallback: int = 10,
) -> Tuple[np.ndarray, str]:
    """The selection logic shared by ``robust_crop_center`` (for the center
    itself) and Stage 8's auto-zoom fitting (which needs the same selected
    points, transformed into camera space). Two-tier fallback if too few
    interior points survive the strict filter:

      1. crop-interior AND original (preferred)
      2. full crop-mask AND original (if too few interior points)
      3. any original point (if the crop mask itself was too sparse)

    Returns ``(selected_points, fallback_used)``.
    """
    selector = is_original & in_crop_interior
    fallback_used = "interior"

    if selector.sum() < min_points_interior and in_crop_mask is not None:
        selector = is_original & in_crop_mask
        fallback_used = "full_crop_mask"

    if selector.sum() < min_points_fallback:
        selector = is_original
        fallback_used = "original_region"

    if selector.sum() == 0:
        raise ValueError(
            "No original (non-synthesized) points available to compute a crop center -- "
            "the source image likely produced an empty/degenerate point cloud."
        )

    return points[selector], fallback_used


def robust_crop_center(
    points: np.ndarray,
    is_original: np.ndarray,
    in_crop_interior: np.ndarray,
    in_crop_mask: Optional[np.ndarray] = None,
    min_points_interior: int = 50,
    min_points_fallback: int = 10,
    canopy_quantile: float = 0.9,
) -> CropCenterResult:
    """Robust median/quantile center of the crop interior, computed *only*
    from ``is_original`` points (never padded/synthesized ones) -- see
    :func:`select_interior_points` for the fallback tiers.
    """
    selected, fallback_used = select_interior_points(
        points, is_original, in_crop_interior, in_crop_mask, min_points_interior, min_points_fallback
    )
    center_x = float(np.median(selected[:, 0]))
    center_z = float(np.median(selected[:, 2]))
    canopy_top_y = float(np.quantile(selected[:, 1], canopy_quantile))

    return CropCenterResult(
        center_x=center_x,
        center_z=center_z,
        canopy_top_y=canopy_top_y,
        interior_point_count=int(selected.shape[0]),
        fallback_used=fallback_used,
    )


def fit_ground_plane_ransac(points: np.ndarray, max_points: int, distance_threshold: float = 0.05, seed: int = 42):
    """Optional RANSAC ground-plane fit via Open3D -- NOT used by the default
    camera-placement path (canopy-top quantile of the crop interior is
    sufficient and avoids adding Open3D to the required dependency chain).
    Provided only because the spec reserves it as an optional diagnostic;
    call it yourself if you want a ground-plane normal/offset for analysis.

    Raises ``ImportError`` with a clear message if Open3D isn't installed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "fit_ground_plane_ransac() requires Open3D ('pip install open3d'), which is NOT "
            "a required dependency of this pipeline otherwise -- it's an optional diagnostic."
        ) from exc

    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    return plane_model, inliers


def draw_crop_center_diagnostic(
    rgb: np.ndarray,
    crop_mask: np.ndarray,
    interior_selector: np.ndarray,
    center_pixel: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Renders the Stage 4 diagnostic: crop mask boundary + interior selection
    + (optionally) the calculated 2D center, overlaid on the source RGB.
    """
    overlay = rgb.copy()

    contours, _ = cv2.findContours(crop_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)  # crop mask boundary in red

    interior_overlay = overlay.copy()
    interior_overlay[interior_selector] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.6, interior_overlay, 0.4, 0)  # green = interior selection

    if center_pixel is not None:
        cv2.drawMarker(
            overlay, center_pixel, (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=3
        )

    return overlay
