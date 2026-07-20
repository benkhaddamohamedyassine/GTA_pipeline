"""Crop/vegetation mask estimation. Not a learned segmentation model -- a
cheap, dependency-light HSV + Excess Green Index heuristic, which is enough
for this pipeline's purposes (finding the crop *interior* for camera
centering) without adding another model to load/unload.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

MIN_COMPONENT_AREA_FRAC = 0.0005  # components smaller than this fraction of the image are noise
MIN_MASK_AREA_FRAC = 0.05  # below this, fall back to the full image rather than a tiny/noisy mask


def estimate_crop_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_green = (hue >= 25) & (hue <= 100) & (sat >= 30) & (val >= 20)

    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    exg_green = (2 * g - r - b) > 15.0

    mask = ((hsv_green | exg_green).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # remove small noise specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small internal holes

    mask_bool = mask > 0
    mask_bool = _remove_tiny_components(mask_bool)
    return mask_bool


def _remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = MIN_COMPONENT_AREA_FRAC * mask.size
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # 0 is background
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = True
    return cleaned


def get_or_estimate_crop_mask(
    rgb: np.ndarray,
    external_mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    min_area_frac: float = MIN_MASK_AREA_FRAC,
) -> tuple[np.ndarray, bool]:
    """Returns ``(mask, used_fallback)``.

    ``external_mask_provider``, if given, is called with the RGB image and may
    return a boolean mask (a user-supplied override, e.g. read from a
    path-mapping callback) or ``None`` to fall through to automatic detection.
    """
    if external_mask_provider is not None:
        provided = external_mask_provider(rgb)
        if provided is not None:
            return provided.astype(bool), False

    mask = estimate_crop_mask(rgb)
    if mask.mean() < min_area_frac:
        return np.ones(rgb.shape[:2], dtype=bool), True
    return mask, False
