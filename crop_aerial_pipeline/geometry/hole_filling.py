"""Small, bounded residual-hole filling for the rendered image -- NOT a
general-purpose inpaint, and never touching pixels that already have a real
rendered value. Large unsupported regions are left clearly marked rather than
silently papered over (Stage 5's source extension is what's responsible for
making large holes rare in the first place).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FillResult:
    filled_image: np.ndarray
    filled_mask: np.ndarray  # valid_mask OR newly-filled
    newly_filled_mask: np.ndarray
    large_unsupported_mask: np.ndarray  # invalid regions too big to fill locally


def fill_residual_holes(
    color_img: np.ndarray,
    valid_mask: np.ndarray,
    max_radius: float,
    smooth_filled: bool = True,
) -> FillResult:
    invalid = ~valid_mask
    if not invalid.any():
        return FillResult(
            filled_image=color_img,
            filled_mask=valid_mask,
            newly_filled_mask=np.zeros_like(valid_mask),
            large_unsupported_mask=np.zeros_like(valid_mask),
        )

    dist, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8) * 255, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    valid_ys, valid_xs = np.where(valid_mask)
    label_map = np.zeros((int(labels.max()) + 1, 2), dtype=np.int32)
    label_map[labels[valid_mask]] = np.stack([valid_ys, valid_xs], axis=1)

    fillable = invalid & (dist <= max_radius)
    fill_ys, fill_xs = np.where(fillable)
    src_yx = label_map[labels[fill_ys, fill_xs]]

    filled_img = color_img.copy()
    filled_img[fill_ys, fill_xs] = color_img[src_yx[:, 0], src_yx[:, 1]]

    if smooth_filled and fillable.any():
        blurred = cv2.GaussianBlur(filled_img, (5, 5), 0)
        filled_img[fillable] = blurred[fillable]  # only newly-filled pixels are touched

    filled_mask = valid_mask | fillable
    large_unsupported_mask = invalid & ~fillable

    return FillResult(
        filled_image=filled_img,
        filled_mask=filled_mask,
        newly_filled_mask=fillable,
        large_unsupported_mask=large_unsupported_mask,
    )


def large_unsupported_fraction(large_unsupported_mask: np.ndarray) -> float:
    return float(large_unsupported_mask.mean())


def crop_inward_if_needed(
    image: np.ndarray,
    large_unsupported_mask: np.ndarray,
    border_frac: float = 0.05,
    max_border_bad_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """If large unsupported regions concentrate near the frame border, crop a
    thin margin inward rather than shipping an output with an obviously bad
    edge. Conservative by design: only triggers when the *border strip*
    itself is mostly unsupported, and only removes ``border_frac`` of each
    side once (not iteratively) to avoid silently shrinking the image a lot.

    Returns ``(cropped_image, cropped_mask, did_crop)``.
    """
    h, w = large_unsupported_mask.shape
    bw, bh = max(1, int(w * border_frac)), max(1, int(h * border_frac))

    border = np.zeros_like(large_unsupported_mask)
    border[:bh, :] = True
    border[-bh:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True

    border_bad_frac = float((large_unsupported_mask & border).sum() / max(border.sum(), 1))
    if border_bad_frac <= max_border_bad_frac:
        return image, large_unsupported_mask, False

    cropped_image = image[bh:-bh, bw:-bw]
    cropped_mask = large_unsupported_mask[bh:-bh, bw:-bw]
    return cropped_image, cropped_mask, True
