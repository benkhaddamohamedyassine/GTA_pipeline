"""Extends the source RGB + depth *before* back-projection, so the renderer
has real (patch-synthesized, crop-consistent) content for what would
otherwise be empty triangular corners -- replacing a large post-warp inpaint
with source extension, and replacing plain mirroring (which produces obvious
mirrored crop-row artifacts) with patch-based texture synthesis by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Protocol, Tuple

import cv2
import numpy as np

from .crop_center import compute_interior_selector

logger = logging.getLogger("crop_aerial_pipeline")

MIN_SAFE_PATCH_PIXELS_FOR_SYNTHESIS = 64 * 64 * 8  # below this, texture synthesis can't find enough source material


@dataclass
class ExtendedSource:
    rgb: np.ndarray
    depth: np.ndarray
    origin_mask: np.ndarray  # True = real (or super-resolved-real) pixel, False = synthesized
    crop_mask: np.ndarray  # crop mask extended into synthesized crop-like regions
    pad_x: int
    pad_y: int
    method: str  # "texture_synthesis" | "reflection"
    fallback_used: bool


class SourceExtender(Protocol):
    def extend(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, pad_x: int, pad_y: int) -> ExtendedSource: ...


def _feather_blend(base: np.ndarray, patch: np.ndarray, x0: int, y0: int, overlap: int) -> None:
    """Alpha-feathers ``patch`` into ``base`` at (x0, y0) over its already-filled
    overlap region, writing in place. Cheap and robust across arbitrary patch
    content (unlike Poisson/seamless cloning, which can misbehave on flat or
    low-texture crop regions).
    """
    ph, pw = patch.shape[:2]
    y1, x1 = y0 + ph, x0 + pw

    alpha = np.ones((ph, pw), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        alpha[:, :overlap] = np.minimum(alpha[:, :overlap], ramp[np.newaxis, :])
        alpha[:overlap, :] = np.minimum(alpha[:overlap, :], ramp[:, np.newaxis])

    region = base[y0:y1, x0:x1].astype(np.float32)
    blended = region * (1 - alpha[..., None]) + patch.astype(np.float32) * alpha[..., None]
    base[y0:y1, x0:x1] = blended.astype(base.dtype)


class TexturePatchSourceExtender:
    """Default extender: samples small patches from the crop interior (never
    from background/boundary pixels) and tiles them across the padded border
    with randomized placement and feather-blended overlaps.
    """

    def __init__(self, patch_size: int = 64, overlap: int = 12, seed: int = 42, safe_quantile: float = 0.5) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self.seed = seed
        self.safe_quantile = safe_quantile

    def extend(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, pad_x: int, pad_y: int) -> ExtendedSource:
        h, w = rgb.shape[:2]
        safe_region = compute_interior_selector(mask, self.safe_quantile)
        ys, xs = np.where(safe_region)

        if len(ys) < 1 or safe_region.sum() < MIN_SAFE_PATCH_PIXELS_FOR_SYNTHESIS:
            logger.info("Not enough crop-interior area for texture synthesis (%d px) -- falling back to reflection.", int(safe_region.sum()))
            fallback = ReflectionSourceExtender().extend(rgb, depth, mask, pad_x, pad_y)
            fallback.fallback_used = True
            return fallback

        out_h, out_w = h + 2 * pad_y, w + 2 * pad_x
        rgb_ext = np.zeros((out_h, out_w, 3), dtype=rgb.dtype)
        depth_ext = np.zeros((out_h, out_w), dtype=np.float32)
        origin_mask = np.zeros((out_h, out_w), dtype=bool)
        crop_mask_ext = np.zeros((out_h, out_w), dtype=bool)

        rgb_ext[pad_y : pad_y + h, pad_x : pad_x + w] = rgb
        depth_ext[pad_y : pad_y + h, pad_x : pad_x + w] = depth
        origin_mask[pad_y : pad_y + h, pad_x : pad_x + w] = True
        crop_mask_ext[pad_y : pad_y + h, pad_x : pad_x + w] = mask

        rng = np.random.default_rng(self.seed)
        candidate_centers = np.stack([ys, xs], axis=1)  # candidate patch-center pixels, all inside safe_region

        self._tile_border(rgb_ext, depth_ext, crop_mask_ext, rgb, depth, candidate_centers, rng, pad_x, pad_y, out_w, out_h)

        # Origin mask is left untouched by tiling (only the original region is "real");
        # everything else in the padded canvas is synthesized, whether or not a patch
        # happened to land exactly there (a not-yet-covered pixel stays at its zero-init
        # value, which downstream hole-filling / the "large unsupported" marking handles).

        return ExtendedSource(
            rgb=rgb_ext,
            depth=depth_ext,
            origin_mask=origin_mask,
            crop_mask=crop_mask_ext,
            pad_x=pad_x,
            pad_y=pad_y,
            method="texture_synthesis",
            fallback_used=False,
        )

    def _tile_border(
        self,
        rgb_ext: np.ndarray,
        depth_ext: np.ndarray,
        crop_mask_ext: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        candidate_centers: np.ndarray,
        rng: np.random.Generator,
        pad_x: int,
        pad_y: int,
        out_w: int,
        out_h: int,
    ) -> None:
        ps, ov = self.patch_size, self.overlap
        stride = max(1, ps - ov)
        h, w = rgb.shape[:2]

        # Grid of top-left corners covering the FULL padded canvas (including corners) --
        # we skip cells that fall entirely inside the already-real original region below.
        for gy in range(0, out_h, stride):
            for gx in range(0, out_w, stride):
                y1, x1 = min(gy + ps, out_h), min(gx + ps, out_w)
                cell_h, cell_w = y1 - gy, x1 - gx
                if cell_h <= 0 or cell_w <= 0:
                    continue
                # Skip cells fully inside the original (real) image -- nothing to synthesize there.
                if gx >= pad_x and x1 <= pad_x + w and gy >= pad_y and y1 <= pad_y + h:
                    continue

                idx = rng.integers(0, len(candidate_centers))
                cy, cx = candidate_centers[idx]
                sy0 = int(np.clip(cy - ps // 2, 0, h - cell_h))
                sx0 = int(np.clip(cx - ps // 2, 0, w - cell_w))
                patch_rgb = rgb[sy0 : sy0 + cell_h, sx0 : sx0 + cell_w]
                patch_depth = depth[sy0 : sy0 + cell_h, sx0 : sx0 + cell_w]

                _feather_blend(rgb_ext, patch_rgb, gx, gy, overlap=min(ov, cell_w, cell_h) if gx > 0 or gy > 0 else 0)
                depth_ext[gy:y1, gx:x1] = patch_depth
                crop_mask_ext[gy:y1, gx:x1] = True


class ReflectionSourceExtender:
    """Fallback extender: reflection-padding with the harsh mirror seam
    smoothed by a light blend/blur band, used when texture synthesis can't
    find enough valid crop-interior patches (or when
    ``SOURCE_EXTENSION_MODE="reflection"`` is set explicitly).
    """

    def __init__(self, seam_band_px: int = 10) -> None:
        self.seam_band_px = seam_band_px

    def extend(self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, pad_x: int, pad_y: int) -> ExtendedSource:
        h, w = rgb.shape[:2]
        rgb_ext = cv2.copyMakeBorder(rgb, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101)
        depth_ext = cv2.copyMakeBorder(depth, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101)
        mask_ext = cv2.copyMakeBorder(mask.astype(np.uint8), pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101) > 0

        origin_mask = np.zeros(rgb_ext.shape[:2], dtype=bool)
        origin_mask[pad_y : pad_y + h, pad_x : pad_x + w] = True

        rgb_ext = self._blur_seam(rgb_ext, origin_mask)

        return ExtendedSource(
            rgb=rgb_ext,
            depth=depth_ext,
            origin_mask=origin_mask,
            crop_mask=mask_ext,
            pad_x=pad_x,
            pad_y=pad_y,
            method="reflection",
            fallback_used=False,
        )

    def _blur_seam(self, rgb_ext: np.ndarray, origin_mask: np.ndarray) -> np.ndarray:
        boundary = origin_mask.astype(np.uint8)
        band = cv2.dilate(boundary, np.ones((self.seam_band_px * 2 + 1, self.seam_band_px * 2 + 1), np.uint8))
        band = (band > 0) & ~origin_mask.astype(bool)  # a thin ring just outside the real region
        blurred = cv2.GaussianBlur(rgb_ext, (0, 0), sigmaX=self.seam_band_px / 2)
        result = rgb_ext.copy()
        result[band] = blurred[band]
        return result
