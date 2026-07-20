"""Stage 5 -- re-estimates relative depth on the COMPLETE AI-outpainted
canvas (Stage 4's output). The extended depth map is aligned back to the
original (Stage 2) depth estimate inside the protected original region via a
robust quantile-based affine fit, then the true original depth values are
pasted back exactly (matching Stage 4's pixel-for-pixel RGB guarantee) -- the
generated border only ever uses reflection/patch-copy/constant/replication
NEVER; it's a genuine second depth pass on genuine (if AI-generated) RGB
content, just brought onto the same scale as the trusted original estimate.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage04_ai_outpaint import AIOutpaintOutput
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, atomic_write_json, atomic_write_npy
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.depth_estimation import DepthAnythingV2Estimator, visualize_depth
from ..models.model_manager import ModelManager

STAGE_NAME = "extended_depth"
MODEL_KEY = "depth"


def _robust_affine_align(source_values: np.ndarray, target_values: np.ndarray, low_q: float = 0.1, high_q: float = 0.9) -> Tuple[float, float]:
    """Solves ``target ~= a * source + b`` using a robust (quantile-pair, not
    true min/max) 2-point affine fit -- standard, cheap histogram-matching
    technique, robust to a handful of outlier pixels in either depth map.
    """
    s_lo, s_hi = np.quantile(source_values, [low_q, high_q])
    t_lo, t_hi = np.quantile(target_values, [low_q, high_q])
    if abs(s_hi - s_lo) < 1e-6:
        return 1.0, float(t_lo - s_lo)
    a = float((t_hi - t_lo) / (s_hi - s_lo))
    b = float(t_lo - a * s_lo)
    return a, b


def _smooth_transition_band(depth: np.ndarray, pad_x: int, pad_y: int, h: int, w: int, feather_px: int) -> np.ndarray:
    """Blurs only a thin band straddling the original/generated boundary, so
    the (already affine-aligned) depth doesn't show a discontinuity right at
    the seam. The protected original region itself is untouched here -- the
    caller re-pastes it afterward as a hard guarantee regardless.
    """
    if feather_px <= 0:
        return depth
    provenance = np.zeros(depth.shape, dtype=bool)
    provenance[pad_y : pad_y + h, pad_x : pad_x + w] = True
    kernel = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8)
    band = cv2.dilate((~provenance).astype(np.uint8), kernel) > 0
    band = band & ~provenance
    blurred = cv2.GaussianBlur(depth, (0, 0), sigmaX=feather_px / 3.0)
    result = depth.copy()
    result[band] = blurred[band]
    return result


def run(
    initial_depth: np.ndarray,
    outpaint: AIOutpaintOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    manager: ModelManager,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    preview_path = paths.stage_output_path("05_extended_depth_preview", relative_path)
    depth_path = paths.sidecar_path("05_extended_depth_preview", relative_path, ".extended_depth.npy")
    alignment_path = paths.sidecar_path("05_extended_depth_preview", relative_path, ".alignment.json")
    outputs = [preview_path] + ([depth_path] if config.SAVE_DEPTH_ARRAY else [])

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[extended_depth] %s -- skipped (resumed)", relative_path)
        if depth_path.exists():
            return np.load(depth_path)

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        estimator = manager.get(MODEL_KEY, lambda: DepthAnythingV2Estimator())
        extended_depth_raw = estimator.estimate(outpaint.rgb)
        if not np.isfinite(extended_depth_raw).all():
            raise ValueError("Extended depth map contains NaN/inf values")

        h, w = initial_depth.shape
        region = (slice(outpaint.pad_y, outpaint.pad_y + h), slice(outpaint.pad_x, outpaint.pad_x + w))
        a, b = _robust_affine_align(extended_depth_raw[region].ravel(), initial_depth.ravel())

        aligned = np.clip(extended_depth_raw * a + b, 0.0, 1.0).astype(np.float32)
        aligned[region] = initial_depth  # hard guarantee: original region matches Stage 2 exactly
        aligned = _smooth_transition_band(aligned, outpaint.pad_x, outpaint.pad_y, h, w, config.AI_OUTPAINT_FEATHER_PX)
        aligned[region] = initial_depth  # re-assert after smoothing touched the band just outside it

        atomic_write_image(visualize_depth(aligned), preview_path)
        if config.SAVE_DEPTH_ARRAY:
            atomic_write_npy(aligned, depth_path)
        atomic_write_json({"affine_a": a, "affine_b": b}, alignment_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[extended_depth] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[extended_depth] %s -- done (affine a=%.3f b=%.3f)", relative_path, a, b)
    return aligned
