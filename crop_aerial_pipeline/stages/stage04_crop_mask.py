"""Stage 4 -- detect the crop/vegetation mask and its distance-transform
*interior* (the part that's actually allowed to influence camera centering
later on). Saves the mask itself plus a diagnostic panel showing the mask
boundary, the selected interior, and (once Stage 7 computes it) the 2D
projection of the calculated center.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from ..config import PipelineConfig
from ..geometry.crop_center import compute_interior_selector, draw_crop_center_diagnostic
from ..io.image_io import atomic_write_image, atomic_write_npy
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.crop_segmentation import get_or_estimate_crop_mask

STAGE_NAME = "crop_mask"


@dataclass
class CropMaskOutput:
    mask: np.ndarray
    interior_selector: np.ndarray
    used_fallback: bool


def run(
    sr_rgb: np.ndarray,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
    external_mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
) -> Optional[CropMaskOutput]:
    mask_visual_path = paths.stage_output_path("04_crop_mask", relative_path, ext_override=".png")
    mask_npy_path = paths.sidecar_path("04_crop_mask", relative_path, ".mask.npy")
    diagnostic_path = paths.sidecar_path("04_crop_mask", relative_path, ".center_diagnostic.jpg")

    if should_skip_stage(record, STAGE_NAME, [mask_visual_path, mask_npy_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[crop_mask] %s -- skipped (resumed)", relative_path)
        mask = np.load(mask_npy_path)
        interior = compute_interior_selector(mask, config.CROP_INTERIOR_QUANTILE)
        return CropMaskOutput(mask=mask, interior_selector=interior, used_fallback=False)

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        mask, used_fallback = get_or_estimate_crop_mask(sr_rgb, external_mask_provider)
        interior = compute_interior_selector(mask, config.CROP_INTERIOR_QUANTILE)

        atomic_write_image((mask.astype(np.uint8) * 255), mask_visual_path)
        atomic_write_npy(mask, mask_npy_path)
        if config.SAVE_DEBUG_VISUALIZATION:
            diagnostic = draw_crop_center_diagnostic(sr_rgb, mask, interior, center_pixel=None)
            atomic_write_image(diagnostic, diagnostic_path)

        record.crop_mask_percentage = float(mask.mean() * 100)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[crop_mask] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[crop_mask] %s -- done (%.1f%% of frame, fallback=%s)",
        relative_path,
        record.crop_mask_percentage,
        used_fallback,
    )
    return CropMaskOutput(mask=mask, interior_selector=interior, used_fallback=used_fallback)
