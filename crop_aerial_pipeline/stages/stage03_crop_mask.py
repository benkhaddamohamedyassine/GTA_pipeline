"""Stage 3 -- detect the crop/vegetation mask and its distance-transform
*interior* on the ORIGINAL validated image (before any outpainting). Only
this original, non-generated interior selection may ever influence camera
X/Z, canopy-height, camera target, or virtual-camera framing -- Stage 6's
re-segmentation of the AI-outpainted canvas is used for rendering/context
only, never for that.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage01_validate import ValidatedImage
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
    validated: ValidatedImage,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
    external_mask_provider: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
) -> Optional[CropMaskOutput]:
    mask_visual_path = paths.stage_output_path("03_crop_mask", relative_path, ext_override=".png")
    mask_npy_path = paths.sidecar_path("03_crop_mask", relative_path, ".mask.npy")
    interior_visual_path = paths.stage_output_path("03_crop_interior", relative_path, ext_override=".png")
    interior_npy_path = paths.sidecar_path("03_crop_interior", relative_path, ".interior.npy")
    diagnostic_path = paths.sidecar_path("03_crop_interior", relative_path, ".center_diagnostic.jpg")

    outputs = [mask_visual_path, mask_npy_path, interior_visual_path, interior_npy_path]
    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[crop_mask] %s -- skipped (resumed)", relative_path)
        return CropMaskOutput(mask=np.load(mask_npy_path), interior_selector=np.load(interior_npy_path), used_fallback=False)

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        mask, used_fallback = get_or_estimate_crop_mask(validated.rgb, external_mask_provider)
        interior = compute_interior_selector(mask, config.CROP_INTERIOR_QUANTILE)

        atomic_write_image((mask.astype(np.uint8) * 255), mask_visual_path)
        atomic_write_npy(mask, mask_npy_path)
        atomic_write_image((interior.astype(np.uint8) * 255), interior_visual_path)
        atomic_write_npy(interior, interior_npy_path)
        if config.SAVE_DEBUG_VISUALIZATION:
            diagnostic = draw_crop_center_diagnostic(validated.rgb, mask, interior, center_pixel=None)
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
