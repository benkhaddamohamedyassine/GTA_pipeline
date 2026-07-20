"""Stage 6 -- re-runs crop segmentation on the AI-outpainted canvas. The
original crop mask is preserved exactly inside the protected original
region (hard-pasted, matching Stage 4/5's guarantees); newly detected crop
pixels are accepted only in the generated region. A separate origin map
tracks which crop-mask pixels came from the trusted original detection vs.
the AI-generated area -- Stage 8/9 must never use the latter for camera
placement (they don't: they gate on Stage 3's original interior selection,
not this stage's output, which exists for rendering/context only).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage03_crop_mask import CropMaskOutput
from .stage04_ai_outpaint import AIOutpaintOutput
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, atomic_write_npy
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.crop_segmentation import estimate_crop_mask

STAGE_NAME = "extended_crop_mask"


@dataclass
class ExtendedCropMaskOutput:
    mask: np.ndarray  # HxW bool, over the full extended canvas
    origin_is_original: np.ndarray  # HxW bool -- True only for pixels carried over from Stage 3's trusted mask


def run(
    original_crop_mask: CropMaskOutput,
    outpaint: AIOutpaintOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[ExtendedCropMaskOutput]:
    mask_path = paths.stage_output_path("06_extended_crop_mask", relative_path, ext_override=".png")
    mask_npy_path = paths.sidecar_path("06_extended_crop_mask", relative_path, ".mask.npy")
    origin_npy_path = paths.sidecar_path("06_extended_crop_mask", relative_path, ".mask_origin.npy")
    outputs = [mask_path, mask_npy_path, origin_npy_path]

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[extended_crop_mask] %s -- skipped (resumed)", relative_path)
        return ExtendedCropMaskOutput(mask=np.load(mask_npy_path), origin_is_original=np.load(origin_npy_path))

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        # estimate_crop_mask() already applies morphological open/close + tiny-component
        # removal internally (models/crop_segmentation.py) -- reused here for free.
        raw_mask = estimate_crop_mask(outpaint.rgb)

        h, w = original_crop_mask.mask.shape
        region = (slice(outpaint.pad_y, outpaint.pad_y + h), slice(outpaint.pad_x, outpaint.pad_x + w))

        extended_mask = raw_mask.copy()
        extended_mask[region] = original_crop_mask.mask  # hard-preserve the trusted original mask exactly

        origin_is_original = np.zeros(extended_mask.shape, dtype=bool)
        origin_is_original[region] = original_crop_mask.mask

        atomic_write_image((extended_mask.astype(np.uint8) * 255), mask_path)
        atomic_write_npy(extended_mask, mask_npy_path)
        atomic_write_npy(origin_is_original, origin_npy_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[extended_crop_mask] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[extended_crop_mask] %s -- done (%.1f%% of extended frame)", relative_path, extended_mask.mean() * 100)
    return ExtendedCropMaskOutput(mask=extended_mask, origin_is_original=origin_is_original)
