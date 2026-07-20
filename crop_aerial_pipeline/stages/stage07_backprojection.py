"""Stage 7 -- back-projects the AI-outpainted RGB (Stage 4) and its aligned
extended depth (Stage 5) into a 3D point cloud. Carries per-point real-vs-
generated provenance, extended crop-mask membership, and -- critically --
ORIGINAL crop-interior membership (Stage 3's selection, embedded at the
correct offset and constant-zero everywhere else), which is the only thing
Stage 8/9 are allowed to use for camera placement/framing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage03_crop_mask import CropMaskOutput
from .stage04_ai_outpaint import AIOutpaintOutput
from .stage06_extended_crop_mask import ExtendedCropMaskOutput
from ..config import PipelineConfig
from ..geometry.backprojection import BackprojectionResult, backproject, depth_norm_to_metric
from ..io.image_io import atomic_write_npz
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "backprojection"


def run(
    outpaint: AIOutpaintOutput,
    extended_depth: np.ndarray,
    original_crop_mask: CropMaskOutput,
    extended_crop_mask: ExtendedCropMaskOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[BackprojectionResult]:
    points_path = paths.sidecar_path("07_point_cloud", relative_path, ".points.npz")
    outputs = [points_path] if config.SAVE_POINT_CLOUD else []

    if outputs and should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[backprojection] %s -- skipped (resumed)", relative_path)
        with np.load(points_path) as npz:
            return BackprojectionResult(
                points=npz["points"],
                colors=npz["colors"],
                pixel_coords=npz["pixel_coords"],
                is_original=npz["is_original"],
                in_crop_mask=npz["in_crop_mask"],
                in_crop_interior=npz["in_crop_interior"],
                is_finite=npz["is_finite"],
            )

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        h, w = original_crop_mask.mask.shape
        interior_mask = np.zeros(outpaint.rgb.shape[:2], dtype=bool)
        interior_mask[outpaint.pad_y : outpaint.pad_y + h, outpaint.pad_x : outpaint.pad_x + w] = (
            original_crop_mask.interior_selector
        )

        depth_m = depth_norm_to_metric(extended_depth)
        result = backproject(
            rgb=outpaint.rgb,
            depth_m=depth_m,
            K=outpaint.K_source,
            stride=config.BACKPROJECT_STRIDE,
            origin_mask=outpaint.provenance,
            crop_mask=extended_crop_mask.mask,
            interior_mask=interior_mask,
        )
        if result.points.shape[0] == 0:
            raise ValueError("Back-projection produced an empty point cloud")

        if config.SAVE_POINT_CLOUD:
            atomic_write_npz(
                points_path,
                points=result.points,
                colors=result.colors,
                pixel_coords=result.pixel_coords,
                is_original=result.is_original,
                in_crop_mask=result.in_crop_mask,
                in_crop_interior=result.in_crop_interior,
                is_finite=result.is_finite,
            )
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[backprojection] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[backprojection] %s -- done (%d points)", relative_path, result.points.shape[0])
    return result
