"""Stage 2 -- conservative super-resolution (Real-ESRGAN by default) before
depth estimation and geometric warping. When disabled, the validated image is
copied through unchanged (same dimensions) so downstream stages don't need to
special-case whether SR ran.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage01_validate import ValidatedImage
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, read_image_rgb
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.model_manager import ModelManager
from ..models.super_resolution import RealESRGANBackend, run_super_resolution

STAGE_NAME = "super_resolution"
MODEL_KEY = "super_resolution"


def _get_backend(manager: ModelManager, config: PipelineConfig) -> RealESRGANBackend:
    return manager.get(
        MODEL_KEY,
        lambda: RealESRGANBackend(
            model_name=config.SUPER_RESOLUTION_MODEL,
            tile=config.SUPER_RESOLUTION_TILE,
            tile_pad=config.SUPER_RESOLUTION_TILE_PAD,
            half_precision=config.SUPER_RESOLUTION_HALF_PRECISION,
        ),
    )


def run(
    validated: ValidatedImage,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    manager: ModelManager,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    output_path = paths.stage_output_path("02_super_resolution", relative_path)

    if should_skip_stage(record, STAGE_NAME, [output_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[super_resolution] %s -- skipped (resumed)", relative_path)
        rgb, _ = read_image_rgb(output_path)
        return rgb

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        if not config.SUPER_RESOLUTION_ENABLED:
            output = validated.rgb
            atomic_write_image(output, output_path, icc_profile=validated.icc_profile)
            record.super_resolved_width, record.super_resolved_height = output.shape[1], output.shape[0]
            record.effective_scale = 1.0
        else:
            backend = _get_backend(manager, config)
            output, info = run_super_resolution(
                backend=backend,
                image=validated.rgb,
                requested_scale=config.SUPER_RESOLUTION_SCALE,
                max_dimension=config.MAX_SUPER_RES_DIMENSION,
                model_name=config.SUPER_RESOLUTION_MODEL,
                tile=config.SUPER_RESOLUTION_TILE,
                device=manager.device,
            )
            atomic_write_image(output, output_path, icc_profile=validated.icc_profile)
            record.super_resolved_width, record.super_resolved_height = info.output_width, info.output_height
            record.effective_scale = info.effective_scale
    except Exception as exc:  # noqa: BLE001 -- any SR failure is recorded, not fatal to the batch
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[super_resolution] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[super_resolution] %s -- done (%dx%d, scale=%.2f)",
        relative_path,
        record.super_resolved_width,
        record.super_resolved_height,
        record.effective_scale,
    )
    return output
