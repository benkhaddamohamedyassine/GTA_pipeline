"""Stage 6 -- back-projects the extended RGB + depth into a 3D point cloud,
carrying per-point provenance (real vs. synthesized) and crop-mask
membership. Fully vectorized (see ``geometry.backprojection``).

This stage has no numbered visual folder of its own (the spec's 8 numbered
folders jump from ``05_source_extended`` to ``06_raw_render``, which is Stage
8's output) -- its machine-readable sidecar (``<stem>.points.npz``) is stored
under ``06_raw_render/`` alongside Stage 7's camera JSON and Stage 8's raw
render, since all three describe the same intermediate 3D/camera state that
leads up to that render.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage05_source_extension import SourceExtensionOutput
from ..config import PipelineConfig
from ..geometry.backprojection import BackprojectionResult, backproject, depth_norm_to_metric
from ..io.image_io import atomic_write_npz
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "backprojection"


def run(
    source_ext: SourceExtensionOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[BackprojectionResult]:
    points_path = paths.sidecar_path("06_raw_render", relative_path, ".points.npz")
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
            )

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        depth_m = depth_norm_to_metric(source_ext.extended.depth)
        result = backproject(
            rgb=source_ext.extended.rgb,
            depth_m=depth_m,
            K=source_ext.K_source,
            stride=config.BACKPROJECT_STRIDE,
            origin_mask=source_ext.extended.origin_mask,
            crop_mask=source_ext.extended.crop_mask,
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
            )
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[backprojection] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[backprojection] %s -- done (%d points)", relative_path, result.points.shape[0])
    return result
