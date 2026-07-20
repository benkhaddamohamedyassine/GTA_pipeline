"""Stage 9 -- small, bounded residual-hole filling only. NOT a general-purpose
inpaint, and NOT SDXL/Stable Diffusion/Flux/ControlNet or any other
text-to-image model -- the source extension in Stage 5 is what's responsible
for supplying most surrounding coverage; this stage just cleans up leftover
point-cloud-subsampling gaps.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage08_render import RenderOutput
from ..config import PipelineConfig
from ..geometry.hole_filling import FillResult, crop_inward_if_needed, fill_residual_holes, large_unsupported_fraction
from ..io.image_io import atomic_write_image, atomic_write_npy, read_image_rgb
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "fill"


def run(
    render: RenderOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[FillResult]:
    filled_path = paths.stage_output_path("08_filled_render", relative_path)
    large_unsupported_path = paths.sidecar_path("08_filled_render", relative_path, ".large_unsupported_mask.npy")

    if should_skip_stage(record, STAGE_NAME, [filled_path, large_unsupported_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[fill] %s -- skipped (resumed)", relative_path)
        filled_image, _ = read_image_rgb(filled_path)
        large_unsupported = np.load(large_unsupported_path)
        return FillResult(
            filled_image=filled_image,
            filled_mask=~large_unsupported,
            newly_filled_mask=np.zeros_like(large_unsupported),  # not persisted; informational only
            large_unsupported_mask=large_unsupported,
        )

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        fill_result = fill_residual_holes(
            color_img=render.result.color_image,
            valid_mask=render.result.valid_mask,
            max_radius=config.SPLAT_RADIUS,
            smooth_filled=True,
        )

        bad_frac = large_unsupported_fraction(fill_result.large_unsupported_mask)
        final_image = fill_result.filled_image
        final_large_unsupported = fill_result.large_unsupported_mask
        if bad_frac > 0:
            cropped_image, cropped_mask, did_crop = crop_inward_if_needed(final_image, final_large_unsupported)
            if did_crop:
                logger.info("[fill] %s -- cropped inward to trim a border-heavy unsupported region", relative_path)
                final_image, final_large_unsupported = cropped_image, cropped_mask

        atomic_write_image(final_image, filled_path)
        atomic_write_npy(final_large_unsupported, large_unsupported_path)

        fill_result = FillResult(
            filled_image=final_image,
            filled_mask=fill_result.filled_mask,
            newly_filled_mask=fill_result.newly_filled_mask,
            large_unsupported_mask=final_large_unsupported,
        )
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[fill] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[fill] %s -- done (%.2f%% large-unsupported)",
        relative_path,
        large_unsupported_fraction(fill_result.large_unsupported_mask) * 100,
    )
    return fill_result
