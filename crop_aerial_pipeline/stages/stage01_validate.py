"""Stage 1 -- read the source image safely, apply EXIF orientation, convert to
RGB, and save a validated copy under ``01_validated`` using the original
basename. Never modifies the source file itself.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from ..config import PipelineConfig
from ..io.image_io import ImageReadError, atomic_write_image, read_image_rgb
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "validate"


@dataclass
class ValidatedImage:
    rgb: np.ndarray
    icc_profile: Optional[bytes]
    original_width: int
    original_height: int


def run(
    source_path: Path,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[ValidatedImage]:
    output_path = paths.stage_output_path("01_validated", relative_path)

    if should_skip_stage(record, STAGE_NAME, [output_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[validate] %s -- skipped (resumed)", relative_path)
        rgb, meta = read_image_rgb(output_path)
        return ValidatedImage(
            rgb=rgb,
            icc_profile=meta["icc_profile"],
            original_width=record.original_width or meta["original_width"],
            original_height=record.original_height or meta["original_height"],
        )

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        rgb, meta = read_image_rgb(source_path)
        atomic_write_image(rgb, output_path, icc_profile=meta["icc_profile"])
    except ImageReadError as exc:
        mark_failed(record, STAGE_NAME, str(exc), time.perf_counter() - start)
        logger.error("[validate] %s -- FAILED: %s", relative_path, exc)
        return None

    record.original_width = meta["original_width"]
    record.original_height = meta["original_height"]
    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[validate] %s -- done (%dx%d)", relative_path, meta["original_width"], meta["original_height"])

    return ValidatedImage(
        rgb=rgb,
        icc_profile=meta["icc_profile"],
        original_width=meta["original_width"],
        original_height=meta["original_height"],
    )
