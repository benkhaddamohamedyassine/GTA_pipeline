"""Stage 3 -- relative depth estimation on the super-resolved image. Saves the
raw float32 array (``<stem>.depth.npy``) plus a colorized preview under the
original basename.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, atomic_write_npy
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.depth_estimation import DepthAnythingV2Estimator, visualize_depth
from ..models.model_manager import ModelManager

STAGE_NAME = "depth"
MODEL_KEY = "depth"


def run(
    sr_rgb: np.ndarray,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    manager: ModelManager,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    preview_path = paths.stage_output_path("03_depth_preview", relative_path)
    depth_path = paths.sidecar_path("03_depth_preview", relative_path, ".depth.npy")
    outputs = [preview_path] + ([depth_path] if config.SAVE_DEPTH_ARRAY else [])

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[depth] %s -- skipped (resumed)", relative_path)
        if depth_path.exists():
            return np.load(depth_path)
        # SAVE_DEPTH_ARRAY was False on the run that produced this cache -- recompute is
        # unavoidable since we never persisted the array itself.

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        estimator = manager.get(MODEL_KEY, lambda: DepthAnythingV2Estimator())
        depth_norm = estimator.estimate(sr_rgb)

        if not np.isfinite(depth_norm).all():
            raise ValueError("Depth map contains NaN/inf values")

        atomic_write_image(visualize_depth(depth_norm), preview_path)
        if config.SAVE_DEPTH_ARRAY:
            atomic_write_npy(depth_norm, depth_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[depth] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[depth] %s -- done", relative_path)
    return depth_norm
