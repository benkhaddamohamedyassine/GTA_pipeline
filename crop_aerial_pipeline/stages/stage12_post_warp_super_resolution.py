"""Stage 12 -- POST-WARP Real-ESRGAN super-resolution. Runs ONLY on Stage
11's completed, hole-filled pseudo-aerial render -- never on the source
photo, never on a render still containing large empty holes. This is the
one and only place upscaling happens in the whole pipeline; see
``models/super_resolution.py``'s module docstring for why it moved here.

Resizes the render-space diagnostic masks (validity, provenance, filled) to
the new output dimensions for preview purposes, using nearest-neighbor
interpolation (never bilinear/Lanczos) since they're binary/categorical, not
continuous -- the raw, full-resolution machine-readable masks from Stages
10-11 remain available unresized at their original render resolution.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage10_render import RenderOutput
from .stage11_fill import FillResult
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, atomic_write_json, read_image_rgb, read_json
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.model_manager import ModelManager
from ..models.super_resolution import RealESRGANPostWarpBackend, run_post_warp_super_resolution

STAGE_NAME = "post_warp_super_resolution"
MODEL_KEY = "post_warp_super_resolution"


def _resize_mask_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8) * 255, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _get_backend(manager: ModelManager, config: PipelineConfig) -> RealESRGANPostWarpBackend:
    return manager.get(
        MODEL_KEY,
        lambda: RealESRGANPostWarpBackend(
            model_name=config.POST_WARP_SUPER_RESOLUTION_MODEL,
            tile=config.POST_WARP_SUPER_RESOLUTION_TILE,
            tile_pad=config.POST_WARP_SUPER_RESOLUTION_TILE_PAD,
            half_precision=config.POST_WARP_SUPER_RESOLUTION_HALF_PRECISION,
            fallback_cpu=config.POST_WARP_SUPER_RESOLUTION_FALLBACK_CPU,
        ),
    )


def run(
    fill_result: FillResult,
    render: RenderOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    manager: ModelManager,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    sr_path = paths.stage_output_path("12_super_resolved_warp", relative_path)
    metrics_path = paths.sidecar_path("12_super_resolved_warp", relative_path, ".super_resolution_metrics.json")
    outputs = [sr_path, metrics_path]

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[post_warp_super_resolution] %s -- skipped (resumed)", relative_path)
        rgb, _ = read_image_rgb(sr_path)
        return rgb

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        if not config.POST_WARP_SUPER_RESOLUTION_ENABLED:
            output = fill_result.filled_image
            atomic_write_image(output, sr_path)
            atomic_write_json(
                {"enabled": False, "effective_scale": 1.0, "output_width": output.shape[1], "output_height": output.shape[0]},
                metrics_path,
            )
        else:
            backend = _get_backend(manager, config)
            output, info = run_post_warp_super_resolution(
                backend=backend,
                image=fill_result.filled_image,
                requested_scale=config.POST_WARP_SUPER_RESOLUTION_SCALE,
                max_dimension=config.POST_WARP_MAX_OUTPUT_DIMENSION,
                model_name=config.POST_WARP_SUPER_RESOLUTION_MODEL,
                tile=config.POST_WARP_SUPER_RESOLUTION_TILE,
                device=manager.device,
            )
            atomic_write_image(output, sr_path)
            atomic_write_json(
                {
                    "enabled": True,
                    "input_width": info.input_width,
                    "input_height": info.input_height,
                    "output_width": info.output_width,
                    "output_height": info.output_height,
                    "requested_scale": info.requested_scale,
                    "effective_scale": info.effective_scale,
                    "model_name": info.model_name,
                    "device": info.device,
                    "tile_size": info.tile_size,
                    "retry_count": info.retry_count,
                    "runtime_seconds": info.runtime_seconds,
                    "used_cpu_fallback": info.used_cpu_fallback,
                },
                metrics_path,
            )

            if config.SAVE_DEBUG_VISUALIZATION:
                out_h, out_w = output.shape[:2]
                for name, mask in (
                    ("validity", render.result.valid_mask),
                    ("provenance", ~render.result.synthesized_mask),
                    ("filled", ~fill_result.large_unsupported_mask),
                ):
                    preview_path = paths.sidecar_path("12_super_resolved_warp", relative_path, f".{name}_mask_preview.png")
                    atomic_write_image(_resize_mask_nearest(mask, out_w, out_h), preview_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[post_warp_super_resolution] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[post_warp_super_resolution] %s -- done (%dx%d)", relative_path, output.shape[1], output.shape[0])
    return output
