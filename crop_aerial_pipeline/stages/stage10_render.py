"""Stage 10 -- renders the extended point cloud (original AND AI-generated
points) from the centered, fitted virtual camera. Fully NumPy-vectorized
z-buffer rasterizer -- no Open3D ``OffscreenRenderer``, no EGL/OpenGL, no
per-point Python loop. Produces a provenance mask recording whether each
*rendered pixel* (not source pixel -- these can differ after z-buffer
resolution) came from original or AI-generated source content.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage07_backprojection import BackprojectionResult
from .stage08_camera import CameraOutput
from .stage09_camera_fitting import CameraFittingOutput
from ..config import PipelineConfig
from ..geometry.rasterizer import RenderResult, depth_buffer_preview, render_zbuffer
from ..io.image_io import atomic_write_image, atomic_write_npy, read_image_rgb
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "render"


@dataclass
class RenderOutput:
    result: RenderResult


def run(
    backprojection: BackprojectionResult,
    camera: CameraOutput,
    camera_fitting: CameraFittingOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[RenderOutput]:
    raw_warp_path = paths.stage_output_path("10_raw_warp", relative_path)
    validity_path = paths.stage_output_path("10_validity_mask", relative_path, ext_override=".png")
    provenance_path = paths.stage_output_path("10_render_provenance", relative_path, ext_override=".png")
    depth_preview_path = paths.sidecar_path("10_raw_warp", relative_path, ".render_depth_preview.jpg")
    provenance_npy_path = paths.sidecar_path("10_render_provenance", relative_path, ".render_provenance.npy")

    outputs = [raw_warp_path, validity_path, provenance_path, provenance_npy_path]

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[render] %s -- skipped (resumed)", relative_path)
        color_image, _ = read_image_rgb(raw_warp_path)
        valid_mask_img, _ = read_image_rgb(validity_path)
        result = RenderResult(
            color_image=color_image,
            valid_mask=valid_mask_img[:, :, 0] > 127,
            synthesized_mask=~np.load(provenance_npy_path),
            depth_buffer=np.full(color_image.shape[:2], np.nan, dtype=np.float32),  # not persisted; preview-only artifact
        )
        return RenderOutput(result=result)

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        result = render_zbuffer(
            points=backprojection.points,
            colors=backprojection.colors,
            is_original=backprojection.is_original,
            extrinsic=camera.pose.extrinsic,
            K=camera_fitting.K_virtual,
            out_w=camera_fitting.out_w,
            out_h=camera_fitting.out_h,
            splat_radius=config.SPLAT_RADIUS,
        )

        atomic_write_image(result.color_image, raw_warp_path)
        atomic_write_image((result.valid_mask.astype(np.uint8) * 255), validity_path)
        render_provenance = ~result.synthesized_mask  # True = pixel came from original source content
        atomic_write_image((render_provenance.astype(np.uint8) * 255), provenance_path)
        atomic_write_npy(render_provenance, provenance_npy_path)
        if config.SAVE_DEBUG_VISUALIZATION:
            atomic_write_image(depth_buffer_preview(result.depth_buffer, result.valid_mask), depth_preview_path)

        record.valid_pixel_percentage = float(result.valid_mask.mean() * 100)
        record.synthesized_pixel_percentage = float(result.synthesized_mask.mean() * 100)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[render] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[render] %s -- done (valid=%.1f%%, from-original=%.1f%%)",
        relative_path,
        record.valid_pixel_percentage,
        100.0 - record.synthesized_pixel_percentage,
    )
    return RenderOutput(result=result)
