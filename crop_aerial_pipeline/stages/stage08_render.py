"""Stage 8 -- solves separate virtual-camera intrinsics (auto-zoomed so the
crop interior fills ``CAMERA_FRAME_FILL`` of the frame) and renders the full
extended point cloud with the vectorized NumPy z-buffer rasterizer. No Open3D
``OffscreenRenderer``, no EGL/OpenGL, no per-point Python loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage05_source_extension import SourceExtensionOutput
from .stage06_backprojection import BackprojectionResult
from .stage07_camera import CameraOutput
from ..config import PipelineConfig
from ..geometry.crop_center import compute_interior_selector, select_interior_points
from ..geometry.backprojection import sample_mask_at_stride
from ..geometry.intrinsics import solve_virtual_intrinsics
from ..geometry.rasterizer import RenderResult, depth_buffer_preview, render_zbuffer
from ..geometry.virtual_camera import transform_points
from ..io.image_io import atomic_write_image, atomic_write_json, atomic_write_npy, read_image_rgb, read_json
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "render"


@dataclass
class RenderOutput:
    result: RenderResult
    K_virtual: np.ndarray
    virtual_hfov_deg: float


def run(
    backprojection: BackprojectionResult,
    source_ext: SourceExtensionOutput,
    camera: CameraOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[RenderOutput]:
    raw_render_path = paths.stage_output_path("06_raw_render", relative_path)
    validity_path = paths.stage_output_path("07_validity_mask", relative_path, ext_override=".png")
    synth_mask_path = paths.sidecar_path("06_raw_render", relative_path, ".synth_mask.npy")
    virtual_intrinsics_path = paths.sidecar_path("06_raw_render", relative_path, ".virtual_intrinsics.json")
    depth_preview_path = paths.sidecar_path("06_raw_render", relative_path, ".depth_buffer_preview.jpg")

    outputs = [raw_render_path, validity_path, synth_mask_path, virtual_intrinsics_path]

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[render] %s -- skipped (resumed)", relative_path)
        color_image, _ = read_image_rgb(raw_render_path)
        valid_mask_img, _ = read_image_rgb(validity_path)
        payload = read_json(virtual_intrinsics_path)
        result = RenderResult(
            color_image=color_image,
            valid_mask=valid_mask_img[:, :, 0] > 127,
            synthesized_mask=np.load(synth_mask_path),
            depth_buffer=np.full(color_image.shape[:2], np.nan, dtype=np.float32),  # not persisted; preview-only artifact
        )
        return RenderOutput(result=result, K_virtual=np.array(payload["K_virtual"]), virtual_hfov_deg=payload["hfov_deg"])

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        h, w = source_ext.extended.rgb.shape[:2]
        interior_selector = compute_interior_selector(source_ext.extended.crop_mask, config.CROP_INTERIOR_QUANTILE)
        in_crop_interior = sample_mask_at_stride(interior_selector, h, w, config.BACKPROJECT_STRIDE)
        interior_points, _ = select_interior_points(
            backprojection.points, backprojection.is_original, in_crop_interior, backprojection.in_crop_mask
        )

        interior_points_cam = transform_points(interior_points, camera.pose.extrinsic)
        original_h = record.original_height or h
        original_w = record.original_width or w
        # Render at the (super-resolved) source's own resolution -- large enough to carry
        # SR's extra detail, without the output ballooning to the padded canvas size.
        out_w = source_ext.extended.rgb.shape[1] - 2 * source_ext.extended.pad_x
        out_h = source_ext.extended.rgb.shape[0] - 2 * source_ext.extended.pad_y

        K_virtual, hfov_deg = solve_virtual_intrinsics(
            interior_points_cam,
            out_w,
            out_h,
            frame_fill=config.CAMERA_FRAME_FILL,
            min_hfov_deg=config.MIN_VIRTUAL_HFOV_DEG,
            max_hfov_deg=config.MAX_VIRTUAL_HFOV_DEG,
        )

        result = render_zbuffer(
            points=backprojection.points,
            colors=backprojection.colors,
            is_original=backprojection.is_original,
            extrinsic=camera.pose.extrinsic,
            K=K_virtual,
            out_w=out_w,
            out_h=out_h,
            splat_radius=config.SPLAT_RADIUS,
        )

        atomic_write_image(result.color_image, raw_render_path)
        atomic_write_image((result.valid_mask.astype(np.uint8) * 255), validity_path)
        atomic_write_npy(result.synthesized_mask, synth_mask_path)
        atomic_write_json({"K_virtual": K_virtual.tolist(), "hfov_deg": hfov_deg}, virtual_intrinsics_path)
        if config.SAVE_DEBUG_VISUALIZATION:
            atomic_write_image(depth_buffer_preview(result.depth_buffer, result.valid_mask), depth_preview_path)

        record.virtual_hfov_deg = hfov_deg
        record.valid_pixel_percentage = float(result.valid_mask.mean() * 100)
        record.synthesized_pixel_percentage = float(result.synthesized_mask.mean() * 100)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[render] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[render] %s -- done (hfov=%.1f deg, valid=%.1f%%, synthesized=%.1f%%)",
        relative_path,
        hfov_deg,
        record.valid_pixel_percentage,
        record.synthesized_pixel_percentage,
    )
    return RenderOutput(result=result, K_virtual=K_virtual, virtual_hfov_deg=hfov_deg)
