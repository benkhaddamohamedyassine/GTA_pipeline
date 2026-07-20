"""Stage 5 -- extend the source RGB + depth on every side *before*
back-projection (patch-based texture synthesis by default, reflection as an
explicit or automatic fallback), and shift the source camera's principal
point to match the larger canvas. This is what replaces the old "large
post-warp Telea inpaint" approach.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage04_crop_mask import CropMaskOutput
from ..config import PipelineConfig
from ..geometry.intrinsics import estimate_source_intrinsics, shift_principal_point
from ..geometry.source_extension import ExtendedSource, ReflectionSourceExtender, TexturePatchSourceExtender
from ..io.image_io import atomic_write_image, atomic_write_json, atomic_write_npy, read_image_rgb, read_json
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "source_extension"


@dataclass
class SourceExtensionOutput:
    extended: ExtendedSource
    K_source: np.ndarray


def _extender_for(config: PipelineConfig):
    if config.SOURCE_EXTENSION_MODE == "reflection":
        return ReflectionSourceExtender()
    return TexturePatchSourceExtender(seed=config.RANDOM_SEED)


def run(
    sr_rgb: np.ndarray,
    depth_norm: np.ndarray,
    crop_mask_out: CropMaskOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[SourceExtensionOutput]:
    rgb_out = paths.stage_output_path("05_source_extended", relative_path)
    depth_out = paths.sidecar_path("05_source_extended", relative_path, ".depth_ext.npy")
    origin_out = paths.sidecar_path("05_source_extended", relative_path, ".origin_mask.npy")
    crop_out = paths.sidecar_path("05_source_extended", relative_path, ".crop_mask_ext.npy")
    meta_out = paths.sidecar_path("05_source_extended", relative_path, ".extension_meta.json")

    outputs = [rgb_out, depth_out, origin_out, crop_out, meta_out]

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[source_extension] %s -- skipped (resumed)", relative_path)
        meta = read_json(meta_out)
        rgb_ext, _ = read_image_rgb(rgb_out)
        extended = ExtendedSource(
            rgb=rgb_ext,
            depth=np.load(depth_out),
            origin_mask=np.load(origin_out),
            crop_mask=np.load(crop_out),
            pad_x=meta["pad_x"],
            pad_y=meta["pad_y"],
            method=meta["method"],
            fallback_used=meta["fallback_used"],
        )
        K_source = np.array(meta["K_source"])
        return SourceExtensionOutput(extended=extended, K_source=K_source)

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        h, w = sr_rgb.shape[:2]
        pad_x = int(round(w * config.SOURCE_PAD_FRAC))
        pad_y = int(round(h * config.SOURCE_PAD_FRAC))

        extender = _extender_for(config)
        extended = extender.extend(sr_rgb, depth_norm, crop_mask_out.mask, pad_x, pad_y)

        if extended.fallback_used and not config.REFLECTION_FALLBACK:
            raise RuntimeError(
                "Texture-synthesis source extension failed and REFLECTION_FALLBACK=False, "
                "so no fallback is permitted."
            )

        K_source = estimate_source_intrinsics(w, h, config.ASSUMED_HFOV_DEG)
        K_source = shift_principal_point(K_source, dx=pad_x, dy=pad_y)

        atomic_write_image(extended.rgb, rgb_out)
        atomic_write_npy(extended.depth, depth_out)
        atomic_write_npy(extended.origin_mask, origin_out)
        atomic_write_npy(extended.crop_mask, crop_out)
        atomic_write_json(
            {
                "pad_x": extended.pad_x,
                "pad_y": extended.pad_y,
                "method": extended.method,
                "fallback_used": extended.fallback_used,
                "K_source": K_source.tolist(),
            },
            meta_out,
        )
        if extended.fallback_used:
            logger.warning("[source_extension] %s -- texture synthesis fell back to reflection padding", relative_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[source_extension] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[source_extension] %s -- done (method=%s)", relative_path, extended.method)
    return SourceExtensionOutput(extended=extended, K_source=K_source)
