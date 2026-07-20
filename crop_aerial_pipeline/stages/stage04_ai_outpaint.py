"""Stage 4 -- AI outpainting (LaMa). Extends the RGB canvas before geometric
back-projection so the renderer has real, plausible surrounding content --
replacing the old classical texture-synthesis/reflection source extension
entirely. The original central image is preserved pixel-for-pixel; AI-
generated pixels are tracked in a provenance map and must never influence
camera positioning (enforced downstream in Stage 8/9, which only ever see
points gated by this provenance mask).

NOTE on "pixel-for-pixel" across a RESUME: within one uninterrupted run this
is exact -- ``run_progressive_outpaint`` hard-pastes ``validated.rgb`` back
into the canvas in memory, no exceptions. On a *resumed* run this stage
reloads its output from disk, and per the spec's "preserve the original
image format" requirement, that file is saved using the source's own
extension -- for a lossy source format (JPEG/WebP) that means a JPEG
round-trip, which is not bit-exact (typically a few intensity levels near
sharp edges, from normal DCT block quantization). This never affects
geometry: depth (Stage 2/5) and the provenance/crop masks are all stored
losslessly (``.npy``/PNG) regardless of the source format, and camera
placement only ever reads those, never raw RGB values. If bit-exact RGB
across a resume matters for your use case, use a lossless source format
(PNG/TIFF).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage01_validate import ValidatedImage
from ..config import PipelineConfig
from ..geometry.intrinsics import estimate_source_intrinsics, shift_principal_point
from ..io.image_io import atomic_write_image, atomic_write_json, atomic_write_npy, read_image_rgb, read_json
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..models.model_manager import ModelManager
from ..models.outpainting import LamaOutpaintBackend, run_progressive_outpaint

STAGE_NAME = "ai_outpaint"
MODEL_KEY = "outpaint"


@dataclass
class AIOutpaintOutput:
    rgb: np.ndarray
    provenance: np.ndarray  # HxW bool -- True = real original pixel
    pad_x: int
    pad_y: int
    K_source: np.ndarray  # source intrinsics, principal point shifted for the extended canvas


def _get_backend(manager: ModelManager, config: PipelineConfig):
    if config.AI_OUTPAINT_BACKEND != "lama":
        raise ValueError(f"Unsupported AI_OUTPAINT_BACKEND {config.AI_OUTPAINT_BACKEND!r}")
    return manager.get(MODEL_KEY, lambda: LamaOutpaintBackend())


def run(
    validated: ValidatedImage,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    manager: ModelManager,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[AIOutpaintOutput]:
    rgb_path = paths.stage_output_path("04_ai_outpaint", relative_path)
    mask_path = paths.stage_output_path("04_ai_outpaint_mask", relative_path, ext_override=".png")
    provenance_visual_path = paths.stage_output_path("04_ai_outpaint_provenance", relative_path, ext_override=".png")
    provenance_npy_path = paths.sidecar_path("04_ai_outpaint_provenance", relative_path, ".outpaint_provenance.npy")
    metrics_path = paths.sidecar_path("04_ai_outpaint", relative_path, ".outpaint_metrics.json")

    outputs = [rgb_path, mask_path, provenance_visual_path, provenance_npy_path, metrics_path]

    if not config.AI_OUTPAINT_ENABLED:
        # Disabled: pass the validated image through unchanged, with everything
        # marked as original (no canvas extension at all).
        if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
            mark_skipped(record, STAGE_NAME)
            logger.info("[ai_outpaint] %s -- skipped (resumed, disabled)", relative_path)
            metrics = read_json(metrics_path)
            return AIOutpaintOutput(
                rgb=validated.rgb, provenance=np.load(provenance_npy_path), pad_x=0, pad_y=0,
                K_source=np.array(metrics["K_source"]),
            )
        mark_running(record, STAGE_NAME)
        start = time.perf_counter()
        provenance = np.ones(validated.rgb.shape[:2], dtype=bool)
        K_source = estimate_source_intrinsics(validated.rgb.shape[1], validated.rgb.shape[0], config.ASSUMED_HFOV_DEG)
        atomic_write_image(validated.rgb, rgb_path)
        atomic_write_image(np.zeros(validated.rgb.shape[:2], dtype=np.uint8), mask_path)
        atomic_write_image((provenance.astype(np.uint8) * 255), provenance_visual_path)
        atomic_write_npy(provenance, provenance_npy_path)
        atomic_write_json({"pad_x": 0, "pad_y": 0, "steps_run": 0, "K_source": K_source.tolist()}, metrics_path)
        mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
        logger.info("[ai_outpaint] %s -- done (disabled, passthrough)", relative_path)
        return AIOutpaintOutput(rgb=validated.rgb, provenance=provenance, pad_x=0, pad_y=0, K_source=K_source)

    if should_skip_stage(record, STAGE_NAME, outputs, config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[ai_outpaint] %s -- skipped (resumed)", relative_path)
        rgb, _ = read_image_rgb(rgb_path)
        metrics = read_json(metrics_path)
        return AIOutpaintOutput(
            rgb=rgb,
            provenance=np.load(provenance_npy_path),
            pad_x=metrics["pad_x"],
            pad_y=metrics["pad_y"],
            K_source=np.array(metrics["K_source"]),
        )

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        backend = _get_backend(manager, config)
        result = run_progressive_outpaint(
            backend=backend,
            image=validated.rgb,
            pad_frac=config.AI_OUTPAINT_PAD_FRAC,
            step_frac=config.AI_OUTPAINT_STEP_FRAC,
            overlap_px=config.AI_OUTPAINT_OVERLAP_PX,
            feather_px=config.AI_OUTPAINT_FEATHER_PX,
            max_side=config.AI_OUTPAINT_MAX_SIDE,
            max_retries=config.AI_OUTPAINT_MAX_RETRIES,
            progressive=config.AI_OUTPAINT_PROGRESSIVE,
            preserve_original=config.AI_OUTPAINT_PRESERVE_ORIGINAL,
        )

        K_source = estimate_source_intrinsics(validated.rgb.shape[1], validated.rgb.shape[0], config.ASSUMED_HFOV_DEG)
        K_source = shift_principal_point(K_source, dx=result.pad_x, dy=result.pad_y)

        atomic_write_image(result.rgb, rgb_path)
        atomic_write_image(((~result.provenance).astype(np.uint8) * 255), mask_path)  # white = AI-generated
        atomic_write_image((result.provenance.astype(np.uint8) * 255), provenance_visual_path)  # white = original
        atomic_write_npy(result.provenance, provenance_npy_path)
        atomic_write_json(
            {
                "pad_x": result.pad_x,
                "pad_y": result.pad_y,
                "steps_run": result.steps_run,
                "downscaled": result.downscaled,
                "K_source": K_source.tolist(),
            },
            metrics_path,
        )
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[ai_outpaint] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[ai_outpaint] %s -- done (pad=%d,%d px, %d steps, downscaled=%s)",
        relative_path, result.pad_x, result.pad_y, result.steps_run, result.downscaled,
    )
    return AIOutpaintOutput(rgb=result.rgb, provenance=result.provenance, pad_x=result.pad_x, pad_y=result.pad_y, K_source=K_source)
