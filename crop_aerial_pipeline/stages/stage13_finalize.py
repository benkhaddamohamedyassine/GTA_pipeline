"""Stage 13 -- non-generative finalization only, applied AFTER post-warp
super-resolution. Because diffusion refinement is exactly what hallucinated
crop geometry in the pipeline this replaces, every operation here is a
small, deterministic, classical image-processing step -- nothing here can
add/remove a crop row, change a field boundary, or alter the camera
viewpoint. No text prompts, no diffusion, no generative inpainting.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage01_validate import ValidatedImage
from .stage11_fill import FillResult
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, read_image_rgb
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..utils.visualization import save_diagnostic_panel

STAGE_NAME = "finalize"


def _gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    mean_per_channel = img.reshape(-1, 3).mean(axis=0)
    overall_mean = mean_per_channel.mean()
    gains = overall_mean / np.clip(mean_per_channel, 1.0, None)
    gains = np.clip(gains, 0.85, 1.15)  # conservative -- a gentle correction, not a hue shift
    return np.clip(img * gains, 0, 255).astype(np.uint8)


def _mild_local_contrast(image: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)


def _edge_preserving_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=25)


def _gentle_unsharp_mask(image: np.ndarray, amount: float = 0.4, sigma: float = 1.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = image.astype(np.float32) + amount * (image.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _weak_color_match(image: np.ndarray, reference: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Reinhard-style mean/std color transfer toward ``reference`` (the
    PRE-super-resolution warped image, Stage 11's output -- NOT the original
    ground-level photo, which no longer shares this image's viewpoint or
    content after outpainting/warping), blended at only ``strength`` so it
    corrects drift introduced by super-resolution without overriding the
    render's own color decisions.
    """
    if reference.shape[:2] != image.shape[:2]:
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    matched = image.astype(np.float32)
    ref = reference.astype(np.float32)
    for c in range(3):
        result_mean, result_std = matched[..., c].mean(), matched[..., c].std() + 1e-6
        ref_mean, ref_std = ref[..., c].mean(), ref[..., c].std() + 1e-6
        matched[..., c] = (matched[..., c] - result_mean) * (ref_std / result_std) + ref_mean

    blended = image.astype(np.float32) * (1 - strength) + np.clip(matched, 0, 255) * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def finalize_conservative(image: np.ndarray, color_reference: Optional[np.ndarray] = None) -> np.ndarray:
    """The default, always-on finalization pipeline: mild contrast, gentle
    white balance, light edge-preserving denoise, gentle unsharp mask, and
    (if ``color_reference`` is given) weak color matching against it. Every
    step is small enough that it cannot plausibly be mistaken for adding or
    removing crop structure -- these are exposure/sharpness corrections, not
    content changes.
    """
    output = _mild_local_contrast(image)
    output = _gray_world_white_balance(output)
    output = _edge_preserving_denoise(output)
    output = _gentle_unsharp_mask(output)
    if color_reference is not None:
        output = _weak_color_match(output, color_reference)
    return output


def lanczos_resize(image: np.ndarray, target_long_side: int) -> np.ndarray:
    """Optional final resize -- not called by default; exposed for callers
    who want a consistent output resolution across a batch."""
    h, w = image.shape[:2]
    scale = target_long_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return image
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _diagnostic_panels(
    validated: ValidatedImage,
    depth_preview_path: Path,
    crop_mask_path: Path,
    interior_path: Path,
    ai_outpaint_path: Path,
    extended_depth_path: Path,
    extended_crop_mask_path: Path,
    raw_warp_path: Path,
    validity_path: Path,
    filled_warp_path: Path,
    super_resolved_path: Path,
    final_image: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    panels: List[Tuple[str, np.ndarray]] = [("1. original", validated.rgb)]
    for label, path in [
        ("2. initial depth", depth_preview_path),
        ("3a. crop mask", crop_mask_path),
        ("3b. crop interior", interior_path),
        ("4. AI outpaint", ai_outpaint_path),
        ("5. extended depth", extended_depth_path),
        ("6. extended crop mask", extended_crop_mask_path),
        ("10. raw warp", raw_warp_path),
        ("10b. validity mask", validity_path),
        ("11. filled warp", filled_warp_path),
        ("12. super-resolved", super_resolved_path),
    ]:
        if path.exists():
            img, _ = read_image_rgb(path)
            panels.append((label, img))
    panels.append(("13. final output", final_image))
    return panels


def run(
    validated: ValidatedImage,
    fill_result: FillResult,
    super_resolved: np.ndarray,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[Path]:
    final_path = paths.results_output_path(relative_path)

    if should_skip_stage(record, STAGE_NAME, [final_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[finalize] %s -- skipped (resumed)", relative_path)
        record.final_output_path = str(final_path)
        return final_path

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        final_image = finalize_conservative(super_resolved, color_reference=fill_result.filled_image)
        atomic_write_image(final_image, final_path, icc_profile=validated.icc_profile)

        if config.SAVE_DEBUG_VISUALIZATION:
            diagnostic_path = paths.run_dir / "diagnostics" / relative_path
            diagnostic_path = diagnostic_path.parent / f"{diagnostic_path.stem}.diagnostic.jpg"
            panels = _diagnostic_panels(
                validated=validated,
                depth_preview_path=paths.stage_output_path("02_initial_depth_preview", relative_path),
                crop_mask_path=paths.stage_output_path("03_crop_mask", relative_path, ext_override=".png"),
                interior_path=paths.stage_output_path("03_crop_interior", relative_path, ext_override=".png"),
                ai_outpaint_path=paths.stage_output_path("04_ai_outpaint", relative_path),
                extended_depth_path=paths.stage_output_path("05_extended_depth_preview", relative_path),
                extended_crop_mask_path=paths.stage_output_path("06_extended_crop_mask", relative_path, ext_override=".png"),
                raw_warp_path=paths.stage_output_path("10_raw_warp", relative_path),
                validity_path=paths.stage_output_path("10_validity_mask", relative_path, ext_override=".png"),
                filled_warp_path=paths.stage_output_path("11_filled_warp", relative_path),
                super_resolved_path=paths.stage_output_path("12_super_resolved_warp", relative_path),
                final_image=final_image,
            )
            save_diagnostic_panel(panels, diagnostic_path)

        record.final_output_path = str(final_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[finalize] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[finalize] %s -- done -> %s", relative_path, final_path)
    return final_path
