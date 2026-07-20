"""Stage 10 -- non-generative finalization only. Because diffusion refinement
is exactly what hallucinated crop geometry in the pipeline this replaces,
every operation here is a small, deterministic, classical image-processing
step -- nothing here can add/remove a crop row, change a field boundary, or
alter the camera viewpoint. No text prompts, no diffusion, no generative
inpainting.
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
from .stage09_fill import FillResult
from ..config import PipelineConfig
from ..io.image_io import atomic_write_image, read_image_rgb
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord
from ..utils.visualization import save_diagnostic_panel

STAGE_NAME = "export"


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


def match_color_to_source(image: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Optional Reinhard-style mean/std color transfer toward the original
    source photo's color statistics. Not called by default (see ``run()``) --
    exposed for callers who want it, e.g. when SR/finalization drifted the
    overall color cast noticeably from the original.
    """
    result = image.astype(np.float32)
    src = source.astype(np.float32)
    for c in range(3):
        result_mean, result_std = result[..., c].mean(), result[..., c].std() + 1e-6
        src_mean, src_std = src[..., c].mean(), src[..., c].std() + 1e-6
        result[..., c] = (result[..., c] - result_mean) * (src_std / result_std) + src_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def lanczos_resize(image: np.ndarray, target_long_side: int) -> np.ndarray:
    """Optional final resize -- not called by default; exposed for callers
    who want a consistent output resolution across a batch."""
    h, w = image.shape[:2]
    scale = target_long_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return image
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def finalize_conservative(image: np.ndarray) -> np.ndarray:
    """The default, always-on finalization pipeline: mild contrast, gentle
    white balance, light edge-preserving denoise, gentle unsharp mask. Every
    step is small enough that it cannot plausibly be mistaken for adding or
    removing crop structure -- these are exposure/sharpness corrections, not
    content changes.
    """
    output = _mild_local_contrast(image)
    output = _gray_world_white_balance(output)
    output = _edge_preserving_denoise(output)
    output = _gentle_unsharp_mask(output)
    return output


def _diagnostic_panels(
    validated: ValidatedImage,
    sr_rgb: np.ndarray,
    depth_preview_path: Path,
    crop_mask_path: Path,
    center_diagnostic_path: Path,
    extended_path: Path,
    raw_render_path: Path,
    validity_path: Path,
    final_image: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """Builds panels 1-8 and 10 from readable image files. Panel 9
    (real-vs-synthesized) is an ``.npy`` boolean array, not an image file --
    the caller (``run()``) inserts it separately rather than passing it
    through this loop, which only knows how to read actual image files.
    """
    panels: List[Tuple[str, np.ndarray]] = [("1. original", validated.rgb), ("2. super-resolved", sr_rgb)]
    for label, path in [
        ("3. depth", depth_preview_path),
        ("4. crop mask", crop_mask_path),
        ("5. interior selection", center_diagnostic_path),
        ("6. source-extended", extended_path),
        ("7. raw render", raw_render_path),
        ("8. validity mask", validity_path),
    ]:
        if path.exists():
            img, _ = read_image_rgb(path)
            panels.append((label, img))
    panels.append(("10. final output", final_image))
    return panels


def run(
    validated: ValidatedImage,
    sr_rgb: np.ndarray,
    fill_result: FillResult,
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
        logger.info("[export] %s -- skipped (resumed)", relative_path)
        record.final_output_path = str(final_path)
        return final_path

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        final_image = finalize_conservative(fill_result.filled_image)
        atomic_write_image(final_image, final_path, icc_profile=validated.icc_profile)

        if config.SAVE_DEBUG_VISUALIZATION:
            diagnostic_path = paths.run_dir / "diagnostics" / relative_path
            diagnostic_path = diagnostic_path.parent / f"{diagnostic_path.stem}.diagnostic.jpg"
            panels = _diagnostic_panels(
                validated=validated,
                sr_rgb=sr_rgb,
                depth_preview_path=paths.stage_output_path("03_depth_preview", relative_path),
                crop_mask_path=paths.stage_output_path("04_crop_mask", relative_path, ext_override=".png"),
                center_diagnostic_path=paths.sidecar_path("04_crop_mask", relative_path, ".center_diagnostic.jpg"),
                extended_path=paths.stage_output_path("05_source_extended", relative_path),
                raw_render_path=paths.stage_output_path("06_raw_render", relative_path),
                validity_path=paths.stage_output_path("07_validity_mask", relative_path, ext_override=".png"),
                final_image=final_image,
            )
            # Panel 9 (real-vs-synthesized) is an .npy boolean array, not an image file --
            # inserted here (before the final-output panel) rather than inside
            # _diagnostic_panels(), which only reads actual image files.
            synth_npy = paths.sidecar_path("06_raw_render", relative_path, ".synth_mask.npy")
            if synth_npy.exists():
                synth_mask = np.load(synth_npy)
                panels.insert(len(panels) - 1, ("9. real-vs-synth", (synth_mask.astype(np.uint8) * 255)))
            save_diagnostic_panel(panels, diagnostic_path)

        record.final_output_path = str(final_path)
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[export] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[export] %s -- done -> %s", relative_path, final_path)
    return final_path
