"""AI outpainting (LaMa), used to extend the source canvas *before*
back-projection. This is what supplies real, plausible surrounding content
for what used to be empty triangular corners -- replacing the old classical
texture-synthesis/reflection source extension entirely. Runs at the source
image's own resolution; it never upscales (Stage 12's post-warp Real-ESRGAN
does that, on the finished render).

LaMa is trained for *inpainting* (filling a masked hole using surrounding
context), not literally "outpainting" -- ``run_progressive_outpaint`` below
drives it as an outpainter by growing the canvas a bit at a time and marking
each new border ring as the "hole" to fill. This is the standard technique
for getting outpainting behavior out of an inpainting model, and doing it
progressively (rather than one giant single-shot border) keeps the
known-to-unknown pixel ratio reasonable at every step, which is what LaMa
actually needs to produce plausible content instead of a blurry mess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

import cv2
import numpy as np

from ..utils import memory

logger = logging.getLogger("crop_aerial_pipeline")


class OutpaintBackend(Protocol):
    def load(self) -> None: ...

    def outpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def unload(self) -> None: ...


class LamaOutpaintBackend:
    """Wraps the ``simple-lama-inpainting`` package (a lightweight, JIT-traced
    LaMa model that auto-downloads its weights on first use). ``mask`` is a
    boolean/uint8 array where truthy pixels are the region to (re)generate.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if memory.is_cuda_available() else "cpu")
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from simple_lama_inpainting import SimpleLama

        self._model = SimpleLama(device=self.device)

    def outpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.load()
        from PIL import Image

        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray((mask.astype(np.uint8)) * 255)

        def _run() -> np.ndarray:
            result = self._model(pil_image, pil_mask)
            return np.array(result.convert("RGB"))

        try:
            return memory.retry_on_cuda_oom(_run, max_retries=2)
        except Exception as exc:
            if memory.is_cuda_oom_error(exc) and self.device == "cuda":
                logger.warning("Repeated CUDA OOM in LaMa -- falling back to CPU for this call.")
                self.device = "cpu"
                self.unload()
                self.load()
                return _run()
            raise

    def unload(self) -> None:
        self._model = None
        memory.clear_memory()


@dataclass
class OutpaintResult:
    rgb: np.ndarray  # full extended canvas, HxWx3 uint8
    provenance: np.ndarray  # HxW bool -- True = real original pixel, False = AI-generated
    pad_x: int
    pad_y: int
    steps_run: int
    downscaled: bool
    metrics: Dict[str, Any] = field(default_factory=dict)


def _downscale_for_backend(image: np.ndarray, mask: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Caps the working resolution fed to LaMa at ``max_side`` -- the result
    is upscaled back afterward, but ONLY the masked (generated) region ever
    uses that upscaled content; real pixels always come from the untouched,
    full-resolution canvas (see the caller). This is a pragmatic resolution
    cap, not true tiled inference -- ``AI_OUTPAINT_TILE_SIZE``/
    ``AI_OUTPAINT_TILE_OVERLAP`` are accepted config fields but not yet wired
    into a genuine tiled-window LaMa pass (seam-stitching tiled outpainting
    correctly is substantially more involved than this pipeline's realistic
    input sizes -- phone photos -- actually need).
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, mask, 1.0
    scale = max_side / longest
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
    return image_small, mask_small, scale


def run_progressive_outpaint(
    backend: OutpaintBackend,
    image: np.ndarray,
    pad_frac: float,
    step_frac: float,
    overlap_px: int,
    feather_px: int,
    max_side: int,
    max_retries: int,
    progressive: bool,
    preserve_original: bool,
) -> OutpaintResult:
    h, w = image.shape[:2]
    target_pad_x = int(round(w * pad_frac))
    target_pad_y = int(round(h * pad_frac))

    if target_pad_x <= 0 and target_pad_y <= 0:
        return OutpaintResult(
            rgb=image.copy(),
            provenance=np.ones((h, w), dtype=bool),
            pad_x=0,
            pad_y=0,
            steps_run=0,
            downscaled=False,
        )

    step_pad_x = max(1, int(round(w * step_frac))) if progressive else target_pad_x
    step_pad_y = max(1, int(round(h * step_frac))) if progressive else target_pad_y

    canvas = image.copy()
    # `settled`: pixels that are "done" -- either the true original, or an
    # earlier step's already-accepted generation. Only `settled == False`
    # pixels (the newest ring) are mandatory to fill each step; the overlap
    # band optionally re-touches settled-but-not-true-original pixels for a
    # smoother blend, but NEVER the true original block (enforced below).
    settled = np.ones((h, w), dtype=bool)
    offset_x = offset_y = 0
    cur_pad_x = cur_pad_y = 0
    steps_run = 0
    any_downscaled = False

    while cur_pad_x < target_pad_x or cur_pad_y < target_pad_y:
        add_x = min(step_pad_x, target_pad_x - cur_pad_x)
        add_y = min(step_pad_y, target_pad_y - cur_pad_y)

        # Reflection here is only a scratch seed so LaMa has non-degenerate pixel
        # values to look at where the mask says "regenerate this" -- every masked
        # pixel is fully overwritten by the model's output below, so this is NOT
        # the "reflection as the extension method" the spec says to avoid; the
        # actual output content comes entirely from LaMa.
        canvas = cv2.copyMakeBorder(canvas, add_y, add_y, add_x, add_x, borderType=cv2.BORDER_REFLECT_101)
        settled = cv2.copyMakeBorder(settled.astype(np.uint8), add_y, add_y, add_x, add_x, borderType=cv2.BORDER_CONSTANT, value=0) > 0
        offset_x += add_x
        offset_y += add_y

        new_ring = ~settled
        mask = new_ring.copy()

        if overlap_px > 0:
            kernel = np.ones((overlap_px * 2 + 1, overlap_px * 2 + 1), np.uint8)
            reblend_band = cv2.dilate(new_ring.astype(np.uint8), kernel) > 0
            reblend_band = reblend_band & settled
            true_original_region = np.zeros_like(settled)
            true_original_region[offset_y : offset_y + h, offset_x : offset_x + w] = True
            reblend_band = reblend_band & ~true_original_region  # NEVER touch the true original
            mask = mask | reblend_band

        work_canvas, work_mask, scale = _downscale_for_backend(canvas, mask, max_side)
        any_downscaled = any_downscaled or scale < 1.0

        filled_small = None
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                filled_small = backend.outpaint(work_canvas, work_mask)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("LaMa outpaint attempt %d/%d failed: %s", attempt + 1, max_retries + 1, exc)
        if filled_small is None:
            raise RuntimeError(f"LaMa outpainting failed after {max_retries + 1} attempts") from last_exc

        filled = (
            cv2.resize(filled_small, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            if scale < 1.0
            else filled_small
        )

        canvas[mask] = filled[mask]
        settled = settled | new_ring

        cur_pad_x += add_x
        cur_pad_y += add_y
        steps_run += 1

    if preserve_original:
        # Hard guarantee, independent of anything the mask discipline above got
        # right or wrong: the true original block is pasted back exactly.
        canvas[offset_y : offset_y + h, offset_x : offset_x + w] = image

    if feather_px > 0:
        canvas = _feather_final_seam(canvas, offset_x, offset_y, h, w, feather_px)
        if preserve_original:
            canvas[offset_y : offset_y + h, offset_x : offset_x + w] = image  # feathering must not leak into the original either

    provenance = np.zeros(canvas.shape[:2], dtype=bool)
    provenance[offset_y : offset_y + h, offset_x : offset_x + w] = True

    return OutpaintResult(
        rgb=canvas,
        provenance=provenance,
        pad_x=offset_x,
        pad_y=offset_y,
        steps_run=steps_run,
        downscaled=any_downscaled,
        metrics={"steps_run": steps_run, "downscaled": any_downscaled, "pad_x": offset_x, "pad_y": offset_y},
    )


def _feather_final_seam(canvas: np.ndarray, offset_x: int, offset_y: int, h: int, w: int, feather_px: int) -> np.ndarray:
    """Light final polish: a thin Gaussian-blended band straddling the true
    original/generated boundary, so the seam doesn't read as a hard edge even
    if LaMa's own blending left a faint one. Only pixels in that thin band are
    touched.
    """
    provenance = np.zeros(canvas.shape[:2], dtype=bool)
    provenance[offset_y : offset_y + h, offset_x : offset_x + w] = True
    kernel = np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8)
    band = cv2.dilate((~provenance).astype(np.uint8), kernel) > 0
    band = band & ~provenance  # stay on the generated side only -- never touch true original pixels
    blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=feather_px / 3.0)
    result = canvas.copy()
    result[band] = blurred[band]
    return result
