"""Conservative super-resolution: sharpens/upsamples the source photo before
depth estimation and geometric warping, WITHOUT inventing content -- no
prompt-based generation, no face/anime-specific weights, no diffusion. Real-
ESRGAN's x2plus weights are trained purely for photorealistic upsampling and
denoising, not generation, which is why it's the default here instead of any
diffusion-based upscaler.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..utils import memory

logger = logging.getLogger("crop_aerial_pipeline")

# (netscale, download url) per supported model -- deliberately a short, explicit
# allowlist rather than accepting an arbitrary model name/path, so a typo'd
# SUPER_RESOLUTION_MODEL fails fast with a clear message instead of a weird
# runtime error deep inside basicsr.
_MODEL_SPECS = {
    "RealESRGAN_x2plus": {
        "netscale": 2,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "num_block": 23,
    },
    "RealESRGAN_x4plus": {
        "netscale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "num_block": 23,
    },
}


class SuperResolutionBackend(Protocol):
    def load(self) -> None: ...

    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray: ...

    def unload(self) -> None: ...


@dataclass
class SuperResolutionRunInfo:
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    effective_scale: float
    tile_size: int
    runtime_seconds: float
    device: str
    model_name: str


def compute_effective_scale(width: int, height: int, requested_scale: float, max_dimension: int) -> float:
    """Caps the requested scale so ``max(width, height) * scale`` never
    exceeds ``max_dimension`` -- large phone photos at 2x-4x can otherwise
    produce enormous images that blow the memory budget of every later stage.
    """
    longest_side = max(width, height)
    if longest_side * requested_scale <= max_dimension:
        return requested_scale
    capped = max_dimension / longest_side
    logger.info(
        "Capping super-resolution scale %.2fx -> %.2fx to keep the longest side <= %d px",
        requested_scale,
        capped,
        max_dimension,
    )
    return max(1.0, capped)


class RealESRGANBackend:
    """Real-ESRGAN backend using the official ``realesrgan``/``basicsr``
    packages. See the package README's troubleshooting section for the known
    ``torchvision.transforms.functional_tensor`` import error on newer
    torchvision versions and how to work around it.
    """

    def __init__(
        self,
        model_name: str = "RealESRGAN_x2plus",
        tile: int = 256,
        tile_pad: int = 16,
        half_precision: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if model_name not in _MODEL_SPECS:
            raise ValueError(f"Unknown SUPER_RESOLUTION_MODEL {model_name!r}; supported: {list(_MODEL_SPECS)}")
        self.model_name = model_name
        self.tile = tile
        self.tile_pad = tile_pad
        self.half_precision = half_precision
        self.device = device or ("cuda" if memory.is_cuda_available() else "cpu")
        self._upsampler = None

    def load(self) -> None:
        if self._upsampler is not None:
            return
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        spec = _MODEL_SPECS[self.model_name]
        arch = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=spec["num_block"], num_grow_ch=32, scale=spec["netscale"]
        )
        self._upsampler = RealESRGANer(
            scale=spec["netscale"],
            model_path=spec["url"],
            model=arch,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=self.half_precision and self.device == "cuda",
            device=self.device,
        )

    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray:
        self.load()
        bgr = np.ascontiguousarray(image[:, :, ::-1])  # RealESRGANer follows cv2's BGR convention

        def _run() -> np.ndarray:
            output_bgr, _ = self._upsampler.enhance(bgr, outscale=scale)
            return output_bgr

        def _on_oom(_attempt: int) -> None:
            self._upsampler.tile = max(64, self._upsampler.tile // 2)
            logger.warning("CUDA OOM in Real-ESRGAN -- reduced tile size to %d", self._upsampler.tile)

        try:
            output_bgr = memory.retry_on_cuda_oom(_run, on_oom=_on_oom, max_retries=3)
        except Exception as exc:
            if memory.is_cuda_oom_error(exc) and self.device == "cuda":
                logger.warning("Repeated CUDA OOM in Real-ESRGAN -- falling back to CPU for this image.")
                self.device = "cpu"
                self.unload()
                self.load()
                output_bgr, _ = self._upsampler.enhance(bgr, outscale=scale)
            else:
                raise

        return np.ascontiguousarray(output_bgr[:, :, ::-1])

    def unload(self) -> None:
        self._upsampler = None
        memory.clear_memory()


def run_super_resolution(
    backend: SuperResolutionBackend,
    image: np.ndarray,
    requested_scale: float,
    max_dimension: int,
    model_name: str,
    tile: int,
    device: str,
) -> tuple[np.ndarray, SuperResolutionRunInfo]:
    h, w = image.shape[:2]
    effective_scale = compute_effective_scale(w, h, requested_scale, max_dimension)

    start = time.perf_counter()
    output = backend.upscale(image, effective_scale)
    runtime = time.perf_counter() - start

    info = SuperResolutionRunInfo(
        input_width=w,
        input_height=h,
        output_width=output.shape[1],
        output_height=output.shape[0],
        effective_scale=effective_scale,
        tile_size=tile,
        runtime_seconds=runtime,
        device=device,
        model_name=model_name,
    )
    return output, info
