"""Post-warp super-resolution: sharpens/upsamples the COMPLETED pseudo-aerial
render (Stage 11's hole-filled output), never the source photo.

This is a deliberate ordering choice (see ``stages/stage12_post_warp_super_
resolution.py`` for the full reasoning): upscaling the source image before
depth estimation and back-projection added memory cost without adding real
geometric information, could amplify foliage detail into unstable monocular
depth, and risked super-resolution artifacts leaking into crop segmentation,
depth, camera placement, or point-cloud reconstruction. Running it after
warping avoids all of that while still sharpening the final image.

WITHOUT inventing content -- no prompt-based generation, no face/anime-
specific weights, no diffusion. Real-ESRGAN's x2plus weights are trained
purely for photorealistic upsampling and denoising, not generation.
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
# POST_WARP_SUPER_RESOLUTION_MODEL fails fast with a clear message instead of
# a weird runtime error deep inside basicsr.
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


def _patch_torchvision_functional_tensor() -> None:
    """Compatibility shim for a well-known ``basicsr``/``realesrgan`` breakage:
    they import ``torchvision.transforms.functional_tensor``, a module
    torchvision removed in 0.17 (its tensor-only implementations, e.g.
    ``rgb_to_grayscale``, were merged into ``torchvision.transforms.
    functional``, which still exposes the same names). Rather than requiring
    every user to pin an old torchvision or hand-patch basicsr's installed
    files after every fresh install, inject a small compatibility module into
    ``sys.modules`` the first time it's needed -- a no-op if the real module
    still exists (older torchvision) or the shim was already installed this
    session.
    """
    import sys

    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401 -- exists (older torchvision); nothing to do

        return
    except ModuleNotFoundError:
        pass

    import types

    import torchvision.transforms.functional as F

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    for name in dir(F):
        if not name.startswith("_"):
            setattr(shim, name, getattr(F, name))
    sys.modules["torchvision.transforms.functional_tensor"] = shim
    logger.info("Patched missing torchvision.transforms.functional_tensor (torchvision>=0.17 compatibility).")


class PostWarpSuperResolutionBackend(Protocol):
    def load(self) -> None: ...

    def upscale(self, image: np.ndarray, scale: float) -> np.ndarray: ...

    def unload(self) -> None: ...


@dataclass
class PostWarpSuperResolutionRunInfo:
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    requested_scale: float
    effective_scale: float
    tile_size: int
    retry_count: int
    runtime_seconds: float
    device: str
    model_name: str
    used_cpu_fallback: bool


def compute_effective_scale(width: int, height: int, requested_scale: float, max_dimension: int) -> float:
    """Caps the requested scale so ``max(width, height) * scale`` never
    exceeds ``max_dimension``, preserving aspect ratio (a single uniform
    scale factor is applied to both axes either way).
    """
    longest_side = max(width, height)
    if longest_side * requested_scale <= max_dimension:
        return requested_scale
    capped = max_dimension / longest_side
    logger.info(
        "Capping post-warp super-resolution scale %.2fx -> %.2fx to keep the longest side <= %d px",
        requested_scale,
        capped,
        max_dimension,
    )
    return max(1.0, capped)


class RealESRGANPostWarpBackend:
    """Real-ESRGAN backend using the official ``realesrgan``/``basicsr``
    packages. See the package README's troubleshooting section for the known
    ``torchvision.transforms.functional_tensor`` compatibility note (handled
    automatically by this class -- see ``_patch_torchvision_functional_tensor``).
    """

    def __init__(
        self,
        model_name: str = "RealESRGAN_x2plus",
        tile: int = 256,
        tile_pad: int = 16,
        half_precision: bool = True,
        fallback_cpu: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if model_name not in _MODEL_SPECS:
            raise ValueError(
                f"Unknown POST_WARP_SUPER_RESOLUTION_MODEL {model_name!r}; supported: {list(_MODEL_SPECS)}"
            )
        self.model_name = model_name
        self.tile = tile
        self.tile_pad = tile_pad
        self.half_precision = half_precision
        self.fallback_cpu = fallback_cpu
        self.device = device or ("cuda" if memory.is_cuda_available() else "cpu")
        self._upsampler = None
        self.last_retry_count = 0
        self.used_cpu_fallback = False

    def load(self) -> None:
        if self._upsampler is not None:
            return
        _patch_torchvision_functional_tensor()
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
        self.last_retry_count = 0
        self.used_cpu_fallback = False

        def _run() -> np.ndarray:
            output_bgr, _ = self._upsampler.enhance(bgr, outscale=scale)
            return output_bgr

        def _on_oom(_attempt: int) -> None:
            self.last_retry_count += 1
            self._upsampler.tile = max(64, self._upsampler.tile // 2)
            logger.warning("CUDA OOM in Real-ESRGAN -- reduced tile size to %d", self._upsampler.tile)

        try:
            output_bgr = memory.retry_on_cuda_oom(_run, on_oom=_on_oom, max_retries=3)
        except Exception as exc:
            if memory.is_cuda_oom_error(exc) and self.device == "cuda" and self.fallback_cpu:
                logger.warning("Repeated CUDA OOM in Real-ESRGAN -- falling back to CPU for this image.")
                self.device = "cpu"
                self.used_cpu_fallback = True
                self.unload()
                self.load()
                output_bgr, _ = self._upsampler.enhance(bgr, outscale=scale)
            else:
                raise

        return np.ascontiguousarray(output_bgr[:, :, ::-1])

    def unload(self) -> None:
        self._upsampler = None
        memory.clear_memory()


def run_post_warp_super_resolution(
    backend: PostWarpSuperResolutionBackend,
    image: np.ndarray,
    requested_scale: float,
    max_dimension: int,
    model_name: str,
    tile: int,
    device: str,
) -> tuple[np.ndarray, PostWarpSuperResolutionRunInfo]:
    h, w = image.shape[:2]
    effective_scale = compute_effective_scale(w, h, requested_scale, max_dimension)

    start = time.perf_counter()
    output = backend.upscale(image, effective_scale)
    runtime = time.perf_counter() - start

    info = PostWarpSuperResolutionRunInfo(
        input_width=w,
        input_height=h,
        output_width=output.shape[1],
        output_height=output.shape[0],
        requested_scale=requested_scale,
        effective_scale=effective_scale,
        tile_size=tile,
        retry_count=getattr(backend, "last_retry_count", 0),
        runtime_seconds=runtime,
        device=device,
        model_name=model_name,
        used_cpu_fallback=getattr(backend, "used_cpu_fallback", False),
    )
    return output, info
