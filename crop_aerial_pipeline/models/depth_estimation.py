"""Monocular relative depth estimation (Depth Anything V2), encapsulated
behind a small interface so the geometry stages never touch a model directly.

IMPORTANT: this produces *relative* (affine-invariant) depth, not metric
depth. There is no physical "meters" scale here -- ``geometry.backprojection.
depth_norm_to_metric`` maps it onto a plausible metric range purely so the
back-projection math has *a* scale to work with. Treat outputs as ordinally
correct (closer things have larger normalized values), not as measurements.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np
from PIL import Image

from ..utils import memory

logger = logging.getLogger("crop_aerial_pipeline")


class DepthEstimator(Protocol):
    def load(self) -> None: ...

    def estimate(self, image: np.ndarray) -> np.ndarray: ...

    def unload(self) -> None: ...


class DepthAnythingV2Estimator:
    """``1.0`` in the returned array means *nearest* to the camera, ``0.0``
    means *farthest* -- this matches Depth Anything's native disparity-like
    convention (larger = closer), kept as-is throughout the pipeline to avoid
    an easy sign-flip bug.
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: Optional[str] = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if memory.is_cuda_available() else "cpu")
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline as hf_pipeline

        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.model_id,
            device=0 if self.device == "cuda" else -1,
        )
        self._torch = torch

    def estimate(self, image: np.ndarray) -> np.ndarray:
        self.load()
        import cv2

        pil_image = Image.fromarray(image)
        with self._torch.inference_mode():
            result = self._pipe(pil_image)

        depth_raw = np.array(result["predicted_depth"], dtype=np.float32)
        if depth_raw.shape != (image.shape[0], image.shape[1]):
            depth_raw = cv2.resize(depth_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        d_min, d_max = float(depth_raw.min()), float(depth_raw.max())
        depth_norm = (depth_raw - d_min) / max(d_max - d_min, 1e-6)

        if not np.isfinite(depth_norm).all():
            raise ValueError("Depth estimation produced NaN/inf values -- source image may be degenerate")

        return depth_norm.astype(np.float32)

    def unload(self) -> None:
        self._pipe = None
        memory.clear_memory()


def visualize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """Colorized [0,1] depth map for the Stage 3 preview output."""
    import matplotlib.cm as cm

    colored = (cm.get_cmap("inferno")(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored
