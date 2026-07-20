"""Lazy model loading/unloading with a single shared policy: don't keep two
large model families (super-resolution, depth) resident on the GPU at once
unless the caller explicitly wants that. This is what implements the "stage-
wise batch execution" lifecycle from the spec (load SR -> run all images ->
unload SR -> load depth -> run all images -> unload depth -> ...).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from ..utils import memory

logger = logging.getLogger("crop_aerial_pipeline")


class ModelManager:
    def __init__(self, preferred_device: Optional[str] = None) -> None:
        self.device = preferred_device or ("cuda" if memory.is_cuda_available() else "cpu")
        self._loaded: Dict[str, Any] = {}

    def get(self, name: str, loader: Callable[[], Any]) -> Any:
        """Returns the named model, constructing it via ``loader()`` the first
        time it's requested and caching it thereafter."""
        if name not in self._loaded:
            logger.info("Loading model %r on device %s (%s)", name, self.device, memory.cuda_memory_summary())
            self._loaded[name] = loader()
        return self._loaded[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def unload(self, name: str) -> None:
        if name not in self._loaded:
            return
        obj = self._loaded.pop(name)
        unload_fn = getattr(obj, "unload", None)
        if callable(unload_fn):
            try:
                unload_fn()
            except Exception:
                logger.exception("Error while unloading model %r (continuing anyway)", name)
        del obj
        memory.clear_memory()
        logger.info("Unloaded model %r (%s)", name, memory.cuda_memory_summary())

    def unload_all(self) -> None:
        for name in list(self._loaded):
            self.unload(name)
