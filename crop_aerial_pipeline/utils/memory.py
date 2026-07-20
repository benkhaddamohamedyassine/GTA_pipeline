"""GPU/CPU memory hygiene helpers: freeing models between batch stages, and a
generic "retry with smaller workload" wrapper for CUDA out-of-memory errors.
"""

from __future__ import annotations

import gc
import logging
from typing import Callable, Optional, TypeVar

logger = logging.getLogger("crop_aerial_pipeline")

T = TypeVar("T")


def _torch():
    # Imported lazily so this module (and anything that only needs clear_memory's
    # no-op CPU behavior) doesn't hard-require torch at import time.
    import torch

    return torch


def is_cuda_available() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def clear_memory() -> None:
    """Run garbage collection and, if CUDA is available, empty its cache. Safe
    to call unconditionally (e.g. between every batch stage)."""
    gc.collect()
    if is_cuda_available():
        _torch().cuda.empty_cache()
        _torch().cuda.synchronize()


def cuda_memory_summary() -> str:
    if not is_cuda_available():
        return "CUDA not available"
    torch = _torch()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    return f"{free_gb:.2f} GiB free / {total_gb:.2f} GiB total"


def is_cuda_oom_error(exc: BaseException) -> bool:
    try:
        torch = _torch()
        if isinstance(exc, torch.cuda.OutOfMemoryError):  # torch >= 2.0
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def retry_on_cuda_oom(
    fn: Callable[..., T],
    *args,
    on_oom: Optional[Callable[[int], None]] = None,
    max_retries: int = 3,
    **kwargs,
) -> T:
    """Calls ``fn(*args, **kwargs)``; on a CUDA OOM error, calls ``on_oom(attempt)``
    (e.g. to shrink a tile size or switch to CPU) and retries, up to
    ``max_retries`` times. Re-raises the last error if every attempt OOMs, and
    re-raises immediately for any non-OOM exception (nothing here should mask a
    real bug as a memory issue).
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised below if not OOM
            if not is_cuda_oom_error(exc):
                raise
            last_exc = exc
            clear_memory()
            logger.warning(
                "CUDA OOM on attempt %d/%d (%s). %s",
                attempt + 1,
                max_retries + 1,
                cuda_memory_summary(),
                "Retrying with reduced workload." if attempt < max_retries else "Giving up.",
            )
            if attempt < max_retries and on_oom is not None:
                on_oom(attempt)
    assert last_exc is not None
    raise last_exc
