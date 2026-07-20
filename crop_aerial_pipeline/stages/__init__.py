"""Thirteen small, single-responsibility stage modules (``stage01_validate``
... ``stage13_finalize``). Each exposes one ``run(...)`` function that:

1. Decides (via :func:`should_skip_stage` below) whether a valid cached output
   already exists and can be loaded instead of recomputed.
2. If not skippable, does the real work and saves its outputs atomically.
3. Records timing/status/errors onto the shared :class:`~..manifest.ImageRecord`.
4. Returns the data the *next* stage needs (not just a boolean) -- so a
   skipped stage is exactly as useful to its caller as a freshly-computed one.

This module holds the small amount of bookkeeping logic shared by all
thirteen stages, so each stage file only contains its own domain logic.

REVISED ORDER: depth/crop-mask run once on the original image (2-3), AI
outpainting extends the canvas (4), depth/crop-mask then re-run on the
*extended* image (5-6), geometry uses the extended data but centers the
camera using only original-image points (7-9), rendering + hole-filling
happen next (10-11), and only THEN does super-resolution run, on the
completed render (12) -- never on the source photo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..io.image_io import is_complete_file
from ..manifest import STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED, ImageRecord

STAGE_ORDER = [
    "validate",
    "initial_depth",
    "crop_mask",
    "ai_outpaint",
    "extended_depth",
    "extended_crop_mask",
    "backprojection",
    "camera",
    "camera_fitting",
    "render",
    "fill",
    "post_warp_super_resolution",
    "finalize",
]


def upstream_ok(record: ImageRecord, stage_name: str) -> bool:
    """True if every stage before ``stage_name`` finished (done or skipped)."""
    idx = STAGE_ORDER.index(stage_name)
    for earlier in STAGE_ORDER[:idx]:
        stage_record = record.stages.get(earlier)
        if stage_record is None or stage_record.status not in (STATUS_DONE, STATUS_SKIPPED):
            return False
    return True


def should_skip_stage(
    record: ImageRecord,
    stage_name: str,
    output_paths: Iterable[Path],
    config_hash: str,
    resume: bool,
    overwrite: bool,
) -> bool:
    """Implements the four resume checks from the spec: prior success,
    upstream success, matching config hash, and (implicitly, via the caller
    re-fingerprinting the source file before calling this) an unchanged
    source file -- callers that detect a source-file change should pass
    ``resume=False`` for that image instead of calling this at all.
    """
    if overwrite or not resume:
        return False
    stage_record = record.stages.get(stage_name)
    if stage_record is None or stage_record.status not in (STATUS_DONE, STATUS_SKIPPED):
        return False
    if not upstream_ok(record, stage_name):
        return False
    if record.config_hash != config_hash:
        return False
    return all(is_complete_file(p) for p in output_paths)


def mark_running(record: ImageRecord, stage_name: str) -> None:
    record.stage(stage_name).status = "running"


def mark_done(record: ImageRecord, stage_name: str, runtime_seconds: float, config_hash: str) -> None:
    stage_record = record.stage(stage_name)
    stage_record.status = STATUS_DONE
    stage_record.runtime_seconds = runtime_seconds
    stage_record.error = None
    record.config_hash = config_hash
    _invalidate_downstream(record, stage_name)


def _invalidate_downstream(record: ImageRecord, stage_name: str) -> None:
    """A stage that just *actually recomputed* (not skipped) may have produced
    different output than last time -- force every later stage to recompute
    too, even if its own output file + config hash still look valid, per the
    spec's "re-run downstream stages when an upstream output changes."
    """
    idx = STAGE_ORDER.index(stage_name)
    for later_stage in STAGE_ORDER[idx + 1 :]:
        if later_stage in record.stages:
            record.stages[later_stage].status = STATUS_PENDING


def mark_skipped(record: ImageRecord, stage_name: str) -> None:
    record.stage(stage_name).status = STATUS_SKIPPED


def mark_failed(record: ImageRecord, stage_name: str, error: str, runtime_seconds: float = 0.0) -> None:
    stage_record = record.stage(stage_name)
    stage_record.status = STATUS_FAILED
    stage_record.error = error
    stage_record.runtime_seconds = runtime_seconds
    record.overall_status = STATUS_FAILED
    record.error = error
