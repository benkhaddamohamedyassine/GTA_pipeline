"""Per-image status tracking and the run-level manifest (``manifest.json`` /
``manifest.csv``) described in the spec. This is the single object every
stage reads from and writes back into -- it's what makes resume, per-image
error isolation, and the returned summary DataFrame all work off one source
of truth.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .io.image_io import atomic_write_json, read_json

STAGE_NAMES = [
    "validate",
    "super_resolution",
    "depth",
    "crop_mask",
    "source_extension",
    "backprojection",
    "camera",
    "render",
    "fill",
    "export",
]

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"  # resumed from a valid cached output
STATUS_FAILED = "failed"


@dataclass
class StageRecord:
    status: str = STATUS_PENDING
    runtime_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ImageRecord:
    relative_path: str
    filename: str
    source_hash: Dict[str, Any] = field(default_factory=dict)

    overall_status: str = STATUS_PENDING
    error: Optional[str] = None
    traceback: Optional[str] = None

    original_width: Optional[int] = None
    original_height: Optional[int] = None
    super_resolved_width: Optional[int] = None
    super_resolved_height: Optional[int] = None
    effective_scale: Optional[float] = None

    crop_mask_percentage: Optional[float] = None
    interior_point_count: Optional[int] = None
    camera_eye: Optional[List[float]] = None
    camera_target: Optional[List[float]] = None
    virtual_hfov_deg: Optional[float] = None
    valid_pixel_percentage: Optional[float] = None
    synthesized_pixel_percentage: Optional[float] = None

    total_runtime_seconds: float = 0.0
    stages: Dict[str, StageRecord] = field(default_factory=lambda: {name: StageRecord() for name in STAGE_NAMES})

    config_hash: Optional[str] = None
    final_output_path: Optional[str] = None

    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord()
        return self.stages[name]

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageRecord":
        d = dict(d)
        stages_raw = d.pop("stages", {})
        record = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        record.stages = {name: StageRecord(**payload) for name, payload in stages_raw.items()}
        return record


class Manifest:
    """In-memory table of :class:`ImageRecord`, keyed by relative path (POSIX
    string form, so it's stable across Windows/Linux and JSON-safe)."""

    def __init__(self) -> None:
        self._records: Dict[str, ImageRecord] = {}

    @staticmethod
    def _key(relative_path: Path) -> str:
        return Path(relative_path).as_posix()

    def get_or_create(self, relative_path: Path, source_hash: Optional[Dict[str, Any]] = None) -> ImageRecord:
        key = self._key(relative_path)
        if key not in self._records:
            self._records[key] = ImageRecord(
                relative_path=key,
                filename=Path(relative_path).name,
                source_hash=source_hash or {},
            )
        elif source_hash is not None:
            self._records[key].source_hash = source_hash
        return self._records[key]

    def get(self, relative_path: Path) -> Optional[ImageRecord]:
        return self._records.get(self._key(relative_path))

    def all_records(self) -> List[ImageRecord]:
        return list(self._records.values())

    def failed_records(self) -> List[ImageRecord]:
        return [r for r in self._records.values() if r.overall_status == STATUS_FAILED]

    # --- persistence -----------------------------------------------------
    def save(self, json_path: Path, csv_path: Path) -> None:
        payload = {key: record.to_dict() for key, record in self._records.items()}
        atomic_write_json(payload, json_path)
        self.to_dataframe().to_csv(csv_path, index=False)

    @classmethod
    def load(cls, json_path: Path) -> "Manifest":
        manifest = cls()
        if not Path(json_path).exists():
            return manifest
        payload = read_json(json_path)
        for key, record_dict in payload.items():
            manifest._records[key] = ImageRecord.from_dict(record_dict)
        return manifest

    @classmethod
    def load_or_create(cls, json_path: Path) -> "Manifest":
        return cls.load(json_path)

    # --- summary -----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for record in self._records.values():
            row = {
                "relative_path": record.relative_path,
                "filename": record.filename,
                "status": record.overall_status,
                "error": record.error,
                "original_width": record.original_width,
                "original_height": record.original_height,
                "super_resolved_width": record.super_resolved_width,
                "super_resolved_height": record.super_resolved_height,
                "effective_scale": record.effective_scale,
                "crop_mask_percentage": record.crop_mask_percentage,
                "interior_point_count": record.interior_point_count,
                "camera_eye": record.camera_eye,
                "camera_target": record.camera_target,
                "virtual_hfov_deg": record.virtual_hfov_deg,
                "valid_pixel_percentage": record.valid_pixel_percentage,
                "synthesized_pixel_percentage": record.synthesized_pixel_percentage,
                "total_runtime_seconds": record.total_runtime_seconds,
                "config_hash": record.config_hash,
                "final_output_path": record.final_output_path,
            }
            for stage_name, stage_record in record.stages.items():
                row[f"stage_{stage_name}_status"] = stage_record.status
                row[f"stage_{stage_name}_seconds"] = stage_record.runtime_seconds
            rows.append(row)
        if not rows:
            return pd.DataFrame(
                columns=["relative_path", "filename", "status", "error", "final_output_path"]
            )
        return pd.DataFrame(rows)
