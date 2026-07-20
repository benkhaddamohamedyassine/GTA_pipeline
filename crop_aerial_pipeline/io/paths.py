"""Computes every path the pipeline reads from or writes to, all derived from a
single ``input_folder`` -- this is what lets ``run_pipeline(input_folder)`` be
the only required argument, and what preserves relative directory structure
(and original basenames) at every stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

STAGE_DIR_NAMES = [
    "01_validated",
    "02_super_resolution",
    "03_depth_preview",
    "04_crop_mask",
    "05_source_extended",
    "06_raw_render",
    "07_validity_mask",
    "08_filled_render",
]

TEMP_DIR_NAME = "_crop_aerial_temp"
RESULTS_DIR_NAME = "_crop_aerial_results"
LOGS_DIR_NAME = "_crop_aerial_logs"

LATEST_RUN_POINTER = "LATEST_RUN.txt"


@dataclass
class PipelinePaths:
    """All paths for one pipeline run, derived from ``input_folder``.

    ``run_id`` identifies one run's temp directory
    (``_crop_aerial_temp/<run_id>/``); resuming reuses the same run_id (see
    :func:`resolve_run_id`), while a fresh, non-resumed run gets a new one.
    """

    input_folder: Path
    run_id: str
    temp_root: Path = field(init=False)
    results_root: Path = field(init=False)
    logs_root: Path = field(init=False)
    run_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.input_folder = Path(self.input_folder).resolve()
        self.temp_root = self.input_folder / TEMP_DIR_NAME
        self.results_root = self.input_folder / RESULTS_DIR_NAME
        self.logs_root = self.input_folder / LOGS_DIR_NAME
        self.run_dir = self.temp_root / self.run_id

    def ensure_directories(self) -> None:
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for stage_name in STAGE_DIR_NAMES:
            (self.run_dir / stage_name).mkdir(parents=True, exist_ok=True)

    def relative_path(self, source_image_path: Path) -> Path:
        """The path (relative to ``input_folder``) that identifies an image
        across every stage/manifest entry -- e.g. ``farm_2/field_b.png``."""
        return Path(source_image_path).resolve().relative_to(self.input_folder)

    def stage_dir(self, stage_name: str) -> Path:
        if stage_name not in STAGE_DIR_NAMES:
            raise ValueError(f"Unknown stage_name {stage_name!r}; expected one of {STAGE_DIR_NAMES}")
        return self.run_dir / stage_name

    def stage_output_path(self, stage_name: str, relative_path: Path, ext_override: Optional[str] = None) -> Path:
        """Where a *visual* stage output for this image lives -- same relative
        subdirectory, same basename (optionally with a forced extension, e.g.
        forcing lossless PNG for a mask), under the given stage's directory.
        """
        relative_path = Path(relative_path)
        out_path = self.stage_dir(stage_name) / relative_path
        if ext_override is not None:
            out_path = out_path.with_suffix(ext_override)
        return out_path

    def sidecar_path(self, stage_name: str, relative_path: Path, suffix: str) -> Path:
        """Machine-readable sidecar next to a stage's visual output, e.g.
        ``sidecar_path("03_depth_preview", "field_a.jpg", ".depth.npy")`` ->
        ``.../03_depth_preview/field_a.depth.npy``.
        """
        relative_path = Path(relative_path)
        stem_path = self.stage_dir(stage_name) / relative_path
        return stem_path.parent / f"{stem_path.stem}{suffix}"

    def results_output_path(self, relative_path: Path) -> Path:
        return self.results_root / Path(relative_path)

    def manifest_json_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def manifest_csv_path(self) -> Path:
        return self.run_dir / "manifest.csv"

    def latest_run_pointer_path(self) -> Path:
        return self.temp_root / LATEST_RUN_POINTER


def resolve_run_id(input_folder: Path, resume: bool, new_run_id_factory) -> str:
    """Picks the run_id for this invocation. If ``resume`` is True and a
    previous run's id was recorded, reuse it (so stage outputs already on disk
    are found and skipped); otherwise mint a new one via
    ``new_run_id_factory()`` (see ``utils.logging_utils.new_run_id``).
    """
    input_folder = Path(input_folder).resolve()
    pointer_path = input_folder / TEMP_DIR_NAME / LATEST_RUN_POINTER

    if resume and pointer_path.exists():
        previous = pointer_path.read_text(encoding="utf-8").strip()
        if previous and (input_folder / TEMP_DIR_NAME / previous).is_dir():
            return previous

    run_id = new_run_id_factory()
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(run_id, encoding="utf-8")
    return run_id
