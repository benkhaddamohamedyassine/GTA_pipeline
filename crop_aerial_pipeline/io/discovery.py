"""Finds every processable source image under ``input_folder``, recursively or
not, while ignoring the pipeline's own working directories and non-image files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}

# Names the pipeline creates beside/inside input_folder -- never treated as input,
# even if RECURSIVE=True, even if the user reruns the pipeline pointed at a
# folder that already contains a previous run's output.
PIPELINE_OWNED_DIR_NAMES: Set[str] = {
    "_crop_aerial_temp",
    "_crop_aerial_results",
    "_crop_aerial_logs",
}


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def discover_images(input_folder: Path, recursive: bool = True) -> List[Path]:
    """Returns absolute paths to every supported, non-hidden image under
    ``input_folder``. Does not open/validate the files -- that's Stage 1's job
    (this stage only has to be fast and correct about *which* files qualify).
    """
    input_folder = Path(input_folder).resolve()
    if not input_folder.is_dir():
        raise NotADirectoryError(f"input_folder does not exist or is not a directory: {input_folder}")

    iterator = input_folder.rglob("*") if recursive else input_folder.glob("*")

    found: List[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if _is_hidden(path):
            continue
        if any(part in PIPELINE_OWNED_DIR_NAMES for part in path.relative_to(input_folder).parts):
            continue
        if not is_supported_image(path):
            continue
        found.append(path)

    return sorted(found)
