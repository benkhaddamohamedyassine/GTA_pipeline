"""Safe, atomic image (and sidecar) I/O.

Reads apply EXIF orientation and normalize to RGB without mutating the
original source file. Writes go to a temporary ``*.part`` path and are only
renamed into place after the write completes -- so a crash mid-write never
leaves a stage output that looks complete but isn't (the resume logic in
``stages/`` relies on this).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageReadError(Exception):
    """Raised for corrupted, zero-sized, or otherwise unreadable source images."""


def read_image_rgb(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads an image, applies EXIF orientation, converts to RGB.

    Returns ``(rgb_array, metadata)`` where metadata includes the *original*
    (pre-orientation-fix, pre-any-processing) width/height, format, and the ICC
    color profile bytes if the file carried one (re-embedded on write where
    practical -- see ``atomic_write_image``).
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise ImageReadError(f"Missing or zero-sized file: {path}")

    try:
        with Image.open(path) as img:
            original_width, original_height = img.size
            original_format = img.format
            icc_profile = img.info.get("icc_profile")

            img = ImageOps.exif_transpose(img)  # bakes EXIF rotation into pixel data
            if img is None:
                raise ImageReadError(f"exif_transpose returned None for {path}")
            rgb = img.convert("RGB")
            array = np.array(rgb)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageReadError(f"Could not read image {path}: {exc}") from exc

    if array.size == 0:
        raise ImageReadError(f"Decoded to an empty array: {path}")

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "original_format": original_format,
        "icc_profile": icc_profile,
    }
    return array, metadata


def _atomic_write(path: Path, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX and Windows (same filesystem)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_image(
    array: np.ndarray,
    path: Path,
    icc_profile: Optional[bytes] = None,
    quality: int = 95,
) -> None:
    path = Path(path)

    def _write(tmp_path: Path) -> None:
        img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        save_kwargs: Dict[str, Any] = {}
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".webp"):
            save_kwargs["quality"] = quality
        if icc_profile is not None and suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif"):
            save_kwargs["icc_profile"] = icc_profile
        # PIL infers format from tmp_path's suffix (".part" broke that), so pass it explicitly.
        format_name = {"jpg": "JPEG", "jpeg": "JPEG", "tif": "TIFF"}.get(suffix.lstrip("."), suffix.lstrip(".").upper())
        img.save(tmp_path, format=format_name, **save_kwargs)

    _atomic_write(path, _write)


def atomic_write_json(data: Dict[str, Any], path: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    _atomic_write(path, _write)


def atomic_write_npy(array: np.ndarray, path: Path) -> None:
    def _write(tmp_path: Path) -> None:
        np.save(tmp_path, array)

    # np.save appends ".npy" if the target doesn't already end with it; make the
    # temp path do the same dance so the final rename target matches exactly.
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(str(tmp_path), array)
        saved_path = tmp_path if tmp_path.exists() else Path(str(tmp_path) + ".npy")
        os.replace(saved_path, path)
    finally:
        for p in (tmp_path, Path(str(tmp_path) + ".npy")):
            if p.exists():
                p.unlink(missing_ok=True)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    def _write(tmp_path: Path) -> None:
        np.savez_compressed(tmp_path, **arrays)

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(str(tmp_path), **arrays)
        saved_path = tmp_path if tmp_path.exists() else Path(str(tmp_path) + ".npz")
        os.replace(saved_path, path)
    finally:
        for p in (tmp_path, Path(str(tmp_path) + ".npz")):
            if p.exists():
                p.unlink(missing_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_complete_file(path: Path) -> bool:
    """A stage output only counts as 'complete' if it exists and isn't a
    leftover ``*.part`` file from an interrupted write."""
    path = Path(path)
    return path.exists() and path.suffix != ".part" and path.stat().st_size > 0
