"""Builds the per-image diagnostic panel (Stage debug visualization) and a
notebook helper to display it. Uses PIL/OpenCV only (no matplotlib) so it stays
cheap to generate for every image in a large batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THUMB_SIZE = (320, 320)
LABEL_HEIGHT = 24
GRID_COLS = 5


def _to_pil(img: np.ndarray) -> Image.Image:
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        return Image.fromarray(img, mode="L").convert("RGB")
    return Image.fromarray(img[..., :3], mode="RGB")


def _thumbnail_with_label(img: np.ndarray, label: str) -> Image.Image:
    pil_img = _to_pil(img)
    pil_img.thumbnail(THUMB_SIZE, Image.LANCZOS)

    canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), color=(24, 24, 24))
    offset_x = (THUMB_SIZE[0] - pil_img.width) // 2
    offset_y = (THUMB_SIZE[1] - pil_img.height) // 2
    canvas.paste(pil_img, (offset_x, offset_y))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, THUMB_SIZE[1] + 4), label, fill=(255, 255, 255), font=font)
    return canvas


def build_diagnostic_panel(panels: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    """``panels`` is an ordered list of (label, image) pairs -- typically the 10
    stage outputs described in the spec. Missing stages can be omitted (e.g. a
    failed image that never reached rendering); the grid just has fewer tiles.
    """
    if not panels:
        raise ValueError("build_diagnostic_panel() needs at least one (label, image) pair")

    tiles = [_thumbnail_with_label(img, label) for label, img in panels]
    cols = min(GRID_COLS, len(tiles))
    rows = (len(tiles) + cols - 1) // cols

    tile_w, tile_h = THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(0, 0, 0))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(tile, (c * tile_w, r * tile_h))

    return np.array(grid)


def save_diagnostic_panel(panels: List[Tuple[str, np.ndarray]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = build_diagnostic_panel(panels)
    Image.fromarray(grid).save(out_path, quality=90)


def show_diagnostic_panel(stem: str, results_root: Path) -> Optional[Image.Image]:
    """Notebook helper: ``show_diagnostic_panel("field_a", paths.temp_run_dir)``.
    Searches recursively (matching the source folder's nested structure) for
    ``<stem>.diagnostic.jpg`` and returns the loaded image (IPython
    auto-displays a returned PIL Image in a notebook cell)."""
    results_root = Path(results_root)
    matches = list(results_root.rglob(f"{stem}.diagnostic.jpg"))
    if not matches:
        print(f"No diagnostic panel found for stem {stem!r} under {results_root}")
        return None
    return Image.open(matches[0])
