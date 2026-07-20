from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_crop_image() -> np.ndarray:
    """A small synthetic 'crop photo': a green vegetation blob roughly centered
    (but offset toward one edge) over a brown soil background -- enough for
    HSV+ExG detection and distance-transform interior selection to behave
    like they would on a real photo, without needing a real dataset.
    """
    h, w = 150, 200
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - w * 0.55) ** 2 / (w * 0.35) ** 2 + (yy - h * 0.5) ** 2 / (h * 0.35) ** 2) < 1.0
    rgb[blob] = [30, 150, 30]
    return rgb


@pytest.fixture
def synthetic_depth() -> np.ndarray:
    h, w = 150, 200
    yy, _ = np.mgrid[0:h, 0:w]
    return np.clip(1.0 - (yy / h) * 0.6, 0, 1).astype(np.float32)


@pytest.fixture
def sample_folder(tmp_path: Path) -> Path:
    """Mimics the nested example from the spec:

        crop_images/
            field_a.jpg
            farm_2/
                field_b.png
    """
    from PIL import Image

    root = tmp_path / "crop_images"
    root.mkdir()
    (root / "farm_2").mkdir()

    Image.new("RGB", (64, 48), color=(10, 120, 10)).save(root / "field_a.jpg")
    Image.new("RGB", (64, 48), color=(10, 120, 10)).save(root / "farm_2" / "field_b.png")

    # Should be ignored by discovery: hidden file, unsupported extension.
    (root / ".hidden.jpg").write_bytes(b"not a real image")
    (root / "notes.txt").write_text("not an image")

    return root
