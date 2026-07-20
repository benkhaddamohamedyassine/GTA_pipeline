"""Correctness of the progressive AI-outpainting driver: the original image
must survive pixel-for-pixel, the provenance mask must exactly identify the
real vs. generated region, and generated content must actually get
produced (not silently left as the raw padding seed).
"""

from __future__ import annotations

import numpy as np

from crop_aerial_pipeline.models.outpainting import run_progressive_outpaint


class _PaintMaskedRedBackend:
    """Fake backend: paints every masked pixel bright red -- makes it trivial
    to detect whether the mask discipline (never touching true-original
    pixels) actually held."""

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def outpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        out[mask] = [255, 0, 0]
        return out


def _run(image, **overrides):
    kwargs = dict(
        backend=_PaintMaskedRedBackend(),
        image=image,
        pad_frac=0.45,
        step_frac=0.12,
        overlap_px=8,
        feather_px=4,
        max_side=2048,
        max_retries=1,
        progressive=True,
        preserve_original=True,
    )
    kwargs.update(overrides)
    return run_progressive_outpaint(**kwargs)


def test_original_region_is_pixel_for_pixel_preserved():
    image = np.full((100, 140, 3), 50, dtype=np.uint8)
    result = _run(image)

    original_block = result.rgb[result.pad_y : result.pad_y + 100, result.pad_x : result.pad_x + 140]
    assert np.array_equal(original_block, image)


def test_provenance_mask_matches_original_rectangle_exactly():
    image = np.full((80, 120, 3), 10, dtype=np.uint8)
    result = _run(image)

    expected = np.zeros(result.rgb.shape[:2], dtype=bool)
    expected[result.pad_y : result.pad_y + 80, result.pad_x : result.pad_x + 120] = True
    assert np.array_equal(result.provenance, expected)


def test_canvas_dimensions_match_requested_padding():
    h, w = 90, 130
    image = np.zeros((h, w, 3), dtype=np.uint8)
    result = _run(image, pad_frac=0.3)

    assert result.rgb.shape == (h + 2 * result.pad_y, w + 2 * result.pad_x, 3)
    assert result.pad_x == int(round(w * 0.3))
    assert result.pad_y == int(round(h * 0.3))


def test_generated_region_actually_gets_generated_content():
    image = np.full((60, 60, 3), 50, dtype=np.uint8)
    result = _run(image)

    corner = result.rgb[0, 0]
    assert not np.array_equal(corner, [50, 50, 50]), "corner still looks like the untouched reflection seed"


def test_zero_padding_is_a_no_op():
    image = np.full((40, 40, 3), 77, dtype=np.uint8)
    result = _run(image, pad_frac=0.0)

    assert result.pad_x == 0 and result.pad_y == 0
    assert result.steps_run == 0
    assert np.array_equal(result.rgb, image)
    assert result.provenance.all()


def test_non_progressive_single_shot_still_preserves_original():
    image = np.full((70, 90, 3), 33, dtype=np.uint8)
    result = _run(image, progressive=False)

    assert result.steps_run == 1
    original_block = result.rgb[result.pad_y : result.pad_y + 70, result.pad_x : result.pad_x + 90]
    assert np.array_equal(original_block, image)
