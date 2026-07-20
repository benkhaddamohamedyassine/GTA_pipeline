from __future__ import annotations

import pytest

from crop_aerial_pipeline.config import PipelineConfig


def test_default_config_is_valid():
    PipelineConfig().validate()  # must not raise


@pytest.mark.parametrize(
    "field, value",
    [
        ("SUPER_RESOLUTION_SCALE", 1.0),  # must be > 1.0
        ("SUPER_RESOLUTION_TILE", 0),
        ("ALTITUDE_M", 0.0),
        ("ALTITUDE_M", -5.0),
        ("TILT_DEGREES", 90.0),
        ("TILT_DEGREES", -1.0),
        ("SOURCE_PAD_FRAC", -0.1),
        ("SOURCE_EXTENSION_MODE", "mirror_everything"),
        ("CROP_INTERIOR_QUANTILE", 0.0),
        ("CROP_INTERIOR_QUANTILE", 1.5),
        ("CAMERA_FRAME_FILL", 0.0),
        ("BACKPROJECT_STRIDE", 0),
        ("MAX_RANSAC_POINTS", 0),
        ("USE_DIFFUSION", True),
    ],
)
def test_invalid_fields_are_rejected(field, value):
    config = PipelineConfig(**{field: value})
    with pytest.raises(ValueError):
        config.validate()


def test_min_max_hfov_ordering_enforced():
    config = PipelineConfig(MIN_VIRTUAL_HFOV_DEG=60.0, MAX_VIRTUAL_HFOV_DEG=50.0)
    with pytest.raises(ValueError):
        config.validate()


def test_config_overrides_are_independent():
    a = PipelineConfig(ALTITUDE_M=12.0)
    b = PipelineConfig()
    assert a.ALTITUDE_M == 12.0
    assert b.ALTITUDE_M == 10.0  # default untouched by `a`'s override
