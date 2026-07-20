"""Camera-centering correctness: the core bug this pipeline fixes (the camera
being pulled toward the visible crop boundary, or toward padded/synthesized
geometry) must not regress.
"""

from __future__ import annotations

import numpy as np

from crop_aerial_pipeline.geometry.backprojection import backproject, depth_norm_to_metric, sample_mask_at_stride
from crop_aerial_pipeline.geometry.crop_center import compute_interior_selector, robust_crop_center
from crop_aerial_pipeline.geometry.intrinsics import estimate_source_intrinsics
from crop_aerial_pipeline.geometry.virtual_camera import place_camera


def test_interior_selector_excludes_boundary_pixels(synthetic_crop_image):
    mask = (synthetic_crop_image[:, :, 1] > 100)  # the green blob
    interior = compute_interior_selector(mask, quantile=0.6)

    assert interior.sum() < mask.sum()  # strictly smaller than the full mask
    assert interior.sum() > 0

    # Every interior pixel must also be a mask pixel (interior <= mask).
    assert np.array_equal(interior & mask, interior)

    # The mask's boundary pixels (eroded-away by one step) must NOT be interior.
    import cv2

    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary = mask & ~eroded
    assert not np.any(interior & boundary)


def test_robust_center_ignores_synthesized_and_boundary_points(synthetic_crop_image, synthetic_depth):
    h, w = synthetic_crop_image.shape[:2]
    mask = synthetic_crop_image[:, :, 1] > 100
    interior = compute_interior_selector(mask, quantile=0.6)

    K = estimate_source_intrinsics(w, h, hfov_deg=70.0)
    depth_m = depth_norm_to_metric(synthetic_depth)

    # Build an origin_mask where the right THIRD of the image is "synthesized"
    # (as if it were padding), and stuff it full of extreme, obviously-wrong
    # color/position so a bug that leaks synthesized points into the center
    # calculation would be very visible in the result.
    origin_mask = np.ones((h, w), dtype=bool)
    origin_mask[:, int(w * 0.66) :] = False

    result = backproject(synthetic_crop_image, depth_m, K, stride=2, origin_mask=origin_mask, crop_mask=mask)
    in_interior = sample_mask_at_stride(interior, h, w, 2)

    center = robust_crop_center(result.points, result.is_original, in_interior, result.in_crop_mask)

    # The real crop blob is centered around x=0.55*w in pixel space, which in this
    # synthetic camera projects to a small, near-zero-ish X in world space -- the key
    # assertion is that the center is NOT dragged toward the (very different, synthetic)
    # geometry on the right third of the frame, which `is_original=False` excludes.
    only_original_center = robust_crop_center(
        result.points[result.is_original],
        np.ones(result.is_original.sum(), dtype=bool),
        in_interior[result.is_original],
        result.in_crop_mask[result.is_original],
    )
    assert np.isclose(center.center_x, only_original_center.center_x)
    assert np.isclose(center.center_z, only_original_center.center_z)


def test_camera_is_vertical_at_zero_tilt():
    pose = place_camera(center_x=3.0, center_z=-1.5, canopy_top_y=0.5, altitude_m=10.0, tilt_degrees=0.0)
    assert np.isclose(pose.eye[0], pose.target[0])
    assert np.isclose(pose.eye[2], pose.target[2])
    assert pose.eye[1] > pose.target[1]  # looking straight down


def test_camera_altitude_and_center_are_respected():
    pose = place_camera(center_x=2.0, center_z=4.0, canopy_top_y=1.0, altitude_m=8.0, tilt_degrees=0.0)
    assert np.isclose(pose.eye[0], 2.0)
    assert np.isclose(pose.eye[2], 4.0)
    assert np.isclose(pose.eye[1], 1.0 + 8.0)


def test_nonzero_tilt_still_shares_no_forced_xz_equality():
    """At nonzero tilt the eye/target X,Z are allowed to differ (only the
    TILT_DEGREES=0 default guarantees equality) -- this just documents that
    place_camera() doesn't silently clamp tilt to zero."""
    pose = place_camera(center_x=0.0, center_z=0.0, canopy_top_y=0.0, altitude_m=10.0, tilt_degrees=7.0)
    assert not np.isclose(pose.eye[2], pose.target[2])
