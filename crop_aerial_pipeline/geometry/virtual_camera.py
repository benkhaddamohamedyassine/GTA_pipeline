"""Virtual camera placement: centered over the crop interior, defaulting to a
true vertical (nadir) view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """World-to-camera 4x4 extrinsic (+Z forward, +Y down in camera space).

    Since the virtual camera looks nearly straight down at zero tilt, world
    +Y is a degenerate up-hint for the cross products below; the *original*
    camera's forward axis ([0,0,1]) is used instead, which stays roughly
    perpendicular to a near-nadir view.
    """
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        up_hint = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-8)
    cam_up = np.cross(forward, right)

    R = np.stack([right, -cam_up, forward], axis=0)
    t = -R @ eye
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic


@dataclass
class VirtualCameraPose:
    eye: np.ndarray
    target: np.ndarray
    extrinsic: np.ndarray
    altitude_m: float
    tilt_degrees: float


def place_camera(center_x: float, center_z: float, canopy_top_y: float, altitude_m: float, tilt_degrees: float) -> VirtualCameraPose:
    """Places the virtual camera ``altitude_m`` above the canopy top, centered
    in X/Z over the crop-interior center. At ``tilt_degrees == 0`` (default),
    eye and target share X and Z exactly -- a true vertical view. Non-zero
    tilt angles the view toward +Z (into the scene the phone was pointed at,
    not back over the photographer), matching how a real oblique drone pass
    is typically flown.
    """
    eye = np.array([center_x, canopy_top_y + altitude_m, center_z])
    tilt_rad = math.radians(tilt_degrees)
    look_dir = np.array([0.0, -math.cos(tilt_rad), math.sin(tilt_rad)])
    target = eye + look_dir * max(altitude_m, 1.0)
    extrinsic = look_at(eye, target)
    return VirtualCameraPose(eye=eye, target=target, extrinsic=extrinsic, altitude_m=altitude_m, tilt_degrees=tilt_degrees)


def transform_points(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """World-space (N,3) points -> camera-space (N,3) via a 4x4 extrinsic."""
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.concatenate([points, ones], axis=1)
    return (extrinsic @ homogeneous.T).T[:, :3]
