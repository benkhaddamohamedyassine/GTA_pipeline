"""Stage 9 -- fits virtual-camera intrinsics (a completely separate matrix
from the source intrinsics) using ONLY the original crop-interior points,
auto-zoomed so their robust angular extent fills ``CAMERA_FRAME_FILL`` of the
output frame. AI-generated extension points may appear in the render (Stage
10), but they are never consulted here to choose the virtual focal length.

Has no numbered visual folder of its own -- its sidecar
(``<stem>.virtual_intrinsics.json``) lives under ``08_camera/``, next to
Stage 8's camera.json, since both describe the same virtual-camera state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage07_backprojection import BackprojectionResult
from .stage08_camera import CameraOutput
from ..config import PipelineConfig
from ..geometry.crop_center import select_interior_points
from ..geometry.intrinsics import solve_virtual_intrinsics
from ..geometry.virtual_camera import transform_points
from ..io.image_io import atomic_write_json, read_json
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "camera_fitting"


@dataclass
class CameraFittingOutput:
    K_virtual: np.ndarray
    virtual_hfov_deg: float
    out_w: int
    out_h: int


def run(
    backprojection: BackprojectionResult,
    camera: CameraOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[CameraFittingOutput]:
    intrinsics_path = paths.sidecar_path("08_camera", relative_path, ".virtual_intrinsics.json")

    if should_skip_stage(record, STAGE_NAME, [intrinsics_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[camera_fitting] %s -- skipped (resumed)", relative_path)
        payload = read_json(intrinsics_path)
        return CameraFittingOutput(
            K_virtual=np.array(payload["K_virtual"]),
            virtual_hfov_deg=payload["hfov_deg"],
            out_w=payload["out_w"],
            out_h=payload["out_h"],
        )

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        out_w = record.original_width
        out_h = record.original_height
        if not out_w or not out_h:
            raise ValueError("record.original_width/original_height not set -- Stage 1 must run first")

        interior_points, _ = select_interior_points(
            backprojection.points,
            backprojection.is_original,
            backprojection.in_crop_interior,
            backprojection.in_crop_mask,
            backprojection.is_finite,
        )
        interior_points_cam = transform_points(interior_points, camera.pose.extrinsic)

        K_virtual, hfov_deg = solve_virtual_intrinsics(
            interior_points_cam,
            out_w,
            out_h,
            frame_fill=config.CAMERA_FRAME_FILL,
            min_hfov_deg=config.MIN_VIRTUAL_HFOV_DEG,
            max_hfov_deg=config.MAX_VIRTUAL_HFOV_DEG,
        )

        atomic_write_json({"K_virtual": K_virtual.tolist(), "hfov_deg": hfov_deg, "out_w": out_w, "out_h": out_h}, intrinsics_path)
        record.virtual_hfov_deg = hfov_deg
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[camera_fitting] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info("[camera_fitting] %s -- done (hfov=%.1f deg)", relative_path, hfov_deg)
    return CameraFittingOutput(K_virtual=K_virtual, virtual_hfov_deg=hfov_deg, out_w=out_w, out_h=out_h)
