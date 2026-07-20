"""Stage 7 -- places the virtual camera above the crop *interior* (never the
padded/synthesized geometry, never the whole point cloud's centroid or
extremes). Defaults to a true vertical (nadir) view.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import mark_done, mark_failed, mark_running, mark_skipped, should_skip_stage
from .stage05_source_extension import SourceExtensionOutput
from .stage06_backprojection import BackprojectionResult
from ..config import PipelineConfig
from ..geometry.backprojection import sample_mask_at_stride
from ..geometry.crop_center import CropCenterResult, compute_interior_selector, robust_crop_center
from ..geometry.virtual_camera import VirtualCameraPose, place_camera
from ..io.image_io import atomic_write_json, read_json
from ..io.paths import PipelinePaths
from ..manifest import ImageRecord

STAGE_NAME = "camera"


@dataclass
class CameraOutput:
    pose: VirtualCameraPose
    crop_center: CropCenterResult


def run(
    backprojection: BackprojectionResult,
    source_ext: SourceExtensionOutput,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> Optional[CameraOutput]:
    camera_path = paths.sidecar_path("06_raw_render", relative_path, ".camera.json")

    if should_skip_stage(record, STAGE_NAME, [camera_path], config_hash, config.RESUME, config.OVERWRITE):
        mark_skipped(record, STAGE_NAME)
        logger.info("[camera] %s -- skipped (resumed)", relative_path)
        payload = read_json(camera_path)
        pose = VirtualCameraPose(
            eye=np.array(payload["eye"]),
            target=np.array(payload["target"]),
            extrinsic=np.array(payload["extrinsic"]),
            altitude_m=payload["altitude_m"],
            tilt_degrees=payload["tilt_degrees"],
        )
        crop_center = CropCenterResult(
            center_x=payload["crop_center"][0],
            center_z=payload["crop_center"][2],
            canopy_top_y=payload["crop_center"][1],
            interior_point_count=payload["interior_point_count"],
            fallback_used=payload.get("fallback_used", "interior"),
        )
        return CameraOutput(pose=pose, crop_center=crop_center)

    mark_running(record, STAGE_NAME)
    start = time.perf_counter()
    try:
        h, w = source_ext.extended.rgb.shape[:2]
        interior_selector = compute_interior_selector(source_ext.extended.crop_mask, config.CROP_INTERIOR_QUANTILE)
        in_crop_interior = sample_mask_at_stride(interior_selector, h, w, config.BACKPROJECT_STRIDE)

        crop_center = robust_crop_center(
            points=backprojection.points,
            is_original=backprojection.is_original,
            in_crop_interior=in_crop_interior,
            in_crop_mask=backprojection.in_crop_mask,
        )
        pose = place_camera(
            center_x=crop_center.center_x,
            center_z=crop_center.center_z,
            canopy_top_y=crop_center.canopy_top_y,
            altitude_m=config.ALTITUDE_M,
            tilt_degrees=config.TILT_DEGREES,
        )

        R = pose.extrinsic[:3, :3]
        t = pose.extrinsic[:3, 3]
        atomic_write_json(
            {
                "eye": pose.eye.tolist(),
                "target": pose.target.tolist(),
                "rotation": R.tolist(),
                "translation": t.tolist(),
                "extrinsic": pose.extrinsic.tolist(),
                "crop_center": [crop_center.center_x, crop_center.canopy_top_y, crop_center.center_z],
                "altitude_m": pose.altitude_m,
                "tilt_degrees": pose.tilt_degrees,
                "interior_point_count": crop_center.interior_point_count,
                "fallback_used": crop_center.fallback_used,
            },
            camera_path,
        )

        record.camera_eye = pose.eye.tolist()
        record.camera_target = pose.target.tolist()
        record.interior_point_count = crop_center.interior_point_count
    except Exception as exc:  # noqa: BLE001
        mark_failed(record, STAGE_NAME, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        logger.exception("[camera] %s -- FAILED", relative_path)
        return None

    mark_done(record, STAGE_NAME, time.perf_counter() - start, config_hash)
    logger.info(
        "[camera] %s -- done (center=(%.2f, %.2f, %.2f), interior_points=%d, fallback=%s)",
        relative_path,
        crop_center.center_x,
        crop_center.canopy_top_y,
        crop_center.center_z,
        crop_center.interior_point_count,
        crop_center.fallback_used,
    )
    return CameraOutput(pose=pose, crop_center=crop_center)
