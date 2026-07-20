"""Top-level orchestration.

``run_pipeline(input_folder)`` is the only thing most callers need. Stage-wise
batch execution keeps at most one model family resident on the GPU at a time:

    1  validate               (no model)
    2  initial_depth          depth model  -> unload
    3  crop_mask               (no model)
    4  ai_outpaint             LaMa         -> unload
    5  extended_depth          depth model  -> unload
    6-11  extended_crop_mask, backprojection, camera, camera_fitting,
          render, fill         (no model -- pure geometry/CV, per-image)
    12 post_warp_super_resolution  Real-ESRGAN -> unload
    13 finalize                (no model)

Super-resolution runs ONLY at step 12, on the completed render -- never on
the source photo (see ``models/super_resolution.py``'s module docstring).

``process_single_image`` runs all thirteen stages for exactly one image,
straight through (no stage-wise batching, since there's only one image).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .config import PipelineConfig
from .io.discovery import discover_images
from .io.paths import STAGE_DIR_NAMES, TEMP_DIR_NAME, PipelinePaths, resolve_run_id
from .manifest import STATUS_DONE, STATUS_FAILED, ImageRecord, Manifest
from .models.model_manager import ModelManager
from .stages import (
    stage01_validate,
    stage02_initial_depth,
    stage03_crop_mask,
    stage04_ai_outpaint,
    stage05_extended_depth,
    stage06_extended_crop_mask,
    stage07_backprojection,
    stage08_camera,
    stage09_camera_fitting,
    stage10_render,
    stage11_fill,
    stage12_post_warp_super_resolution,
    stage13_finalize,
)
from .utils import memory
from .utils.hashing import file_fingerprint, fingerprints_match, hash_config
from .utils.logging_utils import new_run_id, setup_logger

ImageProcessingResult = ImageRecord  # process_single_image() returns one of these


def run_pipeline(input_folder: str, config: Optional[PipelineConfig] = None) -> pd.DataFrame:
    config = config or PipelineConfig()
    config.validate()

    input_folder = Path(input_folder)
    run_id = resolve_run_id(input_folder, config.RESUME, new_run_id_factory=new_run_id)
    paths = PipelinePaths(input_folder=input_folder, run_id=run_id)
    paths.ensure_directories()

    logger = setup_logger(paths.logs_root, run_id)
    logger.info("run_pipeline starting: input_folder=%s run_id=%s recursive=%s", input_folder, run_id, config.RECURSIVE)

    manager = ModelManager()
    logger.info("Device: %s", manager.device)

    config_hash = hash_config(config)
    manifest = Manifest.load_or_create(paths.manifest_json_path()) if config.RESUME else Manifest()

    source_paths = discover_images(input_folder, recursive=config.RECURSIVE)
    logger.info("Discovered %d image(s) under %s", len(source_paths), input_folder)
    if not source_paths:
        logger.warning("No supported images found -- nothing to do.")
        return manifest.to_dataframe()

    records = _prepare_records(manifest, paths, source_paths, config)

    # --- Stage 1: validate (no model) ------------------------------------------------
    validated_by_key = _run_batch_stage(
        "Stage 1/13: validate", source_paths, records,
        lambda source_path, relative_path, record: stage01_validate.run(
            source_path, relative_path, config, paths, record, config_hash, logger
        ),
        manifest, paths,
    )

    # --- Stage 2: initial depth (depth model) -----------------------------------------
    initial_depth_by_key = _run_batch_stage(
        "Stage 2/13: initial depth", source_paths, records,
        lambda source_path, relative_path, record: stage02_initial_depth.run(
            validated_by_key[str(relative_path)], relative_path, config, paths, manager, record, config_hash, logger
        ) if str(relative_path) in validated_by_key else None,
        manifest, paths,
    )
    manager.unload(stage02_initial_depth.MODEL_KEY)
    memory.clear_memory()

    # --- Stage 3: original crop mask + interior (no model) ------------------------------
    crop_mask_by_key = _run_batch_stage(
        "Stage 3/13: crop mask", source_paths, records,
        lambda source_path, relative_path, record: stage03_crop_mask.run(
            validated_by_key[str(relative_path)], relative_path, config, paths, record, config_hash, logger
        ) if str(relative_path) in validated_by_key else None,
        manifest, paths,
    )

    # --- Stage 4: AI outpainting (LaMa) ------------------------------------------------
    outpaint_by_key = _run_batch_stage(
        "Stage 4/13: AI outpaint", source_paths, records,
        lambda source_path, relative_path, record: stage04_ai_outpaint.run(
            validated_by_key[str(relative_path)], relative_path, config, paths, manager, record, config_hash, logger
        ) if str(relative_path) in validated_by_key else None,
        manifest, paths,
    )
    manager.unload(stage04_ai_outpaint.MODEL_KEY)
    memory.clear_memory()

    # --- Stage 5: extended depth (depth model, reloaded) --------------------------------
    extended_depth_by_key = _run_batch_stage(
        "Stage 5/13: extended depth", source_paths, records,
        lambda source_path, relative_path, record: stage05_extended_depth.run(
            initial_depth_by_key[str(relative_path)], outpaint_by_key[str(relative_path)],
            relative_path, config, paths, manager, record, config_hash, logger,
        ) if str(relative_path) in initial_depth_by_key and str(relative_path) in outpaint_by_key else None,
        manifest, paths,
    )
    manager.unload(stage05_extended_depth.MODEL_KEY)
    memory.clear_memory()

    # --- Stages 6-11: geometry + fill, per-image, no persistent model ------------------
    fill_results_by_key: Dict[str, Any] = {}
    render_by_key: Dict[str, Any] = {}
    failed_keys: List[str] = []

    progress = tqdm(list(zip(source_paths, records)), desc="Stages 6-11/13: geometry + fill")
    for source_path, record in progress:
        relative_path = paths.relative_path(source_path)
        key = str(relative_path)
        progress.set_postfix(image=Path(key).name, device=manager.device)

        crop_mask = crop_mask_by_key.get(key)
        outpaint = outpaint_by_key.get(key)
        extended_depth = extended_depth_by_key.get(key)
        if crop_mask is None or outpaint is None or extended_depth is None:
            failed_keys.append(key)
            manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
            continue

        try:
            fill_result, render = _run_geometry_stages(
                crop_mask, outpaint, extended_depth, relative_path, config, paths, record, config_hash, logger
            )
            fill_results_by_key[key] = fill_result
            render_by_key[key] = render
        except Exception as exc:  # noqa: BLE001
            record.overall_status = STATUS_FAILED
            record.error = record.error or f"{type(exc).__name__}: {exc}"
            failed_keys.append(key)
            logger.error("Image failed in geometry stages: %s (%s)", key, record.error)
            if not config.CONTINUE_ON_ERROR:
                manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
                raise
        finally:
            manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())

    # --- Stage 12: post-warp super-resolution (Real-ESRGAN) -----------------------------
    sr_by_key = _run_batch_stage(
        "Stage 12/13: post-warp super-resolution", source_paths, records,
        lambda source_path, relative_path, record: stage12_post_warp_super_resolution.run(
            fill_results_by_key[str(relative_path)], render_by_key[str(relative_path)],
            relative_path, config, paths, manager, record, config_hash, logger,
        ) if str(relative_path) in fill_results_by_key else None,
        manifest, paths,
    )
    manager.unload(stage12_post_warp_super_resolution.MODEL_KEY)
    memory.clear_memory()

    # --- Stage 13: conservative finalization (no model) ---------------------------------
    progress = tqdm(list(zip(source_paths, records)), desc="Stage 13/13: finalize")
    for source_path, record in progress:
        relative_path = paths.relative_path(source_path)
        key = str(relative_path)
        if key not in sr_by_key:
            if key not in failed_keys:
                failed_keys.append(key)
            manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
            continue

        try:
            final_path = stage13_finalize.run(
                validated_by_key[key], fill_results_by_key[key], sr_by_key[key],
                relative_path, config, paths, record, config_hash, logger,
            )
            if final_path is None:
                raise RuntimeError(record.error or "finalize stage failed")
            record.overall_status = STATUS_DONE
            record.total_runtime_seconds = sum(s.runtime_seconds for s in record.stages.values())
        except Exception as exc:  # noqa: BLE001
            record.overall_status = STATUS_FAILED
            record.error = record.error or f"{type(exc).__name__}: {exc}"
            if key not in failed_keys:
                failed_keys.append(key)
            logger.error("Image failed: %s (%s)", key, record.error)
            if not config.CONTINUE_ON_ERROR:
                manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
                raise
        finally:
            manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())

    if failed_keys:
        logger.warning("Batch complete with %d failed image(s):", len(failed_keys))
        for key in failed_keys:
            logger.warning("  FAILED: %s", key)
        print(f"\n{len(failed_keys)} image(s) failed -- see the log and manifest for details:")
        for key in failed_keys:
            print(f"  FAILED: {key}")
    else:
        logger.info("Batch complete: all %d image(s) processed successfully.", len(records))

    return manifest.to_dataframe()


def process_single_image(image_path: str, config: PipelineConfig, paths: PipelinePaths) -> ImageProcessingResult:
    """Runs all thirteen stages for exactly one image, straight through (not
    stage-wise batched). Participates in the same manifest as a full
    ``run_pipeline`` run at the same ``paths``, so it's resumable/inspectable
    the same way.
    """
    config.validate()
    paths.ensure_directories()
    logger = setup_logger(paths.logs_root, paths.run_dir.name)
    config_hash = hash_config(config)

    manifest = Manifest.load_or_create(paths.manifest_json_path())
    relative_path = paths.relative_path(image_path)
    fingerprint = file_fingerprint(Path(image_path))
    record = manifest.get_or_create(relative_path)
    if not fingerprints_match(record.source_hash, fingerprint):
        record.source_hash = fingerprint

    manager = ModelManager()
    try:
        validated = stage01_validate.run(Path(image_path), relative_path, config, paths, record, config_hash, logger)
        if validated is None:
            return record

        initial_depth = stage02_initial_depth.run(validated, relative_path, config, paths, manager, record, config_hash, logger)
        manager.unload(stage02_initial_depth.MODEL_KEY)
        if initial_depth is None:
            return record

        crop_mask = stage03_crop_mask.run(validated, relative_path, config, paths, record, config_hash, logger)
        if crop_mask is None:
            return record

        outpaint = stage04_ai_outpaint.run(validated, relative_path, config, paths, manager, record, config_hash, logger)
        manager.unload(stage04_ai_outpaint.MODEL_KEY)
        if outpaint is None:
            return record

        extended_depth = stage05_extended_depth.run(
            initial_depth, outpaint, relative_path, config, paths, manager, record, config_hash, logger
        )
        manager.unload(stage05_extended_depth.MODEL_KEY)
        if extended_depth is None:
            return record

        fill_result, render = _run_geometry_stages(
            crop_mask, outpaint, extended_depth, relative_path, config, paths, record, config_hash, logger
        )

        sr_output = stage12_post_warp_super_resolution.run(
            fill_result, render, relative_path, config, paths, manager, record, config_hash, logger
        )
        manager.unload(stage12_post_warp_super_resolution.MODEL_KEY)
        if sr_output is None:
            return record

        final_path = stage13_finalize.run(
            validated, fill_result, sr_output, relative_path, config, paths, record, config_hash, logger
        )
        if final_path is None:
            raise RuntimeError(record.error or "finalize stage failed")

        record.overall_status = STATUS_DONE
        record.total_runtime_seconds = sum(s.runtime_seconds for s in record.stages.values())
    except Exception as exc:  # noqa: BLE001
        record.overall_status = STATUS_FAILED
        record.error = record.error or f"{type(exc).__name__}: {exc}"
        logger.exception("process_single_image failed for %s", relative_path)
    finally:
        manager.unload_all()
        manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())

    return record


def clean_temp_files(input_folder: str, keep_manifest: bool = True) -> None:
    """Deletes stage-output subdirectories under every run in
    ``_crop_aerial_temp/`` (never touches ``_crop_aerial_results/``). By
    default keeps each run's ``manifest.json``/``manifest.csv`` so history
    survives the cleanup; set ``keep_manifest=False`` to remove those too
    (and the now-empty run directory, if nothing else remains).
    """
    temp_root = Path(input_folder).resolve() / TEMP_DIR_NAME
    if not temp_root.exists():
        return

    for run_dir in temp_root.iterdir():
        if not run_dir.is_dir():
            continue
        for stage_name in STAGE_DIR_NAMES:
            stage_dir = run_dir / stage_name
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
        diagnostics_dir = run_dir / "diagnostics"
        if diagnostics_dir.exists():
            shutil.rmtree(diagnostics_dir)

        if not keep_manifest:
            for name in ("manifest.json", "manifest.csv"):
                candidate = run_dir / name
                if candidate.exists():
                    candidate.unlink()
            try:
                run_dir.rmdir()
            except OSError:
                pass  # not empty (unexpected extra files) -- leave it, don't force-delete


# --------------------------------------------------------------------------- internals


def _prepare_records(
    manifest: Manifest, paths: PipelinePaths, source_paths: List[Path], config: PipelineConfig
) -> List[ImageRecord]:
    records = []
    for source_path in source_paths:
        relative_path = paths.relative_path(source_path)
        fingerprint = file_fingerprint(source_path)
        record = manifest.get_or_create(relative_path)
        if not fingerprints_match(record.source_hash, fingerprint):
            record.source_hash = fingerprint
        records.append(record)
    return records


def _run_batch_stage(desc: str, source_paths, records, stage_fn, manifest: Manifest, paths: PipelinePaths) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for source_path, record in tqdm(list(zip(source_paths, records)), desc=desc):
        relative_path = paths.relative_path(source_path)
        output = stage_fn(source_path, relative_path, record)
        if output is not None:
            results[str(relative_path)] = output
        manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
    return results


def _run_geometry_stages(
    crop_mask,
    outpaint,
    extended_depth,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
):
    """Stages 6-11: re-segment the extended canvas, back-project, place +
    fit the virtual camera (using only original crop-interior points),
    render, and fill small residual holes. Raises on any stage failure so
    the caller's try/except can mark the image failed without stopping the
    batch (Stage 4's ``CONTINUE_ON_ERROR`` behavior).
    """
    extended_crop_mask = stage06_extended_crop_mask.run(
        crop_mask, outpaint, relative_path, config, paths, record, config_hash, logger
    )
    if extended_crop_mask is None:
        raise RuntimeError(record.error or "extended_crop_mask stage failed")

    backprojection = stage07_backprojection.run(
        outpaint, extended_depth, crop_mask, extended_crop_mask, relative_path, config, paths, record, config_hash, logger
    )
    if backprojection is None:
        raise RuntimeError(record.error or "backprojection stage failed")

    camera = stage08_camera.run(backprojection, relative_path, config, paths, record, config_hash, logger)
    if camera is None:
        raise RuntimeError(record.error or "camera stage failed")

    camera_fitting = stage09_camera_fitting.run(
        backprojection, camera, relative_path, config, paths, record, config_hash, logger
    )
    if camera_fitting is None:
        raise RuntimeError(record.error or "camera_fitting stage failed")

    render = stage10_render.run(
        backprojection, camera, camera_fitting, relative_path, config, paths, record, config_hash, logger
    )
    if render is None:
        raise RuntimeError(record.error or "render stage failed")

    fill_result = stage11_fill.run(render, relative_path, config, paths, record, config_hash, logger)
    if fill_result is None:
        raise RuntimeError(record.error or "fill stage failed")

    return fill_result, render
