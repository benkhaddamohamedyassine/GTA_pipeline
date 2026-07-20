"""Top-level orchestration.

``run_pipeline(input_folder)`` is the only thing most callers need: it
discovers images, runs Stages 1-3 batch-wise (one model family resident on
the GPU at a time), runs Stages 4-10 per-image (pure geometry/CV, no
persistent model needed), keeps a resumable manifest up to date after every
image, and returns a pandas summary.

``process_single_image`` runs all ten stages for exactly one image -- handy
for debugging a single file without a full batch run.
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
    stage02_super_resolution,
    stage03_depth,
    stage04_crop_mask,
    stage05_source_extension,
    stage06_backprojection,
    stage07_camera,
    stage08_render,
    stage09_fill,
    stage10_export,
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
    logger.info("Device: %s", ModelManager().device)

    config_hash = hash_config(config)
    manifest = Manifest.load_or_create(paths.manifest_json_path()) if config.RESUME else Manifest()

    source_paths = discover_images(input_folder, recursive=config.RECURSIVE)
    logger.info("Discovered %d image(s) under %s", len(source_paths), input_folder)
    if not source_paths:
        logger.warning("No supported images found -- nothing to do.")
        return manifest.to_dataframe()

    manager = ModelManager()
    records = _prepare_records(manifest, paths, source_paths, config)

    validated_by_key = _run_batch_stage(
        "Stage 1/10: validate",
        source_paths,
        records,
        lambda source_path, relative_path, record: stage01_validate.run(
            source_path, relative_path, config, paths, record, config_hash, logger
        ),
        manifest,
        paths,
    )

    sr_by_key = _run_batch_stage(
        "Stage 2/10: super-resolution",
        source_paths,
        records,
        lambda source_path, relative_path, record: stage02_super_resolution.run(
            validated_by_key[str(relative_path)], relative_path, config, paths, manager, record, config_hash, logger
        )
        if str(relative_path) in validated_by_key
        else None,
        manifest,
        paths,
    )
    manager.unload(stage02_super_resolution.MODEL_KEY)
    memory.clear_memory()

    depth_by_key = _run_batch_stage(
        "Stage 3/10: depth",
        source_paths,
        records,
        lambda source_path, relative_path, record: stage03_depth.run(
            sr_by_key[str(relative_path)], relative_path, config, paths, manager, record, config_hash, logger
        )
        if str(relative_path) in sr_by_key
        else None,
        manifest,
        paths,
    )
    manager.unload(stage03_depth.MODEL_KEY)
    memory.clear_memory()

    failed_keys: List[str] = []
    progress = tqdm(list(zip(source_paths, records)), desc="Stages 4-10/10: geometry + export")
    for source_path, record in progress:
        relative_path = paths.relative_path(source_path)
        key = str(relative_path)
        progress.set_postfix(image=Path(key).name, device=manager.device, mem=memory.cuda_memory_summary())

        validated = validated_by_key.get(key)
        sr_rgb = sr_by_key.get(key)
        depth_norm = depth_by_key.get(key)
        if validated is None or sr_rgb is None or depth_norm is None:
            failed_keys.append(key)
            manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
            continue

        try:
            _run_geometry_and_export(
                validated, sr_rgb, depth_norm, relative_path, config, paths, record, config_hash, logger
            )
            record.overall_status = STATUS_DONE
            record.total_runtime_seconds = sum(s.runtime_seconds for s in record.stages.values())
        except Exception as exc:  # noqa: BLE001 -- one image's failure must not stop the batch
            record.overall_status = STATUS_FAILED
            record.error = record.error or f"{type(exc).__name__}: {exc}"
            failed_keys.append(key)
            logger.error("Image failed: %s (%s)", key, record.error)
            if not config.CONTINUE_ON_ERROR:
                manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())
                raise
        finally:
            manifest.save(paths.manifest_json_path(), paths.manifest_csv_path())

    manager.unload_all()

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
    """Runs all ten stages for exactly one image (not batch-optimized -- both
    SR and depth models load/unload around this single image). Participates
    in the same manifest as a full ``run_pipeline`` run at the same ``paths``,
    so it's resumable/inspectable the same way.
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

        sr_rgb = stage02_super_resolution.run(validated, relative_path, config, paths, manager, record, config_hash, logger)
        manager.unload(stage02_super_resolution.MODEL_KEY)
        if sr_rgb is None:
            return record

        depth_norm = stage03_depth.run(sr_rgb, relative_path, config, paths, manager, record, config_hash, logger)
        manager.unload(stage03_depth.MODEL_KEY)
        if depth_norm is None:
            return record

        _run_geometry_and_export(validated, sr_rgb, depth_norm, relative_path, config, paths, record, config_hash, logger)
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
            # New file, or its content changed since the last run -- cached stage outputs
            # (if any) are no longer trustworthy; should_skip_stage() will notice the
            # config/upstream checks still pass but callers re-fingerprinting here is what
            # ultimately forces stage 1 to actually re-run (its own output won't match this
            # new hash), which then invalidates every downstream stage via mark_done().
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


def _run_geometry_and_export(
    validated,
    sr_rgb,
    depth_norm,
    relative_path: Path,
    config: PipelineConfig,
    paths: PipelinePaths,
    record: ImageRecord,
    config_hash: str,
    logger: logging.Logger,
) -> None:
    crop_mask_out = stage04_crop_mask.run(sr_rgb, relative_path, config, paths, record, config_hash, logger)
    if crop_mask_out is None:
        raise RuntimeError(record.error or "crop_mask stage failed")

    source_ext = stage05_source_extension.run(
        sr_rgb, depth_norm, crop_mask_out, relative_path, config, paths, record, config_hash, logger
    )
    if source_ext is None:
        raise RuntimeError(record.error or "source_extension stage failed")

    backprojection = stage06_backprojection.run(source_ext, relative_path, config, paths, record, config_hash, logger)
    if backprojection is None:
        raise RuntimeError(record.error or "backprojection stage failed")

    camera = stage07_camera.run(backprojection, source_ext, relative_path, config, paths, record, config_hash, logger)
    if camera is None:
        raise RuntimeError(record.error or "camera stage failed")

    render = stage08_render.run(
        backprojection, source_ext, camera, relative_path, config, paths, record, config_hash, logger
    )
    if render is None:
        raise RuntimeError(record.error or "render stage failed")

    fill_result = stage09_fill.run(render, relative_path, config, paths, record, config_hash, logger)
    if fill_result is None:
        raise RuntimeError(record.error or "fill stage failed")

    final_path = stage10_export.run(
        validated, sr_rgb, fill_result, relative_path, config, paths, record, config_hash, logger
    )
    if final_path is None:
        raise RuntimeError(record.error or "export stage failed")
