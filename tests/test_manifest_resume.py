from __future__ import annotations

from pathlib import Path

import pytest

from crop_aerial_pipeline.config import PipelineConfig
from crop_aerial_pipeline.manifest import STATUS_DONE, STATUS_FAILED, STATUS_PENDING, Manifest
from crop_aerial_pipeline.stages import mark_done, mark_failed, mark_skipped, should_skip_stage
from crop_aerial_pipeline.utils.hashing import fingerprints_match, hash_config, hash_file_content


def _touch(path: Path, content: bytes = b"abc") -> Path:
    path.write_bytes(content)
    return path


def test_should_skip_stage_requires_prior_success(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    output = _touch(tmp_path / "out.jpg")

    # Never run before -> must not skip.
    assert not should_skip_stage(record, "validate", [output], "hash1", resume=True, overwrite=False)


def test_should_skip_stage_true_when_everything_matches(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    output = _touch(tmp_path / "out.jpg")

    mark_done(record, "validate", runtime_seconds=1.0, config_hash="hash1")
    assert should_skip_stage(record, "validate", [output], "hash1", resume=True, overwrite=False)


def test_should_skip_stage_false_when_config_hash_changed(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    output = _touch(tmp_path / "out.jpg")

    mark_done(record, "validate", runtime_seconds=1.0, config_hash="hash1")
    assert not should_skip_stage(record, "validate", [output], "hash2", resume=True, overwrite=False)


def test_should_skip_stage_false_when_output_missing(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    missing_output = tmp_path / "does_not_exist.jpg"

    mark_done(record, "validate", runtime_seconds=1.0, config_hash="hash1")
    assert not should_skip_stage(record, "validate", [missing_output], "hash1", resume=True, overwrite=False)


def test_should_skip_stage_false_when_upstream_not_done(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    output = _touch(tmp_path / "out.jpg")

    # "depth" (stage 3) claims done, but "validate"/"super_resolution" (stages 1-2) never ran.
    mark_done(record, "depth", runtime_seconds=1.0, config_hash="hash1")
    assert not should_skip_stage(record, "depth", [output], "hash1", resume=True, overwrite=False)


def test_should_skip_stage_false_when_overwrite_requested(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    output = _touch(tmp_path / "out.jpg")

    mark_done(record, "validate", runtime_seconds=1.0, config_hash="hash1")
    assert not should_skip_stage(record, "validate", [output], "hash1", resume=True, overwrite=True)


def test_mark_done_invalidates_downstream_stage_statuses():
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))

    for stage_name in ["validate", "super_resolution", "depth", "crop_mask"]:
        mark_done(record, stage_name, runtime_seconds=1.0, config_hash="hash1")
    assert record.stage("crop_mask").status == STATUS_DONE

    # "super_resolution" recomputes (e.g. its own config changed) -> everything
    # after it, including "crop_mask" which had nothing to do with that change,
    # must be forced to re-run.
    mark_done(record, "super_resolution", runtime_seconds=1.0, config_hash="hash2")
    assert record.stage("depth").status == STATUS_PENDING
    assert record.stage("crop_mask").status == STATUS_PENDING
    # Stages BEFORE the recomputed one are untouched.
    assert record.stage("validate").status == STATUS_DONE


def test_failed_stage_marks_overall_record_failed():
    manifest = Manifest()
    record = manifest.get_or_create(Path("field_a.jpg"))
    mark_failed(record, "depth", "boom")
    assert record.overall_status == STATUS_FAILED
    assert record.error == "boom"


def test_manifest_round_trips_through_json(tmp_path: Path):
    manifest = Manifest()
    record = manifest.get_or_create(Path("farm_2/field_b.png"), source_hash={"size": 123, "sha256": "abc"})
    mark_done(record, "validate", runtime_seconds=2.5, config_hash="hash1")
    record.original_width, record.original_height = 800, 600

    json_path = tmp_path / "manifest.json"
    csv_path = tmp_path / "manifest.csv"
    manifest.save(json_path, csv_path)

    reloaded = Manifest.load(json_path)
    reloaded_record = reloaded.get(Path("farm_2/field_b.png"))
    assert reloaded_record is not None
    assert reloaded_record.original_width == 800
    assert reloaded_record.stage("validate").status == STATUS_DONE
    assert reloaded_record.stage("validate").runtime_seconds == 2.5
    assert csv_path.exists()


def test_fingerprints_match_detects_content_change(tmp_path: Path):
    from crop_aerial_pipeline.utils.hashing import file_fingerprint

    path = tmp_path / "image.jpg"
    path.write_bytes(b"version one")
    fp1 = file_fingerprint(path)

    path.write_bytes(b"version two, definitely different bytes")
    fp2 = file_fingerprint(path)

    assert not fingerprints_match(fp1, fp2)
    assert fingerprints_match(fp1, fp1)


def test_hash_config_is_stable_and_sensitive_to_changes():
    a = hash_config(PipelineConfig())
    b = hash_config(PipelineConfig())
    c = hash_config(PipelineConfig(ALTITUDE_M=99.0))
    assert a == b
    assert a != c
