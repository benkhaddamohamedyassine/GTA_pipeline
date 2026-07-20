from __future__ import annotations

from pathlib import Path

from crop_aerial_pipeline.io.paths import PipelinePaths, resolve_run_id


def _paths_for(root: Path) -> PipelinePaths:
    p = PipelinePaths(input_folder=root, run_id="run_test")
    p.ensure_directories()
    return p


def test_required_directories_are_created_beside_input_folder(sample_folder: Path):
    paths = _paths_for(sample_folder)
    assert paths.temp_root == sample_folder / "_crop_aerial_temp"
    assert paths.results_root == sample_folder / "_crop_aerial_results"
    assert paths.logs_root == sample_folder / "_crop_aerial_logs"
    assert paths.temp_root.is_dir()
    assert paths.results_root.is_dir()
    assert paths.logs_root.is_dir()
    for stage_name in [
        "01_validated", "02_initial_depth_preview", "03_crop_mask", "03_crop_interior",
        "04_ai_outpaint", "04_ai_outpaint_mask", "04_ai_outpaint_provenance",
        "05_extended_depth_preview", "06_extended_crop_mask", "07_point_cloud", "08_camera",
        "10_raw_warp", "10_validity_mask", "10_render_provenance",
        "11_filled_warp", "11_filled_mask", "12_super_resolved_warp", "13_finalized",
    ]:
        assert (paths.run_dir / stage_name).is_dir()


def test_relative_path_preserves_nested_structure(sample_folder: Path):
    paths = _paths_for(sample_folder)
    nested = sample_folder / "farm_2" / "field_b.png"
    assert paths.relative_path(nested) == Path("farm_2/field_b.png")


def test_stage_output_path_preserves_basename_and_subdirectory(sample_folder: Path):
    paths = _paths_for(sample_folder)
    relative = Path("farm_2/field_b.png")

    out = paths.stage_output_path("04_ai_outpaint", relative)
    assert out == paths.run_dir / "04_ai_outpaint" / "farm_2" / "field_b.png"
    assert out.name == "field_b.png"  # exact original basename preserved


def test_stage_output_path_extension_override_keeps_stem(sample_folder: Path):
    paths = _paths_for(sample_folder)
    out = paths.stage_output_path("03_crop_mask", Path("field_a.jpg"), ext_override=".png")
    assert out.stem == "field_a"
    assert out.suffix == ".png"


def test_sidecar_path_keeps_stem_and_subdirectory(sample_folder: Path):
    paths = _paths_for(sample_folder)
    sidecar = paths.sidecar_path("02_initial_depth_preview", Path("farm_2/field_b.png"), ".depth.npy")
    assert sidecar == paths.run_dir / "02_initial_depth_preview" / "farm_2" / "field_b.depth.npy"


def test_results_output_path_mirrors_relative_structure(sample_folder: Path):
    paths = _paths_for(sample_folder)
    out = paths.results_output_path(Path("farm_2/field_b.png"))
    assert out == paths.results_root / "farm_2" / "field_b.png"


def test_same_basename_in_different_subfolders_does_not_collide(sample_folder: Path):
    """Two different subfolders can each contain e.g. 'field.jpg' -- their
    stage outputs must not be flattened into the same path."""
    paths = _paths_for(sample_folder)
    out_a = paths.stage_output_path("01_validated", Path("farm_1/field.jpg"))
    out_b = paths.stage_output_path("01_validated", Path("farm_2/field.jpg"))
    assert out_a != out_b
    assert out_a.name == out_b.name == "field.jpg"


def test_resolve_run_id_reuses_previous_run_when_resuming(sample_folder: Path):
    first_id = resolve_run_id(sample_folder, resume=True, new_run_id_factory=lambda: "run_A")
    _paths_for_with_id(sample_folder, first_id)  # simulate that run's directory existing

    second_id = resolve_run_id(sample_folder, resume=True, new_run_id_factory=lambda: "run_B")
    assert second_id == first_id == "run_A"


def test_resolve_run_id_mints_new_id_when_not_resuming(sample_folder: Path):
    first_id = resolve_run_id(sample_folder, resume=True, new_run_id_factory=lambda: "run_A")
    _paths_for_with_id(sample_folder, first_id)

    fresh_id = resolve_run_id(sample_folder, resume=False, new_run_id_factory=lambda: "run_C")
    assert fresh_id == "run_C"
    assert fresh_id != first_id


def _paths_for_with_id(root: Path, run_id: str) -> None:
    p = PipelinePaths(input_folder=root, run_id=run_id)
    p.ensure_directories()
