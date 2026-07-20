from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crop_aerial_pipeline.io.image_io import atomic_write_image, read_image_rgb
from crop_aerial_pipeline.io.paths import PipelinePaths


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".tif", ".webp"])
def test_final_output_keeps_original_extension_and_basename(tmp_path: Path, ext: str):
    paths = PipelinePaths(input_folder=tmp_path, run_id="run_test")
    paths.ensure_directories()

    relative = Path(f"field_a{ext}")
    out_path = paths.results_output_path(relative)
    image = np.full((16, 16, 3), 200, dtype=np.uint8)
    atomic_write_image(image, out_path)

    assert out_path.exists()
    assert out_path.name == f"field_a{ext}"
    reloaded, _ = read_image_rgb(out_path)
    assert reloaded.shape == (16, 16, 3)


def test_no_temp_part_file_left_behind_after_atomic_write(tmp_path: Path):
    out_path = tmp_path / "result.png"
    atomic_write_image(np.zeros((4, 4, 3), dtype=np.uint8), out_path)
    assert out_path.exists()
    assert not (out_path.parent / (out_path.name + ".part")).exists()


def test_duplicate_basenames_in_different_subfolders_both_persist(tmp_path: Path):
    paths = PipelinePaths(input_folder=tmp_path, run_id="run_test")
    paths.ensure_directories()

    image_a = np.full((8, 8, 3), 10, dtype=np.uint8)
    image_b = np.full((8, 8, 3), 250, dtype=np.uint8)

    path_a = paths.results_output_path(Path("farm_1/field.jpg"))
    path_b = paths.results_output_path(Path("farm_2/field.jpg"))
    atomic_write_image(image_a, path_a)
    atomic_write_image(image_b, path_b)

    assert path_a.exists() and path_b.exists()
    reloaded_a, _ = read_image_rgb(path_a)
    reloaded_b, _ = read_image_rgb(path_b)
    assert reloaded_a[0, 0, 0] < 50 < reloaded_b[0, 0, 0]  # confirms they weren't overwritten/mixed up


def test_diagnostic_panel_naming_uses_stem_regardless_of_source_extension(tmp_path: Path):
    from crop_aerial_pipeline.utils.visualization import save_diagnostic_panel

    paths = PipelinePaths(input_folder=tmp_path, run_id="run_test")
    paths.ensure_directories()

    relative = Path("farm_2/field_b.png")
    diagnostic_path = paths.run_dir / "diagnostics" / relative
    diagnostic_path = diagnostic_path.parent / f"{diagnostic_path.stem}.diagnostic.jpg"

    panels = [("original", np.zeros((8, 8, 3), dtype=np.uint8))]
    save_diagnostic_panel(panels, diagnostic_path)

    assert diagnostic_path.exists()
    assert diagnostic_path.name == "field_b.diagnostic.jpg"
