from __future__ import annotations

from pathlib import Path

from PIL import Image

from crop_aerial_pipeline.io.discovery import discover_images


def test_discovers_nested_supported_images(sample_folder: Path):
    found = discover_images(sample_folder, recursive=True)
    names = sorted(p.relative_to(sample_folder).as_posix() for p in found)
    assert names == ["farm_2/field_b.png", "field_a.jpg"]


def test_non_recursive_skips_nested_folders(sample_folder: Path):
    found = discover_images(sample_folder, recursive=False)
    names = sorted(p.relative_to(sample_folder).as_posix() for p in found)
    assert names == ["field_a.jpg"]


def test_ignores_hidden_and_unsupported_files(sample_folder: Path):
    found = discover_images(sample_folder, recursive=True)
    assert not any(p.name.startswith(".") for p in found)
    assert not any(p.suffix == ".txt" for p in found)


def test_ignores_pipeline_owned_directories(sample_folder: Path):
    # Simulate a previous run's temp/results/logs dirs already existing, containing
    # images of their own (which must never be re-discovered as new input).
    for owned in ("_crop_aerial_temp", "_crop_aerial_results", "_crop_aerial_logs"):
        owned_dir = sample_folder / owned / "sub"
        owned_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(owned_dir / "leftover.jpg")

    found = discover_images(sample_folder, recursive=True)
    names = sorted(p.relative_to(sample_folder).as_posix() for p in found)
    assert names == ["farm_2/field_b.png", "field_a.jpg"]


def test_all_required_extensions_are_supported(tmp_path: Path):
    root = tmp_path / "images"
    root.mkdir()
    for ext in ["jpg", "jpeg", "png", "tiff", "tif", "webp"]:
        Image.new("RGB", (8, 8)).save(root / f"sample.{ext}")

    found = discover_images(root, recursive=False)
    assert len(found) == 6
