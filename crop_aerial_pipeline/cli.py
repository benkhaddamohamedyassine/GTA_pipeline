"""Command-line entry point::

    python -m crop_aerial_pipeline /content/drive/MyDrive/crop_images
    python -m crop_aerial_pipeline /path/to/images --altitude 12 --no-super-resolution
    python -m crop_aerial_pipeline /path/to/images --clean-temp
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .config import PipelineConfig
from .runner import clean_temp_files, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crop_aerial_pipeline",
        description="Convert a folder of ground-level crop photos into centered pseudo-aerial views.",
    )
    parser.add_argument("input_folder", help="Folder of source images (a Google Drive-mounted path works fine).")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan nested folders.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any previous run; start fresh.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute every stage even if cached outputs look valid.")
    parser.add_argument("--no-super-resolution", action="store_true", help="Disable Real-ESRGAN super-resolution.")
    parser.add_argument("--altitude", type=float, default=None, help="Override ALTITUDE_M.")
    parser.add_argument("--tilt", type=float, default=None, help="Override TILT_DEGREES.")
    parser.add_argument(
        "--clean-temp", action="store_true", help="Delete temp stage outputs for input_folder and exit (no pipeline run)."
    )
    parser.add_argument(
        "--delete-manifest", action="store_true", help="With --clean-temp, also delete manifest.json/manifest.csv."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.clean_temp:
        clean_temp_files(args.input_folder, keep_manifest=not args.delete_manifest)
        print(f"Cleaned temp stage outputs under {args.input_folder}")
        return 0

    overrides: Dict[str, Any] = {}
    if args.no_recursive:
        overrides["RECURSIVE"] = False
    if args.no_resume:
        overrides["RESUME"] = False
    if args.overwrite:
        overrides["OVERWRITE"] = True
    if args.no_super_resolution:
        overrides["SUPER_RESOLUTION_ENABLED"] = False
    if args.altitude is not None:
        overrides["ALTITUDE_M"] = args.altitude
    if args.tilt is not None:
        overrides["TILT_DEGREES"] = args.tilt

    config = PipelineConfig(**overrides)
    summary = run_pipeline(args.input_folder, config=config)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))

    failed = int((summary["status"] == "failed").sum()) if "status" in summary.columns else 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
