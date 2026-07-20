# crop_aerial_pipeline

Converts a folder of ground-level crop photographs into centered
pseudo-aerial views, without diffusion. You give it one Google Drive folder;
it does the rest.

```python
from crop_aerial_pipeline import run_pipeline
summary = run_pipeline("/content/drive/MyDrive/crop_images")
```

## Why this exists

The previous version of this pipeline centered the virtual camera near the
*visible edge* of the crop (not its interior), which produced a triangular
camera frustum with empty corners that then got filled with smeared/stretched
garbage — and got worse, not better, as altitude increased. Routing the
"finalization" step through SDXL + ControlNet compounded the problem: the
diffusion model changed row spacing, invented plants, and generally
hallucinated crop geometry that wasn't in the source photo.

This rewrite fixes both problems structurally:

- The virtual camera is centered over the **robust interior** of the crop
  (via a distance-transform interior selector + median of only the
  *original*, non-padded, non-synthesized 3D points) — never the boundary,
  never a padded/synthesized point, never the whole point cloud's
  centroid/extremes.
- Missing corner coverage is solved **before** back-projection, by extending
  the source image with patch-based texture synthesis (not mirroring), so the
  renderer has real crop-consistent content to draw from — not by inpainting
  a hole after the fact.
- Finalization is **entirely non-generative**: mild contrast, white balance,
  edge-preserving denoise, gentle unsharp mask. Nothing that could add,
  remove, or move a crop row.

## Project tree

```
crop_aerial_pipeline/
    README.md
    pyproject.toml
    requirements.txt
    crop_aerial_pipeline/
        __init__.py                  # public API: run_pipeline, PipelineConfig, ...
        config.py                    # PipelineConfig dataclass + validate()
        cli.py                       # `python -m crop_aerial_pipeline ...`
        runner.py                    # run_pipeline / process_single_image / clean_temp_files
        manifest.py                  # ImageRecord/StageRecord + Manifest (json/csv)
        io/
            discovery.py             # find_images(): recursive scan, extension/hidden filtering
            paths.py                 # PipelinePaths: every stage path, derived from input_folder
            image_io.py              # EXIF-safe read, atomic write, sidecar helpers
        models/
            model_manager.py         # lazy load/unload, one model family resident at a time
            super_resolution.py      # SuperResolutionBackend protocol + RealESRGANBackend
            depth_estimation.py      # DepthEstimator protocol + DepthAnythingV2Estimator
            crop_segmentation.py     # HSV + Excess-Green-Index crop mask
        geometry/
            intrinsics.py            # source vs. virtual camera intrinsics
            backprojection.py        # vectorized RGB+depth -> point cloud
            crop_center.py           # distance-transform interior + robust center
            virtual_camera.py        # look_at() + camera placement
            rasterizer.py            # vectorized NumPy z-buffer renderer
            hole_filling.py          # bounded nearest-fill (not inpainting)
            source_extension.py      # TexturePatchSourceExtender / ReflectionSourceExtender
        stages/
            stage01_validate.py ... stage10_export.py
        utils/
            logging_utils.py, memory.py, hashing.py, visualization.py
    tests/
        conftest.py, test_config.py, test_paths.py, test_discovery.py,
        test_crop_center.py, test_manifest_resume.py, test_output_naming.py
```

One deliberate deviation from the originally sketched tree: `geometry/`
gained a `source_extension.py` (holding the `SourceExtender` protocol +
both backends) that wasn't in the initial file list — Stage 5's texture
synthesis is real domain logic, not orchestration, so it belongs in
`geometry/` alongside the rest of the math, with `stages/stage05_*.py`
doing only I/O + resume bookkeeping like every other stage.

**Every `.py` file above is complete, real Python** — no stubs, no
pseudocode. The 8 numbered temp folders in the spec map to the 10 pipeline
stages as: stages 1-5 and 8-10 each get their own folder
(`01_validated` ... `05_source_extended`, `06_raw_render`,
`07_validity_mask`, `08_filled_render`); stages 6 (back-projection) and 7
(camera) have no *visual* output, so their machine-readable sidecars
(`<stem>.points.npz`, `<stem>.camera.json`) live under `06_raw_render/`
alongside the render they directly produced.

## Module explanations

- **`config.py`** — one `PipelineConfig` dataclass, every field from the
  spec, with a `.validate()` that raises `ValueError` (listing *every*
  problem, not just the first) before any stage runs.
- **`manifest.py`** — `ImageRecord` (one per source image: per-stage status/
  timing/error, resolution, camera params, pixel percentages, final path) and
  `Manifest` (the whole run's table; `save()`/`load()` to/from
  `manifest.json` + `manifest.csv`; `to_dataframe()` is what
  `run_pipeline()` ultimately returns).
- **`io/discovery.py`** — finds every supported image under `input_folder`,
  recursively or not, skipping the pipeline's own `_crop_aerial_*`
  directories and hidden/unsupported files.
- **`io/paths.py`** — `PipelinePaths`: every path (temp/results/logs roots,
  per-stage directory, per-image visual output, per-image sidecar) is
  derived from `input_folder` + a `run_id`. This is what makes
  `run_pipeline(input_folder)` the only required argument, and what
  guarantees relative-directory-preserving, basename-preserving,
  collision-free output paths for same-named files in different subfolders.
- **`io/image_io.py`** — EXIF-orientation-safe reads with ICC profile
  passthrough; every write goes to a `*.part` file and is atomically renamed
  into place, so a crash mid-write can never look like a "complete" cached
  output to the resume logic.
- **`models/model_manager.py`** — a name -> lazily-constructed-model cache
  with `unload()`/`unload_all()` (garbage collection + `torch.cuda.
  empty_cache()`), used to implement "load Real-ESRGAN, run it on every
  image, unload it, load depth, run it on every image, unload it."
- **`models/super_resolution.py`** — the `SuperResolutionBackend` protocol +
  `RealESRGANBackend` (tiled inference, auto tile-shrink then CPU fallback on
  CUDA OOM, no face/anime weights, no diffusion) + `compute_effective_scale()`
  (caps output size via `MAX_SUPER_RES_DIMENSION`).
- **`models/depth_estimation.py`** — `DepthEstimator` protocol +
  `DepthAnythingV2Estimator` (relative depth; the docstring is explicit that
  `1.0` means *nearest*, and that this is not metric depth).
- **`models/crop_segmentation.py`** — HSV + Excess Green Index vegetation
  mask, morphological cleanup, tiny-component removal, full-image fallback.
- **`geometry/*.py`** — pure math, no disk I/O, no model calls: intrinsics
  (source vs. virtual, auto-zoomed to `CAMERA_FRAME_FILL`), vectorized
  back-projection (carries real-vs-synthesized + crop-mask-membership flags
  per point), the distance-transform interior selector + robust median
  center, camera placement (`TILT_DEGREES=0` -> eye/target share X,Z), the
  z-buffer rasterizer (no Open3D `OffscreenRenderer`, no per-point Python
  loop — see its docstring for how the vectorization works), bounded
  hole-filling, and the two `SourceExtender` backends.
- **`stages/*.py`** — one file per pipeline stage; each `run()` checks
  `should_skip_stage()` first (prior success + upstream success + matching
  config hash + complete output files), loads cached output from disk if so,
  otherwise does the real work, saves atomically, and records
  status/timing/errors onto the shared `ImageRecord`.
- **`runner.py`** — `run_pipeline()` (stage-wise batch execution: all images
  through validate, then all through SR with the model loaded once, unload,
  all through depth, unload, then stages 4-10 per-image since they need no
  persistent model), `process_single_image()` (all 10 stages for one image),
  `clean_temp_files()` (deletes stage-output subfolders, keeps `manifest.*`
  by default, never touches `_crop_aerial_results/`).
- **`cli.py`** — `python -m crop_aerial_pipeline <folder> [--altitude 12] ...`.
- **`utils/*.py`** — logging (one timestamped file per run under
  `_crop_aerial_logs/`), CUDA memory hygiene + OOM-retry wrapper, file/config
  fingerprinting for the resume logic, and the diagnostic-panel builder.

## Installation

```bash
# Core only (discovery/paths/manifest/geometry/config/tests) -- enough to run
# the pipeline with SUPER_RESOLUTION_ENABLED=False and your own DepthEstimator:
pip install numpy opencv-python Pillow pandas tqdm matplotlib

# Full install (adds Real-ESRGAN + Depth Anything V2):
pip install -e ".[models]"
# or, without editable-installing the package:
pip install -r requirements.txt

# For running the test suite:
pip install -e ".[dev]"
pytest tests/ -q
```

## Colab setup

```python
# Cell 1 -- mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 2 -- install (see Troubleshooting below if this errors on basicsr/torchvision)
!pip install -q numpy opencv-python Pillow pandas tqdm matplotlib
!pip install -q torch torchvision transformers
!pip install -q basicsr realesrgan

# Cell 3 -- get the package onto sys.path (copy it into Drive once, then just import
# from there on every future session -- no reinstall needed):
import sys
PACKAGE_PARENT_DIR = "/content/drive/MyDrive/crop_aerial_pipeline"  # the folder containing crop_aerial_pipeline/
sys.path.insert(0, PACKAGE_PARENT_DIR)

from crop_aerial_pipeline import run_pipeline, PipelineConfig
```

## Example batch execution

```python
summary = run_pipeline("/content/drive/MyDrive/crop_images")
display(summary)
```

With overrides:

```python
config = PipelineConfig(
    ALTITUDE_M=12.0,
    SUPER_RESOLUTION_SCALE=2.0,
    SOURCE_PAD_FRAC=0.50,
    CROP_INTERIOR_QUANTILE=0.65,
    CAMERA_FRAME_FILL=0.92,
)
summary = run_pipeline("/content/drive/MyDrive/crop_images", config=config)
```

## Example resume execution

Interrupting a run (Colab disconnect, manual stop, crash) and simply calling
`run_pipeline()` again with the *same* `input_folder` resumes automatically —
`RESUME=True` is the default, and the run reuses the most recent `run_id`
(tracked in `_crop_aerial_temp/LATEST_RUN.txt`), skipping every stage whose
cached output is still valid:

```python
# First attempt gets interrupted partway through...
summary = run_pipeline("/content/drive/MyDrive/crop_images")

# ...just call it again. Already-completed stages are skipped (logged as
# "skipped (resumed)"); only what's missing/invalid actually recomputes.
summary = run_pipeline("/content/drive/MyDrive/crop_images")
```

Force a clean re-run instead (ignore any cache):

```python
config = PipelineConfig(RESUME=False)          # start a brand new run_id
config = PipelineConfig(OVERWRITE=True)         # reuse the run_id, recompute everything anyway
```

Clean up disk space once you're happy with the results (never deletes
`_crop_aerial_results/`):

```python
from crop_aerial_pipeline import clean_temp_files
clean_temp_files("/content/drive/MyDrive/crop_images")             # keeps manifest.json/csv
clean_temp_files("/content/drive/MyDrive/crop_images", keep_manifest=False)
```

Inspect one image's diagnostic panel:

```python
from crop_aerial_pipeline import show_diagnostic_panel
show_diagnostic_panel("field_a", results_root="/content/drive/MyDrive/crop_images/_crop_aerial_temp/<run_id>/diagnostics")
```

## Unit tests

```bash
pytest tests/ -q
```

55 tests, all passing without any GPU/model dependency (they exercise
`config`, `manifest`/resume bookkeeping, `io.discovery`, `io.paths`, and the
`geometry` camera-centering math directly, with synthetic images — the
model-backed stages 2-3 aren't unit-tested here since they need real
weights/GPU; that's an integration-test concern, not a unit-test one).

Covers, per the spec's required areas:

- **Path preservation** — `test_paths.py`, `test_output_naming.py` (nested
  subdirectories, basename/extension preservation, no collisions between
  same-named files in different subfolders).
- **Camera centering** — `test_crop_center.py` (interior selector excludes
  the mask boundary; the robust center ignores synthesized/padded points;
  zero tilt forces eye/target to share X,Z).
- **Configuration validation** — `test_config.py` (every invalid field is
  rejected; valid overrides don't leak between instances).
- **Resume behavior** — `test_manifest_resume.py` (`should_skip_stage()`'s
  four conditions individually; `mark_done()` invalidating downstream
  stages; manifest JSON round-trip; file/config fingerprinting).

## Troubleshooting

### CUDA out-of-memory (Real-ESRGAN or depth estimation)

- `RealESRGANBackend` already retries automatically: on a CUDA OOM it halves
  its tile size (down to a 64px floor) and retries, up to 3 times, then falls
  back to CPU for that image. If you're hitting this constantly, just lower
  `SUPER_RESOLUTION_TILE` (e.g. 128) up front — you'll skip the retries.
- Check `utils.memory.cuda_memory_summary()` (also logged at INFO level
  around every model load/unload) to see free vs. total VRAM at the point of
  failure.
- Make sure a previous stage's model is actually unloaded before the next
  one loads — `run_pipeline()` does this for you via stage-wise batch
  execution; if you're calling `process_single_image()` or the stage modules
  directly in a custom script, call `manager.unload(...)` between stages
  yourself.
- Lower `MAX_SUPER_RES_DIMENSION` if your source photos are already large —
  a 4000px-wide photo at 2x scale is an 8000px output, which is expensive at
  every later stage too, not just SR.
- Depth estimation (Depth Anything V2 "small") is lightweight and rarely
  OOMs on its own; if it does on a very large image, that usually means
  `MAX_SUPER_RES_DIMENSION` let an unexpectedly large image through Stage 2 —
  lower it.

### `ImportError` / `undefined symbol` from `basicsr` after installing a newer `torchvision`

`basicsr` (a `realesrgan` dependency) imports
`torchvision.transforms.functional_tensor`, which was removed in
`torchvision>=0.17`. If you see this error:

```
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

the fix is either:

1. Pin an older `torchvision` before installing `basicsr`/`realesrgan`:
   ```bash
   pip install -q "torchvision<0.17"
   pip install -q basicsr realesrgan
   ```
2. Or patch `basicsr` in place (works with any torchvision version) after
   installing it:
   ```python
   import basicsr, pathlib
   degradations_path = pathlib.Path(basicsr.__file__).parent / "data" / "degradations.py"
   text = degradations_path.read_text()
   text = text.replace(
       "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
       "from torchvision.transforms.functional import rgb_to_grayscale",
   )
   degradations_path.write_text(text)
   ```
   Re-run this after every fresh `pip install basicsr` (a new install
   overwrites the patched file).

### `SUPER_RESOLUTION_ENABLED=False` but I still want depth/geometry to run

That's exactly what it's for — Stage 2 just copies the validated image
through unchanged (same dimensions) when disabled, and every later stage
works identically either way.

### A run says "resumed"/"skipped" for a stage I expected to re-run

Check what changed: `should_skip_stage()` requires the source file's
fingerprint to be unchanged (it is, if you didn't touch the source photo),
the config hash to match (any `PipelineConfig` field change invalidates
*that* stage forward), and every upstream stage to have succeeded. If you
changed a setting only used by a *later* stage (e.g. `CAMERA_FRAME_FILL`,
used starting at Stage 8) and expected Stage 4-7 outputs to stay cached while
Stage 8+ recomputes — that's exactly what happens; check the per-stage
`status`/`config_hash` in `manifest.csv` for that image if it's not
behaving as expected.
