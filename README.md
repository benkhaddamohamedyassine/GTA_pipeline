# crop_aerial_pipeline

Converts a folder of ground-level crop photographs into centered
pseudo-aerial views, without diffusion. You give it one Google Drive folder;
it does the rest.

```python
from crop_aerial_pipeline import run_pipeline
summary = run_pipeline("/content/drive/MyDrive/crop_images")
```

## Why this exists

The original version of this pipeline centered the virtual camera near the
*visible edge* of the crop (not its interior), producing a triangular camera
frustum with empty corners filled by smeared post-warp inpainting or plain
reflection — and it got worse, not better, as altitude increased. Routing
"finalization" through SDXL + ControlNet compounded the problem: diffusion
changed row spacing, invented plants, and hallucinated crop geometry that
wasn't in the source photo.

**This version's pipeline order is deliberate and matters:**

- **Depth, crop masking, and camera placement all happen on the original
  photo (or the original's own crop-interior selection) — never on anything
  AI-generated or upscaled.** The virtual camera is centered over the robust
  interior of the *original* crop, using only points that (a) came from real
  source pixels, (b) belong to the original crop mask, (c) belong to the
  original crop-interior selection, and (d) have finite geometry.
- **Canvas coverage is extended with AI outpainting (LaMa) before
  back-projection**, not classical texture synthesis/reflection and not by
  upscaling — this supplies real, plausible surrounding content for what
  used to be empty triangular corners, with the true original region
  preserved pixel-for-pixel and every generated pixel tracked in a
  provenance map that camera placement is never allowed to see.
- **Super-resolution runs LAST, on the completed, hole-filled render** — not
  on the source photo. Upscaling the source first would inflate depth-
  estimation and point-cloud memory cost without adding real geometric
  information, can amplify foliage detail into unstable monocular depth, and
  risks super-resolution artifacts leaking into segmentation, depth, camera
  placement, or point-cloud reconstruction. Running it after warping avoids
  all of that while still sharpening the final image.
- **Finalization is entirely non-generative**: mild contrast, white balance,
  edge-preserving denoise, gentle unsharp mask, weak color matching against
  the pre-super-resolution warp. Nothing that could add, remove, or move a
  crop row.

## The 13 stages, in order

| # | Stage | Model? | What it does |
|---|-------|--------|---------------|
| 1 | validate | — | EXIF-safe read, RGB, ICC passthrough |
| 2 | initial_depth | Depth Anything V2 | Depth on the **original** image |
| 3 | crop_mask | — | Crop mask + distance-transform interior, on the **original** image |
| 4 | ai_outpaint | LaMa | Extends the canvas; original preserved pixel-for-pixel; provenance tracked |
| 5 | extended_depth | Depth Anything V2 | Depth re-estimated on the extended canvas, aligned back to Stage 2 in the protected region |
| 6 | extended_crop_mask | — | Crop mask re-run on the extended canvas; original preserved exactly in the protected region |
| 7 | backprojection | — | Vectorized RGB+depth → point cloud, carrying provenance + crop-mask + crop-interior membership |
| 8 | camera | — | Centers the virtual camera using ONLY original+interior+finite points; nadir by default |
| 9 | camera_fitting | — | Auto-zooms virtual intrinsics using ONLY original crop-interior points |
| 10 | render | — | Vectorized NumPy z-buffer render of the full (original+generated) point cloud |
| 11 | fill | — | Small bounded residual-hole fill (not diffusion, not broad Telea inpaint) |
| 12 | post_warp_super_resolution | Real-ESRGAN | Upscales the COMPLETED render — the only place upscaling happens |
| 13 | finalize | — | Non-generative corrections only; saved to `_crop_aerial_results/` |

Stages 2-3 and 5-6 look similar (depth + crop mask, twice) — that's
intentional: the first pass establishes trusted, original-only geometry
before anything is AI-generated; the second pass gives the *extended*
canvas coherent depth/mask context for rendering, without ever letting that
second pass's AI-influenced content leak into where the camera gets placed.

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
            outpainting.py           # OutpaintBackend protocol + LamaOutpaintBackend + progressive driver
            super_resolution.py      # PostWarpSuperResolutionBackend protocol + RealESRGANPostWarpBackend
            depth_estimation.py      # DepthEstimator protocol + DepthAnythingV2Estimator
            crop_segmentation.py     # HSV + Excess-Green-Index crop mask
        geometry/
            intrinsics.py            # source vs. virtual camera intrinsics
            backprojection.py        # vectorized RGB+depth -> point cloud (+ provenance/interior flags)
            crop_center.py           # distance-transform interior + robust center
            virtual_camera.py        # look_at() + camera placement
            rasterizer.py            # vectorized NumPy z-buffer renderer
            hole_filling.py          # bounded nearest-fill (not inpainting)
        stages/
            stage01_validate.py ... stage13_finalize.py
        utils/
            logging_utils.py, memory.py, hashing.py, visualization.py
    tests/
        conftest.py, test_config.py, test_paths.py, test_discovery.py,
        test_crop_center.py, test_outpainting.py, test_manifest_resume.py,
        test_output_naming.py
```

**Every `.py` file above is complete, real Python** — no stubs, no
pseudocode.

### Folder-naming note

The canonical `_crop_aerial_temp/<run_id>/` folder list has 18 entries
(`01_validated` ... `13_finalized`) matching each *visual* stage output.
Stages 7 (backprojection) and 9 (camera fitting) produce no image, so their
machine-readable sidecars live inside the nearest related folder instead of
getting their own: Stage 7's `<stem>.points.npz` is under `07_point_cloud/`
(it does get a dedicated folder), and Stage 9's `<stem>.virtual_intrinsics.
json` sits beside Stage 8's `<stem>.camera.json` under `08_camera/`. A few
outputs mentioned in individual stage write-ups but not in the canonical
folder list (Stage 2/5's raw depth `.npy`, Stage 10's render-depth preview)
are similarly nested as sidecars in the nearest listed folder rather than
new top-level directories — each stage module's docstring says exactly
where.

## Module explanations

- **`config.py`** — one `PipelineConfig` dataclass; `.validate()` raises
  `ValueError` listing every problem at once, before any stage runs.
- **`manifest.py`** — `ImageRecord` (per-stage status/timing/error,
  resolution, camera params, pixel percentages, final path) and `Manifest`
  (the whole run's table; `to_dataframe()` is what `run_pipeline()` returns).
- **`io/discovery.py`** — finds every supported image, recursively or not,
  skipping the pipeline's own `_crop_aerial_*` directories and hidden/
  unsupported files.
- **`io/paths.py`** — `PipelinePaths`: every path is derived from
  `input_folder` + a `run_id`, which is what makes `run_pipeline(input_
  folder)` the only required argument and guarantees collision-free,
  relative-structure-preserving output paths.
- **`io/image_io.py`** — EXIF-safe reads with ICC passthrough; every write
  is atomic (`*.part` + rename), so an interrupted write can never look
  "complete" to the resume logic.
- **`models/model_manager.py`** — a name → lazily-constructed-model cache
  with `unload()`, used to implement the stage-wise "load one model family,
  process every image, unload it" lifecycle.
- **`models/outpainting.py`** — `LamaOutpaintBackend` (wraps
  `simple-lama-inpainting`) + `run_progressive_outpaint()`, which grows the
  canvas a bit at a time (marking each new ring as LaMa's "hole" to fill),
  hard-guarantees the true original block is pasted back exactly regardless
  of anything the mask discipline got right or wrong, and returns a
  provenance mask. See its module docstring for why progressive (not
  single-shot) growth, and why reflection-seeding the scratch canvas isn't
  the same thing as "reflection as the extension method."
- **`models/super_resolution.py`** — `RealESRGANPostWarpBackend`: tiled
  inference, auto tile-shrink then CPU fallback on CUDA OOM, no face/anime
  weights, no diffusion. Also auto-patches the `torchvision.transforms.
  functional_tensor` compatibility break (see Troubleshooting).
- **`models/depth_estimation.py`** — `DepthAnythingV2Estimator`; relative
  depth, `1.0` = nearest, explicitly not metric.
- **`models/crop_segmentation.py`** — HSV + Excess Green Index vegetation
  mask, morphological cleanup, tiny-component removal, full-image fallback.
- **`geometry/*.py`** — pure math, no disk I/O, no model calls: intrinsics
  (source vs. auto-zoomed virtual), vectorized back-projection (per-point
  provenance / crop-mask / crop-interior / finite-geometry flags), the
  distance-transform interior selector + robust median center, camera
  placement (nadir by default), the z-buffer rasterizer (no Open3D
  `OffscreenRenderer`, no per-point Python loop), and bounded hole-filling.
- **`stages/*.py`** — one file per pipeline stage; each `run()` checks
  `should_skip_stage()` (prior success + upstream success + matching config
  hash + complete output files) before doing real work, saves atomically,
  and records status/timing/errors onto the shared `ImageRecord`.
- **`runner.py`** — `run_pipeline()` (stage-wise batch execution per the
  table above), `process_single_image()` (all 13 stages for one image,
  straight through), `clean_temp_files()`.
- **`cli.py`** — `python -m crop_aerial_pipeline <folder> [--altitude 12] ...`.
- **`utils/*.py`** — logging, CUDA memory hygiene + OOM-retry wrapper,
  file/config fingerprinting for resume, diagnostic-panel builder.

## Installation

```bash
# Core only (discovery/paths/manifest/geometry/config/tests):
pip install numpy opencv-python Pillow pandas tqdm matplotlib

# Full install (adds Depth Anything V2, LaMa, Real-ESRGAN):
pip install -e ".[models]"
# or:
pip install -r requirements.txt

# Tests:
pip install -e ".[dev]"
pytest tests/ -q
```

## Colab setup

```python
# Cell 1 -- mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 2 -- install
!pip install -q numpy opencv-python Pillow pandas tqdm matplotlib
!pip install -q torch torchvision transformers
!pip install -q basicsr realesrgan simple-lama-inpainting

# Cell 3 -- get the package onto sys.path
import sys
PACKAGE_PARENT_DIR = "/content/drive/MyDrive/crop_aerial_pipeline"  # folder containing crop_aerial_pipeline/
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
    AI_OUTPAINT_PAD_FRAC=0.50,
    CROP_INTERIOR_QUANTILE=0.65,
    CAMERA_FRAME_FILL=0.92,
    POST_WARP_SUPER_RESOLUTION_SCALE=2.0,
)
summary = run_pipeline("/content/drive/MyDrive/crop_images", config=config)
```

## Example resume execution

```python
# First attempt gets interrupted partway through...
summary = run_pipeline("/content/drive/MyDrive/crop_images")

# ...just call it again. Already-completed stages are skipped (logged as
# "skipped (resumed)"); only what's missing/invalid actually recomputes.
summary = run_pipeline("/content/drive/MyDrive/crop_images")
```

Force a clean re-run instead:

```python
config = PipelineConfig(RESUME=False)   # start a brand new run_id
config = PipelineConfig(OVERWRITE=True) # reuse the run_id, recompute everything anyway
```

Clean up disk space (never deletes `_crop_aerial_results/`):

```python
from crop_aerial_pipeline import clean_temp_files
clean_temp_files("/content/drive/MyDrive/crop_images")              # keeps manifest.json/csv
clean_temp_files("/content/drive/MyDrive/crop_images", keep_manifest=False)
```

## Unit tests

```bash
pytest tests/ -q
```

64 tests, all passing without any GPU/model dependency (they exercise
`config`, `manifest`/resume bookkeeping, `io.discovery`, `io.paths`, the
`geometry` camera-centering math, and the AI-outpainting provenance/geometry
guarantees directly, with synthetic images and fake backends — the
model-backed stages (2, 4, 5, 12) aren't unit-tested here since they need
real weights/GPU; that's an integration-test concern).

Covers, per spec: **path preservation** (`test_paths.py`,
`test_output_naming.py`), **camera centering** (`test_crop_center.py`:
interior selector excludes the boundary; robust center ignores
padded/generated points; zero tilt forces eye/target to share X,Z),
**configuration validation** (`test_config.py`), **resume behavior**
(`test_manifest_resume.py`: `should_skip_stage()`'s conditions individually;
`mark_done()` invalidating downstream stages; manifest JSON round-trip;
fingerprinting — plus a regression guard asserting `manifest.STAGE_NAMES`
and `stages.STAGE_ORDER` stay in sync, since these drifting apart is a real
bug this project shipped once), and **AI-outpainting correctness**
(`test_outpainting.py`: original-region pixel exactness, provenance mask
exactness, canvas sizing, generated-content actually being generated).

## Troubleshooting

### CUDA out-of-memory (LaMa, depth, or Real-ESRGAN)

- `RealESRGANPostWarpBackend` retries automatically: on CUDA OOM it halves
  tile size (down to 64px) up to 3 times, then falls back to CPU for that
  image if `POST_WARP_SUPER_RESOLUTION_FALLBACK_CPU=True` (default).
- `LamaOutpaintBackend` retries on CPU after repeated CUDA OOM too.
- `AI_OUTPAINT_MAX_SIDE` caps the resolution LaMa actually processes at
  (default 2048px) regardless of how large `AI_OUTPAINT_PAD_FRAC` makes the
  full canvas — lower it if you're still OOMing during Stage 4.
- Check `utils.memory.cuda_memory_summary()` (logged at INFO around every
  model load/unload) to see free vs. total VRAM at the point of failure.
- `run_pipeline()` already unloads each model family before loading the
  next (Stage 2's depth model, unload; Stage 4's LaMa, unload; Stage 5's
  depth model, reload; unload; Stage 12's Real-ESRGAN, unload) — if you're
  calling stage modules directly in a custom script instead of
  `run_pipeline()`, call `manager.unload(...)` between stages yourself.
- Lower `POST_WARP_MAX_OUTPUT_DIMENSION` if the render is already large.

### `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`

`basicsr` (a `realesrgan` dependency) imports a module torchvision removed
in 0.17+. **You shouldn't need to do anything** —
`RealESRGANPostWarpBackend.load()` patches it automatically (a `sys.modules`
compatibility shim, no pinning, no editing installed files). If you hit an
*different* `ImportError` from elsewhere inside `basicsr`'s dataset-loading
code (it unconditionally imports its entire `data` subpackage on first
import — a known fragility of that library), the immediate unblock is
`PipelineConfig(POST_WARP_SUPER_RESOLUTION_ENABLED=False)` while that gets
sorted out; every other stage works identically either way.

### AI outpainting fell back to reflection, or looks low quality at the borders

Check the log for `"texture synthesis fell back"` or similar — LaMa itself
doesn't have a classical fallback path (unlike the old texture-synthesis
extender it replaced), but `AI_OUTPAINT_MAX_RETRIES` controls how many times
a failed backend call is retried before the stage fails outright for that
image (which `CONTINUE_ON_ERROR=True` then isolates to just that image).
`AI_OUTPAINT_OVERLAP_PX`/`AI_OUTPAINT_FEATHER_PX` control seam blending
width if borders look harsh; `AI_OUTPAINT_STEP_FRAC` controls how gradually
the canvas grows (smaller steps = better quality, more LaMa calls per
image).

### A run says "resumed"/"skipped" for a stage I expected to re-run

`should_skip_stage()` requires the source file's fingerprint to be
unchanged, the config hash to match (any `PipelineConfig` field change
invalidates that stage forward), and every upstream stage to have
succeeded. Check the per-stage `status`/`config_hash` in `manifest.csv` for
that image if it's not behaving as expected.
