"""Single source of truth for every tunable pipeline parameter.

``PipelineConfig`` is a plain, validated dataclass -- no hidden global state.
Every stage receives it explicitly, and its hash (``utils.hashing.hash_config``)
is stored in the manifest so a resumed run can tell whether the user changed a
setting that invalidates previously-cached stage outputs.

REVISED ORDER: super-resolution now runs on the *completed pseudo-aerial
render* (Stage 12), never on the source photo. Extending the image before
back-projection is done by AI outpainting (LaMa, Stage 4), not by upscaling.
See the module docstrings of ``stages/stage04_ai_outpaint.py`` and
``stages/stage12_post_warp_super_resolution.py`` for why.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- Run behavior -----------------------------------------------------
    RECURSIVE: bool = True
    OVERWRITE: bool = False
    RESUME: bool = True
    CONTINUE_ON_ERROR: bool = True

    # --- Camera geometry -----------------------------------------------------
    ASSUMED_HFOV_DEG: float = 70.0  # source (phone) camera -- used only for back-projection
    ALTITUDE_M: float = 10.0
    TILT_DEGREES: float = 0.0

    # --- Stage 4: AI outpainting (LaMa) -- replaces the old classical
    # texture-synthesis/reflection source extension entirely. Upscaling never
    # happens here; this only extends canvas *coverage*, at the source image's
    # own resolution (post-warp super-resolution, Stage 12, is what sharpens
    # the final result).
    AI_OUTPAINT_ENABLED: bool = True
    AI_OUTPAINT_BACKEND: str = "lama"
    AI_OUTPAINT_PAD_FRAC: float = 0.45
    AI_OUTPAINT_STEP_FRAC: float = 0.12  # per-step border growth, as a fraction of the *original* image size
    AI_OUTPAINT_OVERLAP_PX: int = 96  # inward overlap band re-included in each step's inpaint mask, for a seamless blend
    AI_OUTPAINT_FEATHER_PX: int = 48  # final light seam blend width, applied only at the true original/generated boundary
    AI_OUTPAINT_MAX_SIDE: int = 2048  # LaMa runs on a canvas capped to this; result is upscaled back for the generated region only
    AI_OUTPAINT_TILE_SIZE: int = 768
    AI_OUTPAINT_TILE_OVERLAP: int = 128
    AI_OUTPAINT_MAX_RETRIES: int = 3
    AI_OUTPAINT_PROGRESSIVE: bool = True
    AI_OUTPAINT_PRESERVE_ORIGINAL: bool = True

    # --- Stage 3/9: crop-interior centering -----------------------------------
    CROP_INTERIOR_QUANTILE: float = 0.60
    CAMERA_FRAME_FILL: float = 0.90
    MIN_VIRTUAL_HFOV_DEG: float = 20.0
    MAX_VIRTUAL_HFOV_DEG: float = 52.0

    # --- Stage 7/10: point cloud / rendering ----------------------------------
    BACKPROJECT_STRIDE: int = 2
    SPLAT_RADIUS: int = 2
    MAX_RANSAC_POINTS: int = 250_000

    # --- Output verbosity ------------------------------------------------
    SAVE_POINT_CLOUD: bool = True
    SAVE_DEPTH_ARRAY: bool = True
    SAVE_DEBUG_VISUALIZATION: bool = True

    # --- Stage 12: post-warp Real-ESRGAN --------------------------------------
    # Runs ONLY on the completed, hole-filled pseudo-aerial render (Stage 11's
    # output) -- never on the source photo, never on a render still containing
    # large empty holes. See stage12_post_warp_super_resolution.py.
    POST_WARP_SUPER_RESOLUTION_ENABLED: bool = True
    POST_WARP_SUPER_RESOLUTION_MODEL: str = "RealESRGAN_x2plus"
    POST_WARP_SUPER_RESOLUTION_SCALE: float = 2.0
    POST_WARP_SUPER_RESOLUTION_TILE: int = 256
    POST_WARP_SUPER_RESOLUTION_TILE_PAD: int = 16
    POST_WARP_SUPER_RESOLUTION_HALF_PRECISION: bool = True
    POST_WARP_MAX_OUTPUT_DIMENSION: int = 4096
    POST_WARP_SUPER_RESOLUTION_FALLBACK_CPU: bool = True

    # --- Refinement --------------------------------------------------------
    # No diffusion backend exists in this package -- USE_DIFFUSION=True only
    # lets callers assert they're in the safe, non-generative mode.
    USE_DIFFUSION: bool = False
    RANDOM_SEED: int = 42

    def validate(self) -> None:
        """Raises ``ValueError`` listing every invalid field at once, called
        once by ``run_pipeline`` before any stage executes."""
        errors = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        require(0.0 < self.ASSUMED_HFOV_DEG < 180.0, "ASSUMED_HFOV_DEG must be in (0, 180)")
        require(self.ALTITUDE_M > 0.0, "ALTITUDE_M must be > 0")
        require(0.0 <= self.TILT_DEGREES < 90.0, "TILT_DEGREES must be in [0, 90)")

        require(
            self.AI_OUTPAINT_BACKEND in ("lama",),
            "AI_OUTPAINT_BACKEND must be 'lama' (the only backend implemented)",
        )
        require(self.AI_OUTPAINT_PAD_FRAC >= 0.0, "AI_OUTPAINT_PAD_FRAC must be >= 0")
        require(0.0 < self.AI_OUTPAINT_STEP_FRAC <= 1.0, "AI_OUTPAINT_STEP_FRAC must be in (0, 1]")
        require(self.AI_OUTPAINT_OVERLAP_PX >= 0, "AI_OUTPAINT_OVERLAP_PX must be >= 0")
        require(self.AI_OUTPAINT_FEATHER_PX >= 0, "AI_OUTPAINT_FEATHER_PX must be >= 0")
        require(self.AI_OUTPAINT_MAX_SIDE > 0, "AI_OUTPAINT_MAX_SIDE must be > 0")
        require(self.AI_OUTPAINT_TILE_SIZE > 0, "AI_OUTPAINT_TILE_SIZE must be > 0")
        require(self.AI_OUTPAINT_TILE_OVERLAP >= 0, "AI_OUTPAINT_TILE_OVERLAP must be >= 0")
        require(self.AI_OUTPAINT_MAX_RETRIES >= 0, "AI_OUTPAINT_MAX_RETRIES must be >= 0")

        require(0.0 < self.CROP_INTERIOR_QUANTILE <= 1.0, "CROP_INTERIOR_QUANTILE must be in (0, 1]")
        require(0.0 < self.CAMERA_FRAME_FILL <= 1.0, "CAMERA_FRAME_FILL must be in (0, 1]")
        require(0.0 < self.MIN_VIRTUAL_HFOV_DEG < 180.0, "MIN_VIRTUAL_HFOV_DEG must be in (0, 180)")
        require(0.0 < self.MAX_VIRTUAL_HFOV_DEG < 180.0, "MAX_VIRTUAL_HFOV_DEG must be in (0, 180)")
        require(
            self.MIN_VIRTUAL_HFOV_DEG < self.MAX_VIRTUAL_HFOV_DEG,
            "MIN_VIRTUAL_HFOV_DEG must be < MAX_VIRTUAL_HFOV_DEG",
        )

        require(self.BACKPROJECT_STRIDE >= 1, "BACKPROJECT_STRIDE must be >= 1")
        require(self.SPLAT_RADIUS >= 0, "SPLAT_RADIUS must be >= 0")
        require(self.MAX_RANSAC_POINTS > 0, "MAX_RANSAC_POINTS must be > 0")

        require(self.POST_WARP_SUPER_RESOLUTION_SCALE > 1.0, "POST_WARP_SUPER_RESOLUTION_SCALE must be > 1.0")
        require(self.POST_WARP_SUPER_RESOLUTION_TILE > 0, "POST_WARP_SUPER_RESOLUTION_TILE must be > 0")
        require(self.POST_WARP_SUPER_RESOLUTION_TILE_PAD >= 0, "POST_WARP_SUPER_RESOLUTION_TILE_PAD must be >= 0")
        require(self.POST_WARP_MAX_OUTPUT_DIMENSION > 0, "POST_WARP_MAX_OUTPUT_DIMENSION must be > 0")

        require(
            self.USE_DIFFUSION is False,
            "USE_DIFFUSION=True is not supported by this pipeline (no diffusion backend is "
            "implemented -- that is the whole point of this rewrite). Leave it False.",
        )

        if errors:
            raise ValueError("Invalid PipelineConfig:\n  - " + "\n  - ".join(errors))
