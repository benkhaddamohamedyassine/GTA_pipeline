"""Single source of truth for every tunable pipeline parameter.

``PipelineConfig`` is a plain, validated dataclass -- no hidden global state.
Every stage receives it explicitly, and its hash (``utils.hashing.hash_config``)
is stored in the manifest so a resumed run can tell whether the user changed a
setting that invalidates previously-cached stage outputs.
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

    # --- Stage 2: super-resolution ------------------------------------------
    SUPER_RESOLUTION_ENABLED: bool = True
    SUPER_RESOLUTION_MODEL: str = "RealESRGAN_x2plus"
    SUPER_RESOLUTION_SCALE: float = 2.0
    SUPER_RESOLUTION_TILE: int = 256
    SUPER_RESOLUTION_TILE_PAD: int = 16
    SUPER_RESOLUTION_HALF_PRECISION: bool = True
    MAX_SUPER_RES_DIMENSION: int = 4096

    # --- Stage 6/7: camera geometry -----------------------------------------
    ASSUMED_HFOV_DEG: float = 70.0
    ALTITUDE_M: float = 10.0
    TILT_DEGREES: float = 0.0

    # --- Stage 5: source extension ------------------------------------------
    SOURCE_PAD_FRAC: float = 0.45
    SOURCE_EXTENSION_MODE: str = "texture_synthesis"  # or "reflection"
    REFLECTION_FALLBACK: bool = True

    # --- Stage 4/7: crop-interior centering ----------------------------------
    CROP_INTERIOR_QUANTILE: float = 0.60
    CAMERA_FRAME_FILL: float = 0.90

    # --- Stage 8: virtual intrinsics -----------------------------------------
    MIN_VIRTUAL_HFOV_DEG: float = 20.0
    MAX_VIRTUAL_HFOV_DEG: float = 52.0

    # --- Stage 6/8: point cloud / rendering ----------------------------------
    BACKPROJECT_STRIDE: int = 2
    SPLAT_RADIUS: int = 2
    MAX_RANSAC_POINTS: int = 250_000

    # --- Output verbosity ------------------------------------------------
    SAVE_POINT_CLOUD: bool = True
    SAVE_DEPTH_ARRAY: bool = True
    SAVE_DEBUG_VISUALIZATION: bool = True

    # --- Refinement --------------------------------------------------------
    # USE_DIFFUSION exists only so callers can assert the pipeline is running
    # in its safe, non-generative mode. There is no diffusion backend in this
    # package (Stage 10 raises NotImplementedError if this is set True) --
    # diffusion refinement is explicitly what this pipeline replaces.
    USE_DIFFUSION: bool = False
    RANDOM_SEED: int = 42

    def validate(self) -> None:
        """Raises ``ValueError`` on the first invalid field. Called once by
        ``run_pipeline`` before any stage executes, so a typo'd config fails
        fast instead of partway through a long batch.
        """
        errors = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        require(self.SUPER_RESOLUTION_SCALE > 1.0, "SUPER_RESOLUTION_SCALE must be > 1.0")
        require(self.SUPER_RESOLUTION_TILE > 0, "SUPER_RESOLUTION_TILE must be > 0")
        require(self.SUPER_RESOLUTION_TILE_PAD >= 0, "SUPER_RESOLUTION_TILE_PAD must be >= 0")
        require(self.MAX_SUPER_RES_DIMENSION > 0, "MAX_SUPER_RES_DIMENSION must be > 0")

        require(0.0 < self.ASSUMED_HFOV_DEG < 180.0, "ASSUMED_HFOV_DEG must be in (0, 180)")
        require(self.ALTITUDE_M > 0.0, "ALTITUDE_M must be > 0")
        require(0.0 <= self.TILT_DEGREES < 90.0, "TILT_DEGREES must be in [0, 90)")

        require(self.SOURCE_PAD_FRAC >= 0.0, "SOURCE_PAD_FRAC must be >= 0")
        require(
            self.SOURCE_EXTENSION_MODE in ("texture_synthesis", "reflection"),
            "SOURCE_EXTENSION_MODE must be 'texture_synthesis' or 'reflection'",
        )

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

        require(
            self.USE_DIFFUSION is False,
            "USE_DIFFUSION=True is not supported by this pipeline (no diffusion backend is "
            "implemented -- that is the whole point of this rewrite). Leave it False.",
        )

        if errors:
            raise ValueError("Invalid PipelineConfig:\n  - " + "\n  - ".join(errors))
