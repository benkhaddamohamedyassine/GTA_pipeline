"""crop_aerial_pipeline -- ground-level crop photos -> centered pseudo-aerial
views, without diffusion.

    from crop_aerial_pipeline import run_pipeline
    summary = run_pipeline("/content/drive/MyDrive/crop_images")

See the package README for the full module map, Colab setup, and
troubleshooting notes.
"""

from .config import PipelineConfig
from .io.paths import PipelinePaths
from .manifest import ImageRecord, Manifest
from .runner import ImageProcessingResult, clean_temp_files, process_single_image, run_pipeline
from .utils.visualization import show_diagnostic_panel

__all__ = [
    "run_pipeline",
    "process_single_image",
    "clean_temp_files",
    "PipelineConfig",
    "PipelinePaths",
    "ImageRecord",
    "Manifest",
    "ImageProcessingResult",
    "show_diagnostic_panel",
]

__version__ = "0.1.0"
