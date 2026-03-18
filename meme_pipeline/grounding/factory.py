"""Grounder factory."""

from __future__ import annotations

from typing import Any

from meme_pipeline.grounding.base import Grounder
from meme_pipeline.grounding.grounding_dino_wrapper import GroundingDinoGrounder
from meme_pipeline.grounding.yolo_world_wrapper import YoloWorldGrounder
from meme_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)


class NoOpGrounder(Grounder):
    """Fallback grounder that always returns not_found."""

    def ground(self, image_path: str, query: str, *, head_query: str | None = None):
        from meme_pipeline.grounding.base import GroundingResult

        return GroundingResult(
            query=query,
            bbox_xyxy=None,
            score=0.0,
            crop_path=None,
            used_model="none",
            status="not_found",
        )


def load_grounder(config: dict[str, Any]) -> Grounder:
    """Load preferred grounder with automatic fallback."""

    if not config.get("use_grounding", True):
        return NoOpGrounder()
    backend = config.get("grounding_backend", "grounding_dino")
    crop_dir = config.get("grounding_crop_dir")
    dino = GroundingDinoGrounder(
        model_name=config.get("grounding_model_name", "IDEA-Research/grounding-dino-base"),
        box_threshold=float(config.get("grounding_box_threshold", 0.3)),
        text_threshold=float(config.get("grounding_text_threshold", 0.25)),
        crop_dir=crop_dir,
    )
    if backend == "grounding_dino" and dino._model is not None:
        return dino
    if backend == "grounding_dino":
        LOGGER.warning("Falling back from Grounding DINO to YOLO-World.")
    yolo = YoloWorldGrounder(
        model_name=config.get("grounding_fallback_model_name", "yolo_world/l"),
        crop_dir=crop_dir,
    )
    if yolo._model is not None:
        return yolo
    LOGGER.warning("No grounding backend available. Continuing without grounding.")
    return NoOpGrounder()
