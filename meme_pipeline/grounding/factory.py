"""Grounder factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meme_pipeline.grounding.base import Grounder
from meme_pipeline.grounding.grounding_dino_wrapper import GroundingDinoGrounder


class NoOpGrounder(Grounder):
    """Grounder used only when grounding is explicitly disabled."""

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
    """Load Grounding DINO locally or fail clearly."""

    if not config.get("use_grounding", True):
        return NoOpGrounder()
    model_path = Path(config.get("grounding_model_name_or_path", "models/grounding_dino"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Grounding model folder not found: {model_path}. "
            "Set use_grounding=false or place Grounding DINO at the configured path."
        )
    grounder = GroundingDinoGrounder(
        model_name=str(model_path),
        box_threshold=float(config.get("grounding_box_threshold", 0.3)),
        text_threshold=float(config.get("grounding_text_threshold", 0.25)),
        crop_dir=config.get("grounding_crop_dir"),
    )
    if grounder._model is None:
        message = grounder._load_error or "Grounding DINO failed to load."
        raise RuntimeError(
            f"Grounding is enabled but Grounding DINO could not be loaded from {model_path}: {message}"
        )
    return grounder
