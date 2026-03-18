"""Grounding DINO wrapper with graceful fallback behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meme_pipeline.grounding.base import Grounder, GroundingResult
from meme_pipeline.utils.image_utils import save_crop
from meme_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)


class GroundingDinoGrounder(Grounder):
    """Ground phrases with Grounding DINO if available."""

    def __init__(
        self,
        *,
        model_name: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        crop_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.crop_dir = crop_dir
        self._processor = None
        self._model = None
        self._load_error: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            self._load_error = str(exc)
            LOGGER.warning("Grounding DINO unavailable: %s", exc)
            return
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on runtime/model availability
            self._load_error = str(exc)
            LOGGER.warning("Failed to load Grounding DINO model %s: %s", self.model_name, exc)

    def _empty(self, query: str, status: str = "not_found") -> GroundingResult:
        return GroundingResult(
            query=query,
            bbox_xyxy=None,
            score=0.0,
            crop_path=None,
            used_model=self.model_name,
            status=status,  # type: ignore[arg-type]
        )

    def _run_query(self, image_path: str, query: str) -> GroundingResult:
        if self._model is None or self._processor is None:
            return self._empty(query, status="error" if self._load_error else "not_found")
        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            text = query if query.endswith(".") else f"{query}."
            inputs = self._processor(images=image, text=text, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            if len(results["scores"]) == 0:
                return self._empty(query)
            best_index = int(results["scores"].argmax().item())
            score = float(results["scores"][best_index].item())
            if score < self.box_threshold:
                return self._empty(query)
            bbox = [float(value) for value in results["boxes"][best_index].tolist()]
            crop_path = None
            if self.crop_dir:
                filename = f"{Path(image_path).stem}_{query.replace(' ', '_')}.jpg"
                crop_path = save_crop(image_path, bbox, Path(self.crop_dir) / filename)
            return GroundingResult(
                query=query,
                bbox_xyxy=bbox,
                score=score,
                crop_path=crop_path,
                used_model=self.model_name,
                status="ok",
            )
        except Exception as exc:  # pragma: no cover - depends on runtime/model availability
            LOGGER.warning("Grounding DINO query failed for '%s': %s", query, exc)
            return self._empty(query, status="error")

    def ground(self, image_path: str, query: str, *, head_query: str | None = None) -> GroundingResult:
        """Run full phrase, then head noun fallback."""

        result = self._run_query(image_path, query)
        if result.status == "ok" or not head_query or head_query == query:
            return result
        return self._run_query(image_path, head_query)
