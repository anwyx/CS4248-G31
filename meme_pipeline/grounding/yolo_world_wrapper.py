"""YOLO-World or YOLOE open-vocabulary grounding fallback."""

from __future__ import annotations

from pathlib import Path

from meme_pipeline.grounding.base import Grounder, GroundingResult
from meme_pipeline.utils.image_utils import save_crop
from meme_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)


class YoloWorldGrounder(Grounder):
    """Use ultralytics YOLO-World when available."""

    def __init__(self, *, model_name: str, crop_dir: str | None = None) -> None:
        self.model_name = model_name
        self.crop_dir = crop_dir
        self._model = None
        self._load_error: str | None = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            self._load_error = str(exc)
            LOGGER.warning("YOLO-World unavailable: %s", exc)
            return
        try:
            self._model = YOLOWorld(self.model_name)
        except Exception as exc:  # pragma: no cover - runtime dependent
            self._load_error = str(exc)
            LOGGER.warning("Failed to load YOLO-World model %s: %s", self.model_name, exc)

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
        if self._model is None:
            return self._empty(query, status="error" if self._load_error else "not_found")
        try:
            self._model.set_classes([query])
            results = self._model.predict(image_path, verbose=False)
            if not results:
                return self._empty(query)
            boxes = results[0].boxes
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                return self._empty(query)
            score = float(boxes.conf[0].item())
            bbox = [float(value) for value in boxes.xyxy[0].tolist()]
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
        except Exception as exc:  # pragma: no cover - runtime dependent
            LOGGER.warning("YOLO-World query failed for '%s': %s", query, exc)
            return self._empty(query, status="error")

    def ground(self, image_path: str, query: str, *, head_query: str | None = None) -> GroundingResult:
        """Run full phrase, then head noun fallback."""

        result = self._run_query(image_path, query)
        if result.status == "ok" or not head_query or head_query == query:
            return result
        return self._run_query(image_path, head_query)
