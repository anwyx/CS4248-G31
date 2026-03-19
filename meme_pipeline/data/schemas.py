"""Typed schemas for meme pipeline inputs and outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class MetaphorPair(BaseModel):
    """Gold metaphor supervision pair."""

    metaphor: str
    meaning: str


class RawMemeSample(BaseModel):
    """Primary raw JSONL schema for the meme pipeline."""

    category: str = ""
    img_captions: list[str] = Field(default_factory=list)
    meme_captions: list[str] = Field(default_factory=list)
    title: str = ""
    url: str | None = None
    img_fname: str
    post_id: str
    metaphors: list[MetaphorPair] | None = None
    ocr_text: str = ""
    image_path: str = ""

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_keys(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        data = dict(payload)
        if "post_id" not in data and "id" in data:
            data["post_id"] = data["id"]
        if "img_fname" not in data and "image_path" in data:
            data["img_fname"] = Path(str(data["image_path"])).name
        if "img_captions" not in data and "literal_caption" in data:
            literal = data.get("literal_caption") or ""
            data["img_captions"] = [literal] if literal else []
        if "meme_captions" not in data and "gold_meme_caption" in data:
            gold = data.get("gold_meme_caption")
            data["meme_captions"] = [gold] if gold else []
        if "metaphors" not in data and "vehicle_target_pairs" in data:
            metaphors = []
            for pair in data.get("vehicle_target_pairs") or []:
                metaphors.append(
                    {
                        "metaphor": pair.get("vehicle", ""),
                        "meaning": pair.get("target", ""),
                    }
                )
            data["metaphors"] = metaphors
        return data

    @field_validator("title", "ocr_text", mode="before")
    @classmethod
    def _default_empty_string(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("img_captions", "meme_captions", mode="before")
    @classmethod
    def _ensure_list_of_strings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item) for item in value if str(item).strip()]

    @property
    def id(self) -> str:
        """Compatibility alias used across the codebase."""

        return self.post_id


class CandidateVehicle(BaseModel):
    """Vehicle extracted from one image caption."""

    caption_index: int
    surface: str
    normalized: str
    head: str
    start_char: int
    end_char: int
    source: str = "noun_chunk"


class StageAInstance(BaseModel):
    """Single candidate-centric Stage A training/inference item."""

    id: str
    sample_id: str
    image_path: str
    title: str = ""
    ocr_text: str = ""
    img_captions: list[str] = Field(default_factory=list)
    caption_index: int = 0
    vehicle_surface: str
    vehicle_normalized: str
    vehicle_head: str
    bbox_xyxy: list[float] | None = None
    crop_path: str | None = None
    target_label: str = "OTHER"
    target_id: int = 0


class StageAMetaphorMapping(BaseModel):
    """Predicted Stage A mapping for one vehicle candidate."""

    vehicle_surface: str
    vehicle_normalized: str
    vehicle_head: str = ""
    caption_index: int = 0
    bbox_xyxy: list[float] | None = None
    grounding_score: float = 0.0
    predicted_target: str
    predicted_target_id: int
    target_confidence: float
    topk_targets: list[tuple[str, float]] = Field(default_factory=list)


class StageAInferenceRecord(BaseModel):
    """Stage A JSONL output row."""

    id: str
    metaphor_mappings: list[StageAMetaphorMapping] = Field(default_factory=list)


class CaptionCandidate(BaseModel):
    """Candidate Stage B caption."""

    text: str
    score: float
    penalties: dict[str, float] = Field(default_factory=dict)


class StageBInferenceRecord(BaseModel):
    """Stage B JSONL output row."""

    id: str
    predicted_targets: list[tuple[str, float]] = Field(default_factory=list)
    predicted_mappings: list[tuple[str, str]] = Field(default_factory=list)
    candidate_captions: list[CaptionCandidate] = Field(default_factory=list)
    best_caption: str = ""


class GroundingResultModel(BaseModel):
    """Runtime validation helper for grounding results."""

    query: str
    bbox_xyxy: list[float] | None = None
    score: float
    crop_path: str | None = None
    used_model: str
    status: Literal["ok", "not_found", "error"]
