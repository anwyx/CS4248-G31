"""Typed schemas for meme pipeline inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class VehicleTargetPair(BaseModel):
    """Gold metaphor alignment pair."""

    vehicle: str
    target: str


class RawMemeSample(BaseModel):
    """Raw JSONL sample contract."""

    id: str
    image_path: str
    title: str = ""
    ocr_text: str = ""
    literal_caption: str = ""
    gold_meme_caption: str | None = None
    vehicle_target_pairs: list[VehicleTargetPair] | None = None

    @field_validator("title", "ocr_text", "literal_caption", mode="before")
    @classmethod
    def _default_empty_string(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class CandidateVehicle(BaseModel):
    """Vehicle extracted from literal caption."""

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
    literal_caption: str = ""
    vehicle_surface: str
    vehicle_normalized: str
    vehicle_head: str
    bbox_xyxy: list[float] | None = None
    crop_path: str | None = None
    target_label: str = "OTHER"
    target_id: int = 0


class StageAVehiclePrediction(BaseModel):
    """Predicted target for one vehicle."""

    vehicle_surface: str
    vehicle_normalized: str
    bbox_xyxy: list[float] | None = None
    grounding_score: float = 0.0
    predicted_target: str
    predicted_target_id: int
    target_confidence: float
    topk_targets: list[tuple[str, float]] = Field(default_factory=list)


class StageAInferenceRecord(BaseModel):
    """Stage A JSONL output row."""

    id: str
    vehicles: list[StageAVehiclePrediction] = Field(default_factory=list)


class CaptionCandidate(BaseModel):
    """Candidate Stage B caption."""

    text: str
    score: float
    penalties: dict[str, float] = Field(default_factory=dict)


class StageBInferenceRecord(BaseModel):
    """Stage B JSONL output row."""

    id: str
    predicted_targets: list[tuple[str, float]] = Field(default_factory=list)
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
