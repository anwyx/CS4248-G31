"""Grounding abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class GroundingResult:
    """Result schema for a grounded query."""

    query: str
    bbox_xyxy: list[float] | None
    score: float
    crop_path: str | None
    used_model: str
    status: Literal["ok", "not_found", "error"]

    def to_dict(self) -> dict[str, object]:
        """Convert to plain dict."""

        return asdict(self)


class Grounder(ABC):
    """Abstract grounding interface."""

    @abstractmethod
    def ground(self, image_path: str, query: str, *, head_query: str | None = None) -> GroundingResult:
        """Ground a phrase in an image."""
