"""Stage A postprocessing helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def merge_duplicate_metaphor_mappings(
    metaphor_mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge duplicates by normalized vehicle string, keeping highest confidence."""

    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for mapping in metaphor_mappings:
        key = mapping["vehicle_normalized"]
        if key not in merged or mapping["target_confidence"] > merged[key]["target_confidence"]:
            merged[key] = mapping
    return list(merged.values())


def apply_confidence_threshold(
    metaphor_mappings: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Map low-confidence predictions to OTHER."""

    for mapping in metaphor_mappings:
        if mapping["target_confidence"] < threshold:
            mapping["predicted_target"] = "OTHER"
            mapping["predicted_target_id"] = 0
    return metaphor_mappings


def finalize_stage_a_predictions(
    metaphor_mappings: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Apply deduplication, confidence filtering, and sorting."""

    merged = merge_duplicate_metaphor_mappings(metaphor_mappings)
    filtered = apply_confidence_threshold(merged, threshold=threshold)
    return sorted(filtered, key=lambda item: item["target_confidence"], reverse=True)
