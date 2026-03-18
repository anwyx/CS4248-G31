"""Stage A postprocessing helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def merge_duplicate_vehicle_predictions(
    vehicle_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge duplicates by normalized vehicle string, keeping highest confidence."""

    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for prediction in vehicle_predictions:
        key = prediction["vehicle_normalized"]
        if key not in merged or prediction["target_confidence"] > merged[key]["target_confidence"]:
            merged[key] = prediction
    return list(merged.values())


def apply_confidence_threshold(
    vehicle_predictions: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Map low-confidence predictions to OTHER."""

    for prediction in vehicle_predictions:
        if prediction["target_confidence"] < threshold:
            prediction["predicted_target"] = "OTHER"
            prediction["predicted_target_id"] = 0
    return vehicle_predictions


def finalize_stage_a_predictions(
    vehicle_predictions: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Apply deduplication, confidence filtering, and sorting."""

    merged = merge_duplicate_vehicle_predictions(vehicle_predictions)
    filtered = apply_confidence_threshold(merged, threshold=threshold)
    return sorted(filtered, key=lambda item: item["target_confidence"], reverse=True)
