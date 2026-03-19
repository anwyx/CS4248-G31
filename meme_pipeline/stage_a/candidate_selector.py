"""Heuristic ranking for candidate vehicles."""

from __future__ import annotations

from typing import Any

from meme_pipeline.utils.text_norm import normalize_case

ANIMATE_HINTS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "guy",
    "lady",
    "cat",
    "dog",
    "baby",
    "child",
    "student",
    "worker",
    "employee",
    "boss",
    "teacher",
    "friend",
}
ROLE_HINTS = {"boss", "teacher", "officer", "parent", "manager", "worker", "employee", "student"}
EMOTION_HINTS = {
    "confused",
    "angry",
    "sad",
    "happy",
    "worried",
    "scared",
    "tired",
    "crying",
    "smiling",
}


def rank_or_filter_candidates(
    candidates: list[dict[str, Any]],
    title: str,
    ocr_text: str,
    image_captions: list[str] | str,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    """Rank candidate vehicles using lightweight salience heuristics."""

    title_norm = normalize_case(title)
    ocr_norm = normalize_case(ocr_text)
    if isinstance(image_captions, str):
        caption_norm = normalize_case(image_captions)
    else:
        caption_norm = normalize_case(" ".join(image_captions))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        score = 0.0
        normalized = candidate["normalized"]
        head = candidate["head"]
        tokens = normalized.split()
        if head in ANIMATE_HINTS:
            score += 2.0
        if head in ROLE_HINTS:
            score += 1.5
        if any(token in EMOTION_HINTS for token in tokens):
            score += 1.0
        if normalized in title_norm or head in title_norm:
            score += 0.75
        if normalized in ocr_norm or head in ocr_norm:
            score += 0.75
        if normalized in caption_norm:
            score += 0.25
        score += max(0.0, 0.4 - 0.05 * index)
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]["start_char"]))
    return [candidate for _, candidate in ranked[:max_candidates]]
