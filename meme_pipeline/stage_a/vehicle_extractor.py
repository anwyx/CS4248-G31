"""Vehicle extraction from image captions."""

from __future__ import annotations

from typing import Any, Sequence

from meme_pipeline.utils.text_norm import canonicalize_phrase

JUNK_PHRASES = {
    "image",
    "photo",
    "picture",
    "something",
    "scene",
    "meme",
    "text",
    "caption",
}


def load_spacy_or_fail(model_name: str = "en_core_web_sm"):
    """Load spaCy model or raise a clear actionable error."""

    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError("spaCy is required. Install it with `pip install spacy` and download a model.") from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"spaCy model '{model_name}' is missing. Install it with `python -m spacy download {model_name}`."
        ) from exc


def _make_candidate(span: Any, nlp, caption_index: int) -> dict[str, Any] | None:
    surface = span.text.strip()
    normalized = canonicalize_phrase(surface, nlp)
    if not normalized or normalized in JUNK_PHRASES:
        return None
    token_count = len([token for token in normalized.split() if token])
    if token_count < 1 or token_count > 5:
        return None
    head = canonicalize_phrase(getattr(span.root, "text", surface), nlp) or normalized.split()[-1]
    return {
        "caption_index": caption_index,
        "surface": surface,
        "normalized": normalized,
        "head": head,
        "start_char": int(span.start_char),
        "end_char": int(span.end_char),
        "source": "noun_chunk",
    }


def extract_vehicle_candidates(caption_text: str, nlp, *, caption_index: int = 0) -> list[dict[str, Any]]:
    """Extract noun-phrase vehicle candidates from a single image caption."""

    if not caption_text.strip():
        return []
    doc = nlp(caption_text)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    noun_chunks = list(getattr(doc, "noun_chunks", []))
    for chunk in noun_chunks:
        candidate = _make_candidate(chunk, nlp, caption_index)
        if candidate is None or candidate["normalized"] in seen:
            continue
        seen.add(candidate["normalized"])
        candidates.append(candidate)
    if candidates:
        return sorted(candidates, key=lambda item: item["start_char"])
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        normalized = canonicalize_phrase(token.text, nlp)
        if not normalized or normalized in seen or normalized in JUNK_PHRASES:
            continue
        if len(normalized.split()) > 5:
            continue
        seen.add(normalized)
        candidates.append(
            {
                "caption_index": caption_index,
                "surface": token.text,
                "normalized": normalized,
                "head": normalized,
                "start_char": int(token.idx),
                "end_char": int(token.idx + len(token.text)),
                "source": "pos_fallback",
            }
        )
    return sorted(candidates, key=lambda item: item["start_char"])


def extract_vehicle_candidates_from_captions(
    img_captions: Sequence[str],
    nlp,
) -> list[dict[str, Any]]:
    """Extract and deduplicate candidates across all image captions."""

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for caption_index, caption_text in enumerate(img_captions):
        for candidate in extract_vehicle_candidates(caption_text, nlp, caption_index=caption_index):
            if candidate["normalized"] in seen:
                continue
            seen.add(candidate["normalized"])
            merged.append(candidate)
    return sorted(merged, key=lambda item: (item["caption_index"], item["start_char"]))
