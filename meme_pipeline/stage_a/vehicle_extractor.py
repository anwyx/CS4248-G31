"""Vehicle extraction from literal image captions."""

from __future__ import annotations

from typing import Any

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


def _make_candidate(span: Any, nlp) -> dict[str, Any] | None:
    surface = span.text.strip()
    normalized = canonicalize_phrase(surface, nlp)
    if not normalized or normalized in JUNK_PHRASES:
        return None
    token_count = len([token for token in normalized.split() if token])
    if token_count < 1 or token_count > 5:
        return None
    head = canonicalize_phrase(getattr(span.root, "text", surface), nlp) or normalized.split()[-1]
    return {
        "surface": surface,
        "normalized": normalized,
        "head": head,
        "start_char": int(span.start_char),
        "end_char": int(span.end_char),
        "source": "noun_chunk",
    }


def extract_vehicle_candidates(literal_caption: str, nlp) -> list[dict[str, Any]]:
    """Extract noun-phrase vehicle candidates from literal caption."""

    if not literal_caption.strip():
        return []
    doc = nlp(literal_caption)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    noun_chunks = list(getattr(doc, "noun_chunks", []))
    for chunk in noun_chunks:
        candidate = _make_candidate(chunk, nlp)
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
                "surface": token.text,
                "normalized": normalized,
                "head": normalized,
                "start_char": int(token.idx),
                "end_char": int(token.idx + len(token.text)),
                "source": "pos_fallback",
            }
        )
    return sorted(candidates, key=lambda item: item["start_char"])
