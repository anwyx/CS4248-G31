"""Phrase normalization helpers."""

from __future__ import annotations

import re
import string
from difflib import SequenceMatcher

DETERMINERS = ("a", "an", "the", "this", "that", "these", "those")
BOUNDARY_PUNCT = string.punctuation + "“”‘’"


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""

    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_case(text: str) -> str:
    """Lowercase text."""

    return normalize_whitespace(text).lower()


def _trim_boundary_punct(text: str) -> str:
    return text.strip(BOUNDARY_PUNCT + " ")


def strip_determiners(phrase: str) -> str:
    """Remove leading determiners."""

    normalized = normalize_case(_trim_boundary_punct(phrase))
    for determiner in DETERMINERS:
        prefix = determiner + " "
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def lemmatize_phrase_spacy(phrase: str, nlp) -> str:
    """Lightly lemmatize noun/adjective tokens with spaCy."""

    if nlp is None:
        return strip_determiners(phrase)
    doc = nlp(strip_determiners(phrase))
    pieces: list[str] = []
    for token in doc:
        lemma = token.lemma_.strip() or token.text
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
            pieces.append(lemma.lower())
        else:
            pieces.append(token.text.lower())
    return normalize_whitespace(" ".join(pieces))


def canonicalize_phrase(phrase: str, nlp) -> str:
    """Canonicalize phrase for matching and deduplication."""

    cleaned = _trim_boundary_punct(normalize_whitespace(phrase))
    if not cleaned:
        return ""
    cleaned = lemmatize_phrase_spacy(cleaned, nlp)
    cleaned = strip_determiners(cleaned)
    cleaned = _trim_boundary_punct(cleaned)
    if len(cleaned) <= 1:
        return ""
    return cleaned


def remove_duplicate_phrases(phrases: list[str]) -> list[str]:
    """Remove exact and near-duplicate phrases while preserving order."""

    unique: list[str] = []
    seen: list[str] = []
    for phrase in phrases:
        candidate = canonicalize_phrase(phrase, None)
        if not candidate:
            continue
        if candidate in seen:
            continue
        if any(SequenceMatcher(None, candidate, prior).ratio() >= 0.92 for prior in seen):
            continue
        seen.append(candidate)
        unique.append(candidate)
    return unique
