"""Stage B candidate cleanup and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from meme_pipeline.utils.text_norm import normalize_case


def _tokenize(text: str) -> list[str]:
    return [token for token in normalize_case(text).split() if token]


def _forbidden_term_fraction(text: str, forbidden_terms: Sequence[str]) -> float:
    norm_text = normalize_case(text)
    if not forbidden_terms:
        return 0.0
    matches = sum(1 for term in forbidden_terms if term and normalize_case(term) in norm_text)
    return matches / max(len(forbidden_terms), 1)


def _copy_fraction(text: str, source: str) -> float:
    source_tokens = set(_tokenize(source))
    text_tokens = set(_tokenize(text))
    if not source_tokens:
        return 0.0
    return len(source_tokens & text_tokens) / len(source_tokens)


def _target_overlap_score(text: str, targets: Sequence[str]) -> float:
    target_tokens = set(token for target in targets for token in _tokenize(target))
    text_tokens = set(_tokenize(text))
    if not target_tokens:
        return 0.0
    return len(target_tokens & text_tokens) / len(target_tokens)


def _mapping_overlap_score(text: str, mappings: Sequence[tuple[str, str]]) -> float:
    target_side = [mapping[1] for mapping in mappings if len(mapping) == 2]
    return _target_overlap_score(text, target_side)


def _semantic_similarity_proxy(text: str, reference_text: str) -> float:
    if not reference_text.strip():
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return _copy_fraction(text, reference_text)
    try:
        model = _load_sentence_transformer()
    except Exception:
        return _copy_fraction(text, reference_text)
    embeddings = model.encode([text, reference_text], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)


@dataclass
class CandidateScore:
    """Scored candidate output."""

    text: str
    score: float
    penalties: dict[str, float]


def clean_generation_text(text: str) -> str:
    """Strip prompt leakage and boilerplate."""

    cleaned = text.strip()
    for marker in ["Predicted metaphor mappings", "Predicted target concepts", "Task:", "Expected output style:", "User:", "System:"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, maxsplit=1)[0].strip()
    cleaned = cleaned.replace("\n", " ").strip(" -")
    return " ".join(cleaned.split())


def is_valid_candidate(
    text: str,
    *,
    title: str,
    ocr_text: str,
    forbidden_terms: Sequence[str],
) -> bool:
    """Apply Stage B output validity checks."""

    cleaned = clean_generation_text(text)
    if len(cleaned.split()) < 3:
        return False
    if cleaned.lower().startswith("predicted metaphor mappings"):
        return False
    if _copy_fraction(cleaned, title) > 0.8 and len(title.split()) >= 3:
        return False
    if _copy_fraction(cleaned, ocr_text) > 0.8 and len(ocr_text.split()) >= 3:
        return False
    if _forbidden_term_fraction(cleaned, forbidden_terms) > 0.5:
        return False
    return True


def dedupe_candidates(candidates: Sequence[str]) -> list[str]:
    """Remove near-duplicate candidates."""

    seen: list[str] = []
    for candidate in candidates:
        cleaned = clean_generation_text(candidate)
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def rerank_candidates(
    candidates: Sequence[str],
    *,
    mappings: Sequence[tuple[str, str]],
    targets: Sequence[str],
    title: str,
    ocr_text: str,
    image_captions: Sequence[str],
    vehicle_terms: Sequence[str],
    vehicle_penalty_weight: float,
    ocr_penalty_weight: float,
    title_penalty_weight: float,
) -> list[CandidateScore]:
    """Score candidate captions with mapping relevance and copy penalties."""

    mapping_reference = " ; ".join(f"{vehicle} -> {target}" for vehicle, target in mappings)
    caption_reference = " ".join(image_captions)
    scored: list[CandidateScore] = []
    for candidate in dedupe_candidates(candidates):
        length = len(candidate.split())
        verbosity_penalty = 0.0 if 8 <= length <= 28 else min(abs(length - 18) / 18, 1.0)
        vehicle_penalty = _forbidden_term_fraction(candidate, vehicle_terms)
        ocr_penalty = _copy_fraction(candidate, ocr_text)
        title_penalty = _copy_fraction(candidate, title)
        caption_copy_penalty = _copy_fraction(candidate, caption_reference)
        target_coverage = _target_overlap_score(candidate, targets)
        mapping_coverage = _mapping_overlap_score(candidate, mappings)
        semantic_similarity = _semantic_similarity_proxy(candidate, mapping_reference or ", ".join(targets))
        score = (
            target_coverage
            + mapping_coverage
            + semantic_similarity
            - vehicle_penalty_weight * vehicle_penalty
            - ocr_penalty_weight * ocr_penalty
            - title_penalty_weight * title_penalty
            - 0.25 * caption_copy_penalty
            - 0.3 * verbosity_penalty
        )
        scored.append(
            CandidateScore(
                text=candidate,
                score=float(score),
                penalties={
                    "vehicle_copy": float(vehicle_penalty),
                    "ocr_copy": float(ocr_penalty),
                    "title_copy": float(title_penalty),
                    "caption_copy": float(caption_copy_penalty),
                    "verbosity": float(verbosity_penalty),
                },
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored
