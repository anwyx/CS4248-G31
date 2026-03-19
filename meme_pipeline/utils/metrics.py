"""Metrics for Stage A and Stage B."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from meme_pipeline.utils.text_norm import normalize_case


def stage_a_accuracy(labels: Sequence[int], preds: Sequence[int]) -> float:
    """Compute accuracy."""

    return float(accuracy_score(labels, preds)) if labels else 0.0


def stage_a_macro_f1(labels: Sequence[int], preds: Sequence[int]) -> float:
    """Compute macro F1."""

    return float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0


def stage_a_weighted_f1(labels: Sequence[int], preds: Sequence[int]) -> float:
    """Compute weighted F1."""

    return float(f1_score(labels, preds, average="weighted", zero_division=0)) if labels else 0.0


def topk_accuracy(labels: Sequence[int], pred_topk: Sequence[Sequence[int]], k: int) -> float:
    """Compute top-k accuracy."""

    if not labels:
        return 0.0
    correct = 0
    for label, topk in zip(labels, pred_topk):
        if label in list(topk)[:k]:
            correct += 1
    return correct / len(labels)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _normalize_references(references: Sequence[str] | Sequence[Sequence[str]]) -> list[list[str]]:
    if not references:
        return []
    if isinstance(references[0], str):  # type: ignore[index]
        return [[str(item)] for item in references]  # type: ignore[list-item]
    return [[str(item) for item in group if str(item).strip()] for group in references]  # type: ignore[list-item]


def _best_reference(reference_group: Sequence[str], hypothesis: str, scorer) -> float:
    return max((scorer(reference, hypothesis) for reference in reference_group), default=0.0)


def _sentence_bleu(reference: str, hypothesis: str) -> float:
    precisions: list[float] = []
    ref_tokens = normalize_case(reference).split()
    hyp_tokens = normalize_case(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    for n in range(1, 5):
        ref_counts = Counter(_ngrams(ref_tokens, n))
        hyp_counts = Counter(_ngrams(hyp_tokens, n))
        total = max(sum(hyp_counts.values()), 1)
        matches = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append(matches / total if total else 0.0)
    if min(precisions) == 0:
        return 0.0
    brevity_penalty = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4)


def bleu4(references: Sequence[str] | Sequence[Sequence[str]], hypotheses: Sequence[str]) -> float:
    """Compute BLEU-4 with single-reference or multi-reference support."""

    normalized_references = _normalize_references(references)
    if not normalized_references or not hypotheses:
        return 0.0
    scores = [
        _best_reference(reference_group, hypothesis, _sentence_bleu)
        for reference_group, hypothesis in zip(normalized_references, hypotheses)
    ]
    return float(np.mean(scores)) if scores else 0.0


def _rouge_l_single(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_case(reference).split()
    hyp_tokens = normalize_case(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    table = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i, token_a in enumerate(ref_tokens, start=1):
        for j, token_b in enumerate(hyp_tokens, start=1):
            if token_a == token_b:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(references: Sequence[str] | Sequence[Sequence[str]], hypotheses: Sequence[str]) -> float:
    """Compute ROUGE-L using the best score over each reference set."""

    normalized_references = _normalize_references(references)
    if not normalized_references or not hypotheses:
        return 0.0
    scores = [
        _best_reference(reference_group, hypothesis, _rouge_l_single)
        for reference_group, hypothesis in zip(normalized_references, hypotheses)
    ]
    return float(np.mean(scores)) if scores else 0.0


def bertscore(references: Sequence[str] | Sequence[Sequence[str]], hypotheses: Sequence[str]) -> float | None:
    """Compute optional BERTScore using the max F1 over each reference set."""

    try:
        from bert_score import score
    except ImportError:
        return None
    normalized_references = _normalize_references(references)
    if not normalized_references or not hypotheses:
        return 0.0
    scores: list[float] = []
    for reference_group, hypothesis in zip(normalized_references, hypotheses):
        best = 0.0
        for reference in reference_group:
            _, _, f1 = score([hypothesis], [reference], lang="en", verbose=False)
            best = max(best, float(f1.mean().item()))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Compute distinct-n diversity."""

    all_ngrams: list[tuple[str, ...]] = []
    all_tokens = 0
    for text in texts:
        tokens = normalize_case(text).split()
        all_tokens += len(tokens)
        all_ngrams.extend(_ngrams(tokens, n))
    if not all_ngrams or all_tokens == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def distinct_1(texts: Sequence[str]) -> float:
    """Compute distinct-1."""

    return distinct_n(texts, 1)


def distinct_2(texts: Sequence[str]) -> float:
    """Compute distinct-2."""

    return distinct_n(texts, 2)


def copy_rate(texts: Sequence[str], forbidden_sources: Iterable[str]) -> float:
    """Fraction of texts that copy any forbidden source substring."""

    sources = [normalize_case(item) for item in forbidden_sources if item]
    if not texts:
        return 0.0
    copied = 0
    for text in texts:
        norm_text = normalize_case(text)
        if any(source and source in norm_text for source in sources):
            copied += 1
    return copied / len(texts)


def title_copy_rate(texts: Sequence[str], titles: Sequence[str]) -> float:
    """Compute title copy rate."""

    return copy_rate(texts, titles)


def ocr_copy_rate(texts: Sequence[str], ocr_texts: Sequence[str]) -> float:
    """Compute OCR copy rate."""

    return copy_rate(texts, ocr_texts)
