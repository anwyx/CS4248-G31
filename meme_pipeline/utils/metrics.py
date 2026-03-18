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


def bleu4(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute a lightweight corpus BLEU-4."""

    if not references or not hypotheses:
        return 0.0
    precisions: list[float] = []
    hyp_len = 0
    ref_len = 0
    for n in range(1, 5):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = normalize_case(ref).split()
            hyp_tokens = normalize_case(hyp).split()
            hyp_len += len(hyp_tokens) if n == 1 else 0
            ref_len += len(ref_tokens) if n == 1 else 0
            ref_counts = Counter(_ngrams(ref_tokens, n))
            hyp_counts = Counter(_ngrams(hyp_tokens, n))
            matches += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            total += max(sum(hyp_counts.values()), 1)
        precisions.append(matches / total if total else 0.0)
    if min(precisions) == 0:
        return 0.0
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4)


def rouge_l(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Compute average ROUGE-L F score."""

    def lcs_length(a: list[str], b: list[str]) -> int:
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, token_a in enumerate(a, start=1):
            for j, token_b in enumerate(b, start=1):
                if token_a == token_b:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[-1][-1]

    scores: list[float] = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = normalize_case(ref).split()
        hyp_tokens = normalize_case(hyp).split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        lcs = lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def bertscore(references: Sequence[str], hypotheses: Sequence[str]) -> float | None:
    """Compute BERTScore if the optional dependency is installed."""

    try:
        from bert_score import score
    except ImportError:
        return None
    if not references or not hypotheses:
        return 0.0
    _, _, f1 = score(list(hypotheses), list(references), lang="en", verbose=False)
    return float(f1.mean().item())


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
