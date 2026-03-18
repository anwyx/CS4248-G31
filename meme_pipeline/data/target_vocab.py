"""Closed target vocabulary helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from meme_pipeline.data.io import load_raw_samples
from meme_pipeline.utils.logging import get_logger
from meme_pipeline.utils.text_norm import canonicalize_phrase

LOGGER = get_logger(__name__)


class TargetVocab:
    """Closed target vocabulary with bidirectional mappings."""

    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {index: token for token, index in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, label: str) -> int:
        return self.stoi.get(label, self.stoi["OTHER"])

    def decode(self, index: int) -> str:
        return self.itos.get(index, "OTHER")

    def to_dict(self) -> dict[str, dict[str, int] | dict[int, str]]:
        return {"stoi": self.stoi, "itos": self.itos}


def build_target_vocab(
    train_jsonl: str | Path,
    *,
    min_freq: int = 1,
    nlp=None,
) -> TargetVocab:
    """Build target vocabulary from gold Stage A labels."""

    counter: Counter[str] = Counter()
    for sample in load_raw_samples(train_jsonl):
        for pair in sample.vehicle_target_pairs or []:
            normalized = canonicalize_phrase(pair.target, nlp) if nlp is not None else pair.target.lower().strip()
            if normalized:
                counter[normalized] += 1
    stoi = {"OTHER": 0, "NO_TARGET": 1}
    for label, freq in sorted(counter.items()):
        if freq >= min_freq and label not in stoi:
            stoi[label] = len(stoi)
    LOGGER.info("Built target vocab with %s labels", len(stoi))
    return TargetVocab(stoi)


def save_target_vocab(path: str | Path, vocab: TargetVocab) -> None:
    """Persist target vocabulary to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(vocab.to_dict(), handle, ensure_ascii=False, indent=2)


def load_target_vocab(path: str | Path) -> TargetVocab:
    """Load target vocabulary from JSON."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Target vocab not found at {path}. Run Stage A training or provide target_vocab_path."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TargetVocab({key: int(value) for key, value in payload["stoi"].items()})
