"""JSONL and config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import yaml
from pydantic import ValidationError

from meme_pipeline.data.schemas import RawMemeSample
from meme_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)
T = TypeVar("T")


def read_jsonl(
    path: str | Path,
    *,
    parser: Callable[[dict[str, Any]], T] | None = None,
) -> list[T | dict[str, Any]]:
    """Read JSONL, skipping malformed lines with warnings."""

    items: list[T | dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSONL line %s in %s", line_number, path)
                continue
            if parser is None:
                items.append(record)
                continue
            try:
                items.append(parser(record))
            except (ValidationError, ValueError, TypeError) as exc:
                LOGGER.warning(
                    "Skipping invalid record on line %s in %s: %s",
                    line_number,
                    path,
                    exc,
                )
    return items


def load_raw_samples(path: str | Path) -> list[RawMemeSample]:
    """Load validated raw meme samples."""

    return [item for item in read_jsonl(path, parser=RawMemeSample.model_validate)]


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Write JSONL rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump"):
                payload = record.model_dump()
            elif hasattr(record, "dict"):
                payload = record.dict()
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON configuration into a dictionary."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle) or {}


def deterministic_split(
    items: list[T],
    *,
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[T], list[T], list[T]]:
    """Create deterministic train/val/test splits."""

    import random

    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    eval_end = train_end + int(total * eval_ratio)
    return shuffled[:train_end], shuffled[train_end:eval_end], shuffled[eval_end:]
