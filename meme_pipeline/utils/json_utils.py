"""Robust JSON parsing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, ValidationError


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def recover_json_string(text: str) -> str | None:
    """Try to recover a parseable JSON string from noisy model output."""

    if not text:
        return None
    direct = text.strip()
    try:
        json.loads(direct)
        return direct
    except json.JSONDecodeError:
        pass
    extracted = extract_first_json_object(text)
    if extracted is not None:
        return extracted
    repaired = re.sub(r",\s*([}\]])", r"\1", direct)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        return None


def extract_first_json_object(text: str) -> str | None:
    """Extract the first valid JSON object or array substring."""

    if not text:
        return None
    stack: list[str] = []
    start_index: int | None = None
    for index, char in enumerate(text):
        if char in "{[":
            if start_index is None:
                start_index = index
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                continue
            stack.pop()
            if start_index is not None and not stack:
                candidate = text[start_index : index + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    start_index = None
    return None


def parse_json_with_recovery(text: str) -> Any:
    """Parse JSON, attempting light recovery first."""

    recovered = recover_json_string(text)
    if recovered is None:
        raise json.JSONDecodeError("No valid JSON object found", text, 0)
    return json.loads(recovered)


def validate_schema(payload: Any, schema: Type[BaseModel]) -> BaseModel:
    """Validate payload against a pydantic model."""

    return schema.model_validate(payload)


def maybe_validate_schema(payload: Any, schema: Type[BaseModel]) -> BaseModel | None:
    """Validate payload and return None on failure."""

    try:
        return schema.model_validate(payload)
    except ValidationError:
        return None
