"""Batch collation helpers."""

from __future__ import annotations

from typing import Any


def simple_dict_collator(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collate a list of dictionaries into dictionary-of-lists form."""

    if not batch:
        return {}
    keys = batch[0].keys()
    return {key: [item[key] for item in batch] for key in keys}
