"""Stage B dataset construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from meme_pipeline.data.io import load_raw_samples, read_jsonl
from meme_pipeline.utils.text_norm import canonicalize_phrase


def load_stage_a_predictions(path: str | Path) -> dict[str, list[tuple[str, float]]]:
    """Load Stage A outputs keyed by sample id."""

    records = {}
    for record in read_jsonl(path):
        sample_id = record.get("id")
        vehicles = record.get("vehicles", [])
        targets: list[tuple[str, float]] = []
        for vehicle in vehicles:
            target = vehicle.get("predicted_target", "OTHER")
            confidence = float(vehicle.get("target_confidence", 0.0))
            if target and target != "OTHER":
                targets.append((target, confidence))
        records[sample_id] = targets
    return records


@dataclass
class StageBDatasetConfig:
    """Dataset options for Stage B."""

    oracle_target_mode: bool = True
    predicted_stage_a_jsonl: str = ""


class StageBDataset(Dataset):
    """Sample-level dataset for target-conditioned caption training."""

    def __init__(self, raw_jsonl: str | Path, config: StageBDatasetConfig, nlp=None) -> None:
        self.samples = load_raw_samples(raw_jsonl)
        self.config = config
        self.nlp = nlp
        self.stage_a_map = (
            load_stage_a_predictions(config.predicted_stage_a_jsonl)
            if config.predicted_stage_a_jsonl
            else {}
        )
        self.instances = self._build_instances()

    def _extract_targets(self, sample) -> list[str]:
        if self.config.oracle_target_mode and sample.vehicle_target_pairs:
            seen: list[str] = []
            for pair in sample.vehicle_target_pairs:
                target = canonicalize_phrase(pair.target, self.nlp) if self.nlp else pair.target.lower().strip()
                if target and target not in seen:
                    seen.append(target)
            return seen
        return [target for target, _ in self.stage_a_map.get(sample.id, [])]

    def _build_instances(self) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for sample in self.samples:
            if not sample.gold_meme_caption:
                continue
            targets = self._extract_targets(sample)
            vehicles = [pair.vehicle for pair in sample.vehicle_target_pairs or []]
            instances.append(
                {
                    "id": sample.id,
                    "image_path": sample.image_path,
                    "title": sample.title,
                    "ocr_text": sample.ocr_text,
                    "literal_caption": sample.literal_caption,
                    "target_concepts": targets,
                    "vehicle_blacklist": vehicles,
                    "gold_meme_caption": sample.gold_meme_caption,
                }
            )
        return instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.instances[index]
