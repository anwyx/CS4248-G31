"""Stage B dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from meme_pipeline.data.io import load_raw_samples, read_jsonl
from meme_pipeline.utils.text_norm import canonicalize_phrase


def load_stage_a_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Stage A outputs keyed by sample id."""

    records = {}
    for record in read_jsonl(path):
        sample_id = record.get("id")
        mappings = record.get("metaphor_mappings", [])
        normalized_targets: list[tuple[str, float]] = []
        normalized_mappings: list[dict[str, Any]] = []
        for mapping in mappings:
            target = mapping.get("predicted_target", "OTHER")
            confidence = float(mapping.get("target_confidence", 0.0))
            if target and target != "OTHER":
                normalized_targets.append((target, confidence))
            normalized_mappings.append(mapping)
        records[sample_id] = {
            "predicted_targets": normalized_targets,
            "metaphor_mappings": normalized_mappings,
        }
    return records


@dataclass
class StageBDatasetConfig:
    """Dataset options for Stage B."""

    oracle_target_mode: bool = True
    predicted_stage_a_jsonl: str = ""


class StageBDataset(Dataset):
    """Reference-level dataset for target-conditioned caption training."""

    def __init__(
        self,
        raw_jsonl: str | Path,
        config: StageBDatasetConfig,
        *,
        image_root_dir: str | Path | None = None,
        nlp=None,
    ) -> None:
        self.samples = load_raw_samples(raw_jsonl, image_root_dir=image_root_dir)
        self.config = config
        self.nlp = nlp
        self.stage_a_map = (
            load_stage_a_predictions(config.predicted_stage_a_jsonl)
            if config.predicted_stage_a_jsonl
            else {}
        )
        self.instances = self._build_instances()

    def _extract_oracle_mappings(self, sample) -> list[dict[str, Any]]:
        mappings: list[dict[str, Any]] = []
        for pair in sample.metaphors or []:
            vehicle = canonicalize_phrase(pair.metaphor, self.nlp) if self.nlp else pair.metaphor.lower().strip()
            target = canonicalize_phrase(pair.meaning, self.nlp) if self.nlp else pair.meaning.lower().strip()
            mappings.append(
                {
                    "vehicle_surface": pair.metaphor,
                    "vehicle_normalized": vehicle,
                    "predicted_target": target,
                    "target_confidence": 1.0,
                }
            )
        return mappings

    def _extract_targets(self, mappings: list[dict[str, Any]]) -> list[str]:
        seen: list[str] = []
        for mapping in mappings:
            target = mapping.get("predicted_target", "")
            if target and target != "OTHER" and target not in seen:
                seen.append(target)
        return seen

    def _build_instances(self) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for sample in self.samples:
            if not sample.meme_captions:
                continue
            if self.config.oracle_target_mode:
                mappings = self._extract_oracle_mappings(sample)
            else:
                mappings = self.stage_a_map.get(sample.post_id, {}).get("metaphor_mappings", [])
            targets = self._extract_targets(mappings)
            vehicle_blacklist = [mapping.get("vehicle_surface", "") for mapping in mappings]
            mapping_pairs = [
                (mapping.get("vehicle_surface", ""), mapping.get("predicted_target", ""))
                for mapping in mappings
                if mapping.get("predicted_target", "") and mapping.get("predicted_target", "") != "OTHER"
            ]
            for reference_index, meme_caption in enumerate(sample.meme_captions):
                instances.append(
                    {
                        "id": f"{sample.post_id}__ref_{reference_index}",
                        "sample_id": sample.post_id,
                        "image_path": sample.image_path,
                        "title": sample.title,
                        "ocr_text": sample.ocr_text,
                        "img_captions": sample.img_captions,
                        "metaphor_mappings": mapping_pairs,
                        "target_concepts": targets,
                        "vehicle_blacklist": vehicle_blacklist,
                        "gold_meme_caption": meme_caption,
                        "reference_captions": sample.meme_captions,
                    }
                )
        return instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.instances[index]
