"""Stage B target-conditioned generation."""

from __future__ import annotations

from typing import Any, Sequence

from meme_pipeline.stage_b.model import StageBCaptionModel
from meme_pipeline.stage_b.postprocess import clean_generation_text, dedupe_candidates, is_valid_candidate
from meme_pipeline.utils.prompts import build_stage_b_generation_prompt


def serialize_targets(targets: Sequence[tuple[str, float]] | Sequence[str]) -> list[str]:
    """Serialize target concepts as a compact deduplicated list."""

    serialized: list[str] = []
    for item in targets:
        target = item[0] if isinstance(item, tuple) else item
        if target and target not in serialized and target != "OTHER":
            serialized.append(target)
    return serialized


def serialize_mappings(mappings: Sequence[dict[str, Any]] | Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Serialize Stage A mappings into vehicle -> target tuples."""

    serialized: list[tuple[str, str]] = []
    for item in mappings:
        if isinstance(item, tuple):
            pair = item
        else:
            pair = (item.get("vehicle_surface", ""), item.get("predicted_target", ""))
        if pair[0] and pair[1] and pair[1] != "OTHER" and pair not in serialized:
            serialized.append(pair)
    return serialized


class StageBGenerator:
    """Generate candidate meme-meaning captions."""

    def __init__(self, model: StageBCaptionModel) -> None:
        self.model = model

    def _build_prompt(
        self,
        *,
        sample: dict[str, Any],
        target_concepts: Sequence[str],
        metaphor_mappings: Sequence[tuple[str, str]],
        vehicle_blacklist: Sequence[str] | None,
    ) -> str:
        return build_stage_b_generation_prompt(
            title=sample.get("title", ""),
            ocr_text=sample.get("ocr_text", ""),
            img_captions=sample.get("img_captions", []),
            metaphor_mappings=metaphor_mappings,
            target_concepts=target_concepts,
            vehicle_blacklist=vehicle_blacklist,
        )

    def generate_candidates(
        self,
        sample: dict[str, Any],
        *,
        target_concepts: Sequence[str],
        metaphor_mappings: Sequence[tuple[str, str]],
        vehicle_blacklist: Sequence[str] | None = None,
        k: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 64,
    ) -> list[str]:
        """Generate, validate, deduplicate, and retry if needed."""

        prompt = self._build_prompt(
            sample=sample,
            target_concepts=target_concepts,
            metaphor_mappings=metaphor_mappings,
            vehicle_blacklist=vehicle_blacklist,
        )
        candidates: list[str] = []
        for _ in range(k):
            text = self.model.generate_one(
                prompt=prompt,
                image_path=sample["image_path"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            text = clean_generation_text(text)
            if is_valid_candidate(
                text,
                title=sample.get("title", ""),
                ocr_text=sample.get("ocr_text", ""),
                forbidden_terms=vehicle_blacklist or [],
            ):
                candidates.append(text)
        candidates = dedupe_candidates(candidates)
        if candidates:
            return candidates
        simpler_prompt = self._build_prompt(
            sample=sample,
            target_concepts=target_concepts,
            metaphor_mappings=[],
            vehicle_blacklist=vehicle_blacklist,
        )
        retry = self.model.generate_one(
            prompt=simpler_prompt,
            image_path=sample["image_path"],
            max_new_tokens=max_new_tokens,
            temperature=max(0.7, temperature - 0.1),
            top_p=top_p,
        )
        retry = clean_generation_text(retry)
        return [retry] if retry else ["The meme expresses a relatable emotional reaction."]
