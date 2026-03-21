"""Stage B inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from meme_pipeline.data.io import load_config, load_raw_samples, read_jsonl, write_jsonl
from meme_pipeline.data.schemas import CaptionCandidate, StageBInferenceRecord
from meme_pipeline.stage_b.generator import StageBGenerator, serialize_mappings, serialize_targets
from meme_pipeline.stage_b.model import StageBCaptionModel, StageBModelConfig
from meme_pipeline.stage_b.postprocess import rerank_candidates


def load_stage_a_output_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Stage A output rows keyed by sample id."""

    return {record["id"]: record for record in read_jsonl(path)}


class StageBInferencePipeline:
    """Run Stage B conditioned generation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        model_config = dict(config)
        local_qwen_dir = Path(config.get("model_output_dir", "outputs/stage_b_model")) / "qwen_model"
        if local_qwen_dir.exists():
            model_config["model_name_or_path"] = str(local_qwen_dir)
        self.model = StageBCaptionModel(
            config=StageBModelConfig(
                **{key: value for key, value in model_config.items() if key in StageBModelConfig.__annotations__}
            )
        )
        self.generator = StageBGenerator(self.model)

    def _extract_targets_and_mappings(
        self,
        stage_a_record: dict[str, Any] | None,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, str]], list[str]]:
        if not stage_a_record:
            return [], [], []
        kept_targets: list[tuple[str, float]] = []
        vehicle_terms: list[str] = []
        mappings: list[tuple[str, str]] = []
        threshold = float(self.config.get("confidence_threshold", 0.35))
        for mapping in stage_a_record.get("metaphor_mappings", []):
            vehicle_surface = mapping.get("vehicle_surface", "")
            vehicle_terms.append(vehicle_surface)
            confidence = float(mapping.get("target_confidence", 0.0))
            target = mapping.get("predicted_target", "OTHER")
            if target and target != "OTHER" and confidence >= threshold:
                kept_targets.append((target, confidence))
                mappings.append((vehicle_surface, target))
        return kept_targets, mappings, vehicle_terms

    def predict_sample(self, sample, stage_a_record: dict[str, Any] | None = None) -> StageBInferenceRecord:
        """Generate and score candidate captions for one sample."""

        predicted_targets, predicted_mappings, vehicle_terms = self._extract_targets_and_mappings(stage_a_record)
        target_concepts = serialize_targets(predicted_targets)
        mapping_pairs = serialize_mappings(predicted_mappings)
        raw_candidates = self.generator.generate_candidates(
            sample.model_dump(),
            target_concepts=target_concepts,
            metaphor_mappings=mapping_pairs,
            vehicle_blacklist=vehicle_terms,
            k=int(self.config.get("num_candidates", 5)),
            temperature=float(self.config.get("temperature", 0.8)),
            top_p=float(self.config.get("top_p", 0.9)),
            max_new_tokens=int(self.config.get("max_new_tokens", 64)),
        )
        scored = rerank_candidates(
            raw_candidates,
            mappings=mapping_pairs,
            targets=target_concepts,
            title=sample.title,
            ocr_text=sample.ocr_text,
            image_captions=sample.img_captions,
            vehicle_terms=vehicle_terms,
            vehicle_penalty_weight=float(self.config.get("vehicle_penalty_weight", 1.0)),
            ocr_penalty_weight=float(self.config.get("ocr_penalty_weight", 0.5)),
            title_penalty_weight=float(self.config.get("title_penalty_weight", 0.5)),
        )
        candidate_records = [
            CaptionCandidate(text=item.text, score=item.score, penalties=item.penalties)
            for item in scored
        ]
        best_caption = candidate_records[0].text if candidate_records else ""
        return StageBInferenceRecord(
            id=sample.post_id,
            predicted_targets=predicted_targets,
            predicted_mappings=mapping_pairs,
            candidate_captions=candidate_records,
            best_caption=best_caption,
        )

    def predict_jsonl(self, input_path: str, stage_a_outputs: str, output_path: str) -> None:
        """Run Stage B over a JSONL file."""

        samples = load_raw_samples(input_path, image_root_dir=self.config.get("image_root_dir"))
        stage_a_map = load_stage_a_output_map(stage_a_outputs) if stage_a_outputs else {}
        outputs = [
            self.predict_sample(sample, stage_a_map.get(sample.post_id))
            for sample in tqdm(samples, desc="Stage B inference", unit="sample")
        ]
        write_jsonl(output_path, outputs)


def run_stage_b_inference(config_path: str, input_path: str, stage_a_outputs: str, output_path: str) -> None:
    """CLI entry point for Stage B inference."""

    pipeline = StageBInferencePipeline(load_config(config_path))
    pipeline.predict_jsonl(input_path, stage_a_outputs, output_path)
