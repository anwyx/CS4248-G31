"""Stage A inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from meme_pipeline.data.io import load_config, load_raw_samples, write_jsonl
from meme_pipeline.data.schemas import StageAInferenceRecord
from meme_pipeline.data.target_vocab import load_target_vocab
from meme_pipeline.grounding.factory import load_grounder
from meme_pipeline.stage_a.candidate_selector import rank_or_filter_candidates
from meme_pipeline.stage_a.model import StageAMetaphorClassifier, StageAModelConfig
from meme_pipeline.stage_a.postprocess import finalize_stage_a_predictions
from meme_pipeline.stage_a.vehicle_extractor import extract_vehicle_candidates, load_spacy_or_fail
from meme_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)


class StageAInferencePipeline:
    """Inference runner for Stage A."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.nlp = load_spacy_or_fail()
        target_vocab_path = config.get("target_vocab_path") or Path(config.get("model_output_dir", "outputs/stage_a_model")) / "target_vocab.json"
        self.target_vocab = load_target_vocab(target_vocab_path)
        self.grounder = load_grounder(config)
        model_config = dict(config)
        local_backbone = Path(config.get("model_output_dir", "outputs/stage_a_model")) / "backbone"
        if local_backbone.exists():
            model_config["model_name"] = str(local_backbone)
        self.model = StageAMetaphorClassifier(
            config=StageAModelConfig(**{key: value for key, value in model_config.items() if key in StageAModelConfig.__annotations__}),
            num_targets=len(self.target_vocab),
        )
        classifier_path = Path(config.get("model_output_dir", "outputs/stage_a_model")) / "classifier_head.pt"
        if classifier_path.exists():
            self.model.load_classifier_head(classifier_path)
        else:
            LOGGER.warning("Classifier head not found at %s; using randomly initialized head.", classifier_path)

    def predict_sample(self, sample) -> StageAInferenceRecord:
        """Predict targets for one sample."""

        candidates = extract_vehicle_candidates(sample.literal_caption, self.nlp)
        candidates = rank_or_filter_candidates(
            candidates,
            sample.title,
            sample.ocr_text,
            sample.literal_caption,
            int(self.config.get("max_candidates", 5)),
        )
        if not candidates:
            return StageAInferenceRecord(id=sample.id, vehicles=[])
        batch: dict[str, list[Any]] = {
            "id": [],
            "image_path": [],
            "title": [],
            "ocr_text": [],
            "literal_caption": [],
            "vehicle_surface": [],
            "vehicle_normalized": [],
            "vehicle_head": [],
            "bbox_xyxy": [],
        }
        grounding_meta: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            grounding = self.grounder.ground(sample.image_path, candidate["normalized"], head_query=candidate["head"])
            grounding_meta.append(grounding.to_dict())
            batch["id"].append(f"{sample.id}__cand_{index}")
            batch["image_path"].append(sample.image_path)
            batch["title"].append(sample.title)
            batch["ocr_text"].append(sample.ocr_text)
            batch["literal_caption"].append(sample.literal_caption)
            batch["vehicle_surface"].append(candidate["surface"])
            batch["vehicle_normalized"].append(candidate["normalized"])
            batch["vehicle_head"].append(candidate["head"])
            batch["bbox_xyxy"].append(grounding.bbox_xyxy)
        probs = self.model.predict_proba(batch)
        topk_k = min(int(self.config.get("topk_targets", 3)), probs.size(-1))
        topk_values, topk_indices = torch.topk(probs, k=topk_k, dim=-1)
        vehicles: list[dict[str, Any]] = []
        for row_index, candidate in enumerate(candidates):
            best_id = int(topk_indices[row_index, 0].item())
            best_score = float(topk_values[row_index, 0].item())
            vehicles.append(
                {
                    "vehicle_surface": candidate["surface"],
                    "vehicle_normalized": candidate["normalized"],
                    "bbox_xyxy": grounding_meta[row_index]["bbox_xyxy"],
                    "grounding_score": grounding_meta[row_index]["score"],
                    "predicted_target": self.target_vocab.decode(best_id),
                    "predicted_target_id": best_id,
                    "target_confidence": best_score,
                    "topk_targets": [
                        (self.target_vocab.decode(int(index.item())), float(value.item()))
                        for value, index in zip(topk_values[row_index], topk_indices[row_index])
                    ],
                }
            )
        vehicles = finalize_stage_a_predictions(
            vehicles,
            threshold=float(self.config.get("confidence_threshold", 0.35)),
        )
        return StageAInferenceRecord(id=sample.id, vehicles=vehicles)

    def predict_jsonl(self, input_path: str, output_path: str) -> None:
        """Run Stage A inference over a JSONL file."""

        samples = load_raw_samples(input_path)
        records = [self.predict_sample(sample) for sample in samples]
        write_jsonl(output_path, records)


def run_stage_a_inference(config_path: str, input_path: str, output_path: str) -> None:
    """Convenience entry point for CLI."""

    pipeline = StageAInferencePipeline(load_config(config_path))
    pipeline.predict_jsonl(input_path, output_path)
