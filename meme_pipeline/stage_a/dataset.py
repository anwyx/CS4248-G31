"""Stage A dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from meme_pipeline.data.schemas import RawMemeSample, StageAInstance
from meme_pipeline.data.target_vocab import TargetVocab
from meme_pipeline.grounding.base import Grounder
from meme_pipeline.stage_a.candidate_selector import rank_or_filter_candidates
from meme_pipeline.stage_a.vehicle_extractor import extract_vehicle_candidates_from_captions
from meme_pipeline.utils.text_norm import canonicalize_phrase


@dataclass
class StageADatasetConfig:
    """Dataset construction options for Stage A."""

    max_candidates: int = 5
    drop_unlabeled_candidates: bool = False
    use_grounding: bool = True


class StageADataset(Dataset):
    """Candidate-level dataset for Stage A target prediction."""

    def __init__(
        self,
        samples: list[RawMemeSample],
        *,
        nlp,
        target_vocab: TargetVocab,
        config: StageADatasetConfig,
        grounder: Grounder | None = None,
    ) -> None:
        self.samples = samples
        self.nlp = nlp
        self.target_vocab = target_vocab
        self.config = config
        self.grounder = grounder
        if self.config.use_grounding and self.grounder is None:
            raise ValueError("Stage A dataset requires a grounder when use_grounding=true.")
        self.instances = self._build_instances()

    def _build_instances(self) -> list[StageAInstance]:
        instances: list[StageAInstance] = []
        for sample in self.samples:
            candidates = extract_vehicle_candidates_from_captions(sample.img_captions, self.nlp)
            candidates = rank_or_filter_candidates(
                candidates,
                sample.title,
                sample.ocr_text,
                sample.img_captions,
                self.config.max_candidates,
            )
            gold_pairs = sample.metaphors or []
            aligned = {
                canonicalize_phrase(pair.metaphor, self.nlp): canonicalize_phrase(pair.meaning, self.nlp)
                for pair in gold_pairs
            }
            for index, candidate in enumerate(candidates):
                label = aligned.get(candidate["normalized"], "OTHER")
                if label == "OTHER" and self.config.drop_unlabeled_candidates:
                    continue
                bbox_xyxy = None
                crop_path = None
                if self.config.use_grounding and self.grounder is not None:
                    grounding = self.grounder.ground(
                        sample.image_path,
                        candidate["normalized"],
                        head_query=candidate["head"],
                    )
                    bbox_xyxy = grounding.bbox_xyxy
                    crop_path = grounding.crop_path
                instances.append(
                    StageAInstance(
                        id=f"{sample.post_id}__cand_{index}",
                        sample_id=sample.post_id,
                        image_path=sample.image_path,
                        title=sample.title,
                        ocr_text=sample.ocr_text,
                        img_captions=sample.img_captions,
                        caption_index=candidate["caption_index"],
                        vehicle_surface=candidate["surface"],
                        vehicle_normalized=candidate["normalized"],
                        vehicle_head=candidate["head"],
                        bbox_xyxy=bbox_xyxy,
                        crop_path=crop_path,
                        target_label=label or "OTHER",
                        target_id=self.target_vocab.encode(label or "OTHER"),
                    )
                )
        return instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.instances[index].model_dump()
