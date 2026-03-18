"""Stage A training loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from meme_pipeline.data.collators import simple_dict_collator
from meme_pipeline.data.io import deterministic_split, load_config, load_raw_samples
from meme_pipeline.data.target_vocab import TargetVocab, build_target_vocab, save_target_vocab
from meme_pipeline.stage_a.dataset import StageADataset, StageADatasetConfig
from meme_pipeline.stage_a.model import StageAMetaphorClassifier, StageAModelConfig
from meme_pipeline.stage_a.vehicle_extractor import load_spacy_or_fail
from meme_pipeline.utils.logging import get_logger
from meme_pipeline.utils.metrics import (
    stage_a_accuracy,
    stage_a_macro_f1,
    stage_a_weighted_f1,
    topk_accuracy,
)
from meme_pipeline.utils.seed import set_seed

LOGGER = get_logger(__name__)


def _loss_fn(logits: torch.Tensor, labels: torch.Tensor, loss_type: str = "cross_entropy") -> torch.Tensor:
    if loss_type == "focal":
        ce = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** 2 * ce).mean()
    return nn.functional.cross_entropy(logits, labels)


@dataclass
class StageATrainArtifacts:
    """Paths of saved training outputs."""

    model_dir: str
    target_vocab_path: str
    metrics_path: str


class StageATrainer:
    """Train Stage A classifier."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config.get("model_output_dir", "outputs/stage_a_model"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nlp = load_spacy_or_fail()
        set_seed(int(config.get("seed", 42)))

    def _prepare_splits(self):
        train_path = self.config.get("train_jsonl")
        val_path = self.config.get("val_jsonl")
        if train_path:
            train_samples = load_raw_samples(train_path)
            if val_path:
                val_samples = load_raw_samples(val_path)
                return train_samples, val_samples
            train_samples, val_samples, _ = deterministic_split(
                train_samples,
                train_ratio=float(self.config.get("train_split_ratio", 0.8)),
                eval_ratio=float(self.config.get("eval_split_ratio", 0.1)),
                seed=int(self.config.get("seed", 42)),
            )
            return train_samples, val_samples
        raise ValueError("stage_a config must provide train_jsonl for training.")

    def train(self) -> StageATrainArtifacts:
        """Run training and save best checkpoint."""

        train_samples, val_samples = self._prepare_splits()
        target_vocab = build_target_vocab(
            self.config["train_jsonl"],
            min_freq=int(self.config.get("min_target_freq", 1)),
            nlp=self.nlp,
        )
        save_target_vocab(self.output_dir / "target_vocab.json", target_vocab)
        dataset_config = StageADatasetConfig(
            max_candidates=int(self.config.get("max_candidates", 5)),
            drop_unlabeled_candidates=bool(self.config.get("drop_unlabeled_candidates", False)),
        )
        train_dataset = StageADataset(train_samples, nlp=self.nlp, target_vocab=target_vocab, config=dataset_config)
        val_dataset = StageADataset(val_samples, nlp=self.nlp, target_vocab=target_vocab, config=dataset_config)
        model = StageAMetaphorClassifier(
            config=StageAModelConfig(**{key: value for key, value in self.config.items() if key in StageAModelConfig.__annotations__}),
            num_targets=len(target_vocab),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.config.get("train_batch_size", 2)),
            shuffle=True,
            collate_fn=simple_dict_collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.config.get("eval_batch_size", 2)),
            shuffle=False,
            collate_fn=simple_dict_collator,
        )
        optimizer = AdamW(model.parameters(), lr=float(self.config.get("learning_rate", 2e-4)), weight_decay=float(self.config.get("weight_decay", 0.01)))
        patience = int(self.config.get("patience", 2))
        best_macro_f1 = -1.0
        patience_left = patience
        history: list[dict[str, float]] = []
        for epoch in range(int(self.config.get("num_epochs", 5))):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                labels = torch.tensor(batch["target_id"], dtype=torch.long, device=model.device)
                logits = model(batch)
                loss = _loss_fn(logits, labels, self.config.get("loss_type", "cross_entropy"))
                loss.backward()
                if step % int(self.config.get("grad_accum_steps", 1)) == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.item())
            if len(train_loader) % int(self.config.get("grad_accum_steps", 1)) != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            metrics = self.evaluate(model, val_loader)
            metrics["train_loss"] = total_loss / max(len(train_loader), 1)
            metrics["epoch"] = epoch + 1
            history.append(metrics)
            LOGGER.info("Epoch %s metrics: %s", epoch + 1, metrics)
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                patience_left = patience
                model.save(self.output_dir)
                with (self.output_dir / "stage_a_config.json").open("w", encoding="utf-8") as handle:
                    json.dump(self.config, handle, ensure_ascii=False, indent=2)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    LOGGER.info("Early stopping triggered.")
                    break
        metrics_path = self.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        return StageATrainArtifacts(
            model_dir=str(self.output_dir),
            target_vocab_path=str(self.output_dir / "target_vocab.json"),
            metrics_path=str(metrics_path),
        )

    def evaluate(self, model: StageAMetaphorClassifier, loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation data."""

        model.eval()
        gold: list[int] = []
        preds: list[int] = []
        topk: list[list[int]] = []
        hardest_false_positives: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                labels = batch["target_id"]
                probs = model.predict_proba(batch)
                values, indices = torch.topk(probs, k=min(3, probs.size(-1)), dim=-1)
                pred_ids = indices[:, 0].tolist()
                gold.extend(int(item) for item in labels)
                preds.extend(int(item) for item in pred_ids)
                topk.extend(indices.tolist())
                for batch_index, (label, pred, confidence) in enumerate(zip(labels, pred_ids, values[:, 0].tolist())):
                    if int(label) != int(pred):
                        hardest_false_positives.append(
                            {
                                "id": batch["id"][batch_index],
                                "gold": int(label),
                                "pred": int(pred),
                                "confidence": float(confidence),
                            }
                        )
        hardest_false_positives.sort(key=lambda item: item["confidence"], reverse=True)
        LOGGER.info("Hardest false positives: %s", hardest_false_positives[:10])
        return {
            "accuracy": stage_a_accuracy(gold, preds),
            "macro_f1": stage_a_macro_f1(gold, preds),
            "weighted_f1": stage_a_weighted_f1(gold, preds),
            "top3_accuracy": topk_accuracy(gold, topk, 3),
        }


def load_stage_a_trainer_from_config(config_path: str) -> StageATrainer:
    """Load trainer from YAML/JSON config."""

    return StageATrainer(load_config(config_path))
