"""Stage B training loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from meme_pipeline.data.collators import simple_dict_collator
from meme_pipeline.data.io import load_config
from meme_pipeline.stage_a.vehicle_extractor import load_spacy_or_fail
from meme_pipeline.stage_b.dataset import StageBDataset, StageBDatasetConfig
from meme_pipeline.stage_b.model import StageBCaptionModel, StageBModelConfig
from meme_pipeline.utils.logging import get_logger
from meme_pipeline.utils.prompts import build_stage_b_generation_prompt
from meme_pipeline.utils.seed import set_seed

LOGGER = get_logger(__name__)


@dataclass
class StageBTrainArtifacts:
    """Saved Stage B output paths."""

    model_dir: str
    metrics_path: str


class StageBTrainer:
    """Train Stage B caption generator."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config.get("model_output_dir", "outputs/stage_b_model"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nlp = load_spacy_or_fail()
        set_seed(int(config.get("seed", 42)))

    def train(self) -> StageBTrainArtifacts:
        """Run a lightweight SFT-style loop."""

        train_jsonl = self.config.get("train_jsonl")
        if not train_jsonl:
            raise ValueError("stage_b config must provide train_jsonl for training.")
        dataset = StageBDataset(
            train_jsonl,
            config=StageBDatasetConfig(
                oracle_target_mode=bool(self.config.get("oracle_target_mode", True)),
                predicted_stage_a_jsonl=self.config.get("predicted_stage_a_jsonl", ""),
            ),
            nlp=self.nlp,
        )
        if len(dataset) == 0:
            raise ValueError("No Stage B training instances were built. Check gold_meme_caption and target inputs.")
        val_jsonl = self.config.get("val_jsonl")
        if val_jsonl:
            train_dataset = dataset
            val_dataset = StageBDataset(
                val_jsonl,
                config=StageBDatasetConfig(
                    oracle_target_mode=bool(self.config.get("oracle_target_mode", True)),
                    predicted_stage_a_jsonl=self.config.get("predicted_stage_a_jsonl", ""),
                ),
                nlp=self.nlp,
            )
        else:
            total = len(dataset)
            if total < 2:
                train_dataset = dataset
                val_dataset = dataset
            else:
                train_size = max(1, int(total * float(self.config.get("train_split_ratio", 0.8))))
                train_size = min(train_size, total - 1)
                val_size = total - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        model = StageBCaptionModel(
            config=StageBModelConfig(**{key: value for key, value in self.config.items() if key in StageBModelConfig.__annotations__})
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
        optimizer = AdamW(model.parameters(), lr=float(self.config.get("learning_rate", 1e-4)), weight_decay=float(self.config.get("weight_decay", 0.01)))
        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        for epoch in range(int(self.config.get("num_epochs", 3))):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                prompts = [
                    build_stage_b_generation_prompt(
                        title=title,
                        ocr_text=ocr,
                        literal_caption=caption,
                        target_concepts=targets,
                        vehicle_blacklist=vehicles,
                    )
                    for title, ocr, caption, targets, vehicles in zip(
                        batch["title"],
                        batch["ocr_text"],
                        batch["literal_caption"],
                        batch["target_concepts"],
                        batch["vehicle_blacklist"],
                    )
                ]
                loss = model.compute_loss(prompts, batch["image_path"], batch["gold_meme_caption"])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.item())
            val_loss = self.evaluate(model, val_loader)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(len(train_loader), 1),
                "val_loss": val_loss,
            }
            history.append(metrics)
            LOGGER.info("Epoch %s metrics: %s", epoch + 1, metrics)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(self.output_dir)
                with (self.output_dir / "stage_b_config.json").open("w", encoding="utf-8") as handle:
                    json.dump(self.config, handle, ensure_ascii=False, indent=2)
        metrics_path = self.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        return StageBTrainArtifacts(model_dir=str(self.output_dir), metrics_path=str(metrics_path))

    def evaluate(self, model: StageBCaptionModel, loader: DataLoader) -> float:
        """Compute validation loss."""

        model.eval()
        total_loss = 0.0
        count = 0
        for batch in loader:
            prompts = [
                build_stage_b_generation_prompt(
                    title=title,
                    ocr_text=ocr,
                    literal_caption=caption,
                    target_concepts=targets,
                    vehicle_blacklist=vehicles,
                )
                for title, ocr, caption, targets, vehicles in zip(
                    batch["title"],
                    batch["ocr_text"],
                    batch["literal_caption"],
                    batch["target_concepts"],
                    batch["vehicle_blacklist"],
                )
            ]
            with __import__("torch").no_grad():
                loss = model.compute_loss(prompts, batch["image_path"], batch["gold_meme_caption"])
            total_loss += float(loss.item())
            count += 1
        return total_loss / max(count, 1)


def load_stage_b_trainer_from_config(config_path: str) -> StageBTrainer:
    """Load trainer from config path."""

    return StageBTrainer(load_config(config_path))
