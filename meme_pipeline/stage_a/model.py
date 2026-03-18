"""Stage A Qwen3-VL classifier model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from meme_pipeline.stage_a.classifier_head import StageAClassifierHead
from meme_pipeline.utils.image_utils import load_image
from meme_pipeline.utils.logging import get_logger
from meme_pipeline.utils.prompts import build_stage_a_classification_prompt

LOGGER = get_logger(__name__)


@dataclass
class StageAModelConfig:
    """Configuration for Stage A model loading and inference."""

    model_name: str = "Robertp423/Qwen3-VL-4B-Construct"
    fallback_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    freeze_backbone: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    classifier_dropout: float = 0.1
    use_crop_text_hint: bool = True
    use_full_image_only: bool = True


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve configured dtype with CPU-safe fallback."""

    normalized = dtype_name.lower()
    if normalized in {"bfloat16", "bf16"} and torch.cuda.is_available():
        return torch.bfloat16
    if normalized in {"float16", "fp16"} and torch.cuda.is_available():
        return torch.float16
    return torch.float32


class DummyVisionLanguageBackbone(nn.Module):
    """Small deterministic fallback used only for local tests or injected runs."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(2048, hidden_size)

    def encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            token_ids = [ord(char) % 2048 for char in prompt[:512]] or [0]
            ids = torch.tensor(token_ids, dtype=torch.long)
            rows.append(self.embedding(ids).mean(dim=0))
        return torch.stack(rows, dim=0)


class StageAMetaphorClassifier(nn.Module):
    """Qwen3-VL-backed classifier with a lightweight head."""

    def __init__(
        self,
        *,
        config: StageAModelConfig,
        num_targets: int,
        backbone: Any | None = None,
        processor: Any | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_targets = num_targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(config.dtype)
        self.model_name = config.model_name
        self.processor = processor
        self.backbone = backbone
        if self.backbone is None:
            self.backbone, self.processor, self.model_name = self._load_backbone()
        self.hidden_size = hidden_size or self._infer_hidden_size()
        self.classifier = StageAClassifierHead(self.hidden_size, num_targets, config.classifier_dropout)
        self.classifier.to(self.device)
        if isinstance(self.backbone, nn.Module):
            self.backbone.to(self.device)

    def _load_backbone(self):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(
                "transformers>=4.57.0 with Qwen3-VL support is required to load the backbone."
            ) from exc
        for model_name in [self.config.model_name, self.config.fallback_model_name]:
            try:
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.config.device_map,
                    output_hidden_states=True,
                    trust_remote_code=True,
                )
                if self.config.freeze_backbone:
                    for parameter in backbone.parameters():
                        parameter.requires_grad = False
                backbone = self._maybe_apply_lora(backbone)
                LOGGER.info("Loaded Stage A backbone: %s", model_name)
                return backbone, processor, model_name
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                LOGGER.warning("Failed to load model %s: %s", model_name, exc)
        raise RuntimeError(
            f"Unable to load primary model {self.config.model_name} or fallback {self.config.fallback_model_name}."
        )

    def _maybe_apply_lora(self, backbone: nn.Module):
        if not self.config.use_lora:
            return backbone
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            LOGGER.warning("peft is not installed; continuing without LoRA.")
            return backbone
        target_modules = self.config.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        module_names = {name.rsplit(".", 1)[-1] for name, _ in backbone.named_modules()}
        active_modules = [name for name in target_modules if name in module_names]
        if not active_modules:
            LOGGER.warning("No configured LoRA target modules found in backbone; skipping LoRA.")
            return backbone
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=active_modules,
        )
        LOGGER.info("Applying LoRA to modules: %s", ", ".join(active_modules))
        return get_peft_model(backbone, lora_config)

    def _infer_hidden_size(self) -> int:
        if hasattr(self.backbone, "config"):
            for attr in ("hidden_size", "text_config", "d_model"):
                value = getattr(self.backbone.config, attr, None)
                if isinstance(value, int):
                    return value
                if hasattr(value, "hidden_size"):
                    return int(value.hidden_size)
        if isinstance(self.backbone, DummyVisionLanguageBackbone):
            return self.backbone.hidden_size
        return 1024

    def build_prompt(self, batch: dict[str, list[Any]]) -> list[str]:
        """Build classifier prompts for a batch."""

        prompts: list[str] = []
        for index in range(len(batch["vehicle_surface"])):
            bbox = batch.get("bbox_xyxy", [None] * len(batch["vehicle_surface"]))[index]
            bbox_text = str(bbox) if bbox else "None"
            prompts.append(
                build_stage_a_classification_prompt(
                    title=batch.get("title", [""])[index] if batch.get("title") else "",
                    ocr_text=batch.get("ocr_text", [""])[index] if batch.get("ocr_text") else "",
                    literal_caption=batch.get("literal_caption", [""])[index] if batch.get("literal_caption") else "",
                    vehicle_surface=batch["vehicle_surface"][index],
                    vehicle_normalized=batch["vehicle_normalized"][index],
                    vehicle_head=batch["vehicle_head"][index],
                    bbox_or_none=bbox_text,
                )
            )
        return prompts

    def _encode_batch(self, batch: dict[str, list[Any]]) -> torch.Tensor:
        prompts = self.build_prompt(batch)
        if isinstance(self.backbone, DummyVisionLanguageBackbone):
            return self.backbone.encode_prompt(prompts).to(self.device)
        images = [load_image(path) for path in batch["image_path"]]
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        outputs = self.backbone(**prepared, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        attention_mask = prepared.get("attention_mask")
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).expand_as(hidden)
        masked_hidden = hidden * mask
        denom = mask.sum(dim=1).clamp_min(1)
        return masked_hidden.sum(dim=1) / denom

    def forward(self, batch: dict[str, list[Any]]) -> torch.Tensor:
        """Encode a batch and produce target logits."""

        pooled = self._encode_batch(batch)
        return self.classifier(pooled)

    @torch.no_grad()
    def predict_proba(self, batch: dict[str, list[Any]]) -> torch.Tensor:
        """Return target probabilities."""

        self.eval()
        logits = self.forward(batch)
        return torch.softmax(logits, dim=-1)

    def save(self, output_dir: str | Path) -> None:
        """Save classifier head and any adapter/model assets."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.classifier.state_dict(), output_dir / "classifier_head.pt")
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_dir / "processor")
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(output_dir / "backbone")

    def load_classifier_head(self, path: str | Path) -> None:
        """Load classifier head weights."""

        state_dict = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(state_dict)
