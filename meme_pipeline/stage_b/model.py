"""Stage B Qwen3-VL generation and training wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from meme_pipeline.stage_a.model import resolve_dtype
from meme_pipeline.utils.image_utils import load_image
from meme_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class StageBModelConfig:
    """Configuration for Stage B model loading."""

    model_name: str = "Robertp423/Qwen3-VL-4B-Construct"
    fallback_model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


class DummyCaptionBackbone(nn.Module):
    """Simple deterministic caption fallback for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))

    def heuristic_generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "predicted target concepts" in prompt_lower:
            tail = prompt.split("Predicted target concepts:", maxsplit=1)[-1]
            targets = [line.strip("- ").strip() for line in tail.splitlines() if line.strip().startswith("-")]
            if targets:
                return f"The meme conveys {', '.join(targets[:2])} through a relatable reaction."
        return "The meme conveys a relatable emotional reaction under social pressure."


class StageBCaptionModel(nn.Module):
    """Generation model wrapper for Stage B."""

    def __init__(
        self,
        *,
        config: StageBModelConfig,
        backbone: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = resolve_dtype(config.dtype)
        self.backbone = backbone
        self.processor = processor
        self.model_name = config.model_name
        if self.backbone is None:
            self.backbone, self.processor, self.model_name = self._load_backbone()
        if isinstance(self.backbone, nn.Module):
            self.backbone.to(self.device)

    def _load_backbone(self):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(
                "transformers>=4.57.0 with Qwen3-VL support is required to load the Stage B backbone."
            ) from exc
        for model_name in [self.config.model_name, self.config.fallback_model_name]:
            try:
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.config.device_map,
                    trust_remote_code=True,
                )
                backbone = self._maybe_apply_lora(backbone)
                LOGGER.info("Loaded Stage B backbone: %s", model_name)
                return backbone, processor, model_name
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                LOGGER.warning("Failed to load model %s: %s", model_name, exc)
        raise RuntimeError(
            f"Unable to load primary model {self.config.model_name} or fallback {self.config.fallback_model_name}."
        )

    def _maybe_apply_lora(self, backbone):
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
            return backbone
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=active_modules,
        )
        return get_peft_model(backbone, lora_config)

    def compute_loss(self, prompts: list[str], image_paths: list[str], targets: list[str]) -> torch.Tensor:
        """Compute masked causal LM loss for prompt completion."""

        if isinstance(self.backbone, DummyCaptionBackbone):
            return self.backbone.placeholder.sum() * 0
        full_texts = [f"{prompt}\nAnswer: {target}" for prompt, target in zip(prompts, targets)]
        prompt_only = [f"{prompt}\nAnswer:" for prompt in prompts]
        images = [load_image(path) for path in image_paths]
        full_inputs = self.processor(images=images, text=full_texts, return_tensors="pt", padding=True, truncation=True)
        prompt_inputs = self.processor(images=images, text=prompt_only, return_tensors="pt", padding=True, truncation=True)
        full_inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in full_inputs.items()}
        prompt_inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in prompt_inputs.items()}
        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in enumerate(prompt_lengths):
            labels[row, : int(prompt_length)] = -100
        outputs = self.backbone(**full_inputs, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate_one(
        self,
        *,
        prompt: str,
        image_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate one caption candidate."""

        if isinstance(self.backbone, DummyCaptionBackbone):
            return self.backbone.heuristic_generate(prompt)
        image = load_image(image_path)
        inputs = self.processor(images=[image], text=[prompt], return_tensors="pt", padding=True, truncation=True)
        prepared = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        outputs = self.backbone.generate(
            **prepared,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:  # pragma: no cover - processor dependent
            decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:", maxsplit=1)[-1]
        return decoded.strip()

    def save(self, output_dir: str | Path) -> None:
        """Persist model and processor."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_dir / "processor")
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(output_dir / "backbone")
