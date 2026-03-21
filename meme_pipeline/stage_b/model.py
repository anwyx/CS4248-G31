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

    model_name_or_path: str = "models/Qwen3-VL-4B-Instruct"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    vision_max_pixels: int = 512 * 28 * 28  # cap image patches; limits vision encoder attention to O(512²)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None


class DummyCaptionBackbone(nn.Module):
    """Simple deterministic caption fallback used only when tests inject it."""

    def __init__(self) -> None:
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))

    def heuristic_generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "predicted metaphor mappings" in prompt_lower:
            tail = prompt.split("Predicted metaphor mappings:", maxsplit=1)[-1]
            mappings = [line.strip("- ").strip() for line in tail.splitlines() if "->" in line]
            if mappings:
                return f"The meme conveys {mappings[0].split('->')[-1].strip()} in a proud, validated way."
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
        self.model_name_or_path = config.model_name_or_path
        if self.backbone is None:
            self.backbone, self.processor = self._load_backbone()
        if isinstance(self.backbone, nn.Module) and not self.config.device_map:
            self.backbone.to(self.device)

    def _load_backbone(self):
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(
                "transformers>=4.57.0 with Qwen3-VL support is required to load the Stage B backbone."
            ) from exc
        model_path = Path(self.config.model_name_or_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Qwen model folder not found: {model_path}. "
                "Place Qwen/Qwen3-VL-4B-Instruct at the configured local path."
            )
        try:
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                min_pixels=4 * 28 * 28,
                max_pixels=self.config.vision_max_pixels,
                trust_remote_code=True,
                local_files_only=True,
            )
            quantization_config = None
            if self.config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            load_kwargs: dict[str, Any] = dict(
                torch_dtype=self.dtype,
                device_map=self.config.device_map,
                trust_remote_code=True,
                local_files_only=True,
            )
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            adapter_config_path = model_path / "adapter_config.json"
            if adapter_config_path.exists():
                import json
                from peft import PeftModel

                with adapter_config_path.open("r", encoding="utf-8") as handle:
                    adapter_config = json.load(handle)
                base_model_path = Path(adapter_config["base_model_name_or_path"])
                if not base_model_path.exists():
                    raise FileNotFoundError(
                        f"Base model referenced by adapter does not exist locally: {base_model_path}"
                    )
                backbone = Qwen3VLForConditionalGeneration.from_pretrained(str(base_model_path), **load_kwargs)
                backbone = PeftModel.from_pretrained(backbone, str(model_path), local_files_only=True)
            else:
                backbone = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **load_kwargs)
        except Exception as exc:  # pragma: no cover - runtime/model dependent
            raise RuntimeError(f"Failed to load Qwen/Qwen3-VL-4B-Instruct from {model_path}: {exc}") from exc
        if self.config.load_in_4bit and self.config.use_lora:
            from peft import prepare_model_for_kbit_training
            backbone = prepare_model_for_kbit_training(backbone)
        backbone = self._maybe_apply_lora(backbone)
        LOGGER.info("Loaded Stage B backbone from %s", model_path)
        return backbone, processor

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
            LOGGER.warning("No configured LoRA target modules found in backbone; skipping LoRA.")
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
        images = [load_image(path) for path in image_paths]

        def _apply_template(image: Any, text: str, *, generation_prompt: bool = False) -> str:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)

        full_texts = [_apply_template(img, f"{p}\n{t}") for img, p, t in zip(images, prompts, targets)]
        prompt_only = [_apply_template(img, p) for img, p in zip(images, prompts)]
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
        """Generate one caption candidate and decode only new tokens."""

        if isinstance(self.backbone, DummyCaptionBackbone):
            return self.backbone.heuristic_generate(prompt)
        image = load_image(image_path)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        formatted_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=[formatted_text], return_tensors="pt", padding=True, truncation=True)
        prepared = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        outputs = self.backbone.generate(
            **prepared,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        prompt_length = int(prepared["input_ids"].shape[1])
        generated_only = outputs[:, prompt_length:]
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(generated_only, skip_special_tokens=True)[0]
        else:  # pragma: no cover - processor dependent
            decoded = self.processor.decode(generated_only[0], skip_special_tokens=True)
        return decoded.strip()

    def save(self, output_dir: str | Path) -> None:
        """Persist model and processor to one local folder."""

        output_dir = Path(output_dir)
        qwen_dir = output_dir / "qwen_model"
        qwen_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(qwen_dir)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(qwen_dir)
