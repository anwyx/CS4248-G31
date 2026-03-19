"""Optional Stage A explanation mode."""

from __future__ import annotations

from typing import Any

from meme_pipeline.utils.image_utils import load_image
from meme_pipeline.utils.prompts import build_stage_a_debug_generation_prompt


def explain_prediction(
    *,
    model,
    sample: dict[str, Any],
    predicted_target: str,
) -> dict[str, str]:
    """Generate or synthesize a short rationale for debugging."""

    prompt = build_stage_a_debug_generation_prompt(
        title=sample.get("title", ""),
        ocr_text=sample.get("ocr_text", ""),
        img_captions=sample.get("img_captions", []),
        vehicle_surface=sample["vehicle_surface"],
        predicted_target=predicted_target,
    )
    rationale = ""
    backbone = getattr(model, "backbone", None)
    processor = getattr(model, "processor", None)
    if hasattr(backbone, "generate") and processor is not None:
        try:  # pragma: no cover - depends on model runtime
            image = [load_image(sample["image_path"])]
            inputs = processor(images=image, text=[prompt], return_tensors="pt", padding=True)
            outputs = backbone.generate(**inputs, max_new_tokens=48)
            if hasattr(processor, "batch_decode"):
                rationale = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception:
            rationale = ""
    if not rationale:
        rationale = (
            f"The vehicle visually stands in for {predicted_target}, "
            "based on the meme image, image captions, OCR, and title context."
        )
    return {
        "vehicle": sample["vehicle_surface"],
        "predicted_target": predicted_target,
        "rationale": rationale,
    }
