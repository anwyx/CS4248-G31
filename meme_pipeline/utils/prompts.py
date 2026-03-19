"""Centralized prompt templates."""

from __future__ import annotations

from typing import Sequence


def _format_numbered_captions(img_captions: Sequence[str]) -> str:
    if not img_captions:
        return "1. "
    return "\n".join(f"{index + 1}. {caption}" for index, caption in enumerate(img_captions))


def _format_mappings(mappings: Sequence[tuple[str, str]] | Sequence[str]) -> str:
    lines: list[str] = []
    for item in mappings:
        if isinstance(item, tuple):
            lines.append(f"- {item[0]} -> {item[1]}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines) if lines else "- none available"


def build_stage_a_classification_prompt(
    *,
    title: str,
    ocr_text: str,
    img_captions: Sequence[str],
    vehicle_surface: str,
    vehicle_normalized: str,
    vehicle_head: str,
    bbox_or_none: str,
) -> str:
    """Build the structured Stage A classification prompt."""

    return (
        "System:\n"
        "You are a multimodal metaphor analysis model.\n"
        "Given a meme image and contextual text, infer what the highlighted vehicle refers to in the meme's intended meaning.\n\n"
        "User text:\n"
        f"Title: {title}\n"
        f"OCR: {ocr_text}\n"
        f"Image captions:\n{_format_numbered_captions(img_captions)}\n"
        f"Candidate vehicle: {vehicle_surface}\n"
        f"Vehicle normalized: {vehicle_normalized}\n"
        f"Vehicle head noun: {vehicle_head}\n"
        f"Grounding bbox: {bbox_or_none}\n\n"
        "Task:\n"
        "Predict the TARGET category for this candidate vehicle.\n"
        "Return the best target class."
    )


def build_stage_a_debug_generation_prompt(
    *,
    title: str,
    ocr_text: str,
    img_captions: Sequence[str],
    vehicle_surface: str,
    predicted_target: str,
) -> str:
    """Build the optional Stage A explanation prompt."""

    return (
        "System:\n"
        "You briefly justify metaphor interpretations for debugging.\n\n"
        "User:\n"
        f"Title: {title}\n"
        f"OCR: {ocr_text}\n"
        f"Image captions:\n{_format_numbered_captions(img_captions)}\n"
        f"Vehicle: {vehicle_surface}\n"
        f"Predicted target: {predicted_target}\n\n"
        "Task:\n"
        "Explain in one or two concise sentences why this target fits the vehicle in the meme."
    )


def build_stage_b_generation_prompt(
    *,
    title: str,
    ocr_text: str,
    img_captions: Sequence[str],
    metaphor_mappings: Sequence[tuple[str, str]],
    target_concepts: Sequence[str],
    vehicle_blacklist: Sequence[str] | None = None,
) -> str:
    """Build the Stage B target-conditioned generation prompt."""

    prompt = (
        "System:\n"
        "You explain the intended meaning of memes.\n"
        "Use the meme image, image captions, OCR text, and predicted metaphor mappings.\n"
        "Do not describe the image in a purely literal way.\n"
        "Avoid naming the vehicle words unless absolutely necessary.\n"
        "Avoid copying raw OCR and title text verbatim.\n"
        "Write one concise meme-meaning caption.\n\n"
        "User:\n"
        f"Title: {title}\n"
        f"OCR: {ocr_text}\n"
        f"Image captions:\n{_format_numbered_captions(img_captions)}\n"
        f"Predicted metaphor mappings:\n{_format_mappings(metaphor_mappings)}\n"
    )
    if target_concepts:
        prompt += f"Predicted target concepts:\n{_format_mappings(target_concepts)}\n"
    prompt += (
        "\nTask:\n"
        "Explain the intended meaning of this meme in one concise sentence.\n"
        "Use the metaphor mappings and target concepts.\n"
        "Avoid naming the literal vehicle objects if possible.\n"
        "Avoid copying the image template.\n"
        "Avoid excessive specificity not supported by the evidence.\n\n"
        "Expected output style:\n"
        "- one sentence\n"
        "- concise but meaningful\n"
        "- semantic explanation, not object listing\n"
        "- no bullet points\n"
        "- no JSON unless explicitly requested\n"
    )
    if vehicle_blacklist:
        prompt += (
            "\nConstraint:\n"
            "The caption must not contain any of these vehicle strings: "
            + ", ".join(vehicle_blacklist)
        )
    return prompt
