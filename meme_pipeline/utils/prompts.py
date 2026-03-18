"""Centralized prompt templates."""

from __future__ import annotations

from typing import Sequence


def build_stage_a_classification_prompt(
    *,
    title: str,
    ocr_text: str,
    literal_caption: str,
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
        f"Literal image caption: {literal_caption}\n"
        f"Candidate vehicle: {vehicle_surface}\n"
        f"Candidate vehicle normalized: {vehicle_normalized}\n"
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
    literal_caption: str,
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
        f"Literal image caption: {literal_caption}\n"
        f"Vehicle: {vehicle_surface}\n"
        f"Predicted target: {predicted_target}\n\n"
        "Task:\n"
        "Explain in one or two concise sentences why this target fits the vehicle in the meme."
    )


def build_stage_b_generation_prompt(
    *,
    title: str,
    ocr_text: str,
    literal_caption: str,
    target_concepts: Sequence[str],
    vehicle_blacklist: Sequence[str] | None = None,
) -> str:
    """Build the Stage B target-conditioned generation prompt."""

    target_block = "\n".join(f"- {target}" for target in target_concepts) if target_concepts else "- none available"
    prompt = (
        "System:\n"
        "You explain the intended meaning of memes.\n"
        "Use abstract target concepts rather than literal template objects.\n"
        "Do not describe the image in a purely literal way.\n"
        "Do not name the vehicle words unless absolutely necessary.\n"
        "Avoid copying character names, template names, and raw OCR verbatim.\n"
        "Write one concise meme-meaning caption.\n\n"
        "User:\n"
        f"Title: {title}\n"
        f"OCR: {ocr_text}\n"
        f"Literal image caption: {literal_caption}\n"
    )
    if target_concepts:
        prompt += f"Predicted target concepts:\n{target_block}\n"
    prompt += (
        "\nTask:\n"
        "Explain the intended meaning of this meme in one concise sentence.\n"
        "Use the target concepts.\n"
        "Avoid naming the literal vehicle objects.\n"
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
