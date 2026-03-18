from meme_pipeline.utils.prompts import build_stage_b_generation_prompt


def test_stage_b_prompt_contains_targets():
    prompt = build_stage_b_generation_prompt(
        title="When work piles up",
        ocr_text="",
        literal_caption="a cat at a desk",
        target_concepts=["confused person", "social pressure"],
        vehicle_blacklist=["cat"],
    )
    assert "confused person" in prompt
    assert "social pressure" in prompt
    assert "cat" in prompt
