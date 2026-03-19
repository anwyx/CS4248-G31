from meme_pipeline.utils.prompts import build_stage_b_generation_prompt


def test_stage_b_prompt_contains_mapping_information():
    prompt = build_stage_b_generation_prompt(
        title="He did it",
        ocr_text="",
        img_captions=[
            "A woman shows off her engagement ring which Thor approves of.",
            "A couple poses in the top left picture.",
        ],
        metaphor_mappings=[("woman", "meme poster"), ("engagement ring", "meme poster")],
        target_concepts=["meme poster"],
        vehicle_blacklist=["woman", "engagement ring"],
    )
    assert "Image captions" in prompt
    assert "woman -> meme poster" in prompt
    assert "engagement ring -> meme poster" in prompt
