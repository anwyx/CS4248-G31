from meme_pipeline.stage_b.generator import StageBGenerator
from meme_pipeline.stage_b.model import DummyCaptionBackbone, StageBCaptionModel, StageBModelConfig


def test_stage_b_candidate_generation_returns_non_empty_candidate():
    model = StageBCaptionModel(config=StageBModelConfig(), backbone=DummyCaptionBackbone(), processor=None)
    generator = StageBGenerator(model)
    sample = {
        "image_path": "dummy.jpg",
        "title": "He did it",
        "ocr_text": "",
        "img_captions": [
            "A woman shows off her engagement ring which Thor approves of.",
            "A couple poses in the top left picture.",
        ],
    }
    candidates = generator.generate_candidates(
        sample,
        target_concepts=["meme poster"],
        metaphor_mappings=[("woman", "meme poster")],
        vehicle_blacklist=["woman"],
        k=2,
    )
    assert candidates
    assert any(candidate.strip() for candidate in candidates)
