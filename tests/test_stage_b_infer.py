from meme_pipeline.stage_b.generator import StageBGenerator
from meme_pipeline.stage_b.model import DummyCaptionBackbone, StageBCaptionModel, StageBModelConfig


def test_stage_b_candidate_generation_returns_non_empty_candidate():
    model = StageBCaptionModel(config=StageBModelConfig(), backbone=DummyCaptionBackbone(), processor=None)
    generator = StageBGenerator(model)
    sample = {
        "image_path": "dummy.jpg",
        "title": "When work piles up",
        "ocr_text": "",
        "literal_caption": "a cat at a desk",
    }
    candidates = generator.generate_candidates(sample, target_concepts=["confused person"], vehicle_blacklist=["cat"], k=2)
    assert candidates
    assert any(candidate.strip() for candidate in candidates)
