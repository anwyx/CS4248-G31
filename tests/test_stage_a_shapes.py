import torch

from meme_pipeline.stage_a.model import DummyVisionLanguageBackbone, StageAMetaphorClassifier, StageAModelConfig


def test_stage_a_forward_returns_expected_shape():
    model = StageAMetaphorClassifier(
        config=StageAModelConfig(),
        num_targets=7,
        backbone=DummyVisionLanguageBackbone(hidden_size=32),
        processor=None,
        hidden_size=32,
    )
    batch = {
        "image_path": ["image_1.jpg", "image_2.jpg"],
        "title": ["title", "title"],
        "ocr_text": ["", ""],
        "literal_caption": ["cat on chair", "dog near desk"],
        "vehicle_surface": ["cat", "dog"],
        "vehicle_normalized": ["cat", "dog"],
        "vehicle_head": ["cat", "dog"],
        "bbox_xyxy": [None, None],
    }
    logits = model(batch)
    assert isinstance(logits, torch.Tensor)
    assert tuple(logits.shape) == (2, 7)
