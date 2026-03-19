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
        "title": ["He did it", "He did it"],
        "ocr_text": ["", ""],
        "img_captions": [["A woman shows a ring."], ["A man looks proud."]],
        "vehicle_surface": ["A woman", "A man"],
        "vehicle_normalized": ["woman", "man"],
        "vehicle_head": ["woman", "man"],
        "bbox_xyxy": [None, None],
    }
    logits = model(batch)
    assert isinstance(logits, torch.Tensor)
    assert tuple(logits.shape) == (2, 7)
