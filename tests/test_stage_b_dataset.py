import json

from PIL import Image

from meme_pipeline.stage_b.dataset import StageBDataset, StageBDatasetConfig


def test_stage_b_training_expands_multi_reference_meme_captions(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "memes_d079np.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "category": "memes",
                "img_captions": ["A woman shows off her engagement ring which Thor approves of."],
                "meme_captions": [
                    "The meme expresses feeling proud and validated.",
                    "The meme shows someone feeling celebrated after being chosen.",
                ],
                "title": "He did it",
                "img_fname": "memes_d079np.png",
                "metaphors": [{"metaphor": "A woman", "meaning": "Meme poster"}],
                "post_id": "d079np",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = StageBDataset(
        jsonl_path,
        config=StageBDatasetConfig(oracle_target_mode=True),
        image_root_dir=image_root,
        nlp=None,
    )
    assert len(dataset) == 2
    assert dataset[0]["sample_id"] == "d079np"
    assert len(dataset[0]["reference_captions"]) == 2
