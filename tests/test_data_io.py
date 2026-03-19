import json

from PIL import Image

from meme_pipeline.data.io import load_raw_samples


def test_image_path_is_derived_from_image_root_dir(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "memes_d079np.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "post_id": "d079np",
                "img_fname": "memes_d079np.png",
                "img_captions": ["A woman shows off her engagement ring."],
                "title": "He did it",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_raw_samples(jsonl_path, image_root_dir=image_root)
    assert samples[0].image_path == str(image_path)
