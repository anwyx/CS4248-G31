import json

from PIL import Image

from meme_pipeline.stage_a.infer import StageAInferencePipeline
from meme_pipeline.stage_a.model import DummyVisionLanguageBackbone


def test_stage_a_inference_writes_jsonl(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "memes_d079np.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    data_path = tmp_path / "input.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "category": "memes",
                "img_captions": [
                    "A woman shows off her engagement ring which Thor approves of.",
                    "A couple poses in the top left picture.",
                ],
                "meme_captions": ["The meme expresses feeling proud and validated."],
                "title": "He did it",
                "img_fname": "memes_d079np.png",
                "metaphors": [{"metaphor": "A woman", "meaning": "Meme poster"}],
                "post_id": "d079np",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    vocab_dir = tmp_path / "model"
    vocab_dir.mkdir()
    (vocab_dir / "target_vocab.json").write_text(
        '{"stoi":{"OTHER":0,"NO_TARGET":1,"meme poster":2},"itos":{"0":"OTHER","1":"NO_TARGET","2":"meme poster"}}',
        encoding="utf-8",
    )

    class DummyPipeline(StageAInferencePipeline):
        def __init__(self, config):
            from meme_pipeline.grounding.factory import NoOpGrounder
            from meme_pipeline.data.target_vocab import load_target_vocab
            from meme_pipeline.stage_a.model import StageAModelConfig, StageAMetaphorClassifier

            self.config = config
            self.nlp = lambda text: __import__("tests.test_vehicle_extractor", fromlist=["DummyDoc"]).DummyDoc(text)
            self.target_vocab = load_target_vocab(vocab_dir / "target_vocab.json")
            self.grounder = NoOpGrounder()
            self.model = StageAMetaphorClassifier(
                config=StageAModelConfig(),
                num_targets=len(self.target_vocab),
                backbone=DummyVisionLanguageBackbone(hidden_size=16),
                processor=None,
                hidden_size=16,
            )

    pipeline = DummyPipeline(
        {
            "model_output_dir": str(vocab_dir),
            "target_vocab_path": str(vocab_dir / "target_vocab.json"),
            "image_root_dir": str(image_root),
        }
    )
    output_path = tmp_path / "out.jsonl"
    pipeline.predict_jsonl(str(data_path), str(output_path))
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "d079np"
    assert "metaphor_mappings" in payload
