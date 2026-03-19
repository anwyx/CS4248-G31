# Meme Pipeline

This repository implements a two-stage multimodal meme-understanding pipeline:

- Stage A: multimodal metaphor mapping prediction (`vehicle -> target`)
- Stage B: multimodal meme meaning caption generation conditioned on Stage A mappings

Both stages use only `Qwen/Qwen3-VL-4B-Instruct`, loaded from a local folder.

**Local Layout**

Place assets in this structure:

```text
repo_root/
  models/
    Qwen3-VL-4B-Instruct/
    grounding_dino/
  data/
    images/
    train.jsonl
    val.jsonl
    test.jsonl
```

Default config paths:

- Qwen model: `models/Qwen3-VL-4B-Instruct`
- Grounding DINO: `models/grounding_dino`
- images: `data/images`
- splits: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`

**Pipeline**

```text
Stage A input:
  meme image
  image captions (img_captions)
  title
  OCR text if available
  candidate vehicle phrases from image captions
  optional grounding metadata

Stage A output:
  structured metaphor mappings
  example: woman -> meme poster

Stage B input:
  meme image
  image captions
  title
  OCR text if available
  Stage A predicted metaphor mappings

Stage B output:
  candidate meme meaning captions
  reranked best meme meaning caption
```

**Dataset Format**

Primary JSONL schema:

```json
{
  "category": "memes",
  "img_captions": [
    "A woman shows off her engagement ring which Thor approves of.",
    "A couple poses in the top left picture, a close up of a hand with a ring on the upper right, and a smiling man holding a hammer on the bottom."
  ],
  "meme_captions": [
    "Husband feels great after having their wife fall in love with him again after getting amnesia.",
    "The meme poster feels happy for the person who make his wife remember their love even after she forgot all.",
    "meme poster is conveying they feel like thor when a woman says to marrying them"
  ],
  "title": "He did it",
  "url": "https://i.redd.it/dkfj1vnjhuk31.jpg",
  "img_fname": "memes_d079np.png",
  "metaphors": [
    {
      "metaphor": "A woman",
      "meaning": "Meme poster"
    },
    {
      "metaphor": "her engagement ring",
      "meaning": "Meme poster"
    },
    {
      "metaphor": "A couple",
      "meaning": "Meme poster"
    }
  ],
  "post_id": "d079np",
  "ocr_text": ""
}
```

Interpretation:

- `post_id`: sample id
- `img_fname`: image file name
- `image_path`: derived as `Path(image_root_dir) / img_fname`
- `img_captions`: image caption input to Stage A and Stage B
- `meme_captions`: Stage B gold references
- `metaphors`: Stage A gold supervision
- `title`: textual context for both stages
- `ocr_text`: optional OCR text, defaults to empty string if absent

**Installation**

1. Create a Python 3.10+ environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Install the spaCy model with `python -m spacy download en_core_web_sm`.
4. Place local model folders under `models/`.

No online fallback is used for Qwen or Grounding DINO. Missing local folders cause a clear error.

**Stage A**

Stage A always uses:

- full meme image
- all `img_captions`
- `title`
- `ocr_text`
- candidate vehicle phrase
- grounding bbox/crop metadata when enabled

Training supervision:

- vehicle: `metaphors[].metaphor`
- target: `metaphors[].meaning`

Run training:

```bash
python scripts/train_stage_a.py --config meme_pipeline/configs/stage_a.yaml
```

Run inference:

```bash
python scripts/infer_stage_a.py \
  --config meme_pipeline/configs/stage_a.yaml \
  --input data/test.jsonl \
  --output outputs/stage_a_test_predictions.jsonl
```

Example Stage A output:

```json
{
  "id": "d079np",
  "metaphor_mappings": [
    {
      "vehicle_surface": "A woman",
      "vehicle_normalized": "woman",
      "bbox_xyxy": [12, 55, 178, 240],
      "grounding_score": 0.81,
      "predicted_target": "meme poster",
      "predicted_target_id": 5,
      "target_confidence": 0.74,
      "topk_targets": [
        ["meme poster", 0.74],
        ["happy husband", 0.12]
      ]
    }
  ]
}
```

Saved Stage A artifacts:

- `outputs/stage_a_model/qwen_model/` for model + processor
- `outputs/stage_a_model/classifier_head.pt`
- `outputs/stage_a_model/target_vocab.json`

**Stage B**

Stage B always uses:

- full meme image
- all `img_captions`
- `title`
- `ocr_text`
- Stage A predicted metaphor mappings

Training supervision:

- one training instance per reference in `meme_captions`
- all references are preserved for multi-reference evaluation utilities

Run training:

```bash
python scripts/train_stage_b.py --config meme_pipeline/configs/stage_b.yaml
```

Run inference:

```bash
python scripts/infer_stage_b.py \
  --config meme_pipeline/configs/stage_b.yaml \
  --input data/test.jsonl \
  --stage_a_outputs outputs/stage_a_test_predictions.jsonl \
  --output outputs/stage_b_test_predictions.jsonl
```

Example Stage B output:

```json
{
  "id": "d079np",
  "predicted_targets": [
    ["meme poster", 0.74]
  ],
  "predicted_mappings": [
    ["A woman", "meme poster"]
  ],
  "candidate_captions": [
    {
      "text": "The meme expresses feeling proud and validated when love is recognized again.",
      "score": 0.81,
      "penalties": {
        "vehicle_copy": 0.0,
        "ocr_copy": 0.0,
        "title_copy": 0.0
      }
    }
  ],
  "best_caption": "The meme expresses feeling proud and validated when love is recognized again."
}
```

Saved Stage B artifacts:

- `outputs/stage_b_model/qwen_model/` for model + processor

**Full Pipeline**

```bash
python scripts/run_full_stage_ab.py \
  --stage_a_config meme_pipeline/configs/stage_a.yaml \
  --stage_b_config meme_pipeline/configs/stage_b.yaml \
  --input data/test.jsonl \
  --output_dir outputs/full_pipeline/
```

**Common Mistakes**

- Wrong `image_root_dir`: `image_path` is derived from `image_root_dir + img_fname`.
- Missing image file: training and inference fail clearly.
- Missing `models/Qwen3-VL-4B-Instruct`: no silent fallback is used.
- Missing `models/grounding_dino` while `use_grounding=true`: Stage A fails clearly.
- JSONL missing required keys such as `post_id`, `img_fname`, or `img_captions`.
- No `metaphors` in Stage A training data: target vocab build fails clearly.
- No `meme_captions` in Stage B training data: Stage B builds no training instances.
