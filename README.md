# Meme Pipeline

Production-oriented multimodal meme understanding pipeline with two stages:

- Stage A: structured metaphor prediction from vehicle to target
- Stage B: target-conditioned meme meaning caption generation

The primary requested model is `Robertp423/Qwen3-VL-4B-Construct`. If that model is unavailable at runtime, the code logs a warning and falls back to `Qwen/Qwen3-VL-4B-Instruct`.

**Architecture**

```text
raw sample (image + title + optional OCR + optional literal caption)
  -> Stage A vehicle extraction from literal caption
  -> optional phrase grounding (Grounding DINO 1.5, fallback YOLO-World/YOLOE)
  -> Qwen3-VL multimodal classifier + small head
  -> structured vehicle -> target predictions
  -> Stage B target serialization
  -> Qwen3-VL target-conditioned generation
  -> candidate reranking with copy penalties
  -> best meme meaning caption
```

**Installation**

1. Create a Python 3.10+ environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Install spaCy English model with `python -m spacy download en_core_web_sm`.

Optional:

- Install `flash-attn` if supported by your environment.
- Install `ultralytics` for YOLO-World fallback.
- Install `sentence-transformers` for stronger Stage B reranking.

**Model Download**

The code loads models from Hugging Face at runtime through `transformers`. Configure:

- primary Qwen model: `Robertp423/Qwen3-VL-4B-Construct`
- fallback Qwen model: `Qwen/Qwen3-VL-4B-Instruct`
- primary grounding model: Grounding DINO through `grounding_model_name`
- optional grounding fallback: YOLO-World or YOLOE through `grounding_fallback_model_name`

**Input Format**

Each JSONL line should look like:

```json
{
  "id": "sample_1",
  "image_path": "path/to/image.jpg",
  "title": "post title",
  "ocr_text": "optional OCR text",
  "literal_caption": "optional literal description",
  "gold_meme_caption": "optional gold intended-meaning caption",
  "vehicle_target_pairs": [
    {
      "vehicle": "cat",
      "target": "confused person"
    }
  ]
}
```

Malformed lines are skipped with logged line numbers. Missing OCR, literal caption, gold caption, or vehicle annotations are handled without crashing.

**Run Stage A**

Train:

```bash
python scripts/train_stage_a.py --config meme_pipeline/configs/stage_a.yaml
```

Infer:

```bash
python scripts/infer_stage_a.py \
  --config meme_pipeline/configs/stage_a.yaml \
  --input data/test.jsonl \
  --output outputs/stage_a_test_predictions.jsonl
```

Stage A output format:

```json
{
  "id": "sample_1",
  "vehicles": [
    {
      "vehicle_surface": "confused cat",
      "vehicle_normalized": "confused cat",
      "bbox_xyxy": [12, 55, 178, 240],
      "grounding_score": 0.81,
      "predicted_target": "confused person",
      "predicted_target_id": 17,
      "target_confidence": 0.74,
      "topk_targets": [
        ["confused person", 0.74],
        ["overwhelmed employee", 0.11],
        ["helpless student", 0.05]
      ]
    }
  ]
}
```

**Run Stage B**

Train:

```bash
python scripts/train_stage_b.py --config meme_pipeline/configs/stage_b.yaml
```

Infer:

```bash
python scripts/infer_stage_b.py \
  --config meme_pipeline/configs/stage_b.yaml \
  --input data/test.jsonl \
  --stage_a_outputs outputs/stage_a_test_predictions.jsonl \
  --output outputs/stage_b_test_predictions.jsonl
```

Stage B output format:

```json
{
  "id": "sample_1",
  "predicted_targets": [
    ["confused person", 0.74]
  ],
  "candidate_captions": [
    {
      "text": "The meme expresses how someone feels overwhelmed and judged under pressure.",
      "score": 0.81,
      "penalties": {
        "vehicle_copy": 0.0,
        "ocr_copy": 0.1,
        "title_copy": 0.0
      }
    }
  ],
  "best_caption": "The meme expresses how someone feels overwhelmed and judged under pressure."
}
```

**Run Full Pipeline**

```bash
python scripts/run_full_stage_ab.py \
  --stage_a_config meme_pipeline/configs/stage_a.yaml \
  --stage_b_config meme_pipeline/configs/stage_b.yaml \
  --input data/test.jsonl \
  --output_dir outputs/full_pipeline/
```

**Common Failure Cases**

- Missing spaCy model: install `en_core_web_sm`.
- Primary Qwen model unavailable: the loader falls back automatically to `Qwen/Qwen3-VL-4B-Instruct`.
- Grounding backend unavailable: the pipeline continues with null boxes and no crop.
- Missing Stage A target vocab at inference time: provide the trained vocab path or run Stage A training first.
- No Stage A targets above threshold: Stage B falls back to a non-target-conditioned prompt.

**Notes**

- Stage A is a structured target classifier, not free-form explanation generation.
- Stage B is intended-meaning generation and explicitly discourages literal restatement and vehicle-name copying.
- `Qwen3-VL-4B-Construct` is treated as the user-requested primary model, but availability is not assumed.
