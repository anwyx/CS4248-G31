"""CLI for end-to-end Stage A -> Stage B inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from meme_pipeline.stage_a.infer import run_stage_a_inference
from meme_pipeline.stage_b.infer import run_stage_b_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a_config", required=True)
    parser.add_argument("--stage_b_config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_a_output = output_dir / "stage_a_predictions.jsonl"
    stage_b_output = output_dir / "stage_b_predictions.jsonl"
    run_stage_a_inference(args.stage_a_config, args.input, str(stage_a_output))
    run_stage_b_inference(args.stage_b_config, args.input, str(stage_a_output), str(stage_b_output))


if __name__ == "__main__":
    main()
