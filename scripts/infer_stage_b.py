"""CLI for Stage B inference."""

from __future__ import annotations

import argparse

from meme_pipeline.stage_b.infer import run_stage_b_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stage_a_outputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_stage_b_inference(args.config, args.input, args.stage_a_outputs, args.output)


if __name__ == "__main__":
    main()
