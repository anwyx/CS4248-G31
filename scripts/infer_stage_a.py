"""CLI for Stage A inference."""

from __future__ import annotations

import argparse

from meme_pipeline.stage_a.infer import run_stage_a_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_stage_a_inference(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
