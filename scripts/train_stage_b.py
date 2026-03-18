"""CLI for Stage B training."""

from __future__ import annotations

import argparse

from meme_pipeline.stage_b.trainer import load_stage_b_trainer_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    trainer = load_stage_b_trainer_from_config(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
