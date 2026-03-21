"""Download the original MemeCap dataset (eujhwang/meme-cap) and prepare it
for the meme pipeline.

Sources:
  JSON files  — https://github.com/eujhwang/meme-cap  (data/ directory)
  Images      — Google Drive file ID: 1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1

Usage (run from repo root on Google Colab):
    pip install gdown
    python scripts/prepare_data.py

Outputs:
    data/images/      - extracted meme images
    data/train.jsonl  - 90% of memes-trainval.json
    data/val.jsonl    - 10% of memes-trainval.json
    data/test.jsonl   - all of memes-test.json
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GITHUB_REPO = "https://github.com/eujhwang/meme-cap.git"
GDRIVE_FILE_ID = "1o1IB6am0HdYS58CEOmmxra3WjJkrn-M1"

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
IMAGES_ZIP = DATA_DIR / "images.zip"
REPO_CLONE_DIR = Path("/tmp/meme-cap")

VAL_RATIO = 0.1
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some versions store records as a dict keyed by post_id
        return list(data.values())
    raise ValueError(f"Unexpected JSON structure in {path}")


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>5} records → {path}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def clone_repo() -> None:
    if REPO_CLONE_DIR.exists():
        print(f"Repo already cloned at {REPO_CLONE_DIR}, skipping.")
        return
    print("Cloning meme-cap repo for JSON data files...")
    _run(["git", "clone", "--depth", "1", GITHUB_REPO, str(REPO_CLONE_DIR)])


def download_images() -> None:
    if IMAGE_DIR.exists() and any(IMAGE_DIR.iterdir()):
        print(f"Images already present at {IMAGE_DIR}, skipping download.")
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGES_ZIP.exists():
        print("Downloading images from Google Drive (this may take a few minutes)...")
        try:
            import gdown
        except ImportError:
            print("ERROR: gdown not installed. Run: pip install gdown")
            sys.exit(1)
        gdown.download(id=GDRIVE_FILE_ID, output=str(IMAGES_ZIP), quiet=False)

    print(f"Extracting {IMAGES_ZIP} → {IMAGE_DIR} ...")
    with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
        for member in zf.infolist():
            # Flatten any sub-directory structure — put all images directly in IMAGE_DIR
            fname = Path(member.filename).name
            if not fname or member.is_dir():
                continue
            target = IMAGE_DIR / fname
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"  Extracted {sum(1 for _ in IMAGE_DIR.iterdir())} images.")


def build_splits() -> None:
    trainval_path = REPO_CLONE_DIR / "data" / "memes-trainval.json"
    test_path = REPO_CLONE_DIR / "data" / "memes-test.json"

    trainval = _load_json(trainval_path)
    test = _load_json(test_path)

    rng = random.Random(SEED)
    rng.shuffle(trainval)

    n_val = max(1, int(len(trainval) * VAL_RATIO))
    val = trainval[:n_val]
    train = trainval[n_val:]

    _write_jsonl(train, DATA_DIR / "train.jsonl")
    _write_jsonl(val,   DATA_DIR / "val.jsonl")
    _write_jsonl(test,  DATA_DIR / "test.jsonl")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    clone_repo()
    download_images()
    build_splits()
    print("\nDone. Dataset ready under data/")


if __name__ == "__main__":
    main()
