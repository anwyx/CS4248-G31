"""Image loading and crop helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert it to RGB."""

    return Image.open(image_path).convert("RGB")


def save_crop(
    image_path: str | Path,
    bbox_xyxy: list[float],
    output_path: str | Path,
) -> str:
    """Save a crop and return its path."""

    image = load_image(image_path)
    x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
    crop = image.crop((x1, y1, x2, y2))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return str(output_path)
