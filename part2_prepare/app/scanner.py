"""Shared discovery and RGB preprocessing."""
from pathlib import Path
import numpy as np
from PIL import Image


def scan(path, extensions):
    """Yield sorted matching paths recursively; reject missing directories."""
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {path}")
    allowed = {"." + item.lower().lstrip(".") for item in extensions}
    for file in sorted(directory.rglob("*")):
        if file.is_file() and file.suffix.lower() in allowed:
            yield file


def read_image(path, size=64):
    """Return float32 RGB image of shape (size, size, 3), range [0, 1]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)),
                          dtype=np.float32) / 255.0
