"""Generate reproducible toy XOR and image datasets without downloads."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    """Write numeric CSVs and synthetic RGB PNG/JPEG fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/input")
    parser.add_argument("--images", type=int, default=48)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    features = rng.integers(0, 2, (400, 2))
    targets = np.logical_xor(features[:, 0], features[:, 1]).astype(int)
    np.savetxt(root / "xor.csv", np.column_stack((features, targets)),
               delimiter=",", header="feature1,feature2,target", comments="", fmt="%d")
    np.savetxt(root / "predict.csv", [[0, 0], [0, 1], [1, 0], [1, 1]],
               delimiter=",", header="feature1,feature2", comments="", fmt="%d")
    for index in range(args.images):
        folder = root / "images" / ("group_a" if index % 2 else "group_b")
        folder.mkdir(parents=True, exist_ok=True)
        array = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        suffix = "png" if index % 2 else "jpg"
        Image.fromarray(array).save(folder / f"image_{index:03d}.{suffix}")
    print(f"Created XOR CSV, prediction CSV and {args.images} synthetic images")


if __name__ == "__main__":
    main()
