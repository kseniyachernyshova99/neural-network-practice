"""Image preparation and environment diagnostics CLI."""
import argparse
import importlib.metadata
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
from .scanner import scan, read_image


def doctor():
    """Print Python, libraries, working directory and optional CUDA status."""
    print("Python:", platform.python_version(), "cwd:", Path.cwd())
    for name in ("numpy", "Pillow", "matplotlib"):
        try:
            print(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print(name, "NOT INSTALLED")
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run([executable], capture_output=True,
                                    text=True, timeout=15, check=False)
            print("nvidia-smi status:", result.returncode)
            print(result.stdout or result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            print("CUDA check failed:", error)
    else:
        print("CUDA: nvidia-smi unavailable; CPU execution supported")


def main():
    """Parse CLI/ENV options and prepare an image array or inspect environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--path", default=os.getenv("DATA_PATH", "data/input/images"))
    prepare.add_argument("--ext", nargs="+", default=["png", "jpg", "jpeg"])
    prepare.add_argument("--output", default=str(Path(os.getenv(
        "OUTPUT_DIR", "data/output")) / "prepared.npy"))
    prepare.add_argument("--size", type=int, default=64)
    commands.add_parser("doctor")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    if args.command == "doctor":
        doctor()
        return
    try:
        if args.size < 1:
            raise ValueError("Size must be positive")
        started = time.perf_counter()
        files = list(scan(args.path, args.ext))
        if not files:
            raise ValueError("No matching image files")
        result = np.stack([read_image(path, args.size) for path in files])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            np.save(stream, result)
        print(f"files={len(files)} shape={result.shape} dtype={result.dtype}")
        print(f"range=[{result.min():.3f}, {result.max():.3f}] bytes={result.nbytes}")
        print(f"seconds={time.perf_counter() - started:.4f} output={output}")
    except (OSError, ValueError) as error:
        logging.error("Preparation failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
