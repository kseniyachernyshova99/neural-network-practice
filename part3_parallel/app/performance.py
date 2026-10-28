"""Compare complete sequential and parallel image-processing pipelines."""
import argparse
import json
import logging
import os
from pathlib import Path
import time
import numpy as np
from part2_prepare.app.scanner import scan, read_image
from .augment import augment
from .producer_consumer import parallel


def main():
    """Run both implementations on identical inputs and save measured results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=os.getenv("DATA_PATH", "data/input/images"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "data/output"))
    parser.add_argument("--ext", nargs="+", default=["png", "jpg"])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(message)s")
    try:
        if min(args.size, args.readers, args.processes) < 1:
            raise ValueError("Size, readers and processes must be positive")
        started = time.perf_counter()
        files = list(scan(args.path, args.ext))
        if not files:
            raise ValueError("No matching images")
        sequential = np.concatenate([augment((i, read_image(file, args.size)))[1]
                                     for i, file in enumerate(files)])
        sequential_seconds = time.perf_counter() - started
        started = time.perf_counter()
        concurrent, count = parallel(args.path, args.ext, args.size,
                                     args.readers, args.processes)
        parallel_seconds = time.perf_counter() - started
        report = dict(files=count, shape=list(concurrent.shape),
                      sequential_seconds=sequential_seconds,
                      parallel_seconds=parallel_seconds,
                      speedup=sequential_seconds / parallel_seconds,
                      outputs_equal=bool(np.array_equal(sequential, concurrent)),
                      readers=args.readers, processes=args.processes,
                      timing_scope="scan, read, augment, collect; excludes saving")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "augmented.npy", concurrent)
        (output / "performance_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("Processing failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
