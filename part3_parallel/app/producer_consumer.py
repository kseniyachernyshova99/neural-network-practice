"""Bounded producer-consumer queues and a spawn-based process pool."""
import logging
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from part2_prepare.app.scanner import scan, read_image
from .augment import augment


def parallel(path, extensions, size=64, readers=4, processes=2):
    """Return augmented array and successful file count; propagate failures."""
    paths, arrays = queue.Queue(maxsize=8), queue.Queue(maxsize=8)
    failures, outputs = [], []
    lock = threading.Lock()
    processed = 0

    def producer():
        """Scan recursively in one producer thread, then send stop tokens."""
        try:
            for index, file in enumerate(scan(path, extensions)):
                paths.put((index, file))
        except (OSError, ValueError) as error:
            with lock:
                failures.append(str(error))
        finally:
            for _ in range(readers):
                paths.put(None)

    def consumer():
        """Read images in a thread and update the protected shared counter."""
        nonlocal processed
        try:
            while True:
                item = paths.get()
                try:
                    if item is None:
                        return
                    index, file = item
                    try:
                        arrays.put((index, read_image(file, size)))
                        with lock:
                            processed += 1
                        logging.info("%s read %s", threading.current_thread().name,
                                     Path(file).name)
                    except (OSError, ValueError) as error:
                        with lock:
                            failures.append(str(error))
                finally:
                    paths.task_done()
        finally:
            arrays.put(None)

    def receive(result):
        """Collect a process result in the parent via apply_async callback."""
        outputs.append(result)

    if readers < 1 or processes < 1 or size < 1:
        raise ValueError("Readers, processes and size must be positive")
    with mp.get_context("spawn").Pool(processes) as pool:
        with ThreadPoolExecutor(max_workers=readers + 1) as threads:
            futures = [threads.submit(producer)]
            futures.extend(threads.submit(consumer) for _ in range(readers))
            finished, pending = 0, []
            while finished < readers:
                item = arrays.get()
                if item is None:
                    finished += 1
                else:
                    pending.append(pool.apply_async(augment, (item,), callback=receive))
                    if len(pending) >= processes * 2:
                        try:
                            pending.pop(0).get(timeout=120)
                        except (mp.TimeoutError, ValueError, RuntimeError) as error:
                            failures.append(str(error))
                arrays.task_done()
            for future in futures:
                future.result()
            for result in pending:
                result.get(timeout=120)
        pool.close()
        pool.join()
    if failures:
        raise ValueError("; ".join(failures))
    if not outputs:
        raise ValueError("No matching images")
    return np.concatenate([value for _, value in sorted(outputs)]), processed
