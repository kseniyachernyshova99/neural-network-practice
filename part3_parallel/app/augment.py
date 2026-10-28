"""Deterministic CPU augmentation functions safe for process spawning."""
import numpy as np


def augment(job):
    """Return original, rotated and noisy mirrored images for (index, array)."""
    index, array = job
    rng = np.random.default_rng(42 + index)
    angle = int(rng.integers(1, 4))
    noisy = np.clip(np.flip(array, axis=1)
                    + rng.normal(0, .025, array.shape), 0, 1).astype(np.float32)
    return index, np.stack([array, np.rot90(array, angle), noisy])
