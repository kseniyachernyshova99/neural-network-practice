"""CSV loading and leakage-free standardization."""
import numpy as np
from .exceptions import MismatchedDataError


class DatasetManager:
    """Manage a numeric CSV with binary labels in its last column."""

    def load(self, path, test_fraction=0.2, seed=42):
        """Split before fitting the scaler; return train and test arrays."""
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if (values.shape[0] < 4 or values.shape[1] < 2
                or not np.isfinite(values).all()
                or not np.isin(values[:, -1], [0, 1]).all()
                or not 0 < test_fraction < 1):
            raise MismatchedDataError("Need >=4 finite rows, binary labels, 0<test<1")
        indices = np.random.default_rng(seed).permutation(len(values))
        count = max(1, min(len(values) - 1, round(len(values) * test_fraction)))
        test, train = values[indices[:count]], values[indices[count:]]
        self.mean = train[:, :-1].mean(axis=0)
        self.scale = train[:, :-1].std(axis=0)
        self.scale[self.scale == 0] = 1
        return ((train[:, :-1] - self.mean) / self.scale, train[:, -1:],
                (test[:, :-1] - self.mean) / self.scale, test[:, -1:])
