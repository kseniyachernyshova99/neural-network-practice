"""A NumPy multilayer perceptron for binary classification."""
import numpy as np
from .exceptions import InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """Create layers from sizes; activation controls hidden layers."""

    def __init__(self, sizes, activation="sigmoid", seed=42):
        """Initialize random weights and zero biases for supplied sizes."""
        if len(sizes) < 3 or any(type(n) is not int or n < 1 for n in sizes):
            raise InvalidLayerSizeError("Use at least 3 positive layer sizes")
        if sizes[-1] != 1 or activation not in ("sigmoid", "relu"):
            raise InvalidLayerSizeError("Output must be 1; sigmoid or relu")
        self.sizes, self.activation = list(sizes), activation
        self.rng = np.random.default_rng(seed)
        self.weights = [self.rng.normal(0, np.sqrt(2 / left), (left, right))
                        for left, right in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((1, n)) for n in sizes[1:]]
        self.history = []
        self.mean = self.scale = None

    @staticmethod
    def sigmoid(values):
        """Return numerically bounded logistic activations."""
        return 1 / (1 + np.exp(-np.clip(values, -500, 500)))

    def forward(self, features):
        """Return probabilities and cache intermediate activations."""
        features = np.asarray(features, dtype=float)
        if (features.ndim != 2 or features.shape[1] != self.sizes[0]
                or not np.isfinite(features).all()):
            raise MismatchedDataError("Invalid feature dimensions or values")
        self.activations, self.preactivations = [features], []
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            linear = self.activations[-1] @ weights + bias
            self.preactivations.append(linear)
            output = (self.sigmoid(linear)
                      if index == len(self.weights) - 1
                      or self.activation == "sigmoid" else np.maximum(0, linear))
            self.activations.append(output)
        return self.activations[-1]

    def fit(self, features, targets, rate=0.1, epochs=1000, batch_size=16):
        """Train using binary cross entropy; return accumulated loss history."""
        targets = np.asarray(targets).reshape(-1, 1)
        self.forward(features)
        if (len(features) == 0 or len(targets) != len(features)
                or not np.isin(targets, [0, 1]).all()):
            raise MismatchedDataError("Targets must contain binary labels")
        if not np.isfinite(rate) or rate <= 0 or epochs < 1 or batch_size < 1:
            raise MismatchedDataError("Training parameters must be positive")
        for epoch in range(epochs):
            order = self.rng.permutation(len(features))
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                prediction = self.forward(features[indices])
                delta = (prediction - targets[indices]) / len(indices)
                gradients = []
                for layer in reversed(range(len(self.weights))):
                    gradients.append((self.activations[layer].T @ delta,
                                      delta.sum(axis=0, keepdims=True)))
                    if layer:
                        derivative = (self.activations[layer]
                                      * (1 - self.activations[layer])
                                      if self.activation == "sigmoid" else
                                      (self.preactivations[layer - 1] > 0))
                        delta = (delta @ self.weights[layer].T) * derivative
                for layer, (dw, db) in enumerate(reversed(gradients)):
                    self.weights[layer] -= rate * dw
                    self.biases[layer] -= rate * db
            probability = np.clip(self.forward(features), 1e-12, 1 - 1e-12)
            loss = -np.mean(targets * np.log(probability)
                            + (1 - targets) * np.log(1 - probability))
            if not np.isfinite(loss):
                raise MismatchedDataError("Divergence: reduce learning rate")
            self.history.append(float(loss))
            if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"epoch={epoch + 1}/{epochs} loss={loss:.6f}")
        return self.history

    def predict_raw(self, features):
        """Standardize raw features with stored train statistics and predict."""
        if self.mean is None:
            raise MismatchedDataError("Load data or a trained model first")
        return self.forward((np.asarray(features) - self.mean) / self.scale)

    def save(self, path):
        """Save topology, parameters, scaler and history to an NPZ file."""
        if self.mean is None:
            raise MismatchedDataError("Load data before saving the model")
        payload = dict(sizes=self.sizes, activation=self.activation,
                       mean=self.mean, scale=self.scale, history=self.history)
        payload.update({f"w{i}": w for i, w in enumerate(self.weights)})
        payload.update({f"b{i}": b for i, b in enumerate(self.biases)})
        with open(path, "wb") as stream:
            np.savez(stream, **payload)

    @classmethod
    def load(cls, path):
        """Validate and restore a model from a pickle-free NPZ archive."""
        with np.load(path, allow_pickle=False) as saved:
            model = cls(saved["sizes"].tolist(), str(saved["activation"]))
            for i in range(len(model.weights)):
                for prefix, arrays in (("w", model.weights), ("b", model.biases)):
                    value = saved[f"{prefix}{i}"]
                    if value.shape != arrays[i].shape or not np.isfinite(value).all():
                        raise MismatchedDataError("Invalid saved weights")
                    arrays[i] = value.copy()
            model.mean, model.scale = saved["mean"], saved["scale"]
            if (model.mean.shape != (model.sizes[0],)
                    or model.scale.shape != model.mean.shape
                    or not np.isfinite(model.mean).all()
                    or not np.isfinite(model.scale).all()
                    or (model.scale <= 0).any()):
                raise MismatchedDataError("Invalid saved scaler")
            model.history = saved["history"].tolist()
        return model
