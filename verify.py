"""Integration and finite difference checks."""
import numpy as np
from part1_mlp.app.models import NeuralNetwork
from part1_mlp.app.manager import DatasetManager


def main():
    """Verify convergence, persistence and both backpropagation variants."""
    manager = DatasetManager()
    train, target, test, expected = manager.load('data/input/xor.csv')
    model = NeuralNetwork([2, 8, 1])
    model.mean, model.scale = manager.mean, manager.scale
    print('Network:', model.sizes, model.activation)
    print('Train/test:', train.shape, test.shape)
    model.fit(train, target, .3, 1000, 16)
    accuracy = np.mean((model.forward(test) >= .5) == expected)
    print('Test accuracy:', accuracy)
    assert accuracy == 1
    raw = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print('Predictions:', model.predict_raw(raw).ravel())
    model.save('data/output/model.npz')
    restored = NeuralNetwork.load('data/output/model.npz')
    np.testing.assert_allclose(model.predict_raw(raw), restored.predict_raw(raw))
    for activation in ('sigmoid', 'relu'):
        net = NeuralNetwork([2, 3, 1], activation)
        features = np.array([[.3, -.7], [.8, .4]])
        labels = np.array([[1.], [0.]])
        def loss():
            probabilities = net.forward(features)
            return -np.mean(labels * np.log(probabilities)
                            + (1 - labels) * np.log(1 - probabilities))
        originals = [value.copy() for value in net.weights + net.biases]
        numerical = []
        for parameter in net.weights + net.biases:
            gradient = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                before = parameter[index]
                parameter[index] = before + 1e-5
                plus = loss()
                parameter[index] = before - 1e-5
                minus = loss()
                parameter[index] = before
                gradient[index] = (plus - minus) / 2e-5
            numerical.append(gradient)
        net.fit(features, labels, 1e-4, 1, 2)
        for old, new, gradient in zip(originals, net.weights + net.biases, numerical):
            np.testing.assert_allclose((old - new) / 1e-4, gradient, atol=1e-6)
        print('Gradient check OK:', activation)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
