"""Interactive console entry point; run from the project root."""
import numpy as np
from .models import NeuralNetwork
from .manager import DatasetManager
from .exceptions import PracticeError, InputError, MismatchedDataError


def main():
    """Run the menu and convert input/file failures to application errors."""
    model, dataset = None, None
    while True:
        print("1 Create network | 2 Load CSV | 3 Train | 4 Predict")
        print("5 Save model | 6 Load model | 7 Plot loss | 0 Exit")
        try:
            choice = input("> ").strip()
            try:
                if choice == "0":
                    return
                if choice == "1":
                    model = NeuralNetwork(list(map(int, input("Sizes (2 8 1): ").split())),
                                          input("sigmoid/relu: ").strip())
                    dataset = None
                    print("Network created:", model.sizes, model.activation)
                elif choice == "6":
                    model = NeuralNetwork.load(input("Model path: ").strip())
                    dataset = None
                    print("Model loaded")
                elif model is None:
                    raise InputError("Create or load a network first")
                elif choice == "2":
                    manager = DatasetManager()
                    arrays = manager.load(input("CSV path: ").strip(),
                                          float(input("Test fraction (0.2): ")))
                    if arrays[0].shape[1] != model.sizes[0]:
                        raise MismatchedDataError("CSV features do not match network")
                    dataset = arrays
                    model.mean, model.scale = manager.mean, manager.scale
                    print("Train/test shapes:", dataset[0].shape, dataset[2].shape)
                elif choice == "3":
                    if dataset is None:
                        raise InputError("Load CSV first")
                    model.fit(dataset[0], dataset[1],
                              float(input("Learning rate: ")),
                              int(input("Epochs: ")), int(input("Batch size: ")))
                    accuracy = np.mean((model.forward(dataset[2]) >= .5) == dataset[3])
                    print(f"Test accuracy: {accuracy:.3f}")
                elif choice == "4":
                    source = input("m = manual, f = CSV (features only): ")
                    if source == "f":
                        rows = np.loadtxt(input("Path: "), delimiter=",",
                                          skiprows=1, ndmin=2)
                    elif source == "m":
                        rows = np.array([list(map(float, row.split())) for row in
                                         input("Rows separated by ; : ").split(";")])
                    else:
                        raise InputError("Choose m or f")
                    for probability in model.predict_raw(rows).ravel():
                        print(f"p={probability:.6f} class={int(probability >= .5)}")
                elif choice == "5":
                    model.save(input("Path (data/output/model.npz): "))
                    print("Model saved")
                elif choice == "7":
                    import matplotlib.pyplot as plt
                    plt.plot(model.history)
                    plt.xlabel("Epoch")
                    plt.ylabel("Binary cross entropy")
                    plt.tight_layout()
                    plt.show()
                else:
                    raise InputError("Unknown menu item")
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise InputError(str(error)) from error
        except PracticeError as error:
            print("ERROR:", error)
        except (EOFError, KeyboardInterrupt):
            print("Exiting")
            return


if __name__ == "__main__":
    main()
