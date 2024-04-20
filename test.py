"""test model"""

from npnn import Tensor

from dataset import load_mnist
from utils import load_model


def test_model(model, dataset="val"):
    test_images, test_labels = load_mnist("./data", dataset)
    test_size = len(test_images)
    accuracy = 0
    for x, y in zip(test_images, test_labels):
        y_hat = model(Tensor(x)).data.argmax()
        if y_hat == y:
            accuracy += 1
    accuracy /= test_size
    return accuracy


if __name__ == "__main__":
    best_model = load_model(r"checkpoints\2024_0419(1713542060)\best_model.xz")
    metric = test_model(best_model, dataset="test")
    print(f"test done, metric = {metric}")
