"""test model"""

from npnn import Tensor

from dataset import load_mnist
from utils import load_model

MINI_BATCHSIZE = 16


def test_model(model, dataset="val"):
    test_images, test_labels = load_mnist("./data", dataset)
    test_size = len(test_images)
    accuracy = 0
    for batch in range(len(test_labels) // MINI_BATCHSIZE):
        x = test_images[
            MINI_BATCHSIZE * batch: MINI_BATCHSIZE * (batch + 1), :, None
        ]
        y = test_labels[
            MINI_BATCHSIZE * batch: MINI_BATCHSIZE * (batch + 1), None, None
        ]
        y_hat = model(Tensor(x)).data.argmax(axis=1, keepdims=True)
        accuracy += (y == y_hat).sum().item()
    accuracy /= test_size
    return accuracy


if __name__ == "__main__":
    best_model = load_model(r"checkpoints\2024_0419(1713542060)\best_model.xz")
    metric = test_model(best_model, dataset="test")
    print(f"test done, metric = {metric}")
