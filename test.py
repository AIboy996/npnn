"""test model"""

from dataset import load_mnist
from nnn import Tensor


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
