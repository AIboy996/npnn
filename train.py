"""train model"""

import nnn.functional as F
from nnn.optim import Adam
from nnn import Tensor

from dataset import load_mnist
from model import FNN
from test import test_model

IMAGE_SIZE = 28 * 28
NUM_CLASS = 10

model = FNN(
    in_size=IMAGE_SIZE,
    out_size=NUM_CLASS,
    hidden_size=[256,],
    activation=F.ReLU,
)
optimizer = Adam(model.parameters(), lr=0.01)

train_images, train_labels = load_mnist("./data", "train")
# trun into one hot
train_labels_onehot = F.one_hot(train_labels, NUM_CLASS)

# model's last layer is LogSoftmax, so we use NLL Loss function here
# this is equivalent to CrossEntropy Loss
criterion = F.NLL()

for epoch in range(3):
    dataset_size = len(train_images)
    mini_batch_size = 1
    for b in range(dataset_size // mini_batch_size):
        mean_loss = 0
        for x, y in zip(
            train_images[mini_batch_size * b : mini_batch_size * (b + 1),],
            train_labels_onehot[mini_batch_size * b : mini_batch_size * (b + 1),],
        ):
            optimizer.zero_grad()
            y_hat = model(Tensor(x))
            y = Tensor(y)
            loss = criterion(y_hat, y)
            loss.backward()
            mean_loss += loss.data[0]
            optimizer.step()
        mean_loss /= mini_batch_size
        accuracy = test_model(model)
        print(f"mean_loss={mean_loss : .4f}, accuracy={accuracy: .4f}")
        break
    break
