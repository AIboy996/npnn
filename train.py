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
    hidden_size=[512, 256],
    activation=F.ReLU,
)
optimizer = Adam(model.parameters(), lr=0.001)

train_images, train_labels = load_mnist("./data", "train")
# trun into one hot
train_labels_onehot = F.one_hot(train_labels, NUM_CLASS)

# model's last layer is LogSoftmax, so we use NLL Loss function here
# this is equivalent to CrossEntropy Loss
criterion = F.NLL()

for epoch in range(2):
    epoch_loss = 0
    dataset_size = len(train_images)
    for x, y in zip(train_images, train_labels_onehot):
        y_hat = model(Tensor(x))
        y = Tensor(y)
        loss = criterion(y_hat, y)
        loss.backward()
        epoch_loss += loss.data
        optimizer.step()
    epoch_loss /= dataset_size
    print(f"{epoch_loss=}")
    accuracy = test_model(model)
    print(f"{accuracy=}")