"""train model"""

import logging
import os
import time

import nnn.functional as F
from nnn.optim import Adam
from nnn import Tensor

from dataset import load_mnist
from model import FNN
from test import test_model
from utils import save_model

logger = logging.getLogger(__name__)
if not os.path.exists("./logs"):
    os.mkdir("./logs")
if not os.path.exists("./checkpoints"):
    os.mkdir("./checkpoints")
date = time.strftime(r"%Y_%m%d")
hashcode = f"{date}({int(time.time())})"
logging.basicConfig(
    filename=f"./logs/{hashcode}.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)

IMAGE_SIZE = 28 * 28
NUM_CLASS = 10
TOTAL_EPOCH = 5
MINI_BATCHSIZE = 5

model = FNN(
    in_size=IMAGE_SIZE,
    out_size=NUM_CLASS,
    hidden_size=[
        256,
    ],
    activation=F.ReLU,
)
optimizer = Adam(model.parameters(), lr=0.01)

train_images, train_labels = load_mnist("./data", "train")
# trun into one hot
train_labels_onehot = F.one_hot(train_labels, NUM_CLASS)

# model's last layer is LogSoftmax, so we use NLL Loss function here
# this is equivalent to CrossEntropy Loss
criterion = F.NLL()

best = 0
for epoch in range(TOTAL_EPOCH):
    dataset_size = len(train_images)
    for b in range(dataset_size // MINI_BATCHSIZE)[:50]:
        mean_loss = 0
        optimizer.zero_grad()
        for x, y in zip(
            train_images[MINI_BATCHSIZE * b : MINI_BATCHSIZE * (b + 1),],
            train_labels_onehot[MINI_BATCHSIZE * b : MINI_BATCHSIZE * (b + 1),],
        ):
            y_hat = model(Tensor(x))
            y = Tensor(y)
            loss = criterion(y_hat, y)
            loss.backward()
            mean_loss += loss.data[0]
        # take one setp on gradient direction on each mini-batch data
        optimizer.step()
        mean_loss /= MINI_BATCHSIZE
        accuracy = test_model(model, dataset="val")
        if accuracy > best:
            best = accuracy
            file_name = save_model(model, f"./checkpoints/{hashcode}", "best_model")
            logger.info(f"find better model, saved to {file_name}.")
        logger.info(f"train loss={mean_loss : .4f}, valid accuracy={accuracy: .4f}")
