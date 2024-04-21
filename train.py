"""train model"""

import logging
import os
import time
from typing import Literal

import npnn.functional as F
from npnn.optim import Adam
from npnn import Tensor, np

from dataset import load_mnist
from model import FNN
from test import test_model
from utils import save_model, load_model


IMAGE_SIZE = 28 * 28
NUM_CLASS = 10
TOTAL_EPOCH = 1


class Trainer:
    def __init__(
        self,
        hidden_size: list[int] = [256, 128, 64],
        activation=F.ReLU,
        regularization: Literal[None, "l1", "l2"] = None,
        regular_strength: float = 0.01,
        lr: float = 0.01,
        batch_size: int = 8,
    ) -> None:
        self.batch_size = batch_size
        self.model = FNN(
            in_size=IMAGE_SIZE,
            out_size=NUM_CLASS,
            hidden_size=hidden_size,
            activation=activation,
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            regularization=regularization,
            regular_strength=regular_strength,
        )
        train_images, train_labels = load_mnist("./data", "train")
        self.images = train_images
        # trun into one hot
        self.labels = F.one_hot(train_labels, NUM_CLASS)
        # model's last layer is LogSoftmax, so we use NLL Loss function here
        # this is equivalent to CrossEntropy Loss
        self.criterion = F.NLL()
        date = time.strftime(r"%Y_%m%d")
        self.train_hashcode = f"{date}({int(time.time())})"
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger()
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        logging.basicConfig(
            filename=f"./logs/{self.train_hashcode}.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        return logger

    def train(self):
        self.best_metric = 0
        self.best_model_path = ""
        for epoch in range(TOTAL_EPOCH):
            self.train_epoch(epoch)
        self.logger.info(f"training done, best metric = {self.best_metric}")

    def test(self):
        best_model = load_model(self.best_model_path)
        metric = test_model(best_model, dataset="test")
        self.logger.info(f"test done, metric = {metric}")
        return metric

    def train_epoch(self, epoch):
        dataset_size = len(self.images)
        total_loss = 0
        early_stop_count = 0
        for batch in range(dataset_size // self.batch_size):
            self.optimizer.zero_grad()
            x = self.images[
                self.batch_size * batch: self.batch_size * (batch + 1), :, None
            ]
            y = self.labels[
                self.batch_size * batch: self.batch_size * (batch + 1), :, None
            ]
            y_hat = self.model(Tensor(x))
            y = Tensor(y)
            loss = self.criterion(y_hat, y)
            loss.backward()
            total_loss += loss.data.mean().item()  # batch mean loss
            # take one setp on gradient direction on each mini-batch data
            self.optimizer.step()
            # do validation each 50 batch
            if batch % 50 == 1:
                metric = test_model(self.model, dataset="val")
                if metric > self.best_metric:
                    early_stop_count = 0
                    self.best_metric = metric
                    file_name = save_model(
                        self.model,
                        f"./checkpoints/{self.train_hashcode}",
                        "best_model",
                    )
                    self.best_model_path = file_name
                    self.logger.info(
                        f"{epoch=}, {batch=}, train loss={total_loss/batch : .4f}, valid metric={metric: .4f}.\n"
                        f"Find better model, saved to {file_name}.",
                    )
                else:
                    early_stop_count += 1
                    self.logger.info(
                        f"{epoch=}, {batch=}, train loss={total_loss/batch : .4f}, valid metric={metric: .4f}"
                    )
            if early_stop_count > (20000 // self.batch_size // 50):
                f"{epoch=}, Early stop since metric have no improvement for {early_stop_count} consecutive batches."
                break
        return total_loss / batch

    def __exit__(self, exc_type, exc_value, traceback):
        # close logging handler
        file_handler = self.logger.handlers[0]
        self.logger.removeHandler(file_handler)
        file_handler.close()
        if np.__name__ == 'cupy':
            # cupy free memory
            np.get_default_memory_pool().free_all_blocks()

    def __enter__(self):
        return self


if __name__ == "__main__":
    with Trainer(
        hidden_size=[256], activation=F.ReLU, regularization=None, batch_size=32
    ) as trainer:
        trainer.train()
        trainer.test()
