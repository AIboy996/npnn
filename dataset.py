"""
Provide a dataloader.
Fashion MINIST dataset refer to https://github.com/zalandoresearch/fashion-mnist
"""

import os
import gzip
from typing import Literal
from urllib.request import urlretrieve
from hashlib import md5

from npnn import np


def download_minist(dataset_path="./data"):
    data_urls = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    ]
    md5_checksums = [
        "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "25c81989df183df01b3e8a0aad5dffbe",
        "bef4ecab320f06d8554ea6380940ec79",
        "bb300cfdad3c16e7a12a480ee83cd310",
    ]
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        print(f"Downloading data into {os.path.abspath(dataset_path)}")
        for idx, url in enumerate(data_urls):
            save_path = f'{dataset_path}/{url.split("/")[-1]}'
            try:
                urlretrieve(url, save_path)
                with open(save_path, "rb") as f:
                    assert md5(f.read()).hexdigest() == md5_checksums[idx]
            except AssertionError:
                print(f"Downloading {url} failed, Checksum Wrong.")
            except Exception as e:
                print(f"Downloading {url} failed.")
                print(repr(e))
                continue
    else:
        for idx, url in enumerate(data_urls):
            save_path = f'{dataset_path}/{url.split("/")[-1]}'
            with open(save_path, "rb") as f:
                assert (
                    md5(f.read()).hexdigest() == md5_checksums[idx]
                ), "Checksum Wrong."


def load_mnist(
    dataset_path, kind: Literal["train", "val", "test"] = "train", train_prop=0.9
):
    """Load Fashion MNIST data from `dataset_path`"""

    download_minist(dataset_path)
    load_file = "t10k" if kind == "test" else "train"
    labels_path = os.path.join(dataset_path, "%s-labels-idx1-ubyte.gz" % load_file)
    images_path = os.path.join(dataset_path, "%s-images-idx3-ubyte.gz" % load_file)

    with gzip.open(labels_path, "rb") as lbpath:
        # shape = (60000,)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        # shape = (60000, 784)
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )
    # default is 54000, train:val:test = 5.4 : 0.6 : 1
    train_len = int(train_prop * images.shape[0])
    if kind == "train":
        return images[:train_len, :], labels[:train_len]
    elif kind == "val":
        return images[train_len:], labels[train_len:]
    elif kind == "test":
        return images, labels
    else:
        raise ValueError("Invalid dataset name.")


if __name__ == "__main__":
    images, labels = load_mnist("./data", "test")
    breakpoint()
