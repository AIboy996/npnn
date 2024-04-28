"""visualization"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import FNN
from utils import load_model


def viz_trainlog(
    log_file="./logs/search2/2024_0423(1713857657).json", ax=None, alpha=0.2, drop=10
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    with open(log_file) as f:
        dic = json.load(f)
    # ewm smoothing with alpha
    train_loss = pd.Series(dic["train_loss"])[drop:].ewm(alpha=alpha).mean()
    valid_metric = pd.Series(dic["valid_metric"])[drop:].ewm(alpha=alpha).mean()
    x = np.arange(len(train_loss))
    # plot
    ax_twin = ax.twinx()
    ax.plot(x, train_loss, linestyle="dashed", label="train loss")
    ax_twin.plot(
        x, valid_metric, linestyle="dotted", color="black", label="valid metric"
    )
    ax.legend(loc=2)
    ax_twin.legend(loc=1)


def viz_model(model="2024_0423(1713857657)"):
    model: FNN = load_model(f"checkpoints/{model}/best_model.xz")
    parameters = model.parameters()
    num = len(parameters)
    fig, axes = plt.subplots(ncols=num, nrows=1, figsize=(10, 10))
    for idx, param in enumerate(parameters):
        # param.data are cupy array, so we should call get turn it into numpy array
        axes[idx].imshow(param.data.get()[0], cmap="winter", aspect="equal")
        axes[idx].set_axis_off()
    return fig


def viz_search(file="search_result1.csv", figsize=[(22, 10), (22, 10)]):
    df = pd.read_csv(file)
    lr_num = len(df["learning_rate"].unique())
    hidden_num = len(df["hidden_size"].unique())
    regular_num = len(df["regular_strength"].unique())
    # 固定学习率看正则化
    fig1, axes = plt.subplots(
        figsize=figsize[0], nrows=hidden_num, ncols=lr_num, sharex=True
    )
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.suptitle("Accuracy to regular strength\nhidden_size and lr fixed")
    for row, hidden in enumerate(df["hidden_size"].unique()):
        for col, lr in enumerate(df["learning_rate"].unique()):
            df_plot = df[(df.hidden_size == hidden) & (df.learning_rate == lr)][
                ["accuracy", "regular_strength"]
            ]
            ax = axes[row, col]
            ax.scatter(np.arange(len(df_plot.regular_strength)), df_plot.accuracy)
            ax.set_xticks(
                np.arange(len(df_plot.regular_strength)), df_plot.regular_strength
            )
            if row == hidden_num - 1:
                ax.set_xlabel(f"{lr=}", color="red")
            if col == 0:
                ax.set_ylabel(f"{hidden}", rotation=0, labelpad=20, color="blue")

    # 固定正则化看学习率
    fig2, axes = plt.subplots(
        figsize=figsize[1], nrows=hidden_num, ncols=regular_num, sharex=True
    )
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.suptitle("Accuracy to lr\nhidden_size and regular fixed")
    for row, hidden in enumerate(df["hidden_size"].unique()):
        for col, regular in enumerate(df["regular_strength"].unique()):
            df_plot = df[(df.hidden_size == hidden) & (df.regular_strength == regular)][
                ["accuracy", "learning_rate"]
            ]
            ax = axes[row, col]
            ax.scatter(np.arange(len(df_plot.learning_rate)), df_plot.accuracy)
            ax.set_xticks(np.arange(len(df_plot.learning_rate)), df_plot.learning_rate)
            if row == hidden_num - 1:
                ax.set_xlabel(f"{regular=}", color="red")
            if col == 0:
                ax.set_ylabel(f"{hidden}", rotation=0, labelpad=20, color="blue")
    return fig1, fig2


if __name__ == "__main__":
    viz_search()
    plt.show()
