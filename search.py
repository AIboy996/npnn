"""search hyperparameter"""

import json
from pathlib import Path
import pandas as pd

from train import train

lst = []
# control batch size due to GPU memory limit
for hidden_size, batch_size in [
    # ([512, 256], 2),
    # ([256, 128], 4),
    # ([256, 64], 8),
    # ([128, 64], 8),
    ([384], 3),
    ([256], 8),
    ([128], 16),
]:
    for learning_rate in range(1, 11):
        learning_rate *= 0.002
        for regular in [
            (None, 0),
            ("l2", 0.1),
            ("l2", 0.01),
            ("l2", 0.005),
            ("l2", 0.001),
        ]:

            print(f"searching {learning_rate=}, {hidden_size=}, {regular=}")
            regularization, regular_strength = regular
            train(
                hidden_size=hidden_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                regularization=regularization,
                regular_strength=regular_strength,
            )
l = []
for json_file in Path("./logs").glob("*.json"):
    with open(json_file) as f:
        train_log = json.load(f)
        del train_log["train_loss"]
        del train_log["valid_metric"]
        l.append(train_log)

df = pd.DataFrame(l)
df.to_csv("./search_result.csv")
