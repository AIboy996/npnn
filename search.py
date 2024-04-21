"""search hyperparameter"""

import pandas as pd

from train import Trainer

lst = []
for lr in range(1, 11):
    lr *= 0.005
    for hidden_size, batch_size in [
        ([128], 32),
        ([256], 8),
        ([128, 64], 32),
        ([256, 64], 8),
        ([256, 128], 8),
        ([512, 128], 2),
        ([512, 256], 2),
        ([512, 384], 1),
    ]:
        # control batch due to GPU memory limit
        for regular in [
            (None, 0),
            ("l2", 0.5),
            ("l2", 0.1),
            ("l2", 0.01),
            ("l2", 0.001),
        ]:
            print(f"searching {lr=}, {hidden_size=}, {regular=}")
            regularization, regular_strength = regular
            with Trainer(
                hidden_size=hidden_size,
                regularization=regularization,
                regular_strength=regular_strength,
                lr=lr,
                batch_size=batch_size,
            ) as trainer:
                train_hashcode = trainer.train_hashcode
                trainer.train()
                metric = trainer.test()
            train_log = dict(
                train_id=train_hashcode,
                accuracy=metric,
                learning_rate=lr,
                hidden_size=hidden_size,
                batch_size=batch_size,
                regularization=str(regularization),
                regular_strength=regular_strength,
            )
            lst.append(train_log)
            print(train_hashcode, metric)

df = pd.DataFrame(lst)
df.to_csv("./search_result.csv")
