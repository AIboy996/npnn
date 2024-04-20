"""search hyperparameter"""

import pandas as pd

from train import Trainer

lst = []
for lr in range(1, 11):
    lr *= 0.005
    for hidden_size in [
        [512, 384],
        [512, 256],
        [512, 128],
        [256, 128],
        [256, 64],
    ]:
        for regular in [
            (None, 0),
            ("l1", 0.5),
            ("l1", 0.1),
            ("l1", 0.01),
            ("l1", 0.001),
            ("l2", 0.5),
            ("l2", 0.1),
            ("l2", 0.01),
            ("l2", 0.001),
        ]:
            regularization, regular_strength = regular
            trainer = Trainer(
                hidden_size=hidden_size,
                regularization=regularization,
                regular_strength=regular_strength,
                lr=lr,
            )
            train_hashcode = trainer.train_hashcode
            trainer.train()
            metric = trainer.test()
            train_log = dict(
                train_id=train_hashcode,
                accuracy=metric,
                hidden_size=hidden_size,
                regularization=str(regularization),
                regular_strength=regular_strength,
                learning_rate=lr,
            )
            lst.append(train_log)
            print(train_hashcode, metric)

df = pd.DataFrame(lst)
df.to_csv('./search_result.csv')
