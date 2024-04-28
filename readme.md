# [npnn](https://pypi.org/project/npnn/)
> NumPy Neural Network

[![PyPI - Version](https://img.shields.io/pypi/v/npnn)](https://pypi.org/project/npnn/)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/npnn)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/npnn)
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dm/npnn) -->


## What's npnn?
> `npnn` is a a **torch-like** Python module for **gradient descent based machine learning** implemented with `numpy`. 

### Dependency
Basically `npnn` only depends on `numpy`(the latest version 1.26.4 is verified).

If you have CUDA devices available, then you can easily get a acceleration by installing suitable version of `cupy`.  In this case `npnn` will use `cupy` api rather than `numpy` api.

For example, my PC have CUDA v12.x (x86_64), so I use command:
```bash
pip install cupy-cuda12x
pip install npnn
```
or in short:
```bash
pip install npnn[cuda12x]
```
check [cupy documentation](https://docs.cupy.dev/en/stable/install.html#installing-cupy) for more information.


### API references

See [npnn WIKI](https://github.com/AIboy996/npnn/wiki).

### Known issues

See [npnn known-issues](https://github.com/AIboy996/npnn/wiki#known-issues).

## Work with npnn!
> Here we will construct a image classification neural network with npnn.

BTW, this is a course assignment of *DATA620004, School of Data Science, Fudan University*.

### Task
Construct and Train a neural network on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) to do image classification.

- Implement gradient backpropagation algorithm by hand,you can use `numpy` but **DO NOT** use `pytorch` or `tensorflow` to do autograd.

- Submit source code including at least four parts: `model definition`, `training`, `parameters searching` and `testing`.

### Implementation

- `dataset.py`: provide Fashion MNIST dataset
- `model.py`: model definition
- `train.py`: model training
- `search.py`: parameters searching
- `test.py`: model testing
- `viz.py`: visualization
- `utils.py`: some misc function, such as `save_model`

run `search.py`, you can get a table like:

no|train_id|accuracy|hidden_size|batch_size|learning_rate|regularization|regular_strength
--|--|--|--|--|--|--|--
0|2024_0423(1713841292)|0.8306|[384]|3|0.002|None|0.0
1|2024_0423(1713845802)|0.8145|[384]|3|0.002|l2|0.1
2|2024_0423(1713849349)|0.8269|[384]|3|0.002|l2|0.01
3|2024_0423(1713853939)|0.8255|[384]|3|0.002|l2|0.005
4|2024_0423(1713857657)|0.8373|[384]|3|0.002|l2|0.001

train log file and saved model weights can be found in `./logs` and `./checkpoints` folder.

### Experiments

See [report.ipynb](report.ipynb) or more readable version: [report.pdf](report.pdf).

## LICENSE

MIT