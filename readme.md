# [npnn](https://pypi.org/project/npnn/0.0.1/)
> NumPy Neural Network

[![PyPI - Version](https://img.shields.io/pypi/v/npnn)](https://pypi.org/project/npnn/0.0.1/)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/npnn)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/npnn)
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dm/npnn) -->


## What's `npnn`?
> `npnn` is a a torch-like Python module for **gradient descent based machine learning** implemented with NumPy. 


## Work with `npnn`!
> construct a image classification neural network with npnn

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
- `utils.py`: some misc function, such as `save_mode`