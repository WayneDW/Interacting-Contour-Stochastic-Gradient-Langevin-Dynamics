

### Requirement

* Python 3.8.10 or similar
* [PyTorch = 1.10.2](https://pytorch.org/) or similar
* numpy
* CUDA


## Mode exploration on MNIST dataset

This code is only conducted based on a single chain to show how the interacting random field function outperforms the vanilla alternatives.

SGD
```python
$ python3 bayes_cnn.py -sn 500 -lr 1e-6 -classes 5 -zeta 0 -T 0
```

SGLD
```python
$ python3 bayes_cnn.py -sn 500 -lr 1e-6 -classes 5 -zeta 0 -T 0.1
```

ICSGLD or CSGLD based on the novel random field function
```python
$ python3 bayes_cnn.py -sn 500 -lr 1e-6 -classes 5 -zeta 3e4 -T 0.1
```
