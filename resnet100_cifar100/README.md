

### Requirement

* Python 3.8.10 or similar
* [PyTorch = 1.10.2](https://pytorch.org/) or similar
* numpy
* CUDA


## Uncertainty Estimation via Image Data

Folder: **resnet100_cifar100**.

How to train baseline algorithms on CIFAR100 using ResNet20:

m-SGDxP4 (a low temperature version of SGHMC)
```python
$ python3 bayes_cnn.py -sn 500 -model resnet -depth 20 -c sghmc -T 1e-10 -period 0 -warm 0.85 -burn 0.98
```

Replica exchange method xP4
```python
$ python3 bayes_cnn.py -sn 500 -model resnet -depth 20 -c replica -correction 4000 -warm 0.8 -burn 0.8
```

cycSGHMC & cycSWAG xT4
```python
$ python3 bayes_cnn.py -sn 2000 -chains 1 -model resnet -depth 20 -c cswag -period 0 -cycle 10 -warm 0.94 -burn 0.94
```

**ICSGHMCxP4**

ResNet20 
```python
$ python3 bayes_cnn.py -sn 500 -model resnet -depth 20 -c csghmc -stepsize 0.003 -zeta 3e6 -part 200 -div 200 -bias 2e4 -warm 0.75 -burn 0.80
```

ResNet32
```python
$ python3 bayes_cnn.py -sn 500 -model resnet -depth 32 -c csghmc -stepsize 0.003 -zeta 3e6 -part 200 -div 200 -bias 2e4 -warm 0.75 -burn 0.80
```

ResNet56
```python
$ python3 bayes_cnn.py -sn 500 -model resnet -depth 56 -c csghmc -stepsize 0.003 -zeta 3e6 -part 200 -div 200 -bias 1e4 -warm 0.75 -burn 0.80
```

WRN-16-8
```python
$ python3 bayes_cnn.py -sn 500 -model wrn -depth 0 -c csghmc -stepsize 0.003 -zeta 3e6 -part 200 -div 60 -bias 0 -warm 0.8 -burn 0.80
```
