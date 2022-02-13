# Interacting-Contour-Stochastic-Gradient-Langevin-Dynamics
A pleasantly parallel adaptive importance sampling algorithms for simulations of multi-modal distributions (ICLR'22). An embarrassingly parallel multiple-chain contour stochastic gradient Langevin dynamics (CSGLD) sampler  with efficient interactions, where the interacting parallel version [\[Link\]](https://openreview.net/pdf?id=IK9ap6nxXr2) can be theoretically more efficient than a single-chain CSGLD [\[Link\]](https://arxiv.org/pdf/2010.09800.pdf) with an equivalent computational budget.


```
@inproceedings{ICSGLD,
  title={Interacting Contour Stochastic Gradient Langevin Dynamics},
  author={Wei Deng and Siqi Liang and Botao Hao and Guang Lin and Faming Liang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```



### Requirement

* Python 3.8.10 or similar
* [PyTorch = 1.10.2](https://pytorch.org/) or similar
* numpy
* CUDA

## Simulations

Please check the **notebook (ipynb)** files for the ICSGLD algorithm and the baselines, respectively.

## Deep Contextual Bandits on Mushroom tasks

Folder: **contextual_bandits**.

All the algorithms run 2,000 epochs based on 4 parallel chains.

#### Const SGD (SGLD with temperature 0)

```python
$ python3 main.py -c sgld -lr 1e-6
```

#### EpsGreedy (with decaying learning rates and uniform exploration)

```python
$ python3 main.py -c sgld -decay 0.999 -epsilon 0.003 -lr 1e-6
```

#### Dropout

```python
$ python3 main.py -c sgld -lr 1e-6 -rate 0.5 -samples 5
```


#### pSGLD (preconditioner requires a different learning rate)
```python
$ python3 main.py -c sgld -precondition 1 -lr 3e-7 -T 0.3
```

#### ICSGLD xP4

```python
$ python3 main.py -c csgld -precondition 1 -lr 3e-7 -T 0.3 -part 200 -div 10 -sz 0.03 -zeta 20
```



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
