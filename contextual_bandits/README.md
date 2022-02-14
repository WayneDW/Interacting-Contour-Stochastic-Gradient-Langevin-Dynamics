# Interacting-Contour-Stochastic-Gradient-Langevin-Dynamics
An embarrassingly parallel dynamic importance sampler with efficient interactions, where the interacting parallel version [\[Link\]](https://openreview.net/pdf?id=IK9ap6nxXr2) can be theoretically more efficient than a single-chain CSGLD [\[Link\]](https://arxiv.org/pdf/2010.09800.pdf) with an equivalent computational budget.


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


## Deep Contextual Bandits on Mushroom tasks


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
