# Interacting-Contour-Stochastic-Gradient-Langevin-Dynamics
A pleasantly parallel adaptive importance sampling algorithms for simulations of multi-modal distributions (ICLR'22). An embarrassingly parallel multiple-chain contour stochastic gradient Langevin dynamics (CSGLD) sampler  with efficient interactions, where the interacting parallel version [Link](https://openreview.net/pdf?id=IK9ap6nxXr2) can be theoretically more efficient than a single-chain CSGLD [link](https://arxiv.org/pdf/2010.09800.pdf) with an equivalent computational budget.


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

Move to the folder: **contextual_bandits**.

All the algorithms run 2,000 epochs based on 4 parallel chains.

#### Const SGD (SGLD with temperature 0)

```python
$ python main.py -c sgld -lr 1e-6
```

#### EpsGreedy (with decaying learning rates and uniform exploration)

```python
$ python main.py -c sgld -decay 0.999 -epsilon 0.003 -lr 1e-6
```

#### Dropout

```python
$ python main.py -c sgld -lr 1e-6 -rate 0.5 -samples 5
```


#### pSGLD (preconditioner requires a different learning rate)
```python
$ python main.py -c sgld -precondition 1 -lr 3e-7 -T 0.3
```

#### ICSGLD xP4

```python
$ python main.py -c csgld -precondition 1 -lr 3e-7 -T 0.3 -part 200 -div 10 -sz 0.03 -zeta 20
```
