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

The **first** sampling algorithm that efficiently explores deep energy landscapes via a fixed learning rate.

This is a simple MNIST dataset, but so far the existing algorithms cannot even freely explore multiple modes while maintaining exploitations using a fixed learning rate.

<p float="left">
  <img src="./figures/ICSGLD_losses_path.gif" width="200" title="SGLD"/>
  <img src="./figures/SGD_SGLD_CSGLD.png" width="600" title="SGLD"/>
</p>



### Requirement

* Python 3.8.10 or similar
* [PyTorch = 1.10.2](https://pytorch.org/) or similar
* numpy
* CUDA

