---
layout: page
title: About
permalink: /about/
---

I am a research engineer at FAIR, Meta, researching the use of language models in mathematics and theoretical physics. I graduated (a long time ago) from Ecole Polytechnique and ENSAE, where I majored in statistics. After a career in media, advertising, and software development, I had the good fortune of landing a short term contract as a visiting entrepreneur in Meta (best job title ever!), and publishing a [paper], with Guillaume Lample(https://arxiv.org/abs/1912.01412), demonstrating that transformers can be trained to perform symbolic integration, with the same accuracy as computer algebras. Then, one thing led to another, and I became, at the fresh age of 55, a full-time research scientist, working on AI4Science. My recent scientific news can usually be found on [my twitter account](https://twitter.com/f_charton). I can be contacted at [fcharton@gmail.com](mailto:fcharton@gmail.com)

### Publications on AI for Science

* [End-to-end symbolic regression with transformers (2022)](https://arxiv.org/abs/2204.10532), with Pierre-Alexandre Kamienny, Stéphane d'Ascoli and Guillaume Lample: transformers can predict functions from their values, first attempt at a model that uses both numeric and symbolic tokens.
* [Deep Symbolic Regression for Recurrent Sequences (2021)](https://arxiv.org/abs/2201.04600), with Stéphane d'Ascoli, Pierre-Alexandre Kamienny and Guillaume Lample: recovering underlying recurrence relations from a sequence of numbers. When predicting the next terms in a sequence (e.g. IQ tests), discovering the law (symbolic regression) and then using it to predict outperforms direct prediction. 
* [Linear algebra with transformers (2021)](https://arxiv.org/abs/2112.01898): learning basic operations on matrices (transposition, addition, multiplication), eigenvalue and singular value decomposition and matrix inversion. First results about out-of-distribution generalization: models can generalize if their training distribution is chosen wisely.
* [A deep language model to predict metabolic network equilibria (2021)](https://arxiv.org/abs/2112.03588), with Amaury Hayat, Sean McQuade, Nathaniel Merrill and Benedetto Piccoli: predicting properties of transport graphs, existence of an equilibrium, and flows at the equilibrium.
* [Learning advanced mathematical computations from examples (2020)](https://arxiv.org/abs/2006.06462), with Amaury Hayat and Guillaume Lample: learning proposerties of differential systems, convergence at a critical point (aka the Spectral Mapping Theorem), controllability of overparametrized systems, integrability of some partial differential equations ([code](https://github.com/facebookresearch/MathsFromExamples)). 
* [Deep learning for symbolic mathematics (2019)](https://arxiv.org/abs/1912.01412), with Guillaume Lample: transformers can learn to integrate functions, and solve first and second order ordinary differential equations ([code](https://github.com/facebookresearch/SymbolicMathematics)).

