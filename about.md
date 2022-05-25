---
layout: page
title: About
permalink: /about/
---

I am a research engineer in Meta AI, working on applying AI (and especially transformers) to problems of mathematics.
This blog presents unpublished results, not worth a research paper but interesting nevertheless. I intend to update it monthly (see [my twitter account](https://twitter.com/f_charton) for announcements).

Each experiment documents an attempt to use transformers to solve a specific math problem. I usually document the problem, the data generation, and model evaluation procedure, the main results and a few lessons learned. I tend to reuse the same code base, derived from [the one developped our paper on dynamical systems](https://github.com/facebookresearch/MathsFromExamples), and will eventually open source it. A high level description of the models and training procedures can be found [here](/models/).

### Why AI for maths?

### My publications on AI for Maths

* [Deep learning for symbolic mathematics (2019)](https://arxiv.org/abs/1912.01412), with Guillaume Lample: transformers can learn to integrate functions, and solve first and second order ordinary differential equations ([code](https://github.com/facebookresearch/SymbolicMathematics)).
* [Learning advanced mathematical computations from examples (2020)](https://arxiv.org/abs/2006.06462), with Amaury Hayat and Guillaume Lample: learning proposerties of differential systems, convergence at a critical point (aka the Spectral Mapping Theorem), controllability of overparametrized systems, integrability of some partial differential equations ([code](https://github.com/facebookresearch/MathsFromExamples)). 
* [A deep language model to predict metabolic network equilibria (2021)](https://arxiv.org/abs/2112.03588), with Amaury Hayat, Sean McQuade, Nathaniel Merrill and Benedetto Piccoli: predicting properties of transport graphs, existence of an equilibrium, and flows at the equilibrium.
* [Linear algebra with transformers (2021)](https://arxiv.org/abs/2112.01898): learning basic operations on matrices (transposition, addition, multiplication), eigenvalue and singular value decomposition and matrix inversion. First results about out-of-distribution generalization: models can generalize if their training distribution is chosen wisely.
* [Deep Symbolic Regression for Recurrent Sequences (2021)](https://arxiv.org/abs/2201.04600), with Stéphane d'Ascoli, Pierre-Alexandre Kamienny and Guillaume Lample: recovering underlying recurrence relations from a sequence of numbers. When predicting the next terms in a sequence (e.g. IQ tests), discovering the law (symbolic regression) and then using it to predict outperforms direct prediction. 
* [End-to-end symbolic regression with transformers (2022)](https://arxiv.org/abs/2204.10532), with Pierre-Alexandre Kamienny, Stéphane d'Ascoli and Guillaume Lample: transformers can predict functions from their values, first attempt at a model that uses both numeric and symbolic tokens.

### Interested? 
contact me:
[fcharton@gmail.com](mailto:fcharton@gmail.com)
