---
layout: post
title: Computing the roots of polynomials
---
TLDR: Transformers can be trained to predict the roots of polynomials from their coefficients: 

### The problem
In general, a **polynomial** of degree $n$ with real coefficients,
$P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$ ($a_i$ $\in$ $\mathbb{R}$), has $n$ **roots** ($x_i$ such that $P(x_i) = 0$). The roots are real numbers or pairs of conjugate complex numbers (i.e. pairs of the form ($a+ib$, $a-ib$), $a$, $b$ $\in$ $\mathbb R$). For small values of $n$, there are formulas to compute the roots from the coefficients. For instance, for degree 2, the roots of $P(x) = ax^2+bx+c$ can be computed as $x_0=\frac{ -b + \sqrt{b^2-4ac}}{2a}$ and $x_1=\frac{-b-\sqrt{b^2-4ac}}{2a}$ if $b^2-4ac>=0$, and $\frac{-b+i\sqrt{4ac-b^2}}{2a}$ and $\frac{-b-i\sqrt{4ac-b^2}}{2a}$ if $b^2-4ac<0$. When $n>4$, Galois and Abel have proven that no such formula exists, but numerical algorithms can compute approximate values of the roots.

In this post, I use transformers to predict approximate values of $x_i$ from $a_i$.

### Data generation
Models are trained from a dataset of polynomials and their roots, with degrees uniformly distributed in {$d_{min},d_{max}$}. To generate a random polynomial, I sample uniformly in $[-A,A]$ (choosing $A=10$) $n$ real numbers $r_0,\dots, r_{n-1}$. With probability $p_c$, $r_{2k}$ and $r_{2k+1}$ are interpreted as the pair of conjugate complex roots $x_{2k}=r_{2k}+i r_{2k+1}$ and $x_{2k+1}=r_{2k}-i r_{2k+1}$, and with probaility $1-p_c$ as two real roots $x_{2k}=r_{2k}$ and $x_{2k+1}=r_{2k+1}$. This yields $n$ (complex) roots $x_0, \dots ,x_{n-1}$, from which we reconstruct a real polynomial $P$.

This generates a dataset of polynomials, with degrees uniformly distributed in {$d_{min},d_{max}$}, a fixed proportion $p_c$ of complex roots, real roots uniformly distributed over $[-10,10]$ and complex roots uniformly over the corresponding square in the complex plane.

Each polynomial $P(x)$ is represented as a sequence of $n+1$ real numbers, and its roots as $2n$ real numbers (the real and imaginary parts of the roots). Real numbers are rounded to three significant digits and encoded as sequences of three tokens (sign, mantissa, exponent), with mantissa in {$0$, $999$}, and exponent in {$E-100$, $E100$} (see encoding P1000 in my [paper on linear algebra](https://arxiv.org/abs/2112.01898)). Each sequence of real numbers is preceded by a token indicating its length. Overall a polynomial with degree $n$ is encoded as a input sequence of $1 +3(n+1) = 3n+4$ tokens, and its root as an output sequence of $6n+1$ tokens.

### Models and training

The main architecture is a transformer with 4 layers, 512 dimensions and 8 attention heads in the encoder and decoder. It is trained on a cross-entropy loss, using the Adam optimiser with learning rate $lr=10^{-4}$, over batches of 64 examples, with linear warmup during the first 1000 optimisation steps, and cosine scheduling (with a very long period of 2,000,000 steps) afterwards. At the end of every peoch (300,000 examples), the models is tested on 10,000 random examples. A prediction is considered correct if it can be decoded as a sequence of $2n$ roots, and the maximal relative prediction error for all roots $\|pred-correct\|/\|correct\|$, ($pred$ the predicted root, $correct$ the correct value) is below a tolerance level (5%).

### Results

Our best models achieve 61% accuracy of datasets of polynomials for degree 3 to 6. Accuracy saturates after 400 epochs (120 million examples). Prediction accuracy decreases with the degree of the polynomial, from 84% for degree 6, to 37% for degree 6. The following table compares the performance of our two best 4-layer models, with models 

**Accuracy as a function of model depth and scheduling** 
|Degree | 4/4 | 4/4 | 6/6 | 2/2 | 1/1 | 4/4 no scheduling |
|---|---|---|---|---|---|---| 
|3 | 84 | 86| 85| 87 | 85 | 82 |
|4 | 73 | 72| 71 | 72 | 68 | 66 |
|5 | 50| 50| 50 | 48 | 43 | 45 |
|6 | 37| 34| 35 |  32 | 25 | 28 |
|Average | 61 | 61| 60 | 60 | 56 | 55 |



### Additional experiments

### Discussion
