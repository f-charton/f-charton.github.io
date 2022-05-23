---
layout: post
title: Computing the roots of polynomials
---
TLDR: Transformers can be trained to predict the roots of polynomials from their coefficients: 

### The problem
In general, a **polynomial** of degree $n$ with real coefficients,
$P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$ ($a_i \in \mathbb{R}$), has $n$ **roots** ($x_i$ such that $P(x_i) = 0$). The roots can be real or complex. When complex, they come in conjugate pairs (i.e. pairs of the form $(a+ib, a-ib)$, with $a$ and $b$ real numbers). For small values of $n$, formulas allow to compute the roots from the coefficients. For instance, for degree 2, the roots of $P(x) = ax^2+bx+c$ can be computed as $x_0=\frac{b+\sqrt{b^2-4ac}}{2a}$ and $x_1=\frac{b-\sqrt{b^2-4ac}}{2a}$ if $b^2-4ac>=0$, and $\frac{b+i\sqrt{4ac-b^2}}{2a}$ and $\frac{b-i\sqrt{4ac-b^2}}{2a}$ if $b^2-4ac<0$. When $n>4$, Galois and Abel have proven that no such formula exists, but numerical algorithms can compute approximate values of the roots.

In this post, I use transformers to predict the $x_i$ from the $a_i$.

### Data generation
Numpy has the algorithms exist to solve polynomials, 

A degree $n$ polynomial can be represented as a sequence of $n+1$ coefficients ($a_0,\dots a_n$)

### Models and training

### Results
