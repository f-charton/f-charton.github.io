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
Models are trained from a dataset of polynomials and their roots, with degrees uniformly distributed in {$d_{min},d_{max}$}. To generate a random polynomial, I sample $n$ real numbers $r_0,\dots, r_{n-1}$. For each even index $2k$ in {$0,n-1$}, $r_{2k}$ and $r_{2k+1}$ are interpreted with probability $p_c$ as the complex conjugates $x_{2k}=r_{2k}+i r_{2k+1}$ and $x_{2k+1}=r_{2k}-i r_{2k+1}$, and with probaility $1-p_c$ as two real roots $x_{2k}=r_{2k}$ and $x_{2k+1}=r_{2k+1}$. This yields $n$ (complex) roots $x_0, \dots ,x_{n-1}$, from which we reconstruct a real polynomial $P$ (all coefficients are real because the conjugate of all roots are roots too).

This procedure generates a dataset of polynomials, with degrees uniformly distributed in {$d_{min},d_{max}$}, a fixed proportion $p_c$ of complex roots, and roots uniformly distributed over a segment of the real line (and a square of the complex plane).

$P(x)$ is then represented as a sequence of $n+1$ real numbers, and its roots as $2n$ real numbers (the real and imaginary parts of the roots). Real numbers are rounded to three significant digits, and encoded as sequences of three tokens (sign, mantissa, exponent), with mantissa in {$0,999$}, and exponent in {$E-100,E100$} (see encoding P1000 in my [paper on linear algebra](https://arxiv.org/abs/2112.01898)). Each sequence of real numbers is preceded by a token indicating its length. Overall for a polynomial with degree $n$, input length is $1 +3(n+1) = 3n+4$, and output length is $6n+1$.

### Models and training

### Results
