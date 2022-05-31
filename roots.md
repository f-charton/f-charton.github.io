---
layout: post
title: Computing the roots of polynomials
---
TLDR: Transformers can be trained to predict the roots of polynomials from their coefficients: 

### The maths
A **polynomial** of degree $n$ with real coefficients is a function of the form : $P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$, with all $a_i$ in $\mathbb{R}$). A degree $n$ polynomial $P$ has $n$ **roots** : values $x_i$ such that $P(x_i) = 0$, that allow to  **factorize** $P$ as $P(x) = a_n (x-x_1)(x-x_2)\dots(x-x_n)$ (several roots may have the same value). When all the coefficients ($a_i$) are real, the roots are either real numbers, or pairs of conjugate complex numbers, ($a+ib$, $a-ib$), $a$, $b$ $\in$ $\mathbb R$. This post focuses on predicting the roots of a polynomial from its coefficients (i.e. the $x_i$ from the $a_i$). 

For $n\leq 4$, formulas allow to compute the roots of a polynomial from its coefficients. For instance, the roots of a degree 2 polynomial $P(x) = ax^2+bx+c$ can be computed as $x_0=\frac{ -b + \sqrt{b^2-4ac}}{2a}$ and $x_1=\frac{-b-\sqrt{b^2-4ac}}{2a}$ if $b^2-4ac>=0$, and $\frac{-b+i\sqrt{4ac-b^2}}{2a}$ and $\frac{-b-i\sqrt{4ac-b^2}}{2a}$ if $b^2-4ac<0$. For $n > 4$, it was proven (by Galois and Abel) that no such formula (involving radicals) can be found. There are, on the other hand, numerical algorithms to compute approximate values of the roots.

### Encoding the problem
Transformers process sequences. The input polynomial is encoded as a sequence of $n+2$ numbers: the (integer) degree and the $n+1$ real coefficients. Degrees are represented as symbolic tokens ('N1' to 'N100'), and real numbers are rounded to three significant digits, written in scientific notation, and encoded as sequences of three tokens: sign ('+' or '-'), mantissa (from '0' to '999'), and exponent (from 'E-100' to 'E100'). See my [paper on linear algebra](https://arxiv.org/abs/2112.01898) for more details (this is the P1000 encoding). For a polynomial of degree $n$, the input sequence has $3n+4$ tokens.

The $n$ output roots are encoded as a sequence of $2n$ real numbers (the real and imaginary parts of the $n$ roots), encoded on three tokens each, and preceded by a symbolic token defining the length of the sequence ($2n$). The roots of a polynomial of degree $n$ are represented as a sequence of $6n+1$ tokens. Overall, the models use a vocabulary of about 1120 tokens.

Here is some minimal Python code for encoding real numbers, and sequences of reals or complex numbers.

    def write_float(self, value):
        precision = self.float_precision # significant digts - 1
        m, e = (f"%.{precision}e" % np.abs(value)).split("e")
        i, f = m.split(".")
        i = i + f
        ipart = int(i)
        expon = int(e) - precision
        if expon < -100:
            ipart = 0
        if ipart == 0:
            value = 0.0
            expon = 0
        res = ["+"] if value >= 0.0 else ["-"]
        res = res + [str(ipart)] 
        return res + ["E" + str(expon)]
        
    def encode(self, vect, cplx=False):
        lst = []
        l = len(vect)
        lst.append("V" + str(l))
        for val in vect:
            if cplx:
                lst.extend(self.write_float(val.real))
                lst.extend(self.write_float(val.imag))
            else:
                lst.extend(self.write_float(val))
        return lst



### Generating data
Models are trained and tested from sets of polynomials, with degrees uniformly distributed in {$d_{min},d_{max}$}. By default, I use $d_{min}=3$ and $d_{max}=6$. Polynomials are generated from their roots: after randomly choosing a degree $n$, I select $n$ real numbers $r_0,\dots, r_{n-1}$ from a uniform distribution over $[-A,A]$ (all results below are with $A=10$). Each pair ($r_{2i}$, $r_{2i+1}$) is then interpreted with probability $p_c$ is the two conjugate complex roots $x_{2k}=r_{2k}+i r_{2k+1}$ and $x_{2k+1}=r_{2k}-i r_{2k+1}$, and with probability $1-p_c$ as tthe wo real roots $x_{2k}=r_{2k}$  and $x_{2k+1}=r_{2k+1}$.

To generate a random polynomial, I sample uniformly in $[-A,A]$ (choosing $A=10$) $n$ real numbers $r_0,\dots, r_{n-1}$. With probability $p_c$, $r_{2k}$ and $r_{2k+1}$ are interpreted as the pair of conjugate complex roots $x_{2k}=r_{2k}+i r_{2k+1}$ and $x_{2k+1}=r_{2k}-i r_{2k+1}$, and with $1-p_c$ as two real roots $x_{2k}=r_{2k}$ and $x_{2k+1}=r_{2k+1}$. This yields $n$ (complex) roots $x_0, \dots ,x_{n-1}$, from which the real polynomial $P$ can be reconstructed.

This procedure generates polynomials, with degrees uniformly distributed in {$d_{min},d_{max}$}, a fixed proportion $p_c$ of complex roots, real roots uniformly distributed over $[-10,10]$ and complex roots uniformly over the corresponding square in the complex plane. All polynomials are normalised (i.e. verify $a_n=1$).

Here is the minimal code for generating polynomials and roots, using the random number generator `self.rng`, with degrees from `self.min_degree` to `self.max_degree`. The parameter `self.sort_roots` decides whether the roots are output in random order (i.e. in the order they were generated), or sorted from largest to smallest (real part first, imaginary part second).

    def generate(self, rng):
        degree = rng.randint(self.min_degree, self.max_degree + 1)
        roots = rng.rand(degree) * self.max_root
        roots = roots.astype(complex)
        for i in range(degree//2):
            cplex = (rng.rand() < self.prob_complex)
            if cplex:
                roots[2 * i] = complex(roots[2 * i], roots[2 * i + 1])
                roots[2 * i + 1] = np.conj(roots[2 * i])
        poly = np.real(np.poly(roots))
        if self.sort_roots:
            roots = np.sort_complex(roots)
        return poly, roots



### Models, training and evaluation

My main architecture is a sequence to sequence (two-tower) transformer with 4 layers, 512 dimensions and 8 attention heads in the encoder and decoder. Models are trained (supervisedly, with teacher forcing) using a cross-entropy loss, over batches of 64 examples, using the Adam optimizer with a learning rate of $lr=5.10^{-5}$, with linear warmup during the first 10,000 optimisation steps, and cosine scheduling (with a very long period of 2,000,000 steps) afterwards. I am using a code base derived from [my paper on dynamic systems](https://github.com/facebookresearch/MathsFromExamples), but default Pytorch implementations for transformers would certainly produce similar results.

At the end of every epoch (300,000 examples), the models is evaluated on a random test set on 10,000 random examples (a different one for each epoch). 
A prediction is considered correct if it can be decoded as a sequence of $n$ roots, and all relative prediction errors $\|pred-correct\|/\|correct\|$, ($pred$ the predicted root, $correct$ the correct value) are below a certain tolerance level (5%). With this **maximal relative error (max-err)** metric, a prediction is correct is all predicted roots fall within 5% of the correct values. 

Two weaker, but meaningful, alternative measures are the **minimal relative error (min-err)**, which decides that a prediction is correct if at least one root is predicted to 5% tolerance, and the **average relative error (avg-err)** the proportion of the $n$ roots predicted within tolerance. These alternative metrics can be turned into practical techniques for finding all roots. By computing $P(x)$ for predicted roots, one can determine which predictions were correct, divide the polynomial by the corresponding ($x-root$) terms, and iterate.

### Main results

Models trained on a dataset of polynomials of degree 3 to 6 (each degree in equal proportion) reach a max-err accuracy of 61.3% after 400 epochs (120 million examples): all roots can be predicted with less than 5% relative error in more than 61% of the test cases.   
Max-err accuracy drops to 41.4, 27.2 and 14.0% for 2, 1 and 0.5% tolerance. 
Min-err accuracy is 97.2%, and avg-err accuracy 79.9%: the model recovers all roots 61% of the time, at least one root almost every time (97%), and will correctly predict 80% of the roots on average. 

Max-err accuracy decreases as the degree of the polynomial goes up: from 86% for degree 3 to 36.5 for degree 6 polynomials. However, min-err accuracy and the number of roots correctly precdicted (avg-err + degree) are stable for all degrees. Whereas **predicting all roots** becomes more difficult as the number of roots to be predicted increases, the difficulty of **predicting just a few roots** (a constant number of them) seems constant for all degrees. 

**Table 1 - Accuracy as a function of degree (roots of polynomials of degree 3-6)** 
|Degree | All roots (max-err) | One root (min-err) | % of roots (avg-err) | # roots predicted |
|---|---|---|---|---|
|3 | 86.1 | 97.6 | 91.8 | 2.8 | 
|4 | 71.0 | 97.2 | 83.5 | 3.3 | 
|5 | 49.1| 97.5 | 73.4 | 3.7 | 
|6 | 36.5| 96.4 | 62.3 |  3.7 | 
|Average | 61.3 | 97.2 | 79.9 | - | 

Models trained from different data sets (degree 3 to 6, vs only degree 6, vs degree 5 to 8 ...) achieve similar performances

**Table 2 - max-err accuracy per degree, for different datasets** 
|Degree | 3 | 4 | 5 | 6 | 7 | 8 | 3-4 | 3-6 | 3-8 | 5-6 | 5-8 |
|-------|---|---|---|---|---|---|-----|-----|-----|-----|-----|
| 3 | 84.1  | - | - | - | - | - | 84.5| 86.1| 85.4| -   | -   | 
| 4 | - | 70.7  | - | - | - | - | 71.8| 71.0| 71.1| -   | -   |
| 5 | - | - | 50.3  | - | - | - | -   | 49.1| 49.6| 51.2| 49.2|
| 6 | - | - | - | 36.0  | - | - | -   | 36.5| 36.9| 35.8| 35.7|
| 7 | - | - | - | - | 18.8. | - | -   | -   | 17.4| -   | 18.8|
| 8 | - | - | - | - | - | 10.1. | -   | -   |  9.6| -   | 9.0 |



**Table 3 -  Number of roots predicted for different datasets**
|Degree | 3 | 4 | 5 | 6 | 7 | 8 | 3-4 | 3-6 | 3-8 | 5-6 | 5-8 |
|-------|---|---|---|---|---|---|-----|-----|-----|-----|-----|
| 3 | 84.1  | - | - | - | - | - | 84.5| 2.8 | 85.4| -   | -   | 
| 4 | - | 70.7  | - | - | - | - | 71.8| 3.3 | 71.1| -   | -   |
| 5 | - | - | 50.3  | - | - | - | -   | 3.7 | 51.2| 49.2|
| 6 | - | - | - | 36.0  | - | - | -   | 3.7 | 36.9| 35.8| 35.7|
| 7 | - | - | - | - | 18.8. | - | -   | -   | 17.4| -   | 18.8|
| 8 | - | - | - | - | - | 10.1. | -   | -   |  9.6| -   | 9.0 |



Accuracy saturates after 400 epochs (120 million examples). Prediction accuracy decreases with the degree of the polynomial, from 84% for degree 6, to 37% for degree 6. The following table compares the performance of our two best 4-layer models, with models 


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
