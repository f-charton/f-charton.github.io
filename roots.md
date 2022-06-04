---
layout: post
title: Computing the roots of polynomials
---
TLDR: Transformers can be trained to predict the roots of polynomials from their coefficients: 

It has been observed that neural networks struggle with basic arithmetic and exact computations. For instance, transformers perform poorly on a task like multiplication of two large numbers (represented as sequences of digits in some number base). 
In a [paper](https://arxiv.org/abs/2112.01898) published last year, I showed that transformers could be trained to predict, from examples only, the approximate solutions of various problems of linear algebra, from basic operations on matrices to eigendecomposition and inversion. 

Here are results on a slightly more advanced numerical problem: finding the roots of polynomials.
I am using the same architectures and encoding as in my paper on linear algebra. 

### The maths
A **polynomial** of degree $n$ with real coefficients is a function of the form : $P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$, with all $a_i$ in $\mathbb{R}$). A degree $n$ polynomial $P$ has $n$ **roots** : values $x_i$ such that $P(x_i) = 0$, that allow to  **factorize** $P$ as $P(x) = a_n (x-x_1)(x-x_2)\dots(x-x_n)$ (roots can be multiple, i.e. have the same value). When all the coefficients ($a_i$) are real, the roots are either real numbers, or pairs of conjugate complex numbers, ($a+ib$, $a-ib$), $a$, $b$ $\in$ $\mathbb R$. In these experiments, we are predicting the roots of a polynomial from its coefficients (i.e. the $x_i$ from the $a_i$). 

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
In the main series of experiments, I am training transformers on a set of polynomials of degrees 3 to 6 (each degree sampled in equal proportion). After 400 epochs (120 million examples), the models reach a max-err accuracy of 61.3%: all roots are predicted to less than 5% relative accuracy for more than 61% of the (random) test cases. Higher precision can be achieved: I have 41.4, 27.2 and 14.0% accuracy at 2, 1 and 0.5% tolerance, and better performance could be reached with more training (the learning curves still increase after 400 epochs). Note also that there exist very efficient (numerical) techniques to improve approximate solutions.  
Min-err accuracy is 97.2%: the model recovers at least one roots almost all the time. Avg-err accuracy is 79.9%: on average, 80% of the roots are correctly predicted.

Max-err accuracy decreases as the degree of the polynomial grows, from 86.1% for degree 3 to 36.5%  for degree 6. The task becomes more difficult since there are more roots to predict. On the other hand, min-err accuracy is the same for all degrees, and the number of roots correctly precdicted (avg-err * degree) increases slowly with the degree. 
In other words, **predicting all roots** is more difficult as the number of roots to be predicted increases, but the difficulty of **predicting one root** and the **number of roots predicted** is constant (or slightly decreasing)for all degrees. 

**Table 1 - Accuracy as a function of degree (roots of polynomials of degree 3-6)** 
|Degree | All roots (max-err) | One root (min-err) | % of roots (avg-err) | # roots predicted |
|---|---|---|---|---|
|3 | 86.1 | 97.6 | 91.2 | 2.7 | 
|4 | 71.0 | 97.2 | 83.5 | 3.3 | 
|5 | 49.1| 97.5 | 75.4 | 3.8 | 
|6 | 36.5| 96.4 | 68.4 |  4.1 | 
|Average | 61.3 | 97.2 | 79.9 | - | 

### Varying the training set

In my [paper on linear algebra](https://arxiv.org/abs/2112.01898), I observed that training models on sets mixing problems of different sizes could improve accuracy. A model trained on 10x10 matrices only could not learn their eigenvalues, but a model trained on a mixture of 5x5 to 15x15 matrices would learn eigenvalues for all dimensions (with high accuracy).

In this second series of experiments, I compare models trained on sets of polynomials of same degree (six datasets with degree 3, 4, 5, 6, 7 and 8), and mixtures of polynomials of different degrees (3-4, 3-6, 3-8, 5-6 5-8, all uniformly sampled with respect to the degree). All modesl are trained for about 400 epochs. Table 2 summarize the max-err accuracy (proportion of polynomials with all roots predicted correctly), for all degrees in the test set. For each degree (line in Table 2) accuracies are constant, and independent of the training set, e.g. all degree 3 polynomials are predicted with 85% accuracy.
 
**Table 2 - max-err accuracy per degree, for different datasets** 
|Degree | 3 | 4 | 5 | 6 | 7 | 8 | 3-4 | 3-6 | 3-8 | 5-6 | 5-8 |
|-------|---|---|---|---|---|---|-----|-----|-----|-----|-----|
| 3 | 84.1  | - | - | - | - | - | 84.5| 86.1| 85.4| -   | -   | 
| 4 | - | 70.7  | - | - | - | - | 71.8| 71.0| 71.1| -   | -   |
| 5 | - | - | 50.3  | - | - | - | -   | 49.1| 49.6| 51.2| 49.2|
| 6 | - | - | - | 36.0  | - | - | -   | 36.5| 36.9| 35.8| 35.7|
| 7 | - | - | - | - | 18.8  | - | -   | -   | 17.4| -   | 18.8|
| 8 | - | - | - | - | - | 10.1  | -   | -   |  9.6| -   | 9.0 |

The same observation holds for the number of roots predicted. As degree increases, the number of roots seem to saturate around 4.

**Table 3 -  Number of roots predicted for different datasets**
|Degree | 3 | 4 | 5 | 6 | 7 | 8 | 3-4 | 3-6 | 3-8 | 5-6 | 5-8 |
|-------|---|---|---|---|---|---|-----|-----|-----|-----|-----|
| 3 | 2.7   | - | - | - | - | - | 2.7 | 2.7 | 2.7 | -   | -   | 
| 4 | - | 3.5   | - | - | - | - | 3.4 | 3.3 | 3.3 | -   | -   |
| 5 | - | - | 3.8   | - | - | - | -   | 3.8 | 3.7 | 3.8 | 3.7 |
| 6 | - | - | - | 4.1   | - | - | -   | 4.1 | 4.1 | 4.1 | 4.1 |
| 7 | - | - | - | - | 4.2   | - | -   | -   | 4.1 | -   | 4.2 |
| 8 | - | - | - | - | - | 4.1   | -   | -   | 4.0 | -   | 4.1 |

All the results in tables 2 and 3 were obtained after training on the same number of examples (about 120 million). This means that a model trained on polynomials of degree 3 to 8 saw about 20 million examples of each degrees, yet achieve similar results to models trained on one degree, over 120 million examples. 

### Larger degrees

Results scale to larger degrees: table 4 presents six models, trained on polynomials of degree 5, 8, 10, 15, 20 and 25. 

**Table 4 - Accuracy as a function of degree** 
|Degree | All roots (max-err) | One root (min-err) | % of roots (avg-err) | # roots predicted |
|---|---|---|---|---|
|5 | 49.1| 97.5 | 75.4 | 3.8 | 
|8 | 36.5| 96.4 | 68.4 |  4.1 | 
|10 | 36.5| 96.4 | 68.4 |  4.1 | 
|15 | 36.5| 96.4 | 68.4 |  4.1 | 
|20 | 36.5| 96.4 | 68.4 |  4.1 | 
|25 | 36.5| 96.4 | 68.4 |  4.1 | 

### Sorted and unsorted roots

In my basic train sets, the root of the poynomial are sorted in decreasing order. Tabel 4 compares their accuracy with models trained on datasets where the roots are left in random order. For small degrees (3 and 4), root order has no impact on accuracy. For larger degrees, sorting the roots brings a small gain in accuracy. This result confirms an observation we made in [our paper on recurrences](https://arxiv.org/abs/2201.04600): training from simplified expresssions did not improve accuracy. Here, not sorting the roots means that the "correct solution" (i.e. the output during supervised training) is only correct up to a permutation of the $n$ roots. Intuitively, this should make the training **much harder*** (and in fact, the cross entropy loss is larger), that it is not the case is an intriguing finding. 

**Table 4 - Sorted and unsorted roots, max-err accuracy** 
|Degree | 3-6 sorted | 3-6 unsorted | 3-8 sorted | 3-8 unsorted | 5-8 sorted | 5-8 unsorted |
|---|---|---|---|---|---|---|
|3 | 86.1| 87.2 | 85.4 | 85.5 | -    | - | 
|4 | 71.0| 70.7 | 71.1 | 69.9 | -    | - |
|5 | 49.1| 47.8 | 49.6 | 49.3 | 49.2 | 48.9 |
|6 | 36.5| 31.7 | 36.9 | 31.9 | 35.7 | 32.3 |
|7 | -   | -    | 17.4 | 16.4 | 18.8 | 16.3 |
|8 | -   | -    | 9.6. |  5.6 |  9.0 | 7.0  |

### Data usage, and batch size

So far, all models were trained using batches of 64 examples, and needed 400 epochs, or 120 million samples, to achieve high accuracy. This is a very large training set. Better data efficiency is possible by reducing the batch size. Table 5 indicates the number of epochs and examples needed to train a model to 58% (max-err) accuracy (over polynomials of degree 3 to 6), for different batch sizes. With batches of 4 examples the mode needs 12.6 million examples, almost 10 times less that when using batches of 128. Note that smaller batches result in slower learning, since the optimizer, a slow operation, is called more often.

Final accuracy tends to decrrease with larger batches: models with 256, 512 and 1024 batches never reached 58% accuracy. In these experiments, the besst accuracies were achieved with batch size between 32 and 64. 

**Table 5 - batch size, number or epochs, and millions of examples, to reach 58% accuracy**
|Batch size|Epochs|Millions of examples|
|---|---|---|
|4|42|12.6|
|8|63|18.9|
|12|68|20.4|
|16|91|27.3|
|24|120|36|
|32|157|47.1|
|48|208|62.4|
|64|271|81.3|
|96|333|99.9|
|128|399|119.7|


### Impact of model dimension

**Accuracy as a function of model depth and scheduling** 
|Degree | 4/4 | 4/4 | 6/6 | 2/2 | 1/1 | 4/4 no scheduling |
|---|---|---|---|---|---|---| 
|3 | 84 | 86| 85| 87 | 85 | 82 |
|4 | 73 | 72| 71 | 72 | 68 | 66 |
|5 | 50| 50| 50 | 48 | 43 | 45 |
|6 | 37| 34| 35 |  32 | 25 | 28 |
|Average | 61 | 61| 60 | 60 | 56 | 55 |

### Asymmetric architectures


### Shared layers and universal transformers

The [universal transformer](https://arxiv.org/abs/1807.03819) is a shared layer model: one layer (in the encoder and/or the decoder) is iterated through several times, by feeding its output back into its input. This can allow for more complex calculations than what can be done with one transformer layer only, while keeping the number of trainable parameters low. The looping mechanism also constrains the inner layer of the transformer to stick to the same representation for their input and output. In the original paper, the number of iterations was either fixed, or controlled by a technique called [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (ACT). While experimenting with universal transformers, I have noticed that ACT was very hard to train (i.e. very unstable with respect to model initialization), and that fixingthe number of loops to a large value usually did not work. 

I am using here a technique proposed by [Csordas et al.](https://arxiv.org/abs/2110.07732), which adds a copy-gate (in pure LSTM fashion) to the output of the self-attention mechanism in the shared layer. Depending on the token and output of the attention mechanism, the token will either be processed by the layers, or just copied (and possibly fed back into the shared layer). 

I experiment on polynomials of degrees 3 to 6, with transformers with 1 or 2 layers, and one shared layer in the encoder and/or the decoder. Shared layers are gated, and iterated though 4, 8 or 12 times. 

Over several problems, I have noticed that whereas large numbers of iterations 



Instead of deeper transformers, I experimented with shared 

### Discussion

These results are of little interest for mathematicians, or people who actually need to compute the roots of polynomials. We already have efficient algorithms for this. 

