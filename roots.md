---
layout: post
title: Computing the roots of polynomials
---
TLDR: Transformers can be trained to predict the roots of polynomials from their coefficients: 

It has been observed by many authors that neural networks struggle with basic arithmetic. For instance, transformers perform poorly on a task like the multiplication of two large numbers (represented as sequences of digits in some number base). This has long been considered a "hard" limitation of language models (and, for some authors, a proof that hybrid systems, mixing rule-based and gradient-based techniques, were needed.

In a [paper published last year](https://arxiv.org/abs/2112.01898), I showed that transformers can learn to perform approximate computations from examples only. They can predict the solutions of various problems of linear algebra, from basic (arithmetic) operations on matrices to advanced (non linear) computations, like eigen decomposition and inversion. 

This post features additional results, involving a slightly more advanced numerical problem: finding the roots of polynomials. 
I am using the same architectures and encoding as in my paper on linear algebra.

### The maths
A **polynomial** of degree $n$ with real coefficients is a function of the form : $P(x) = a_n x^n + a_{n-1} x^{n-1} + \dots + a_1 x + a_0$, with all $a_i$ in $\mathbb{R}$). A degree $n$ polynomial $P$ has $n$ **roots** : values $x_i$ such that $P(x_i) = 0$, that allow to  **factorize** $P$ as $P(x) = a_n (x-x_1)(x-x_2)\dots(x-x_n)$. The roots can be multiple, i.e. we can have $x_i=x_j$ for different values of $i$ and $j$. When all the coefficients $a_i$ are real, the roots are either real numbers, or pairs of conjugate complex numbers, ($a+ib$, $a-ib$), with $a$, $b$ $\in$ $\mathbb R$. 

We want to predict the roots of a polynomial from its coefficients (i.e. the $x_i$ from the $a_i$). 

For $n\leq 4$, there exists formulas $x_i=f(a_1,a_2,...a_n)$. For instance, the roots of a degree 2 polynomial $P(x) = ax^2+bx+c$ can be computed as $x_0=\frac{ -b + \sqrt{b^2-4ac}}{2a}$ and $x_1=\frac{-b-\sqrt{b^2-4ac}}{2a}$ if $b^2-4ac>=0$, and $\frac{-b+i\sqrt{4ac-b^2}}{2a}$ and $\frac{-b-i\sqrt{4ac-b^2}}{2a}$ if $b^2-4ac<0$. 
However, it was proven (by Galois and Abel), that no such general formula, involving common operations and radicals, can exist for $n > 4$. But there are numerical algorithms to compute approximate values of the roots.

Most existing algorithms (Newton, Halley, Laguerre) compute one root $x_i$ at a time. The polynomial is then divided by the factor $(x-x_i)$, and the process is repeated. Techniques exist (e.g. Aberth) to compute all roots in one step. Here, I train transformers to predict all roots.

### Encoding the problem
Transformers process sequences, and the first step in using them is to rewrite the problem and solution as sequences of tokens. Here, the input is a $n$ degree polynomial, encoded as a sequence of $n+2$ numbers: the (integer) degree and the $n+1$ real coefficients. The degree is represented as a symbolic tokens (from 'N1' to 'N100'). Real coefficients are rounded to three significant digits, written in scientific notation, and encoded as sequences of three tokens: sign ('+' or '-'), mantissa (from '0' to '999'), and exponent (from 'E-100' to 'E100'). This is the P10 encoding from my [paper on linear algebra](https://arxiv.org/abs/2112.01898). For a polynomial of degree $n$, the input sequence has $3n+4$ tokens.

The $n$ output roots are encoded as a sequence of $2n$ real numbers: the real and imaginary parts of the $n$ roots. They are encoded as before: a symbolic token defining the length of the sequeunce ($2n$), and the $2n$ real numbers, each represented by three tokens (sign, mantissa, exponent). For a polynomial of degree $n$, the output sequence has length $6n+1$.

Overall, the vocabulary has about 1120 tokens. Here is some minimal Python code for encoding real numbers, and sequences of reals or complex numbers.

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
All models are trained and tested on sets of polynomials, with degrees uniformly distributed in {$d_{min},d_{max}$}. By default, I set $d_{min}=3$ and $d_{max}=6$. To generate a polynomial, I first generate its $n$ roots $x_1$ ... $x_n$ and reconstruct the polynomial as $P(x)=(x-x_i)(x-x_2)...(x-x_n)$. This has the double advantage of being very fast, and not needing some external root finding algorithm.

Each polynomial is generated as follows. First a degree $n$ is randomly selected in {$d_{min},d_{max}$}. Then, $n$ real numbers $r_0,\dots, r_{n-1}$ are sampled from a uniform distribution over $[-A,A]$ (I set $A=10$). Finally, each pair ($r_{2i}$, $r_{2i+1}$) is interepreted with probability $p_c$ (I set $p_c=0.5$) as the two conjugate complex roots $x_{2k}=r_{2k}+i r_{2k+1}$ and $x_{2k+1}=r_{2k}-i r_{2k+1}$, and with probability $1-p_c$ as the two real roots $x_{2k}=r_{2k}$  and $x_{2k+1}=r_{2k+1}$. This creates $n$ (complex) roots $x_0 \dots x_{n-1}$, from which the real polynomial $P$ can be reconstructed.

This procedure generates polynomials, with degrees uniformly distributed in {$d_{min},d_{max}$}, a fixed proportion $p_c=0.5$ of complex roots, real roots uniformly distributed over $[-10,10]$ and complex roots uniformly over the corresponding square in the complex plane. All polynomials are normalised (i.e. verify $a_n=1$).

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

All the results in tables 2 and 3 were obtained after training on the same number of examples (about 120 million). This means that a model trained on polynomials of degree 3 to 8 saw about 20 million examples of each degrees, yet achieve similar results to models trained on polynomials of one degree, with 120 million examples. This clearly demonstrates the benefit of mixing degrees in the datasets.

### Larger degrees

Table 4 presents models trained on polynomials of degree 5, 8, 10, 15, 20 and 25. As the degree increases, predicting all roots becomes a very hard task. Max-err accuracy drops to zero after degree 10. Still, the models can predict at least one root in over 92% of the test cases, and prefict an average of 3 to 4 roots for all degrees.

In other words, whereas **predicting all roots** does not scale to large degrees, **predicting some roots** does not seem to be constrained. 

**Table 4 - Accuracy as a function of degree** 
|Degree | All roots (max-err) | One root (min-err) | % of roots (avg-err) | # roots predicted |
|---|---|---|---|---|
|5 | 49.1| 97.5 | 75.4 | 3.8 | 
|8 | 10.1| 95.3 | 51.1 |  4.1 | 
|10 | 0.6| 93.1 | 34.4 |  3.4 | 
|15 | 0| 92.8 | 22.6 |  3.4 | 
|20 | 0| 92.7 | 15.9 |  3.2 | 
|25 | 0| 95.5 | 15.3 |  3.8 | 

### Sorted and unsorted roots

In my basic train sets, the root of the poynomial are sorted in decreasing order. Table 5 compares their accuracy with models trained on datasets where the roots are left in random order. For small degrees (3 and 4), root order has no impact on accuracy. For larger degrees, sorting the roots brings a small gain in accuracy. This result is slighty counter-intuitive: when training on unsorted roots, the model sees one out of $n$ possible permutations of the $n$ roots. This results in a higher cross-entropy loss, and should make the training much harder.

This confirms an observation from [our paper on recurrences](https://arxiv.org/abs/2201.04600): training the model on simplified expressions (i.e. $2x+1$ vs $x+2+x-1$) made no difference in accuracy. 

The discussion on the importance of simplification has been ongoing since my first paper ([on integration](https://arxiv.org/abs/1912.01412)). In a review, [Ernest Davis commented](https://arxiv.org/abs/1912.05752) that working from simplified functions made the problem easier ("No integration without simplification!"), and I considered that a fair point. The results from the paper on recurrences, which suggested that simplification was orthogonal to the problem we were solving (and therefore had no bearing on it), came as a surprise. This result on sorting roots seems to confirm it (or, at least, go in the same direction).


**Table 5 - Sorted and unsorted roots, max-err accuracy** 
|Degree | 3-6 sorted | 3-6 unsorted | 3-8 sorted | 3-8 unsorted | 5-8 sorted | 5-8 unsorted |
|---|---|---|---|---|---|---|
|3 | 86.1| 87.2 | 85.4 | 85.5 | -    | - | 
|4 | 71.0| 70.7 | 71.1 | 69.9 | -    | - |
|5 | 49.1| 47.8 | 49.6 | 49.3 | 49.2 | 48.9 |
|6 | 36.5| 31.7 | 36.9 | 31.9 | 35.7 | 32.3 |
|7 | -   | -    | 17.4 | 16.4 | 18.8 | 16.3 |
|8 | -   | -    | 9.6. |  5.6 |  9.0 | 7.0  |

### Data usage, and batch size

So far, all models were trained using batches of 64 examples, and needed 400 epochs, or 120 million samples, to achieve high accuracy. This is a very large training set. Better data efficiency is possible by reducing the batch size. Table 6 indicates the number of epochs and examples needed to train a model to 58% (max-err) accuracy (over polynomials of degree 3 to 6), for different batch sizes. With batches of 4 examples the mode needs 12.6 million examples, almost 10 times less that when using batches of 128. Note that smaller batches result in slower learning, since the optimizer, a slow operation, is called more often.

Final accuracy tends to decrrease with larger batches: models with 256, 512 and 1024 batches never reached 58% accuracy. In these experiments, the besst accuracies were achieved with batch size between 32 and 64. 

**Table 6 - batch size, number or epochs, and millions of examples, to reach 58% accuracy**
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

Model size has little impact on performance. Apart from 1-layer transformers, which prove too shallow to achieve good performance, 2, 4, 6 and 8 layers models result in the same accuracy (in table 7, the larger models are a little lower, because they take longer to train). Embedding dimension and the number of attention heads also see to have little impact on accuracy. 

**Table 7 - max-err accuracy as a function of model depth, dimension and attention heads** 
|   | 1/1 | 2/2 | 4/4 | 6/6 | 8/8 |
|---|---|---|---|---|---|
|240 dimensions 8 heads  | 48.4 | 58.4 | 60.6 | 60.1 | 59.0 | 
|480 dimensions 6 heads  | -    | 59.3 | 60.4 | 59.0 | - | 
|480 dimensions 8 heads  | 52.2 | 60.5 | 60.6 | 59.7 | 58.5 |
|480 dimensions 10 heads | -    | 60.2 | 60.1 | 59.3 | - |
|480 dimensions 12 heads | -    | 60.5 | 60.0 | 59.0 | - |
|480 dimensions 16 heads | -    | 60.7 | 60.5 | 59.4 | - |
|720 dimensions 6 heads  | -    | 60.1 | 60.1 | 58.6 | - | 
|720 dimensions 8 heads  | 54.9 | 60.5 | 60.1 | 59.3 | 57.8 | 
|720 dimensions 10 heads | -    | 60.7 | 60.3 | 58.4 | - | 
|720 dimensions 12 heads | -    | 60.2 | 59.8 | 58.8 | - | 
|720 dimensions 16 heads | -    | 60.6 | 59.5 | 58.9 | - | 

### Asymmetric architectures

Asymmetric models, with a deep encoder and shallow decoder, have proven their worth for linear algebra. 

|   | 240 dimensions | | | 480 dimensions | | |
|---|---|---|---|---|---|---|
|Encoder   | 4 heads  | 6 heads | 8 heads | 4 heads  | 6 heads | 8 heads | 
|480 dimensions 8 heads  | 52.2 | 60.5 | 60.6 | 59.7 | 58.5 |
|480 dimensions 10 heads | -    | 60.2 | 60.1 | 59.3 | - |
|480 dimensions 12 heads | -    | 60.5 | 60.0 | 59.0 | - |
|720 dimensions 8 heads  | 54.9 | 60.5 | 60.1 | 59.3 | 57.8 | 
|720 dimensions 10 heads | -    | 60.7 | 60.3 | 58.4 | - | 
|720 dimensions 12 heads | -    | 60.2 | 59.8 | 58.8 | - | 



### Shared layers and universal transformers

The [universal transformer](https://arxiv.org/abs/1807.03819) is a shared layer model: one layer (in the encoder and/or the decoder) is iterated through several times, by feeding its output back into its input. This can allow for more complex calculations than what can be done with one transformer layer only, while keeping the number of trainable parameters low. The looping mechanism also constrains the inner layer of the transformer to stick to the same representation for their input and output. In the original paper, the number of iterations was either fixed, or controlled by a technique called [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (ACT). While experimenting with universal transformers, I have noticed that ACT was very hard to train (i.e. very unstable with respect to model initialization), and that fixingthe number of loops to a large value usually did not work. 

I am using here a technique proposed by [Csordas et al.](https://arxiv.org/abs/2110.07732), which adds a copy-gate (in pure LSTM fashion) to the output of the self-attention mechanism in the shared layer. Depending on the token and output of the attention mechanism, the token will either be processed by the layers, or just copied (and possibly fed back into the shared layer). 

I experiment on polynomials of degrees 3 to 6, with transformers with 1 or 2 layers, and one shared layer in the encoder and/or the decoder. Shared layers are gated, and iterated though 4, 8 or 12 times. 

Over several problems, I have noticed that whereas large numbers of iterations 



Instead of deeper transformers, I experimented with shared 

### Discussion

These results are of little interest for mathematicians, or people who actually need to compute the roots of polynomials. We already have efficient algorithms for this. 

