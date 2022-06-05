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

The $n$ output roots are encoded as a sequence of $2n$ real numbers: the real and imaginary parts of the $n$ roots. They are encoded as before: a symbolic token defining the length of the sequence ($2n$), and the $2n$ real numbers, each represented by three tokens (sign, mantissa, exponent). For a polynomial of degree $n$, the output sequence has length $6n+1$.

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

My main architecture is a sequence to sequence (two-tower) [transformer](https://arxiv.org/abs/1706.03762) with 4 layers, 512 dimensions and 8 attention heads in the encoder and decoder. Models are trained (supervisedly, with teacher forcing) using a cross-entropy loss, over batches of 64 examples, using the Adam optimizer with a learning rate of $lr=5.10^{-5}$, with linear warmup during the first 10,000 optimisation steps, and cosine scheduling (with a very long period of 2,000,000 steps) afterwards. I am using a code base derived from [my paper on dynamic systems](https://github.com/facebookresearch/MathsFromExamples), but default Pytorch implementations for transformers would certainly produce similar results.

At the end of every epoch (300,000 examples), the models is evaluated on a random test set on 10,000 random examples (a different one for each epoch). 
A prediction is considered correct if it can be decoded into a sequence of $n$ roots, and all relative prediction errors $\|pred-correct\|/\|correct\|$, ($pred$ the predicted root, $correct$ the correct value) are below a certain tolerance level (5%). With this **maximal relative error (max-err)** metric, a prediction is correct when all predicted roots fall within 5% of the correct values. 

I also introduce three alternative measures. With the **minimal relative error (min-err)** metrics, a prediction is correct if the relative prediction error is below the 5% tolerance level for at least one root. Whereas max-err uses the maximum of relative prediction errors and min-err their minimum, **average relative error (avg-err)** measures the proportion of the $n$ roots predicted within tolerance: min-err therefore corresponds to avg-err=$\frac{1}{n}$ and max_err to avg-err=1. Finally, the **number of roots predicted (n-roots)** is the product of the average error by the degree, this allows for comparison between polynomials of different degrees.

### Main results
For the main experiments, transformers are trained on a set of polynomials of degrees 3 to 6 (with an equal proportion of each degree). After 400 epochs (120 million examples), the best model achieve a max-err accuracy of 61.3%: all roots are predicted within 5% relative error in more than 61% of the (random) test cases. At 2, 1 and 0.5%, the model achieves 41.4, 27.2 and 14.0% accuracy. The learning curves for strict tolerance are not saturated after 400 epochs, so more training would improve these results. Since these approximate solutions can be refined using efficient numerical techniques, I do not think such additional training is warranted.

Min-err accuracy is 97.2%: the model almost always recovers at least one root. Finally, avg-err accuracy is 79.9%: on the whole test set, 4 roots out of 5 are correctly predicted.

As the degree of the polynomial increas, max-err accuracy drops, from 86.1% for degree 3 to 36.5%  for degree 6. This comes to no surprise. As the number of roots to predict within tolerance increases, the task becomes more difficult. On the other hand, min-err accuracy and the number of roots correctly predicted (n-roots) are independent of the degree (n-root slightly increases with the degree).

In other words, while  **all** roots becomes more difficult as the degree (and the number of roots to be predicted) increases, the model capability to predict **at least one** root (and in fact three to four), keeps constant as degree grows. 

**Table 1 - Accuracy as a function of degree (roots of polynomials of degree 3-6)** 
|Degree | All roots (max-err) | One root (min-err) | % of roots (avg-err) | # roots predicted |
|---|---|---|---|---|
|3 | 86.1 | 97.6 | 91.2 | 2.7 | 
|4 | 71.0 | 97.2 | 83.5 | 3.3 | 
|5 | 49.1| 97.5 | 75.4 | 3.8 | 
|6 | 36.5| 96.4 | 68.4 |  4.1 | 
|Average | 61.3 | 97.2 | 79.9 | - | 

This observation remains valid for larger degrees (table 1b). Predicting all roots becomes impossible for degrees higher than 10, and max-err accuracy drops to zero. But all models can predict one root in 93% of the test cases, and on average, 3 to 4 roots are correctly predicted for all degrees.

**Table 1b - Larger degrees** 
|Degree | All roots (max-err) | One root (min-err) | % of roots (avg-err) | # roots predicted  (n-roots)|
|---|---|---|---|---|
|5 | 49.1| 97.5 | 75.4 | 3.8 | 
|8 | 10.1| 95.3 | 51.1 |  4.1 | 
|10 | 0.6| 93.1 | 34.9 |  3.5 | 
|15 | 0| 92.8 | 22.6 |  3.4 | 
|20 | 0| 92.7 | 15.9 |  3.2 | 
|25 | 0| 95.5 | 15.6 |  3.9 | 

(note: 400 epochs for degree 5 and 8, 200 for 10, 120 for 15 and 60 for 20 and 25)


### Different training sets

When studying [transformers in linear algebra](https://arxiv.org/abs/2112.01898), I observed that mixing problems of different sizes in the training set could improve accuracy. Specifically, models trained on 10x10 matrices only never seemed to learn to predict eigenvalues. But models trained on a mixtures of 5x5 to 15x15 matrices would learn to predict eigenvalues for all dimensions.

These observations do not seem to transfer to polynomial root finding. Table 2 and 3 compare models trained on different datasets: polynomials of same degree (six datasets with degree 3, 4, 5, 6, 7 and 8), and (uniform) mixtures of different degrees (3-4, 3-6, 3-8, 5-6 5-8). All models were trained for about 400 epochs. In both tables, for a given degree in the test polynomial (i.e. line in the table), accuracy are constant over all training sets:  ee.g. all degree 3 polynomials are predicted with 85% max_err accuracy, no matter they were trained on degree 3 polynomials only, or on a mixture of degree 3 to 8.
 
**Table 2 - max-err accuracy per degree, for different datasets** 
|Degree | 3 | 4 | 5 | 6 | 7 | 8 | 3-4 | 3-6 | 3-8 | 5-6 | 5-8 |
|-------|---|---|---|---|---|---|-----|-----|-----|-----|-----|
| 3 | 84.1  | - | - | - | - | - | 84.5| 86.1| 85.4| -   | -   | 
| 4 | - | 70.7  | - | - | - | - | 71.8| 71.0| 71.1| -   | -   |
| 5 | - | - | 50.3  | - | - | - | -   | 49.1| 49.6| 51.2| 49.2|
| 6 | - | - | - | 36.0  | - | - | -   | 36.5| 36.9| 35.8| 35.7|
| 7 | - | - | - | - | 18.8  | - | -   | -   | 17.4| -   | 18.8|
| 8 | - | - | - | - | - | 10.1  | -   | -   |  9.6| -   | 9.0 |

**Table 3 -  Number of roots predicted for different datasets (n-roots)**
|Degree | 3 | 4 | 5 | 6 | 7 | 8 | 3-4 | 3-6 | 3-8 | 5-6 | 5-8 |
|-------|---|---|---|---|---|---|-----|-----|-----|-----|-----|
| 3 | 2.7   | - | - | - | - | - | 2.7 | 2.7 | 2.7 | -   | -   | 
| 4 | - | 3.5   | - | - | - | - | 3.4 | 3.3 | 3.3 | -   | -   |
| 5 | - | - | 3.8   | - | - | - | -   | 3.8 | 3.7 | 3.8 | 3.7 |
| 6 | - | - | - | 4.1   | - | - | -   | 4.1 | 4.1 | 4.1 | 4.1 |
| 7 | - | - | - | - | 4.2   | - | -   | -   | 4.1 | -   | 4.2 |
| 8 | - | - | - | - | - | 4.1   | -   | -   | 4.0 | -   | 4.1 |

This makes a strong case for mixing degrees in the training sets. 
All the results in tables 2 and 3 were obtained from datasets of similar size (about 120 million examples), but mixture sets present the model with **much less** examples of each degree. For instance, a polynomial of degree 6 is predicted with 36% max-err accuracy after being presented with 120 million degree 6 polynomials (in the 6-only training set), and 36.9% after seeing only 20 million degree 6 polynomials in the 3-8 training set. This strongly suggests that the models learns from polynomials of different degrees, i.e. that training on degrees 3 to 6 does not amount to learning 4 different problems.

### Sorted and unsorted roots

In the basic train sets, the root in the output are sorted in decreasing order: to each input polynomial corresponds the unique output that is presented at training. If the roots were not sorted, there would be $n!$ possible outputs for a degree $n$ polynomial, and the model would be trained on one, selected randomly. Intuitively, this should lead to a much harder task.

Table 4 compares max-err accuracy (our "hardest" metric) on three different datasets, for models trained from sorted and unsorted roots. The evaluation procedure does not change: predicted roots are sorted before relative errors are computed. On the unsorted samples, the training loss is much higher, but sorting makes little difference in accuracy. In fact, for small degrees (3 and 4), root order has no impact on accuracy. For larger degrees, accuracy is improved by sorting the roots, but models trained on unsorted roots still perform pretty well (for instance, their min-err and n-roots accuracy is unchanged).

This confirms an observation from [our paper on recurrences](https://arxiv.org/abs/2201.04600): training the model on simplified or unsimplified expressions (e.g. $2x+1$ vs $x+2+x-1$) made no difference in accuracy. 

The debate on the importance of simplification has been ongoing since my first paper ([on integration](https://arxiv.org/abs/1912.01412)). In a review, [Ernest Davis commented](https://arxiv.org/abs/1912.05752) that working from simplified functions made the problem easier ("No integration without simplification!"). Back then, I considered this a fair point. The results from the paper on recurrences, which suggested that simplification was orthogonal to the problem we were solving (and therefore had no bearing on it), came as a surprise. This result on sorting roots seems to confirm it (or, at least, go in the same direction).


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

So far, all my models achieve high accuracy after about 400 epochs, or 120 million samples. This is a very large training set. As noted, training on mixture of polynomials of different degrees can improve data efficiency: a model trained on 3 to 8-degree polynomials predicts the roots of a 6-degree polynomial just as well as one trained on 6-degree polynomials only.

Better data efficiency is also possible by reducing the batch size. All models are trained using batches of 64 examples. Table 5 indicates the number of epochs and examples needed to train a model to 58% (max-err) accuracy over polynomials of degree 3 to 6, for different batch sizes. Small batches results in much slower learning, since the optimizer, a costly operation, is called more often. Yet, only 12.6 million examples are needed with batches of 4, and 120 millions with batches 128. I tried larger batches (256, 512 and 1024) but they never reached 58% accuracy. 

When training data is generated, data efficiency is not an issue, but these results suggest that if we were learning from limited "real world" data, for instance learning a black-box computation from a system, or having to request all our data from an external (and slow) API, reducing batch size would be in order. 

Note that the results in table 5 are for 58% accuracy, if we consider the best possible accuracy, batch size of 32 and 64 seem optimal. 

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

### Model architecture: impact of dimension

Our default model has 512 dimensions, 4 layers and 8 attention heads. Table 6 presents max-err accuracy after 300-500 epochs, for different model sizes: 1 to 8 layers, 240 to 720 dimensions, and 6 to 16 heads (note: the transformer implementation I use demands that dimension is a multiple of the number of heads, I choose multiples of 240 for the dimensions so as to test 6, 8, 10, 12 and 16 heads).

Apart for 1-layer models, which prove too shallow, model depth, dimension and number of heads seem to have very little impact on performance. Actually, the largest models seem to perform a little worse, but this is probably an effect of their slower training speed.

**Table 6 - max-err accuracy as a function of model depth, dimension and attention heads** 
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

In the [graph](https://arxiv.org/abs/2112.03588) and linear algebra paper, I have found that asymmetric models, with a deep encoder and shallow decoder, proved efficient. For matrix inversion, the hardest problem from linear algebra, the best models had 6 layers in the encoder layer, and only one in the decoder.

Table 7 presents the performance of asymmetric architectures, with 4 to 6 layers and 480 or 720 dimensions in the encoder, and one layer and 240 or 480 dimensions in the decoder, after 400 to 500 epochs. There is a marginal improvement in accuracy compared to symmetric models. Note that the best decoders are very small: 1 layer, 240 dimensions, and 4 attention heads.

**Table 7 - max-err and min-err accuracy for some asymmetric architectures**
|Encoder layers |Encoder dimensions |Encoder heads |Decoder layers | Decoder dimensions |Decoder heads | max-err | min-err | 
|---|---|---|---|---|---|---|---|
|6  | 720| 12 | 1 | 240 | 4 | 62.0 | 97.3 |
|6  | 720| 12 | 1 | 480 | 4 | 61.6 | 97.2 |
|4  | 720| 12 | 1 | 240 | 4 | 61.8 | 97.3 |
|4  | 720| 8  | 1 | 480 | 8 | 61.3 | 97.3 |
|6  | 480| 12 | 1 | 240 | 6 | 61.7 | 97.4 |
|6  | 480| 8  | 1 | 480 | 4 | 61.6 | 97.3 |
|4  | 480| 10 | 1 | 240 | 4 | 61.4 | 97.2 |
|4  | 480| 12 | 1 | 480 | 8 | 61.3 | 97.2 |

### Shared layers and universal transformers

The [universal transformer](https://arxiv.org/abs/1807.03819) is a shared layer model: one layer (in the encoder and/or the decoder) is iterated through several times, by feeding its output back into its input. This can allow for more complex calculations than what can be done with one transformer layer only, while keeping the number of trainable parameters low. The looping mechanism also constrains the inner layer of the transformer to keep the representation of their input and output compatible. In the original paper, the number of iterations was either fixed, or controlled by a technique called [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (ACT). Experimenting with universal transformers, I found ACT to be hard to train (i.e. very unstable with respect to model initialization), and that having a large, and fixed, number of loops usually did not work. 

To control the loops, I am using a technique proposed by [Csordas et al.](https://arxiv.org/abs/2110.07732), which adds a copy-gate (in LSTM fashion) to the output of the self-attention mechanism in the shared layer. Depending on the token and output of the attention mechanism, the token will either be processed by the shared layer, or just copied (i.e. that loop is skipped for this token). 

I experiment on polynomials of degrees 3 to 6, with transformers with 1 or 2 layers, and one shared layer in the encoder and/or the decoder. Shared layers are gated, and iterated though 4, 8 or 12 times. Encoders and decoder have 512 dimensions and 8 attention heads.

Only transformers with a shared encoder and no shared decoder seem to learn. With models with a shared decoder, the cross-entropy loss on predicted sequences is reduced at training, but max-err accuracy never reaches 1%, and min-err accuracy stays around 10%. All models with shared encoder achieve a max-err accuracy between 60.2 and 61.6% after 400 epochs, on par with our best symmetric models. Table 8 presents detailed results. Hyper parameters have little impact on performance.

**Table 8 - Universal transformers (shared layer in encoder only)**
|Encoder layers |Shared layer |Max loops | Decoder layers| max-err | min-err | avg-err | 
|---|---|---|---|---|---|---|
|1  | 1| 4 | 1| 60.8 | 97.2 | 79.7 | 
|1  | 1| 8 | 1| 60.9 | 97.2 | 79.8 | 
|1  | 1| 12 | 1| 61.0 | 97.4 | 79.9 | 
|2  | 1| 4 | 1| 60.8 | 97.0 | 79.5 | 
|2  | 1| 4 | 2| 61.3 | 97.1 | 79.9 | 
|2  | 1| 8 | 1 |60.6 | 97.2 | 79.6 | 
|2  | 1| 8 | 2 |61.0 | 97.1 | 79.7 | 
|2  | 1| 12 | 1 |61.1 | 97.1 | 79.8 | 
|2  | 1| 12 | 2 |61.1 | 97.2 | 79.7 | 
|2  | 2| 4 | 1| 61.1 | 97.2 | 80.0 | 
|2  | 2| 4 | 2| 61.3 | 97.1 | 79.7 | 
|2  | 2| 8 | 1 |61.4 | 97.3 | 79.6 | 
|2  | 2| 8 | 2 |61.6 | 97.2 | 79.9 | 
|2  | 2| 12 | 1 |60.7 | 97.2 | 79.7 | 
|2  | 2| 12 | 2 |60.6 | 97.0 | 79.3 | 


### Conclusions

These results are of little interest for mathematicians, or people who actually need to compute the roots of polynomials. We already have efficient algorithms for this. 

