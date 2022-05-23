---
layout: page
title: Models
permalink: /models/
---

In all problems, I generate pairs of problems and solutions, encode them as sequences, and use them to train (supervised) sequence to sequence transformers. Models use the "two tower" architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762): an encoder and a decoder linked by cross-attention layers. Compared to large language models, my transformers are small: they use a few (1 to 8) layers in the encoder and decoder, and (typically) 512 dimensions and 8 attention heads. 

For training, the loss is the cross-entropy between model prediction and the correct sequence  (computed at the token level). The models do not understand mathematics, they only predicts sequences. Adam is used as the optimizer, usually with linear warmup and scheduling (inverse square root or cosine).

Trained models are tested on held-out test sets (usually 10 000 examples). A prediction is considered correct if the output of the model can decode into a valid solution (e.g. a number, vector, or matrix of numbers), which solves the problem within a certain tolerance.

### Number Encodings

Schemes for encoding mathematical expressions are discussed in [Deep learning for symbolic mathematics](https://arxiv.org/abs/1912.01412). For numbers, folllowing the discussion in [Linear algebra with transformers](https://arxiv.org/abs/2112.01898), we use the following methods.

#### Integers (bounded)
Integers between 0 and Q (e.g. in modular arithmetic) are encoded as single tokens ('0', '1', ... 'Q')
#### Integers (not bounded)
Encoded as a sign ('+' or '-') followed by the representation of their absolute value in base b (typically, b=10 or 1000): -3124 is represented as the sequence '-', '3', '1', '2', '4' in base 10, and '-', '3', '124' in base 1000
#### Real numbers
Represented in scientific notation with a certain number of significant digits (typically 3), and encoded as triplet: (sign, mantissa, exponent), mantissa and exponents are bounded integers (encoded as above). -3.14 = -314 . 10^-2 is encoded as '-', '314', 'E-2' (other encodings exist, see the paper on linear algebra)
#### Complex numbers
Represented as pairs of real numbers
 
