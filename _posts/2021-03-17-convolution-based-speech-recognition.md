---
title: "(Under Construction) A Convolution Based Speech Recognition Implementation"
categories:
  - Posts
tags:
  - TensorFlow
  - Machine Learning
collections:
  - Reflection
excerpt: "Reimplementing Wav2Letter in TensorFlow."
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial.

# Relevant Links

Wav2Letter on Facebook Research
* [https://research.fb.com/downloads/wav2letter/](https://research.fb.com/downloads/wav2letter/)

Paper: Wav2Letter: an End-to-End ConvNet-based Speech Recognition System
* [https://arxiv.org/abs/1609.03193](https://arxiv.org/abs/1609.03193)

Paper: Letter-Based Speech Recognition with Gated ConvNets
* [https://arxiv.org/abs/1712.09444](https://arxiv.org/abs/1712.09444)

Paper: Language Modeling with Gated Convolutional Networks
* [https://paperswithcode.com/paper/language-modeling-with-gated-convolutional](https://paperswithcode.com/paper/language-modeling-with-gated-convolutional)

GitHub PyTorch Implementation
* [https://github.com/facebookresearch/wav2letter/tree/master/recipes/conv_glu](https://github.com/facebookresearch/wav2letter/tree/master/recipes/conv_glu)
* Gated ConvNet Network Architecture: [https://github.com/facebookresearch/wav2letter/blob/master/recipes/conv_glu/librispeech/network.arch](https://github.com/facebookresearch/wav2letter/blob/master/recipes/conv_glu/librispeech/network.arch)

# Gated Convolution

This paper: Language Modeling with Gated Convolutional Networks
* [https://paperswithcode.com/paper/language-modeling-with-gated-convolutional](https://paperswithcode.com/paper/language-modeling-with-gated-convolutional)

has information on implementing gated convolutions used gated linear units.
They present this equation:
$$hl(X) = (X*W + b) \otimes \sigma(X*V + c)$$

I believe that the desired results for a gated convolutional layer can be achieved in TensorFlow with this code. I also added batch normalisation because in my experiments the outputs could sometimes explode to high values otherwise.

```python
class GatedConvolution(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, dropout_rate, padding='causal'):
    super(GatedConvolution, self).__init__()

    self.convolution = tf.keras.layers.Conv1D(
    filters=filters, kernel_size=kernel_size, padding=padding,
    )

    self.gated = tf.keras.layers.Conv1D(
    filters=filters, kernel_size=kernel_size, padding=padding,
    )

    self.multiply = tf.keras.layers.Multiply()
    
    self.norm = tf.keras.layers.BatchNormalization()
    
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
  def call(self, x, training):

    convolution_output = self.convolution(x)  # (batch_size, input_seq_len, d_model)

    gate_output = tf.keras.activations.sigmoid(self.gated(x))
    
    output = self.multiply([convolution_output, gate_output])
    
    output = self.norm(output, training=training)

    output = self.dropout(output, training=training)
    
    return output
```

# Network Structure
The facebook research paper presents 3 different network architectures. I'll start by implementing the one that was designed for the WSJ dataset. Note that the PyTorch implementations are available on their GitHub and it was there I found this architecture file:

* [https://github.com/facebookresearch/wav2letter/blob/master/recipes/conv_glu/wsj/network.arch](https://github.com/facebookresearch/wav2letter/blob/master/recipes/conv_glu/wsj/network.arch)

```
WN 3 C NFEAT 200 13 1 -1
GLU 2
DO 0.25

WN 3 C 100 200 3 1 -1
GLU 2
DO 0.25

WN 3 C 100 200 4 1 -1
GLU 2
DO 0.25

WN 3 C 100 250 5 1 -1
GLU 2
DO 0.25

WN 3 C 125 250 6 1 -1
GLU 2
DO 0.25

WN 3 C 125 300 7 1 -1
GLU 2
DO 0.25

WN 3 C 150 350 8 1 -1
GLU 2
DO 0.25

WN 3 C 175 400 9 1 -1
GLU 2
DO 0.25

WN 3 C 200 450 10 1 -1
GLU 2
DO 0.25

WN 3 C 225 500 11 1 -1
GLU 2
DO 0.25

WN 3 C 250 500 12 1 -1
GLU 2
DO 0.25

WN 3 C 250 500 13 1 -1
GLU 2
DO 0.25

WN 3 C 250 600 14 1 -1
GLU 2
DO 0.25

WN 3 C 300 600 15 1 -1
GLU 2
DO 0.25

WN 3 C 300 750 21 1 -1
GLU 2
DO 0.25

RO 2 0 3 1
WN 0 L 375 1000
GLU 0
DO 0.25

WN 0 L 500 NLABEL
```

It is difficult to work out exactly what this is communicating but I believe that (WN 3 C) represent convolutions, (GLU 2) represents gated linear units and (DO) represents dropout.


