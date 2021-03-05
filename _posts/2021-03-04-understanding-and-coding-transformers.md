---
title: "(Under Construction) Understanding and Coding Transformers"
categories:
  - Posts
tags:
  - TensorFlow
  - Machine Learning
collections:
  - Reflection
excerpt: "Examining the associated paper as well as various codebases."
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial.

As part of building an automatic speech recognition tensorflow model I am researching and building transformers.

# Some Useful Links

## The Paper "Attention Is All You Need"

[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf)

## TensorFlow

Tutorial: Transformer model for language understanding

* [https://www.tensorflow.org/tutorials/text/transformer](https://www.tensorflow.org/tutorials/text/transformer)

Multi-head attention API

* [https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention?version=nightly](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention?version=nightly)

Position encoding

* [https://github.com/tensorflow/examples/blob/master/community/en/position_encoding.ipynb](https://github.com/tensorflow/examples/blob/master/community/en/position_encoding.ipynb)

## Tensor2Tensor

* [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

This was mentioned in the transformer paper as the original codebase.

"Tensor2Tensor, or T2T for short, is a library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research."

It has since been replaced by Trax.

**Automated Speech Recognition with the Transformer model**

This was the only tutorial under the docs in github

* [https://cloud.google.com/tpu/docs/tutorials/automated-speech-recognition](https://cloud.google.com/tpu/docs/tutorials/automated-speech-recognition)

Tensor2Tensor Transformer

* [https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)

Speech recognition transformer notebook

* [https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/asr_transformer.ipynb](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/asr_transformer.ipynb)

Inside `transformer.py` there are parameters for an automatic speech recognition transformer.

```python
@registry.register_hparams
def transformer_librispeech_v1():
  """HParams for training ASR model on LibriSpeech V1."""
  hparams = transformer_base()

  hparams.num_heads = 4
  hparams.filter_size = 1024
  hparams.hidden_size = 256
  hparams.num_encoder_layers = 5
  hparams.num_decoder_layers = 3
  hparams.learning_rate = 0.15
  hparams.batch_size = 6000000

  librispeech.set_librispeech_length_hparams(hparams)
  return hparams


@registry.register_hparams
def transformer_librispeech_v2():
  """HParams for training ASR model on LibriSpeech V2."""
  hparams = transformer_base()
  
  hparams.num_heads = 2
  hparams.filter_size = 1536
  hparams.hidden_size = 384
  hparams.num_encoder_layers = 6
  hparams.num_decoder_layers = 4
  hparams.learning_rate = 0.15
  hparams.batch_size = 16

  hparams.max_length = 1240000
  hparams.max_input_seq_length = 1550
  hparams.max_target_seq_length = 350
  hparams.daisy_chain_variables = False
  hparams.ffn_layer = "conv_relu_conv"
  hparams.conv_first_kernel = 9
  hparams.weight_decay = 0
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.relu_dropout = 0.2

  return hparams
```

## Trax

[https://github.com/google/trax](https://github.com/google/trax)

"Trax is an end-to-end library for deep learning that focuses on clear code and speed. It is actively used and maintained in the Google Brain team".

Trax Documentation

* [https://trax-ml.readthedocs.io/en/latest/](https://trax-ml.readthedocs.io/en/latest/)

Trax Transformer

* [https://github.com/google/trax/blob/master/trax/models/transformer.py](https://github.com/google/trax/blob/master/trax/models/transformer.py)

## TensorFlow Addons

Multi-head attention

* [https://www.tensorflow.org/addons/api_docs/python/tfa/layers/MultiHeadAttention](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/MultiHeadAttention)

# Background: Attention

Tutorial on TensorFlow Site

* [https://www.tensorflow.org/tutorials/text/nmt_with_attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention)

A Blog post found by google searching "attention networks"

* [https://buomsoo-kim.github.io/attention/2020/01/01/Attention-mechanism-1.md/](https://buomsoo-kim.github.io/attention/2020/01/01/Attention-mechanism-1.md/)

## Seq2Seq: Motivated attention

Blog post linked by the other Buomsoo Kim post

* [https://buomsoo-kim.github.io/attention/2020/01/09/Attention-mechanism-2.md/](https://buomsoo-kim.github.io/attention/2020/01/09/Attention-mechanism-2.md/)

Google AI Blog (Building Your Own Neural Machine Translation System in TensorFlow)

* [https://ai.googleblog.com/2017/07/building-your-own-neural-machine.html](https://ai.googleblog.com/2017/07/building-your-own-neural-machine.html)

The actual tutorial link and has some good theory on seq2seq

* [https://github.com/tensorflow/nmt](https://github.com/tensorflow/nmt)

There is also a seq2seq google repository

* [https://google.github.io/seq2seq/](https://google.github.io/seq2seq/)

* [https://github.com/google/seq2seq](https://github.com/google/seq2seq)

* [https://github.com/google/seq2seq/blob/master/docs/nmt.md](https://github.com/google/seq2seq/blob/master/docs/nmt.md)

TensorFlow Addon Seq2Seq Tutorial

* [https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt](https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt)
