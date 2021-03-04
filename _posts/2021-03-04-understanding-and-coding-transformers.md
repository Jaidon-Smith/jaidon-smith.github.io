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

## TensorFlow Tutorial

* Transformer model for language understanding

[https://www.tensorflow.org/tutorials/text/transformer](https://www.tensorflow.org/tutorials/text/transformer)

## Tensor2Tensor

[https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

This was mentioned in the transformer paper as the original codebase.

"Tensor2Tensor, or T2T for short, is a library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research."

It has since been replaced by Trax.

* Automated Speech Recognition with the Transformer model

This was the only tutorial under the docs in github

[https://cloud.google.com/tpu/docs/tutorials/automated-speech-recognition](https://cloud.google.com/tpu/docs/tutorials/automated-speech-recognition)

Tensor2Tensor Transformer

[https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)

Speech recognition transformer notebook

[https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/asr_transformer.ipynb](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/asr_transformer.ipynb)

Inside `transformer.py` there are parameters for a automatic speech recognition transformer.

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

  hparams.max_length = 1240000
  hparams.max_input_seq_length = 1550
  hparams.max_target_seq_length = 350
  hparams.batch_size = 16
  hparams.num_decoder_layers = 4
  hparams.num_encoder_layers = 6
  hparams.hidden_size = 384
  hparams.learning_rate = 0.15
  hparams.daisy_chain_variables = False
  hparams.filter_size = 1536
  hparams.num_heads = 2
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

[https://trax-ml.readthedocs.io/en/latest/](https://trax-ml.readthedocs.io/en/latest/)

Trax Transformer

[https://github.com/google/trax/blob/master/trax/models/transformer.py](https://github.com/google/trax/blob/master/trax/models/transformer.py)
