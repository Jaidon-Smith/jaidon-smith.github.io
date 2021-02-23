---
title: "(Under Construction) Building a Japanese SentencePiece Tokenization"
categories:
  - Posts
tags:
  - Natural Language Processing
  - Machine Learning
  - Japanese
collections:
  - Reflection
excerpt: "The first step of Japanese Speech to text is to work out the tokens"
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial

The [Wiki40b japanese linear model](https://tfhub.dev/google/wiki40b-lm-ja/1) on tensorflow hub uses 32k sentence piece tokens.

# Comparing Wiki40b tokens to the transcripts of my audio data
One of my Japanese speech datasets is of about 100h length and I believe to contain some domain specific characters. I want to briefly analyse the extent of this.

Getting the tokens from Wiki40b

```python
!pip install --quiet tensorflow-text
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

def create_token_explorer():
  num_tokens = 32000
  token_explorer = list(range(num_tokens))
  for i in range(num_tokens - 1):
    token_explorer.insert(num_tokens - 1 - i, 4)
  return np.array([token_explorer])

token_explorer = create_token_explorer()

tf.disable_eager_execution()
g = tf.Graph()
with g.as_default():
  module = hub.Module("https://tfhub.dev/google/wiki40b-lm-ja/1")
  detoken_text = module(dict(token_ids=token_explorer), signature="detokenization",
                        as_dict=True)
  detoken_text = detoken_text["text"]

  init_op = tf.group([tf.global_variables_initializer(),
                      tf.tables_initializer()])

# Initialize session.
with tf.Session(graph=g) as session:
  session.run(init_op)
  detoken_text = session.run([
    detoken_text])
  
tokens = detoken_text[0][0].decode().split('_START_SECTION_')[8:]
```

As an aside the code only works with tensorflow v1, this is the exception when using with tensorflow v2.

![image1](/assets/images/2021-02-19-building-japanese-sentence-piece-tokenization/image1.jpg)

## Characters That Are in My Dataset but Not in Wiki40b
My text contains 13,305 unique characters, of which only 3,822 could appear in wiki40b tokenization.

Here are some of the characters that could not be represented:
* ('曠', 4245)
* ('昿', 4246)
* ('曦', 4247)
* ('曩', 4248)
* ('曵', 4250)
* ('曷', 4251)
* ('朏', 4252)
* ('朖', 4253)
* ('朞', 4254)
* ('朦', 4255)
* ('朮', 4258)
* ('朿', 4259)
* ('朶', 4260)
* ('杁', 4261)
* ('朸', 4262)
* ('朷', 4263)
* ('杆', 4264)
* ('杞', 4265)
* ('杠', 4266)
* ('杙', 4267)
* ('杤', 4269)
* ('枉', 4270)

They are mostly obscure kanji which are used in alternate spellings or domain specific.

## Characters that are in Wiki40b but not my dataset
There are 4288 unique characters in Wiki40b of which 3822 are representable in my dataset.

Here are some of the characters that could not be represented:
* ('ヰ', 3783)
* ('И', 3784)
* ('ป', 3791)
* ('»', 3798)
* ('M', 3801)
* ('기', 3803)
* ('ς', 3817)
* ('t', 3820)
* ('Ø', 3838)
* ('ε', 3840)
* ('เ', 3849)
* ('อ', 3856)
* ('Ἀ', 3857)
* ('τ', 3896)
* ('x', 3904)
* ('p', 3919)
* ('에', 3929)
* ('ж', 3947)
* ('D', 3948)
* ('ヲ', 3953)
* ('ว', 3963)
* ('โ', 3966)
* ('f', 4001)
* ('하', 4009)
 
While containing some rarer Japanese kana such as 'ヰ' and 'ヲ', for the most part this list mostly consists of non Japanese characters.
 
## Implications of above analysis
The above analysis I think confirms my concern that it may be necessary to redo the SentencePiece tokenisation for my use. Japanese is a very character rich language unlike English and this tokenisation does not cover enough of the rarer characters. I think what may have occured is that the tokenization was based on a small subset of wiki40b which is fine for English but may not cover enough characters for Japanese.
 
All that being said, it still may be possible to just add the extra characters to the wiki40b tokenization so I have to decide now between doing that or performing the SentencePiece from scratch.
 
# Building a SentencePiece Tokenization
SentencePiece may need to be installed with pip.
```
pip install sentencepiece
```
 
I have my text dataset saved on Google Drive as about 1200 text files
```python
import os
files = os.listdir('/gdrive/MyDrive/Japanese/Bible/chapters')
files = ['/gdrive/MyDrive/Japanese/Bible/chapters/' + i for i in files if '.txt' in i]
```
I then train an SPM model, with 32k pieces like wiki40b in order to compare them
```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(input=files, model_prefix='m', vocab_size=32000)
```
The model can be loaded and the tokens can be inspected
```python
sp = spm.SentencePieceProcessor(model_file='m.model')
tokens = sp.id_to_piece(list(range(32000)))
```
This does work and builds up 32k tokens but it only considers a subset of the text and as a result about 11k characters in my dataset are not representable.
What I have done is prepare the list of characters that appear in my text (called `the_list_names`) and I am going to force the tokenization to be exactly these characters.
I admit that I no longer need SentencePiece for this kind of tokenization but it means that my later work could support it just by changing the model it loads.

> Note: was `the_list_names_space_dropped` is `the_list_names` with space token dropped.
```python
spm.SentencePieceTrainer.train(input=files, model_prefix='letters', vocab_size=13310, user_defined_symbols=the_list_names_space_dropped)
```
# SentencePiece in TensorFlow
Documentation for using SentencePiece in TensorFlow:
* [https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/SentencepieceTokenizer.md](https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/SentencepieceTokenizer.md)
* [https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/Tokenizer.md](https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/Tokenizer.md)

Imports
```python
!pip install --quiet tensorflow-text

import tensorflow_text as text
from tensorflow.python.platform import gfile
```
Getting the tokenizer
```python
model_file = '/gdrive/MyDrive/Japanese/Bible/letters.model'
model = gfile.GFile(model_file, 'rb').read()

tokenizer = text.SentencepieceTokenizer(model=model)
```


