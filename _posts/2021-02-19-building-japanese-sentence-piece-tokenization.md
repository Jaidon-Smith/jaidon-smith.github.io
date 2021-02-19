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



