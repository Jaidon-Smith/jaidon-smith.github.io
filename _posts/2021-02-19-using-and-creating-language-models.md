---
title: "(Under Construction) Using and Creating Language Models"
categories:
  - Posts
tags:
  - Audio
  - Machine Learning
collections:
  - Reflection
excerpt: "Big data complicates things"
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial

For my current project I need a Japanese speech to text, the english Silero-STT would be appropiate however it needs to work for Japanese.
This is why (as well as just being a good exercise) I am recreating the described work.

The focus of this blog post is to examine language models and how they fit into a speech to text.

I just realised that language model only fits into the decoder part of the model which does not need to be utilised for my project. The SentencePiece tokens would still be relevant.
