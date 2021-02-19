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

As an aside the code only works with tensorflow v1, 

![image1](/assets/images/2021-02-19-building-japanese-sentence-piece-tokenization/image1.jpg)



