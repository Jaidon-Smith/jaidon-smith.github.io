---
title: "(Under Construction) Preparing Japanese Audio Datasets for TensorFlow"
categories:
  - Posts
tags:
  - Natural Language Processing
  - Machine Learning
  - Japanese
collections:
  - Reflection
excerpt: "Preparing JSUT and Mozilla Common Voice for TensorFlow"
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial

[These datasets to be used with TensorFlow are available here.](https://github.com/Jaidon-Smith/public-datasets)

To to set up these datasets we will follow this guide:
* [TensorFlow Datasets writing custom datasets](https://www.tensorflow.org/datasets/add_dataset)

# JSUT
[https://sites.google.com/site/shinnosuketakamichi/publication/jsut](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

JSUT is a japanese speech dataset consisting of about 10h of a single female speaker. The transcipt was designed to cover common use words.

# Common Voice Version 6
TensorFlow datasets only has version 1 of this dataset which does not have Japanese.

Version 6 has 5h total Japanese speech with 3h of it validated.

# Tatoeba Japanese
Japanese sentences that contain audio on [Tatobeta](https://tatoeba.org/eng/downloads).

Consists of about 1h of Japanese speech made up from 1525 sentences.

Zip file can be downloaded [here](https://drive.google.com/file/d/1WTZKU5uv8D8eoUJcwrIa5m-2ya_fmcjt/view?usp=sharing).
