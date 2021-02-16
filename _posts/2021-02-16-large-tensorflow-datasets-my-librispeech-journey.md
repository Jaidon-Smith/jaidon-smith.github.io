---
title: "(Under Construction) Large Tensorflow Datasets - My LibriSpeech Journey"
categories:
  - posts
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

My recent project requires working with a great many audio files and Tensorflow datasets offers what looks to be a good mechanism for using them. I planned to use an established tensorflow dataset LibriSpeech as a model for my own, I didn't expect however that it would be complicated to get working. This is my recount of trying to get the LibriSpeech dataset to work.

# Issue 1: Download Speed Unworkably Slow
On my machine with mediocre internet it was estimated to take 12 hours to download. I proceeded to set up a Google Cloud Platform Virtual Machine which along with other advantages for machine learning projects, could download the dataset in less than an hour.

# Issue 2: The extraction code had a bug
I proceeded to run the code to download and extract the dataset.
'''python
import tensorflow_datasets as tfds
ds = tfds.load('librispeech', split='train_clean100', shuffle_files=True, data_dir='./')
'''
