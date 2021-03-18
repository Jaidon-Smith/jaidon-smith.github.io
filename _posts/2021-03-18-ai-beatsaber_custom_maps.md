---
title: "(Under Construction) AI Custom BeatSaber Maps and Magenta"
categories:
  - Posts
tags:
  - TensorFlow
  - Machine Learning
  - Transformers
  - BeatSaber
  - Magenta
collections:
  - Reflection
excerpt: "Investigating the premise of developing a machine learning model for BeatSaber map generation."
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial.

The VR rythm game BeatSaber has a very strong custom map creating community. While there are many well designed maps and many great songs, it can be difficult to find a map that satifies both of these requirments at the same time. It is for this reason that it would be desirable to have an AI generate maps for any song that the user likes.

There has already been some activity on this front ([https://beatsage.com/](https://beatsage.com/)). I am not going to directly compare my output to them as I intend this as more of a fun learning exercise rather than being result focused.

I have long been interested in [Google AI's Magenta Project](https://magenta.tensorflow.org/). I expect to some extent that working with the BeatSaber note events will be similar to working with MIDI. Magenta have done a lot of work with MIDI for piano transcription as well as musical score generation. Part of this exercise is also to investigate if any of their models can be adapted to this problem.
