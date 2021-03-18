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

# Dataset and Map Files

Custom maps can be downloaded from [https://bsaber.com/](https://bsaber.com/). Songs can be sorted according to rating which in my experience finds the highest quality maps. They can also be filtered according to difficulty for which I will select Expert+. While there are certainly some good lower difficulty levels, I find these to be the most interesting.

A map file comes as a zip file containing the song as well as a .data for the level. The data is binary encoded json. The most useful sequence is that of the `_notes`. A typical note looks like this:

```
  {
      "_time": x,
      "_lineIndex": x,
      "_lineLayer": x,
      "_type": x,
      "_cutDirection": x
  }
```
There are other kinds of notes in the game but to start I will only consider coloured notes.

* _time: represents the time in seconds that the note occured and is a decimal.
* _lineIndex and _lineLayer represent positioning. Notes can occupy a grid of 3 rows and 4 columns. _lineIndex is the column, _lineLayer is the row. The bottom left corner represents (0,0)
* _type: 0 if the note is red, 1 if the note is blue.

** _cutDirection **
* Down: 1
* Left: 2
* Up: 0
* Right: 3
* LD: 6
* LU: 4
* RU: 5
* RD: 7
* Neutral: 8

In this setting that means that there are 2 * 12 * 9 = 216 possible single events.
