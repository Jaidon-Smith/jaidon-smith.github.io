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

## Obtaining a dataset

Custom maps can be downloaded from [https://bsaber.com/](https://bsaber.com/). Songs can be sorted according to rating which in my experience finds the highest quality maps. They can also be filtered according to difficulty for which I will select Expert+. While there are certainly some good lower difficulty levels, I find these to be the most interesting.

For starters I will collect 5000 of these songs. It is hard to say how many hours this will be but I expect it to be somewhere between 100 and 200 hours.

## A typical map file

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

**_cutDirection**
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

# Representation

I am interested in approaching this problem using a transformer model so I have been reading this magenta paper:

**MUSIC TRANSFORMER: GENERATING MUSIC WITH LONG-TERM STRUCTURE**

* [https://magenta.tensorflow.org/music-transformer](https://magenta.tensorflow.org/music-transformer)
* [https://arxiv.org/pdf/1809.04281.pdf](https://arxiv.org/pdf/1809.04281.pdf)

In the paper they discuss different data representations for music. BeatSaber songs more resemble event based MIDI than Bach chorales. In the paper they state that their representation is based on another paper: 

**This Time with Feeling: Learning Expressive Musical Performance**

* [https://arxiv.org/pdf/1808.03715.pdf](https://arxiv.org/pdf/1808.03715.pdf)

In that paper they say this:

> A MIDI excerpt is represented as a sequence of events from the following vocabulary of 413 different events:
> * 128 NOTE-ON events: one for each of the 128 MIDI pitches. Each one starts a new note.
> * 128 NOTE-OFF events: one for each of the 128 MIDI pitches. Each one releases a note.
> * 125 TIME-SHIFT events: each one moves the time step forward by increments of 8 ms up to 1 second.
> * 32 VELOCITY events: each one changes the velocity applied to all subsequent notes (until the next velocity event).

The use of a time-shift with multiple values is interesting.

While it would be possible to express BeatSaber notes a similar way (ie. an event for the colour, different event for direction), I think I will still combine them into one event. But I will probably handle the timing a similar way.

## What aspects of the notes depend on the music?

It would be beneficial I think to split the model into two stages, one that depends on the music and one from the output of the first stage. This would mean we could reduce the size of the representation in the model that depends on the music.

**Depends on Music**

Timing

Note Colour

> While not every BeatSaber map does this, my favourite maps are when the blue and the red are highlighting different aspects of the song.

**Unsure**

Note Location

> Evidence For Independent
> * Louder dynamics often result in larger swings. I originally thought that location would be needed to capture this but I have realised that multiple notes in the same location can also mean that there is a large swing.
>
> Evidence For Dependent
> * For song consistency (ie. verse 1 and verse 2 have a similar structure)
> * Cyclic actions, (ie. You repeat the same action every bar, just given that notes are there you can't determine the start of the bar)

**Independent of Music**

Direction

> The swing direction depends more on the structure of the notes rather than the music.

If I include location, there would be 18 possible events not considering time events.

If I do not consider location then there would be 2 possible events not considering time events. I suppose that multiple notes occuring at the same time has two possible representations. You could have multiple events or include the number of notes in the event. I would probably go with multiple events.

Going forward I will consider Note Location independent of the music and see what kind of results I get but I may revisit this.

# Understanding and implementing RELATIVE POSITIONAL SELF-ATTENTION

This section is about understanding and implementing the sytem employed in the Music Transformer for encoding relative postion information.

From the Music Transformer Paper:

> As the Transformer model relies solely on positional sinusoids to represent timing information, Shaw
et al. (2018) introduced relative position representations to allow attention to be informed by how far
two positions are apart in a sequence.

They have a good explanationnation of their method but I think it would be good to entirely remove the concept of heads to make it clearer.

**The original**

Let $$D$$ be the transformer depth dimension.

$$L$$ is the number of input vectors.

Then $$E^r$$ of shape $$(L, D)$$

$$R$$ of shape $$(L, L, D)$$

The queries Q with shape $$(L, D)$$.

$$Q_{reshaped}$$ is Q but reshaped to $$(L, 1, D)$$

$$S^{rel} = QR^T$$ has shape $$(L, L)$$

$$Attention(Q,K,V)=softmax_{k}(\frac{QK^{T} + S^{rel}}{d})V$$

$$Attention(Q,K,V)=softmax_{k}(\frac{QK^{T} + Q_{reshaped}R^T}{d})V$$

But what I am finding strange about this is that $$Q$$ is shape $$(L, D)$$ and $$K^T$$ is shape $$(D, L)$$ so it is natural that $$QK^T$$ is shape $$(L, L)$$.

However $$R$$ is shape $$(L, L, D)$$ instead of the $$(L, D)$$ of $$K$$.

The answer to this is understanding what it actually means to multiply $$Q_{reshaped}$$ and $$R^T$$ with shapes $$(L, 1, D)$$ and $$(L, L, D)^T$$ and show that it has shape $$(L, L)$$ as required.

In Tensorflow, the matmul function only acts on the last two dimensions and all of the earlier ones behave like a batch dimension. So essentially for every $$L$$ in the first dimension multiply a shape $$(1, D)$$ with $$(L, D)^T$$ with the transpose making it shape $$(D, L)$$

This will mean that the output has shape $$(L, 1, L)$$ and has to be reshaped into $$(L, L)$$ before we 



