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

# JSUT
[https://sites.google.com/site/shinnosuketakamichi/publication/jsut](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

JSUT is a japanese speech dataset consisting of about 5h of a single female speaker. The transcipt was designed to cover common use words.

To to set up this dataset we will follow this guide:
* [TensorFlow Datasets writing custom datasets](https://www.tensorflow.org/datasets/add_dataset)

Run this command from command line to create a dummy dataset that we will modify
```
tfds new jsut_beta
```

This creates a python file called `jsut_beta.py`

```python
"""jsut_beta dataset."""

import tensorflow_datasets as tfds

# TODO(jsut_beta): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(jsut_beta): BibTeX citation
_CITATION = """
"""


class JsutBeta(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for jsut_beta dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(jsut_beta): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # e.g. ('image', 'label')
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(jsut_beta): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={},
        ),
    ]

  def _generate_examples(self):
    """Yields examples."""
    # TODO(jsut_beta): Yields (key, example) tuples from the dataset
    yield 'key', {}

```

To see how I am meant to complete this, I examine the code of a similar kind of dataset, ljspeech

```python
# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LJSpeech dataset."""

import os

import tensorflow.compat.v2 as tf

import tensorflow_datasets.public_api as tfds

_CITATION = """\
@misc{ljspeech17,
  author       = {Keith Ito},
  title        = {The LJ Speech Dataset},
  howpublished = {\\url{https://keithito.com/LJ-Speech-Dataset/}},
  year         = 2017
}
"""

_DESCRIPTION = """\
This is a public domain speech dataset consisting of 13,100 short audio clips of
a single speaker reading passages from 7 non-fiction books. A transcription is
provided for each clip. Clips vary in length from 1 to 10 seconds and have a
total length of approximately 24 hours.
The texts were published between 1884 and 1964, and are in the public domain.
The audio was recorded in 2016-17 by the LibriVox project and is also in the
public domain.
"""

_URL = "https://keithito.com/LJ-Speech-Dataset/"
_DL_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class Ljspeech(tfds.core.GeneratorBasedBuilder):
  """LJSpeech dataset."""

  VERSION = tfds.core.Version("1.1.0")

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "id": tf.string,
            "speech": tfds.features.Audio(sample_rate=22050),
            "text": tfds.features.Text(),
            "text_normalized": tfds.features.Text(),
        }),
        supervised_keys=("text_normalized", "speech"),
        homepage=_URL,
        citation=_CITATION,
        metadata=tfds.core.MetadataDict(sample_rate=22050),
    )

  def _split_generators(self, dl_manager):
    extracted_dir = dl_manager.download_and_extract(_DL_URL)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"directory": extracted_dir},
        ),
    ]

  def _generate_examples(self, directory):
    """Yields examples."""
    metadata_path = os.path.join(directory, "LJSpeech-1.1", "metadata.csv")
    with tf.io.gfile.GFile(metadata_path) as f:
      for line in f:
        line = line.strip()
        key, transcript, transcript_normalized = line.split("|")
        wav_path = os.path.join(directory, "LJSpeech-1.1", "wavs",
                                "%s.wav" % key)
        example = {
            "id": key,
            "speech": wav_path,
            "text": transcript,
            "text_normalized": transcript_normalized,
        }
        yield key, example
```

I found out that when you are developing these datasets on a colab runtime, it is necessary to be able to reload the importing of the changed dataset.
```python
import importlib
importlib.reload(jsut_beta.jsut_beta)
```
