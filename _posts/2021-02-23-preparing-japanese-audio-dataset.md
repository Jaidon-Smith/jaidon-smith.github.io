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
