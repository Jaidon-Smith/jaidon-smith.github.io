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

```python
import tensorflow_datasets as tfds
ds = tfds.load('librispeech', split='train_clean100', shuffle_files=True, data_dir='./')
```

However this terminated unsuccessfully. I actually forked the tensorflow datasets repository on github and found the location of exception, it was a one line fix in `tensorflow_datasets/audio/librispeech.py`.

```python
with tf.io.gfile.GFile(os.path.join(path, transcript_file)) as f:
```

changed to:

```python
with tf.io.gfile.GFile(transcript_file) as f:
```

Without this fix Librispeech just flat out doesn't work and it made me wonder why the master was in this state. I actually found an issue on github where someone had made the exact same change as me but unfortunately in their pull request had many other unrelated changes so no one had reviewed it.

# Issue 3: Using GCP DataFlow
When running the extration on my machine, about 10m in the apache runner announces it is out of memory. I think this pretty well confirms my suspicion that it won't really be possible to extract the dataset on a single machine. The dataset is designed to be extracted using parallel computation ([https://www.tensorflow.org/datasets/beam_datasets](https://www.tensorflow.org/datasets/beam_datasets)), so I think the next step is to set up GCP DataFlow.

I work through some guides to get a feel for DataFlow and Apache Beam
* [https://cloud.google.com/dataflow/docs/quickstarts](https://cloud.google.com/dataflow/docs/quickstarts)
* [https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)

