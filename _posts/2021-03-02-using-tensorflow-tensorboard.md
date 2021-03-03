---
title: "(Under Construction) Using TensorFlow TensorBoard"
categories:
  - Posts
tags:
  - TensorFlow
  - Machine Learning
collections:
  - Reflection
excerpt: "Realtime visualisation is important for ML research."
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial.

When doing machine learning research, it is important to be able to have real time visualisation. TensorBoard allows such visualisation. This post will document my experiments utilising TensorBoard.

# TensorBoard Guides
I Looked at these guides:
* [TensorBoard Getting Started Guide](https://www.tensorflow.org/tensorboard/get_started)
* [TensorBoard Scalars: Logging training metrics in Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras)
* [Displaying image data in TensorBoard](https://www.tensorflow.org/tensorboard/image_summaries)
* [Examining the TensorBoard Graph](https://www.tensorflow.org/tensorboard/graphs)

These TensorFlow API references are also relevant:
* [Writing to logs: tf.summary](https://www.tensorflow.org/api_docs/python/tf/summary?version=nightly)
* [Understannding callbacks in keras: tf.keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks)

# Logging Audio
* `speech_4` was a list of 1D numpy arrays representinng mono audio.
* `transcripts` was a list of strings representing the spoken characters in the corresponding audio.
```python
# Clear out any prior log data.
!rm -rf logs

# Sets up a timestamped log directory.
logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer(logdir)

for i in range(4):
  print(i)
  audio = speech_4[i]
  audio = audio.reshape((1, -1, 1))
  description = '**Transcript**: {}\n\n'.format(transcripts[i])
  with file_writer.as_default():
    tf.summary.audio("Audio " + str(i+1), audio, 48000, step=0, description=description)
```
How the audio appears in TensorBoard:
![image1](/assets/images/2021-03-02-using-tensorflow-tensorboard/image1.jpg)
* The Description can be viewed by clicking the information.

