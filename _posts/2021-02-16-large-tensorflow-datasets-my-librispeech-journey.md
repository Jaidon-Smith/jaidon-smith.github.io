---
title: "(Under Construction) Large Tensorflow Datasets - My LibriSpeech Journey"
categories:
  - posts
tags:
  - Audio
  - Machine Learning
  - Google Cloud Platform
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

# Issue 2: The generation code had a bug
I proceeded to run the code to download and generate the dataset.

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
When running the extration on my machine, about 10m in the apache runner announces it is out of memory. I think this pretty well confirms my suspicion that it won't really be possible to generate the dataset on a single machine. The dataset is designed to be generated using parallel computation ([https://www.tensorflow.org/datasets/beam_datasets](https://www.tensorflow.org/datasets/beam_datasets)), so I think the next step is to set up GCP DataFlow.

## Learning Resources for DataFlow

I work through some guides to get a feel for DataFlow and Apache Beam
* [https://console.cloud.google.com/dataflow/jobs?walkthrough_tutorial_id=dataflow_index](https://console.cloud.google.com/dataflow/jobs?walkthrough_tutorial_id=dataflow_index)
* [https://cloud.google.com/dataflow/docs/quickstarts](https://cloud.google.com/dataflow/docs/quickstarts)
* [https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)

Also useful to read Apache Beam Documentation
* [https://beam.apache.org/documentation/sdks/python/](https://beam.apache.org/documentation/sdks/python/)
* [https://beam.apache.org/documentation/programming-guide/](https://beam.apache.org/documentation/programming-guide/)

Uses Google Cloud Storage and gsutil
* [https://cloud.google.com/storage/docs/gsutil](https://cloud.google.com/storage/docs/gsutil)
* [Cloud Storage Guides](https://cloud.google.com/storage/docs/quickstart-console)

In order to understand what DataFlow is achieving, I researched the underpinning idea of MapReduce
* [Brief video that demonstrates how map reduce can achieve parallelism](https://www.youtube.com/watch?v=43fqzaSH0CQ&ab_channel=internet-class)

# Generating the LibriSpeech dataset using DataFlow
It is now time to use the above knowledge to generate the dataset. Like the dataflow tutorial example I execute all of these commands from the GCP console

**Setting up the virtual machine**
```
pip3 install --upgrade virtualenv \ --user
python3 -m virtualenv env
source env/bin/activate
```

**Installing Apache Beam and tensorflow-datasets**
```
pip3 install --quiet \ apache-beam[gcp]
pip3 install tensorflow-datasets
```

**Create Storage Bucket**
```
gsutil mb gs://general-304503
```

**Set some parameters**
```
DATASET_NAME=librispeech
DATASET_CONFIG=
GCP_PROJECT=general-304503
GCS_BUCKET=gs://general-304503
```

**You will then need to create a file to tell Dataflow to install tfds on the workers**
```
echo "tensorflow_datasets[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

**Finally, you can launch the job using the command below**
```
tfds build $DATASET_NAME \
--data_dir=$GCS_BUCKET/tensorflow_datasets \
--beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt"
```

## Google Cloud Storage Buckets
However only a couple of minutes into the downloading the console crashes. I think this is because the GCP console is not designed for long running computations (like downloading a dataset). What this means is I will have to download the dataset to GCP bucket from a compute instance before executing the above commands.

According to the [GCP compute engine documentation for mounting a bucket](https://cloud.google.com/compute/docs/disks/gcs-buckets#mount_bucket) I have to install [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md)

**Add the gcsfuse distribution URL as a package source and import its public key:**
```
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
```

**Update the list of packages available and install gcsfuse**
```
sudo apt-get update
sudo apt-get install gcsfuse
```

**Mount the storage bucket**
```
mkdir ~/general-304503
gcsfuse general-304503 ~/general-304503
```

After mounting I found that I could not interact with the files in the bucket. I needed to change the storage API permissions in the compute engine from Read Only to Full

![image2](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image2.jpg)

After getting the mounting to work I attempted to copy the downloaded dataset to the bucket however it was extremely slow (Estimated to take 48h). I think this is because the gcsfuse and probably just gcs buckets in general are not very efficient when it comes to many small files (After extraction there are almost 300,000 files). I don't think buckets are going to be appropiate so I will look into using disks with DataFlow.

## DataFlow with Disks





