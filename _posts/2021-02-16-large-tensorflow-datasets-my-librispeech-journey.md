---
title: "Large Tensorflow Datasets - My LibriSpeech Journey"
categories:
  - Posts
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

I actually found an issue on github where someone had made the exact same change as me but their pull request had many other unrelated changes so no one had reviewed it.

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
pip3 install --upgrade virtualenv \--user
python3 -m virtualenv env
source env/bin/activate
```

**Installing Apache Beam and tensorflow-datasets**
```
pip3 install apache-beam[gcp]
pip3 install tensorflow-datasets
```

**Create Storage Bucket**
```
gsutil mb gs://general-304503
```

**Set parameters**
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

After mounting I found that I could not interact with the files in the bucket. I needed to change the storage API permissions in the compute engine from Read Only to Full:

![image2](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image2.jpg)

After getting the mounting to work I attempted to copy the downloaded dataset to the bucket however it was extremely slow (Estimated to take 48h). I think this is because the gcsfuse and probably just gcs buckets in general are not very efficient when it comes to many small files (After extraction there are almost 300,000 files). I looked briefly into using disks with DataFlow but I am not sure how to mount the same disk to different workers and in fact the documentation seems to suggest this is not possible.

> Persistent disk resources
The Dataflow service is currently limited to 15 persistent disks per worker instance when running a streaming job. Each persistent disk is local to an individual Compute Engine virtual machine. Your job may not have more workers than persistent disks; a 1:1 ratio between workers and disks is the minimum resource allotment.

Instead I am going to see what I can do to optimise gcsfuse.

## Using gsutil to upload to a bucket
In my research to see if there was some option for more efficient upload in gcsfuse, I realised that I could perform the task with ```gsutil cp```.
I copy the code from the documentation given here: [https://cloud.google.com/storage/docs/gsutil/commands/cp](https://cloud.google.com/storage/docs/gsutil/commands/cp).

Use the -r option to copy an entire directory tree. For example, to upload the directory tree dir:

```
gsutil cp -r dir gs://my-bucket
```
If you have a large number of files to transfer, you can perform a parallel multi-threaded/multi-processing copy using the top-level gsutil -m option (see gsutil help options):

```
gsutil -m cp -r dir gs://my-bucket
```
I executed the second command to upload the downloaded dataset from the disk to the bucket and it took about 20m so this is much better.

## Peforming the generation on DataFlow
I first tried imitating the DataFlow tutorial where all commands were executed from the console but I found that I could not install tensorflow-datasets there. I discovered that the DataFlow tutorial can actually also be exectuted from a virtual machine with API permissions so going forward thats what I am using.

My first attempt yielded this exception:

![image3](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image3.jpg)

I realised that this meant that I needed to define a region for the DataFlow execution. This wasn't in the pipeline options in the [tensorflow apache guide](https://www.tensorflow.org/datasets/beam_datasets) but was in the dataflow tutorial.

**Here is the change that needs to be made to the `tfds build` command to define the region:**
```
tfds build $DATASET_NAME \
--data_dir=$GCS_BUCKET/tensorflow_datasets \
--beam_pipeline_options=\
"runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,"\
"staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,"\
"requirements_file=/tmp/beam_requirements.txt,region=us-central1"
```

Finally it is about to create the job on GCP, however about 16m into the execution there is an exception. Here is the graph where the blocks that failed are visible:

![image4](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image4.jpg)

This is the exception that occured:

![image5](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image5.jpg)

Before I start trying to debug this, I will check if installing tfds-nightly instead of tensorflow-datasets will fix the problem. That will involve executing these commands:

```
pip3 uninstall tensorflow-datasets
pip3 install tfds-nightly
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt
```

I however get this error:
> ERROR: Could not find a version that satisfies the requirement tfds-nightly[librispeech]

**What I am going to try is installing the entirety of tfds-nightly on the workers.**
```
echo "tfds-nightly" > /tmp/beam_requirements.txt
```

Still returns errors. I think what is happening is that the tfds command is not updated when I uninstall tensorflow-datasets and install tfds-nightly. So I need to look at doing this without the tfds command.

I found instructions to do this in the [ReadMe for Google Research text-to-text transfer transformer](https://github.com/google-research/text-to-text-transfer-transformer#c4).

```
pip install tfds-nightly[c4]

echo 'tfds-nightly[c4]' > /tmp/beam_requirements.txt

python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=c4/en \
  --data_dir=gs://$MY_BUCKET/tensorflow_datasets \
  --beam_pipeline_options="project=$MY_PROJECT,job_name=c4,staging_location=gs://$MY_BUCKET/binaries,temp_location=gs://$MY_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,experiments=shuffle_mode=service,region=$MY_REGION"
```
They used a different dataset but it can be modified to fit my use.
```
pip3 install tfds-nightly[$DATASET_NAME]

echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt

python -m tensorflow_datasets.scripts.download_and_prepare \
  --datasets=$DATASET_NAME \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options="project=$GCP_PROJECT,job_name=test2,staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,runner=DataflowRunner,requirements_file=/tmp/beam_requirements.txt,region=us-central1"
```

It is still failing. I now think that is is the actual installation of tfds-nightly on the workers that is failing. This is what it is trying to execute.
```
INFO[stager.py]: Executing command: ['/home/jsjsrobert500/env/bin/python', '-m', 'pip', 'download', '--dest', '/tmp/dataflow-requirements-cache', '-r', '/tmp/beam_requirements.txt', '--exists-action', 'i', '--no-bin
ary', ':all:']
```
I test the corresponding command in a terminal.
```
echo "tfds-nightly[$DATASET_NAME]" > /tmp/beam_requirements.txt

/home/jsjsrobert500/env/bin/python -m pip download --dest /tmp/dataflow-requirements-cache -r /tmp/beam_requirements.txt --exists-action i --no-binary :all:
```
If the beam requirements is `tensorflow-datasets` there is not problem but with the nightly we get these errors:

![image6](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image6.jpg)

Now it is easier to see what is happening. Clearly pip is iterating through the nightly releases and rejecting each of them for some reason. Here is a full printout of one of the warnings:
```
Using cached tfds-nightly-4.2.0.dev202102220106.tar.gz (3.1 MB)
  WARNING: Generating metadata for package tfds-nightly produced metadata for project name tensorflow-datasets. Fix your #egg=tfds-nightly fragments.
WARNING: Discarding https://files.pythonhosted.org/packages/89/71/fa9a318e54c55bc8cfa7c2dad06d0f592c00e25ce17064f8ed01b1db4b27/tfds-nightly-4.2.0.dev202102220106.tar.gz#sha256=1a6eb0e1e9647dffaaa6c83d380807c1f47d69e
9d242c333e42d4304f000c1d3 (from https://pypi.org/simple/tfds-nightly/) (requires-python:>=3.6). Requested tensorflow-datasets from https://files.pythonhosted.org/packages/89/71/fa9a318e54c55bc8cfa7c2dad06d0f592c00e2
5ce17064f8ed01b1db4b27/tfds-nightly-4.2.0.dev202102220106.tar.gz#sha256=1a6eb0e1e9647dffaaa6c83d380807c1f47d69e9d242c333e42d4304f000c1d3 (from -r /tmp/beam_requirements.txt (line 1)) has inconsistent name: filename 
has 'tfds-nightly', but metadata has 'tensorflow-datasets'
```

I found a [github issue](https://github.com/tensorflow/datasets/issues/2827) where someone mentioned a similar problem involving metadata. One of the suggested solutions was downgrading pip to a version lower than 20.
```
/home/jsjsrobert500/env/bin/python -m pip install pip==19.0.1
```
![image7](/assets/images/2021-02-16-large-tensorflow-datasets-my-librispeech-journey/image7.jpg)

After doing this pip no longer rejects all of the nightly packages.

However the problem where the DataFlow job crashes after about 15m is still present.

# Issue 4

After making the change on my fork of tensorflow datasets. I put the clone command I usually use to install packages from git in the requirements:
```
echo "git+https://github.com/Jaidon-Smith/datasets.git" > /tmp/beam_requirements.txt
```
However after running this and checking the logs I realised that the workers can not use git to install so I will have to explore another way of allowing them to obtain the package.

[This post](https://adamj.eu/tech/2019/03/11/pip-install-from-a-git-repository/) gives instructions for installing from a tarball so that git does not have to be installed.

**Installing via tarballs**

An alternative that avoids Git is to install from a tarball URL, that the major hosted Git solutions provide, for example:

```
# GitHub
python -m pip install https://github.com/django/django/archive/45dfb3641aa4d9828a7c5448d11aa67c7cbd7966.tar.gz
# GitLab
python -m pip install https://gitlab.com/pycqa/flake8/-/archive/3.7.7/flake8-3.7.7.tar.gz
# Bitbucket
python -m pip install https://bitbucket.org/hpk42/tox/get/2.3.1.tar.gz
```

I looked into creating these tarballs but then I found that github automatically creates them. Here is the pip command to install from the tarball.
```
python -m pip install https://github.com/Jaidon-Smith/datasets/archive/master.tar.gz
```
Here is the corresponding command to create the requirements.
```
echo "https://github.com/Jaidon-Smith/datasets/archive/master.tar.gz" > /tmp/beam_requirements.txt
```



