---
title: "(Under Construction) Integrating Google Cloud Storage into my AI workflow"
categories:
  - Posts
tags:
  - TensorFlow
  - Machine Learning
  - Google Cloud
collections:
  - Reflection
excerpt: "While I like the affordability of Google Drive and the simplicity of mounting disks, this post discusses ways to use Google 
Cloud Storage."
toc: true
toc_sticky: true
---
> Note: The purpose of this post is as a personal reflection and not as a tutorial.

# Storage Pricing 

When doing machine learning with Google Colab I have been mounting my Google Drive and giving a folder from it as the data directory. This solution is not going to scale very well with large datasets or other places to run the code such as an AI Notebook so I am going to investigate using Google Cloud Storage.

I still want to integrate Google Drive into my workflow however because from what I have been able to see, Google Drive is more affordable for long term dataset storage as the price per month is comparable to coldline at $6.25 per TB with no data retrieval costs.

**Google Coud Storage pricing per month per terabyte**

![image1](/assets/images/2021-03-15-dataset-cloud-storage-solutions/image1.jpg)

**pricing per per terabyte to retreive the data**

![image2](/assets/images/2021-03-15-dataset-cloud-storage-solutions/image2.jpg)

# Moving A Folder from Google Drive to Google Cloud

The first step is to move a dataset stored in Google Drive to Google Cloud.

The process would go something like this:

* Allocate a VM with enough disk space for the dataset.
* Download the dataset from Google Drive
* Upload the dataset to Google Cloud Storage
While downloading and uploading could be done by visiting the sites, it would be better to automate this process by utilising the APIs.

There have been other posts by people regarding this very problem and a common stated solution is to use Colab to mount the Google Drive and then upload from there to cloud storage. However this method does still use the disk space of a Colab which is only 100GB so it could possibly break down for transferring a large dataset.


