---
title: "AIKaraoke"
categories:
  - Posts
tags:
  - Machine Learning
  - Google Cloud Platform
collections:
  - Reflection
excerpt: "Some deployment details regarding AIKaraoke, a site that transforms any YouTube video into a karaoke version with on-screen lyrics and backing track.
"
toc: true
toc_sticky: true
---

AIKaraoke is a site that transforms any YouTube video into a karaoke version with on-screen lyrics and backing track.
The site is available here: [https://jaidon-smith.github.io/portfolio/AIKaraoke/](https://jaidon-smith.github.io/portfolio/AIKaraoke/).

This post is focused on the cloud infrastructure rather than the machine learning algorithms. 

When hosting this site I wanted an infrastructure that would allow for automatic scaling to 0 when there are no users and automatic scaling up for many users. It would require a cloud platform to achieve the necessary optimisation of resources and therefore I used Google Cloud Platform for this task.

# The Infrastructure

* The frontend uses App Engine where I run Django in a standard Python environment. In the future I would consider swapping this out for React or Angular and using a Javascript runtime, this would allow me to create a more visually appealing frontend.

* The AI processing happens in Cloud Run. This is great for resource use optimisation and scaling to 0 in that machines only need to exist if there is a job to do. This design choice also allowed me to install linux command line tools that are not available in a standard App Engine runtime.

* Cloud Run is evoked by Cloud Task. This means that the processing happens asynchronously, or that the user doesn't have to wait for the response that the processing is done. It happens in the background and the video page becomes active once the processing is finished.

* At the moment there is no need for a database so I have just been using Google Cloud Storage for file storage. I may user firestore if I want a NoSQL database in the future.

