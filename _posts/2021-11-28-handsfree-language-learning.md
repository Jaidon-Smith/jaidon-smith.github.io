---
title: "Hands-Free Language Learning App"
categories:
  - Posts
tags:
  - Google Cloud Platform
  - Twilio
collections:
  - Reflection
excerpt: "My work on a hands-free app for learning foreign words.
"
toc: true
toc_sticky: true
---
# Introduction
Foreign language learning has been a hobby of mine for some time. While there are a lot of tools and apps for doing this, most of these require regular physical interaction with a device and access to a screen. Increasingly I have wished that I could so some of this learning during activities where it is impracticle to interact with a device or view a screen (Such as exercising or driving).

In response to this problem many companies and individuals put out lessons to teach basic grammer or vocablulary in the form of audio files. In my own experience I haven't found these to be all that effective. I would attribute that to these main reasons:
* The lesson is fixed and cannot be adapted depending on whether you get a question correct or incorrect. This can mean the lesson can spend too much time on something you already know, or go too fast through something you are less familiar with. Also if the lessons employ some kind of SRS intending you to do a lesson once a day, this gets broken if you take a long break.
* The course content is fixed. This might not be such an issue for some, but is not ideal if you are the kind of person used to the flexibility of Anki allowing you to define the vocabulary or phrases from your favorite media that you want to learn.

# Language Listening Project
My first attempt to build something useful to me for language learning while exercising I called 'Language Listening'.
Essentially you would provide to the program a list of English and Foreign Word pairs and it would produce and audio file that:
* Used Text to Speech to say the english word and then say the foreign word twice afterwards.
* The order of the words would be randomised.
* There would be music in the background.

One weekend I prepared an audio file of about 2h of words and then left the house to exercise, jogging and walking for about 5h.
I ended up playing parts of the track multiple times and had these thoughts:
* Without requireing user interaction, playing the file has limited long term usefullness for language learning. Eventually I found myself just beginning to ignore it.
* Once you've played it a few times you start to get used to some of the word orderings and it doesn't feel random anymore.

This confirmed to me that the second iteration of this should employ some kind of speech recognition to be more interactive.

# Hands-Free Foreign Vocabulary Learning
This is the plan for the next iteration of this idea.

* The user provides a list of English and Foreign Word pairs and begins a session.
* The app says the english word.
* The user has limited time to say the Foreign pronounication.
* Speech recognition used to provide feedback to the user.
* Uses an SRS system, so if the word is determined to be correct, it will take longer to appear again.

Repeat these steps for the next word.

