
VAE-MODEL 

2023

Laura Fernández Galindo & María Sánchez Ruiz

# Introduction

VAE is form by an autoencoder and a decoder, the articularity of this generative model is that the encoding distribution is regularised during the training in this way 
the model is able to generate data from that space. 
## Our generative model

First of all, we have in working in the generation of our own VAE to do that we have to create our own dataset  preprocessing and train the VAE. In this repository we
are going to show all the steps we have made to achive that goal.

## Download the dataset

1. Firstly, we download the CSV from [VGGSOUND](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) 
As you can see in the CSV is saved all the identifiers of the Youtube video we want to download, as we want only to download videos that have any sound of animal,
we have creat a script that search for this videos and download for internet. To execute that script yo should do:

     ```console
     python download_video.py
     ```
     >__Note__ It is important that you change the path where you are going to save of the download videos. Our first aprouch is to download videos from birds, cat and dogs.

2. Secondly, we download more animal sound from [Freesound](https://freesound.org/) and some image from [Imagenet](https://www.image-net.org/). Our goal is to create a VAE that is able to generate an image from a sound and that 
why we download videos to have all this data.

## Preprocessing data

1. It is important that we separte audio from the video, to do that you should run:
     ```console
     python video_to_audio.py
     ```    
      >__Note__ It is important that you change the path where you are going to save of the new audio.
2. The next step is to extract the segment of our audios that have the mayor likelihood of being a Animal. And in this way when we introduce to our our vae all our audios is going
to have the extrac of audios that have the less noise as posible. Additionaly, we want to extract the image of the video so in this script we take the image of the frame with more
probability to be an animal. To do this you should run:

     ```console
     python animal_sound_detector.py
     ``` 

     >__Note__ It is important that you change the path where you are going to save of the new audio.
     
3. Lastly, we take a segment of 3 seconds from the new animal audio and we convert all of that data in spectograms. 
     ```console
     python Preprocess_audio_VAE.py
     ``` 

     >__Note__ It is important that you change all the paths.
     
 ## Generation of data
 
- Now we have all our data preprocess we introduce our path in a our VAE. Our VAE start to train in each epoch and when the process end you can see a generative audio
from the latent space. 
     ```console
     python VAE.py
     ``` 

     >__Note__ It is important that you change all the paths. Also, is important that you create a file where the model is going to be save as model.pth so can train your
     own model with your own database

## Referencias

