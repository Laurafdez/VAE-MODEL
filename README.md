
VAE-MODEL 

2023

Laura Fernández Galindo & María Sánchez Ruiz

# Introduction

An important deep generative model that learns from untargeted data is the Variational Autoencoder (VAE). This model is formed by two neural networks: the encoder and the decoder. The flow that this model follows is that input data is mapped to the latent space by the encoder, and the decoder network maps the latent space back to the input data.

The particularity of this model is that the encoder network learns to produce a distribution over the latent space rather than a single point. This distribution usually is a Gaussian distribution with a mean and a standard deviation.

The model learns how to maximize the probability of the input data given the distribution over the latent space. This is achieved by minimizing a loss function that consists of two parts: the reconstruction loss and the regularization loss.

The VAE has many applications in a wide range of fields, such as images, texts, and audio. One of the main advantages of the VAE is that it allows for the generation of new data points that are similar to the training data.

## Our generative model

Firstly, we have been working on generating our own VAE. To do that, we needed to create our own dataset, preprocess it, and then train the VAE. In this repository, we will show all the steps we took to achieve that goal.

## Download the dataset

1. Firstly, we download the CSV from [VGGSOUND](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) As you can see in the CSV, all the identifiers of the YouTube videos we want to download are saved. As we only want to download videos that have animal sounds, we created a script that searches for these videos and downloads them from the internet. To execute that script, you should do:

     ```console
     python download_video.py
     ```
     >__Note__ It is important that you change the path where you are going to save the downloaded videos. Our first approach is to download videos of birds, cats, and dogs.

2. Secondly, we downloaded more animal sounds from [Freesound](https://freesound.org/) and some images from [Imagenet](https://www.image-net.org/). Our goal is to create a VAE that can generate an image from a sound. That's why we downloaded videos to have all this data.

## Preprocessing data

1. It is important that we separte the audio from the video. To do that, you should run:
     ```console
     python video_to_audio.py
     ```    
      >__Note__ It is important that you change the path where you are going to save of the new audio.
      
2. The next step is to extract the segments of our audios that have the highest likelihood of being an animal sound. By doing this, when we introduce all our audios to our VAE, they will have the segments with the least noise possible. Additionally, we want to extract the image of the video, so in this script, we take the image of the frame with the highest probability of being an animal. To do this, you should run:

     ```console
     python animal_sound_detector.py
     ``` 

     >__Note__ It is important that you change the path where you are going to save the new audio.
     
3. Lastly, we take a segment of 3 seconds from the new animal audio, and we convert all of that data in spectograms. 
     ```console
     python Preprocess_audio_VAE.py
     ``` 

     >__Note__ It is important that you change all the paths.
     
 ## Generation of data
 
- Now we have all our data preprocessed, we introduce our path into our VAE. Our VAE starts to train in each epoch, and when the process ends, you can see a generative audio from the latent space. 
     ```console
     python VAE.py
     ``` 

     >__Note__ It is important that you change all the paths. Also, it is important that you create a file where the model is going to be saved as "model.pth" so you can train your own model with your own database.

## References

Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
Doersch, C. (2016). Tutorial on Variational Autoencoders. arXiv preprint arXiv:1606.05908.
Kulkarni, T. D., Whitney, W. F., Kohli, P., & Tenenbaum, J. B. (2015). Deep convolutional inverse graphics network. In Advances in neural information processing systems (pp. 2539-2547).
Gregor, K., Danihelka, I., Graves, A., Rezende, D., & Wierstra, D. (2015). DRAW: A recurrent neural network for image generation. arXiv preprint arXiv:1502.04623.
Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2016). Generating sentences from a continuous space. arXiv preprint arXiv:1511.06349.
