#Este VAE está inspirado en el el VAE de Mateo
import os
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import base64
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#import sounddevice as sd
import plotly.graph_objects as go
import soundfile as sf

# >> USER SPECIFIC PATHS <<
# Set the path to the directory containing the audio files
chunks_folder = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audio_npy"
# Set the path to the directory where the spectrograms will be saved
save_path = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audio_npy"
# Set the path to the directory where the min/max values will be saved
minmax_save_path = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audio_minmax"
# Set the path to the directory where a trained model is stored
model_strockes = "./media/byo/Data/MariaTFM/audioanimales/downloads/model/newmodel.pth"
model_dog ="/media/byo/Data/MariaTFM/audioanimales/model_checkpoint_best_val_loss.pt"
# Set the path to the directory where points in the latent space are stored
latent_spaces_path = "/media/byo/Data/MariaTFM/audioanimales/downloads/model/new_latent_spaces.npy"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# >> VAE <<
# ENCODER
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalEncoder, self).__init__()

        # define convolutional layers for encoding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(5, 5), stride=(2, 2))
        self.batch1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.batch2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
        self.batch4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1))

        # define distribution parameters
        self.mu = nn.Linear(32 * 31 * 3, latent_dim)  # map from flattened output to latent_dim
        self.var = nn.Linear(32 * 31 * 3, latent_dim)  # map from flattened output to latent_dim
        self.N = torch.distributions.Normal(0, 1)  # normal distribution for sampling z
        self.N.loc = self.N.loc.cuda()  # move loc parameter to GPU for sampling
        self.N.scale = self.N.scale.cuda()  # move scale parameter to GPU for sampling
        self.kl = 0  # initialize KL divergence to 0

    def forward(self, x):
        x = x.to(device)  # move input to GPU
        x = F.relu(self.batch1(self.conv1(x)))  # apply convolutional layer, batch norm, and ReLU activation
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.batch4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)  # flatten output for linear layers

        mu = self.mu(x)  # map flattened output to latent mean
        sigma = torch.exp(self.var(x))  # map flattened output to latent standard deviation
        z = mu + sigma*self.N.sample(mu.shape)  # sample z from normal distribution using reparametrization trick
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()  # compute KL divergence between Q(z|x) and N(0,1)

        return z
        
# DECODER
class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        # linear layer to map latent code to 3D tensor
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * 31 * 3),
            nn.ReLU(True)
        )

        # unflatten 3D tensor to 4D tensor
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 31, 3))

        # transposed convolutional layers to gradually increase spatial resolution
        self.dec1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(64)
        self.dec2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        self.batch2 = nn.BatchNorm2d(128)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), output_padding=1)
        self.batch3 = nn.BatchNorm2d(256)
        self.dec4 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), output_padding=1)
        self.batch4 = nn.BatchNorm2d(512)
        self.dec5 = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=(5, 5), stride=(2, 2), output_padding=1)

    def forward(self, x):
        # linear layer to map latent code to 3D tensor
        x = self.decoder_lin(x)
        # unflatten 3D tensor to 4D tensor
        x = self.unflatten(x)
        # transposed convolutional layers to gradually increase spatial resolution
        x = F.relu(self.batch1(self.dec1(x)))
        x = F.relu(self.batch2(self.dec2(x)))
        x = F.relu(self.batch3(self.dec3(x)))
        x = F.relu(self.batch4(self.dec4(x)))
        x = self.dec5(x)
        # apply sigmoid activation function to ensure output is between 0 and 1
        x = torch.sigmoid(x)
        return x

# VARIATIONAL AUTOENCODER
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

# TRAIN VAE
# Set default dtype and random seed
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
# Set hyperparameters
lr = 1e-4
beta = 0.00001
num_epochs = 5000
latent_dim=20
# Set up model and optimizer
vae = VariationalAutoencoder(latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
# Set up device
vae.to(device)
# Set up train and validation data loaders
dogs = []
for dirpath, dirname, filenames in os.walk(save_path):
    for f in filenames:
        dogs.append(np.load(os.path.join(dirpath, f), allow_pickle=True))


#dogs = [np.nan_to_num(dog, nan=0) for dog in dogs]
inputs = dogs
#print(inputs[0].shape)
inputs = [i[np.newaxis, :512, :64].astype('float32') for i in inputs]
#print(inputs[0].shape)
for i in inputs:
    print(i)
x_train, x_test = train_test_split(inputs, shuffle=True, test_size=0.2)

train_loader = DataLoader(x_train, batch_size=2, shuffle=True)
valid_loader = DataLoader(x_test, batch_size=2, shuffle=True)
#valid_loader = train_loader

# Define training and validation loops
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode dataloaderfor both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    reconstruction_error_epoch = 0.0
    kl_error_epoch = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x in tqdm(dataloader):
        # Move tensor to the proper device
        x = x.to(device)
        
        x_hat = vae(x)

            


        # Evaluate loss
        reconstruction_error = ((torch.abs(x - x_hat)) ** 2).sum()
        kl_error = vae.encoder.kl.sum() * beta
        loss = reconstruction_error + kl_error
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update loss values
        reconstruction_error_epoch += reconstruction_error.item()
        kl_error_epoch += kl_error.item()
        train_loss += loss.item()

    # Compute mean loss values
    return train_loss / len(dataloader.dataset), reconstruction_error_epoch / len(dataloader.dataset), kl_error_epoch / len(dataloader.dataset)


def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Decode data
            x_hat = vae(x)
            # loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss = ((torch.abs(x - x_hat)) ** 2).sum() + vae.encoder.kl.sum() * beta
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def train(num_epochs, vae, device, train_loader, optimizer, valid_loader):
    best_val_loss=float('inf')
    for epoch in range(num_epochs):
        train_loss, reconstruction_error, kl_error = train_epoch(vae, device, train_loader, optimizer)
        val_loss = test_epoch(vae, device, train_loader)
        print('\n EPOCH {}/{} \t train loss {:.3f} (recon error {:.3f}, kl error {:.3f}) \t val loss {:.3f}'.format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                reconstruction_error,
                                                                                                                kl_error,
                                                                                                                 val_loss))

        if val_loss < best_val_loss:
            # save the model's state when the validation loss is better than the training loss
            checkpoint_path = "./checkpoints/dogs_vNewaudios.pt"
            torch.save(vae.state_dict(), checkpoint_path)
            print('model saved')
            best_val_loss = val_loss

# Sample and decode using spectrogram
def sample_and_decode_vae(vae, latent_dim, n_samples=1, sr=48_000, hop_length=512, n_fft=1024):
    # load the pre-trained weights into the model
    vae.load_state_dict(torch.load("./model_checkpoint_best_val_loss.pt"))
    vae.load_state_dict(torch.load(model_dog))

    # set the model to evaluation mode
    vae.eval()
    # Sample noise vectors from a normal distribution
    spectogram_path = "/media/byo/Data/MariaTFM/VIDEOANIMALES/audio"
    
     # Set up train and validation data loaders
    dog = []
    for dirpath, dirname, filenames in os.walk(spectogram_path):
        for f in filenames:
            dog.append(np.load(os.path.join(dirpath, f), allow_pickle=True))


    #dogs = [np.nan_to_num(dog, nan=0) for dog in dogs]
    input = dog
    #print(inputs[0].shape)
    input = [i[np.newaxis, :512, :64].astype('float32') for i in input]
    #print(inputs[0].shape)
    for i in input:
        print(i)
   
    input = np.array(input)
    input = torch.from_numpy(input)
    decoded = vae(input)
    decoded = decoded.cpu()


    # Unnormalize the data
    decoded = denormalize(decoded, 10, -80)
    decoded = decoded.cpu().detach().numpy()

    
    # Reshape the data to be a spectrogram
    decoded = decoded[0, 0, ...]
    
    librosa.display.specshow(decoded, sr=sr, hop_length=512, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-scaled Spectrogram')
    plt.show()


    # Use Griffin-Lim algorithm to reconstruct the audio signal from the magnitude spectrogram
    
    audio = []
    for i in range(n_samples):
        audio_signal = librosa.griffinlim(decoded, hop_length=hop_length)
        audio_signal = librosa.util.normalize(audio_signal, norm=np.inf)
        audio.append(audio_signal)

    return audio

# Sample and decode using latent space
def sample_and_decode_vae2(vae, latent_dim, n_samples=1, sr=48_000, hop_length=512, n_fft=1024):
    # Sample noise vectors from a normal distribution
    noise = np.random.normal(size=(n_samples, latent_dim)).astype('float32')
    
   # latent = torch.randn(128, 4, device=device)

    # reconstruct images from the latent vectors
    #decoded = vae(latent)
    # Use the decoder to generate audio signals from the noise vectors
    decoded = vae.decoder(torch.from_numpy(noise).to(device))
    decoded = decoded.cpu()

    # Unnormalize the data
    decoded = denormalize(decoded, 10, -80)
    decoded = decoded.cpu().detach().numpy()
    
    # Reshape the data to be a spectrogram
    decoded = decoded[0, 0, ...]
    
    librosa.display.specshow(decoded, sr=sr, hop_length=512, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-scaled Spectrogram')
    plt.show()

    # Use Griffin-Lim algorithm to reconstruct the audio signal from the magnitude spectrogram
    
    audio = []
    for i in range(n_samples):
        audio_signal = librosa.griffinlim(decoded, hop_length=hop_length)
        audio_signal = librosa.util.normalize(audio_signal, norm=np.inf)
        audio.append(audio_signal)

    return audio

def normalize(signal):
    return (signal - signal.min()) / (signal.max() - signal.min())
    
# Function to denormalize an audio signal
def denormalize(signal, max_val, min_val):
    return signal * (max_val - min_val) + min_val

#Para coger del espacio latente:

if __name__ == "__main__":
    #train(num_epochs, vae, device, train_loader, optimizer, valid_loader)

    # decode an audio using encoder-decoder
    sr=48000
    audio=sample_and_decode_vae2(vae, latent_dim, n_samples=1, sr=sr, hop_length=512, n_fft=1024)

    for i, audio_signal in enumerate(audio):
       filename = f"dog{i}.wav"
       sf.write(filename, audio_signal, sr, format='WAV', subtype='PCM_24')
    
    sr=48000
    audio1=sample_and_decode_vae(vae, latent_dim, n_samples=1, sr=sr, hop_length=512, n_fft=1024)

    for i, audio_signal in enumerate(audio1):
       filename = f"animal_{i}.wav"
       sf.write(filename, audio_signal, sr, format='WAV', subtype='PCM_24')
