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
chunks_folder = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audios"
# Set the path to the directory where the spectrograms will be saved
save_path = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audio_npy"
# Set the path to the directory where the min/max values will be saved
minmax_save_path = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audio_minmax"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# >> PREPROCESS <<
# Function to normalize an audio signal between 0, 1
def normalize(signal):
    if (signal.max() - signal.min()) == 0:
        return np.zeros(signal.shape)
    return (signal - signal.min()) / (signal.max() - signal.min())
    
# Function to denormalize an audio signal
def denormalize(signal, max_val, min_val):
    return signal * (max_val - min_val) + min_val

# Loop over all the files in the directory
for dirpath, dirname, filenames in os.walk(chunks_folder):
    for f in tqdm(filenames):
        # Load the audio file and resample to a sampling rate of 48 kHz
        audio, sr = librosa.load(os.path.join(dirpath, f), sr=48_000)

        # Fix the length of the audio to 48k x 3 (porque queremos audio de 3secs) samples
        audio = librosa.util.fix_length(audio, size=int(sr*3))

        # Normalize the audio signal
        audio = librosa.util.normalize(audio)

        # Compute the short-time Fourier transform (STFT) of the audio
        stft = librosa.stft(audio, n_fft=1024, hop_length=512)
        if np.sum(np.isnan(audio)) > 0:
            print('caca')

        # Convert the STFT to a log-magnitude spectrogram
        log_spectrogram = librosa.amplitude_to_db(np.abs(stft))

        # Normalize the log-magnitude spectrogram between 0 and 1
        spec_norm = normalize(log_spectrogram)
        if np.sum(np.isnan(spec_norm)) > 0:
            print('caca')
        # Save the normalized spectrogram to a .npy file
        file_name = f.split('.wav')[0]
        np.save(os.path.join(save_path, file_name + '.npy'), spec_norm)

        # Compute the min/max values of the log-magnitude spectrogram
        min_max = [np.min(log_spectrogram), np.max(log_spectrogram)]

        # Save the min/max values to a .npy file
        np.save(os.path.join(minmax_save_path, file_name + '.npy'), min_max)
