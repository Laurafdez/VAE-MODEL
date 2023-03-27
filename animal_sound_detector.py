import librosa

import os
import librosa
import soundfile as sf
import numpy as np

import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import cv2
from numpy import *
import librosa
import soundfile as sf


def plot_sound_event_detection_result(framewise_output, sr, audio_file):
    """Visualization of sound event detection result. 
   This function detects the seconds of the audio where there is a higher probability of finding an animal sound. 
   As we want to extract both audio and images from that point. A line is added where this step is performed.

    Args:
        framewise_output (int): .
        sr (int): sampling frequency.
        audio_file: audio path to be analyzed
    """
    
    # Label that we want our audio to be more likely to be heard.
    target_label = 'Animal'
    target_label_index = np.where(np.array(labels) == target_label)[0][0]
    target_probs = framewise_output[:, target_label_index]
    

    # Extract the x values that have a y value greater than 0.6
    x_values = np.where(target_probs > 0.6)[0]


    # Extract the pieces of audio that are most likely to have audio of an animal.
    consecutives = []
    temp = [x_values[0]]

    for i in range(1, len(x_values)):
        if x_values[i] - x_values[i-1] == 1:
            temp.append(x_values[i])
        else:
            consecutives.append(temp)
            temp = [x_values[i]]

    consecutives.append(temp)

    durations = []
    for c in consecutives:
        first = c[0]
        last = c[-1]
        duration = len(c)
        
        secondinit = first
        secondfin = last
        duracion_segmento = secondfin - secondinit
        durations.append(duracion_segmento)

    indice_max_duracion = durations.index(max(durations))
    segmento_long = consecutives[indice_max_duracion]
    first_long = segmento_long[0]
    last_long = segmento_long[-1]
    
    secondinit_long = first_long
    secondfin_long = last_long
    duracion = secondfin_long - secondinit_long

    
    # Load the audio file
    audio, sr = librosa.core.load(audio_file, sr=None)

    # Define the start and end times of the segment of interest (in seconds).
    start_time = secondinit_long
    end_time = secondfin_long + duracion

    # Calculate sample rates for start and end times
    start_index = int(start_time * sr)
    end_index = int(end_time * sr)

    # Trimming audio
    audio_segment = audio[start_index:end_index]


    if len(audio_segment) > 0:
        # Get the original file's name and extension
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        file_extension = os.path.splitext(os.path.basename(audio_file))[1]
        output_folder = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_audios"
        image_output_folder="/media/byo/Data/MariaTFM/VIDEOANIMALES/Cropped_images"
        videos = "/media/byo/Data/MariaTFM/videoanimales/downloads/{}.mp4".format(file_name)


        # Concatenate the output folder, the original file's name, and extension
        output_filename = os.path.join(output_folder, f"{file_name}_cropped{file_extension}")

        # Saving the trimmed audio segment to the output file
        sf.write(output_filename, audio_segment, sr)

        cap = cv2.VideoCapture(videos)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        ret, frame = cap.read()
        cap.release()
        
        # Save the image in the output folder
        image_output_path = os.path.join(image_output_folder, f"{file_name}_cropped.jpg")
        cv2.imwrite(image_output_path, frame)

    else:
        print(f"The trimmed audio segment of the {audio_file} file has a length of 0, it will not be saved in the output file.")




# Path of the folder containing the audio files
audio_folder = "/media/byo/Data/MariaTFM/VIDEOANIMALES/Audios_of_Videos"  
audio_extensions = (".mp3", ".wav", ".mp4")  # add the extensions of the audio files you want to analyse

# Get a list of all audio files in the folder
audio_files = []
for file in os.listdir(audio_folder):
    if os.path.splitext(file)[1].lower() in audio_extensions:
        audio_files.append(os.path.join(audio_folder, file))

# Process each audio file in the list.
for audio_file in audio_files:
    try:
        y, sr = librosa.load(audio_file, sr=None)
        (audio, _) = librosa.core.load(audio_file, sr=sr, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        # Audio tagging
        at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(audio)

        # Sound event detection
        sed = SoundEventDetection(checkpoint_path=None, device='cuda')
        framewise_output = sed.inference(audio)

        print('Processing file:', audio_file)
        
        plot_sound_event_detection_result(framewise_output[0], sr, audio_file)
    except Exception as e:
        print(f"An exception occurred while processing the file {audio_file}: {str(e)}")
        continue

