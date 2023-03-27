import os
import librosa
import soundfile as sf
import numpy as np
from moviepy.editor import *
import cv2

import os
import librosa
import soundfile as sf
import numpy as np
from moviepy.editor import *
import cv2

# Path to the folder with video files
video_folder = '/media/byo/Data/MariaTFM/videoanimales/downloads'

# Path to the folder where the audio files will be stored
audio_folder = '/media/byo/Data/MariaTFM/VIDEOANIMALES/Audios_of_Videos'

# Create the audio folder if it does not exist
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# Iterating through the files in the video folder
for file_name in os.listdir(video_folder):
    full_file_path = os.path.join(video_folder, file_name)
    
    # Verify that the file is a video file
    if os.path.isfile(full_file_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov')):
       # Check that video files are not empty
        if os.path.getsize(full_file_path) > 0: 
            # Extract audio from video
            audio = AudioFileClip(full_file_path)
            
            # Check that the audio file is not empty
            if audio.duration != 0:
                # Create output path and save the audio file
                audio_file_path = os.path.join(audio_folder, file_name.replace(os.path.splitext(file_name)[-1], '.mp3'))
                audio.write_audiofile(audio_file_path)
