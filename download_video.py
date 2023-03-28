import pandas as pd
#import re
df = pd.read_csv("./vggsound.csv")
from pytube import YouTube
import moviepy.editor as mp
#import librosa
#import numpy as np

import time
from pytube.exceptions import VideoPrivate
from pytube.exceptions import VideoUnavailable
from pytube.exceptions import RegexMatchError


filtered_dog = df[df.iloc[:, 2].str.contains('dog')]
filtered_cat = df[df.iloc[:, 2].str.contains(r'\bcat\b ')]
filtered_bird = df[df.iloc[:, 2].str.contains('bird')]
dataset = pd.concat([filtered_dog, filtered_cat, filtered_bird])



def download_one_material(yt,i,category):
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    stream.download(output_path='videoanimales\\downloads\\', filename='video_{}_{}.mp4'.format(category,i))
    print(f"Video {i} downloaded successfully!")
    time.sleep(10)


def download_material_category(material,category):
    i=0
    for id in material.iloc[:, 0]:
      try:
        i+=1
        yt = YouTube('https://www.youtube.com/watch?v='+ id) 
        
        download_one_material(yt,i,category)
        
      except VideoPrivate:
        print(f"El video con el identificador {id} es privado y no se puede descargar.")
        continue
        time.sleep(10)
      except VideoUnavailable:
        print(f"El video con el identificador {id} ya no esta disponible.")
        continue
        time.sleep(10)
    
      except (RegexMatchError, AttributeError, KeyError):
        print(f"El video con el identificador {id} no es valido.")
        continue
        time.sleep(10)
        
      except Exception:
          attempt = 0
          while (attempt<3):
            try: 
                  print(f"Error en la descarga, repetir {attempt}/3 y saltar.")
                  download_one_material(yt,i,category)
                  attempt=3
                  
            except Exception:
                  attempt+=1

          continue





download_material_category(filtered_dog,'dog')
download_material_category(filtered_cat,'cat')
download_material_category(filtered_bird,'bird')
