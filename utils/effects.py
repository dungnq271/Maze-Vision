import os
import glob
from random import randint
import cv2
from PIL import Image
import numpy as np
import pygame

images_dir = 'images/'
audio_dir = 'audio/'
images = os.listdir(images_dir)


def add_effect(w, h):
    idx = randint(0, len(images) - 1)
    file = images[idx]
    image = np.array(Image.open(images_dir + file).convert('RGB'))
    image = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
    audio = audio_dir + file.split('.')[0] + '.mp3'
    return image, audio


def play_audio(audio):
    pygame.mixer.init()
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()


