# Install: pip install TTS
from TTS.api import TTS
import pygame

# Use a better model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph")

def coqui_advanced(text):
    # Add some basic SSML-like control
    tts.tts_to_file(text=text, file_path="output.wav")
    
    pygame.mixer.init()
    pygame.mixer.music.load("output.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

coqui_advanced("This sounds much more natural than pyttsx3")