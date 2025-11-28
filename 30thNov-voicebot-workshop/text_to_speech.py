# section3_text_to_speech.py
from gtts import gTTS
import playsound

def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    playsound.playsound(filename)

if __name__ == "__main__":
    text_to_speech("Hello! This is your voice assistant speaking.")
