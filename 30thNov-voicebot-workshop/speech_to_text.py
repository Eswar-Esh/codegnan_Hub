import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import numpy as np

def record_audio(filename="input.wav", duration=5, fs=44100):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio_int16 = np.int16(audio * 32767)
    write(filename, fs, audio_int16)
    print(f"Audio saved as {filename}")

def speech_to_text(filename="input.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I could not understand."
        except sr.RequestError:
            return "API unavailable"

if __name__ == "__main__":
    record_audio()
    print("You said:", speech_to_text())
