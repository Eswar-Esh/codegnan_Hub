import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
from dotenv import load_dotenv
import requests
import numpy as np
import time
import tempfile
import webbrowser
from pytube import Search   # pip install pytube
import vlc                  # pip install python-vlc

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_RECORD_DURATION = 8
MIC_TIMEOUT = 1.5
MIC_PHRASE_TIME_LIMIT = 10


# ---------------- AUDIO CAPTURE ----------------
def capture_audio_with_sounddevice(duration=DEFAULT_RECORD_DURATION, fs=44100):
    print(f"🎤 (Fallback) Recording {duration}s")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    return sr.AudioData(audio_int16.tobytes(), fs, 2)


def capture_audio_with_microphone(recognizer, microphone):
    print("🎤 Listening...")
    try:
        with microphone as source:
            audio_data = recognizer.listen(source, timeout=MIC_TIMEOUT, phrase_time_limit=MIC_PHRASE_TIME_LIMIT)
        return audio_data
    except sr.WaitTimeoutError:
        return None


# ---------------- STT ----------------
def speech_to_text(recognizer, audio_data):
    try:
        return recognizer.recognize_google(audio_data)
    except:
        return None


# ---------------- GEMINI ----------------
def get_ai_response(user_text):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Reply in 2 short lines.\nUser: {user_text}"}],
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        candidates = data.get("candidates", [])
        for c in candidates:
            text = c["content"]["parts"][0]["text"]
            return text
        return "I couldn't understand that."
    except Exception as e:
        return f"Error: {e}"


# ---------------- TTS ----------------
def text_to_speech(text):
    print("🔊 Speaking:", text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        path = tmp.name
    tts = gTTS(text=text, lang="en")
    tts.save(path)
    playsound.playsound(path)
    os.remove(path)


# ---------------- SYSTEM COMMANDS ----------------

def play_youtube_song(song_name):
    print("🎵 Searching YouTube for:", song_name)
    
    search = Search(song_name)
    result = search.results[0]  # first result
    
    video_url = f"https://www.youtube.com/watch?v={result.video_id}"
    print("▶️ Playing:", video_url)

    webbrowser.open(video_url)

    return f"Playing {song_name} from YouTube!"


def handle_commands(user_text):
    text = user_text.lower()

    # ✅ ---- Added Shutdown Command ----
    if "shutdown" in text:
        return "shutdown_now"
    # -----------------------------------

    # 1. Play song from YouTube
    if "play" in text and "song" in text:
        parts = text.split("play")
        if len(parts) > 1:
            song_name = parts[1].strip()
            if song_name == "" or song_name == "a song":
                song_name = "popular music"
        else:
            song_name = "music"

        reply = play_youtube_song(song_name)
        return reply

    # 2. Open website
    if "open" in text and "website" in text:
        words = text.split()
        for w in words:
            if ".com" in w or ".in" in w or ".org" in w:
                webbrowser.open(f"https://{w}")
                return f"Opening {w}"

        webbrowser.open("https://google.com")
        return "Opening Google."

    return None


# ---------------- MAIN LOOP ----------------
def chat_loop():
    recognizer = sr.Recognizer()

    try:
        microphone = sr.Microphone()
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
        print("🎧 Microphone ready!")
    except:
        microphone = None
        print("⚠ No microphone, fallback mode.")

    print("🤖 Assistant ready. Say 'shutdown' to exit.")

    while True:
        if microphone:
            audio_data = capture_audio_with_microphone(recognizer, microphone)
            if audio_data is None:
                continue
        else:
            audio_data = capture_audio_with_sounddevice()

        user_text = speech_to_text(recognizer, audio_data)

        if not user_text:
            print("Didn't catch that...")
            continue

        print("🗣️ You said:", user_text)

        # ---- Check system commands first ----
        command_reply = handle_commands(user_text)

        # ✅ ---- Shutdown Logic Added ----
        if command_reply == "shutdown_now":
            text_to_speech("Shutting down now. Goodbye!")
            break
        # -----------------------------------

        if command_reply:
            text_to_speech(command_reply)
            continue

        # ---- Otherwise go to Gemini ----
        bot_reply = get_ai_response(user_text)
        text_to_speech(bot_reply)


if __name__ == "__main__":
    chat_loop()
