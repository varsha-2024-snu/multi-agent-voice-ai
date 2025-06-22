import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import requests
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.clients.listen import PrerecordedOptions
import re

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables")

dg_client = DeepgramClient(DEEPGRAM_API_KEY)

def trans_file(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio:
            buffer = audio.read()

        response = dg_client.listen.prerecorded.v("1").transcribe_file(
            {
                "buffer": buffer,
                "mimetype": "audio/wav"
            },
            PrerecordedOptions(
                model="nova",
                language="en-US",
                smart_format=True
            )
        )
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

def clean_text_for_tts(text: str) -> str:
    # Remove markdown and extra whitespace
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # remove **bold**
    text = re.sub(r'\n+', ' ', text)              # replace newlines with space
    text = re.sub(r'\s+', ' ', text)              # collapse multiple spaces
    return text.strip()

def record_and_trans(duration=5, sample_rate=16000):
    print("Listening... Speak now.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, sample_rate, recording)
        audio_file_path = f.name

    transcript = trans_file(audio_file_path)
    os.unlink(audio_file_path)
    print("You said:", transcript)
    return transcript

def text_to_speech(text, agent="default"):
    text = clean_text_for_tts(text)
    voice_map = {
    "optimist": "aura-thalia-en",
    "realist": "aura-apollo-en",
    "default": "aura-helena-en"
}

    voice = voice_map.get(agent, "aura-2-helena-en")
    url = f"https://api.deepgram.com/v1/speak?model={voice}"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"TTS Error: {response.text}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(response.content)
        f.flush()
        audio = AudioSegment.from_mp3(f.name)
        play(audio)
