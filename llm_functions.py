"""Functions that the LLM can call."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
import requests

def speak(text: str) -> None:
    """Vocalizes the given text using ElevenLabs API and plays it."""

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ELEVENLABS_API_KEY before calling talk().")

    voice_id = "pqHfZKP75CvOlQylNhV4"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5,
        },
    }

    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()

    audio_file = Path("speech.mp3")
    with open(audio_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    try:
        subprocess.run(["afplay", str(audio_file)], check=True)
    finally:
        audio_file.unlink()
