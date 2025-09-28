"""Herdora speech pipeline.

Generates a sarcastic response via OpenAI and vocalizes it with ElevenLabs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import requests

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "avatar_system_prompt.txt"
DEFAULT_USER_PROMPT = (
    "Share your bleakest assessment of the current code, then begrudgingly offer to help."
)
DEFAULT_OUTPUT_DIR = Path(os.getenv("HERDORA_OUTPUT_DIR", "output"))
DEFAULT_OUTPUT_FILE = os.getenv("HERDORA_OUTPUT_FILE", "herdora-latest.mp3")
DEFAULT_VOICE_SETTINGS: Dict[str, float | bool] = {
    "stability": 0.2,
    "similarity_boost": 0.8,
    "style": 0.65,
    "use_speaker_boost": True,
}

__all__ = ["talk"]


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing system prompt. Ensure prompts/avatar_system_prompt.txt exists."
        ) from exc


def _build_messages(user_prompt: str) -> list[dict[str, str]]:
    persona = _load_system_prompt()
    system_content = (
        f"{persona}\n\n"
        "Stay focused on code critique and deliver a single spoken reply Herdora can voice. "
        "Keep it under three sentences, weave in vocal cues like *sighs* or *mutters*, and end with "
        "a begrudging offer to assist."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]


def _request_openai_completion(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY before calling talk().")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages)
        return (response.choices[0].message.content or "").strip()
    except ImportError:
        import openai  # type: ignore

        openai.api_key = api_key
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return (response["choices"][0]["message"]["content"] or "").strip()


def _synthesize_speech(
    text: str,
    *,
    voice_settings: Optional[Dict[str, float | bool]] = None,
    output_dir: Path,
    output_file: str,
) -> Path:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ELEVENLABS_API_KEY before calling talk().")

    voice_id = "IKne3meq5aSn9XLyUdCD"
    if not voice_id:
        raise EnvironmentError(
            "Set ELEVENLABS_VOICE_ID to the ElevenLabs voice ID that best matches Herdora."
        )

    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")

    payload: Dict[str, object] = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {**DEFAULT_VOICE_SETTINGS, **(voice_settings or {})},
    }

    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
        "xi-api-key": api_key,
    }

    response = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(
            f"ElevenLabs request failed: {response.status_code} {response.reason} | {response.text}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / output_file
    with file_path.open("wb") as audio_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_file.write(chunk)

    return file_path


def talk(
    user_prompt: str = DEFAULT_USER_PROMPT,
    *,
    voice_settings: Optional[Dict[str, float | bool]] = None,
    output_dir: Optional[Path] = None,
    output_file: Optional[str] = None,
) -> Dict[str, object]:
    """Generate Herdora's sarcastic take and voice it via ElevenLabs."""

    messages = _build_messages(user_prompt)
    script = _request_openai_completion(messages)
    if not script:
        raise RuntimeError("OpenAI returned an empty reply for Herdora's script.")

    path = _synthesize_speech(
        script,
        voice_settings=voice_settings,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        output_file=output_file or DEFAULT_OUTPUT_FILE,
    )

    return {
        "file_path": str(path),
        "bytes": path.stat().st_size,
        "text": script,
    }
