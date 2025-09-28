"""Herdora speech pipeline.

Generates a sarcastic response via Herdora (OpenAI-compatible) or OpenAI,
then vocalizes it with ElevenLabs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List, Union

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

# Herdora / OpenAI defaults
DEFAULT_HERDORA_BASE_URL = os.getenv("HERDORA_BASE_URL", "https://pygmalion.herdora.com/v1")
DEFAULT_HERDORA_MODEL = os.getenv("HERDORA_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

__all__ = ["talk"]


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing system prompt. Ensure prompts/avatar_system_prompt.txt exists."
        ) from exc


def _build_messages(user_prompt: str, image_url: Optional[str] = None) -> List[dict]:
    """Build OpenAI-style messages. If image_url is provided, make multimodal."""
    persona = _load_system_prompt()
    system_content = (
        f"{persona}\n\n"
        "Stay focused on code critique and deliver a single spoken reply Herdora can voice. "
        "Keep it under three sentences, weave in vocal cues like *sighs* or *mutters*, and end with "
        "a begrudging offer to assist."
    )

    user_content: Union[str, List[dict]]
    if image_url:
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        user_content = user_prompt

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _request_openai_completion(messages: List[dict]) -> str:
    """Route to Herdora if HERDORA_API_KEY is set; otherwise use OpenAI.

    Herdora path requires `openai>=1.0.0` (the `OpenAI` client import).
    """
    herdora_key = os.getenv("HERDORA_API_KEY")
    if herdora_key:
        # --- Herdora (OpenAI-compatible) path ---
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Herdora route requires `openai` >= 1.0.0. Install with `pip install -U openai`."
            ) from exc

        client = OpenAI(base_url=DEFAULT_HERDORA_BASE_URL, api_key=herdora_key)
        model = DEFAULT_HERDORA_MODEL

        resp = client.chat.completions.create(model=model, messages=messages)
        return (resp.choices[0].message.content or "").strip()

    # --- Fallback: regular OpenAI path (supports your previous envs) ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set HERDORA_API_KEY (preferred) or OPENAI_API_KEY before calling talk()."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Prefer the modern client if available
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=messages)
        return (resp.choices[0].message.content or "").strip()
    except ImportError:
        # Legacy SDK fallback
        import openai  # type: ignore

        openai.api_key = api_key
        resp = openai.ChatCompletion.create(model=model, messages=messages)
        return (resp["choices"][0]["message"]["content"] or "").strip()


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

    # Hardcoded default Herdora voice; override via env if you like
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "IKne3meq5aSn9XLyUdCD")
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
    image_url: Optional[str] = None,
    voice_settings: Optional[Dict[str, float | bool]] = None,
    output_dir: Optional[Path] = None,
    output_file: Optional[str] = None,
) -> Dict[str, object]:
    """Generate Herdora's sarcastic take and voice it via ElevenLabs.

    If HERDORA_API_KEY is set, calls the Herdora gateway (OpenAI-compatible).
    Otherwise, uses standard OpenAI credentials.

    Args:
        user_prompt: Main user text prompt.
        image_url: Optional URL for a vision-enabled prompt (multimodal).
        voice_settings: Optional ElevenLabs voice overrides.
        output_dir: Where to write the MP3 (defaults to HERDORA_OUTPUT_DIR or ./output).
        output_file: MP3 filename (defaults to HERDORA_OUTPUT_FILE or 'herdora-latest.mp3').
    """
    messages = _build_messages(user_prompt, image_url=image_url)
    script = _request_openai_completion(messages)
    if not script:
        raise RuntimeError("The model returned an empty reply for Herdora's script.")

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
