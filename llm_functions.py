"""LLM-accessible helper functions.

This module exposes callable tools the language model may invoke. Each tool is
registered with both a JSON schema (for the model) and a Python implementation
(for the runtime). The goal is to keep this registry small and auditable so more
capabilities can be layered in safely over time (e.g. Tavus video synthesis).
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import tempfile

import requests

# ---------------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisteredFunction:
    """Defines a function that can be surfaced to the LLM."""

    name: str
    schema: Mapping[str, Any]
    implementation: Callable[..., Dict[str, Any]]


_REGISTERED_FUNCTIONS: Dict[str, RegisteredFunction] = {}


def register_function(definition: RegisteredFunction) -> None:
    """Add a function to the registry, guarding against duplicate names."""

    if definition.name in _REGISTERED_FUNCTIONS:
        raise ValueError(f"Function '{definition.name}' is already registered.")
    _REGISTERED_FUNCTIONS[definition.name] = definition


def function_schemas() -> Iterable[Mapping[str, Any]]:
    """Return JSON schemas for all registered functions (for the LLM)."""

    return (definition.schema for definition in _REGISTERED_FUNCTIONS.values())


def call_registered_function(name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Invoke a registered function with sanitized keyword arguments."""

    try:
        definition = _REGISTERED_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Function '{name}' is not registered.") from exc

    impl = definition.implementation
    impl_signature = impl.__code__.co_varnames[: impl.__code__.co_argcount]

    filtered_kwargs: Dict[str, Any] = {
        key: value for key, value in arguments.items() if key in impl_signature
    }
    return impl(**filtered_kwargs)


# ---------------------------------------------------------------------------
# speech(...) implementation
# ---------------------------------------------------------------------------


_TEMP_AUDIO_DIR = tempfile.TemporaryDirectory(prefix="herdora_audio_")
_AUDIO_OUTPUT_DIR = Path(_TEMP_AUDIO_DIR.name)


@atexit.register
def _cleanup_temp_audio_dir() -> None:
    """Ensure temporary audio artifacts are removed when the process exits."""

    try:
        _TEMP_AUDIO_DIR.cleanup()
    except Exception:
        # Avoid masking the shutdown sequence if cleanup fails for any reason.
        pass


def _build_voice_settings(
    *,
    stability: float | None,
    similarity_boost: float | None,
    style: float | None,
) -> Dict[str, Any]:
    """Compose the ElevenLabs voice settings payload."""

    voice_settings: Dict[str, Any] = {}
    if stability is not None:
        voice_settings["stability"] = stability
    if similarity_boost is not None:
        voice_settings["similarity_boost"] = similarity_boost
    if style is not None:
        voice_settings["style"] = style
    return voice_settings


def _play_audio(filepath: Path) -> None:
    """Attempt to play audio on the current platform."""

    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", str(filepath)], check=True)
        elif sys.platform.startswith("linux"):
            subprocess.run(["aplay", str(filepath)], check=True)
        elif sys.platform.startswith("win"):
            subprocess.run(["powershell", "-Command", f"(New-Object Media.SoundPlayer '{filepath}')"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Failing to play audio should not break the overall flow.
        pass


def speech(
    text: str,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    playback: bool = True,
) -> Dict[str, Any]:
    """Convert expressive text (plain or SSML) into speech via ElevenLabs.

    Parameters align with ElevenLabs' current REST API. Only ``text`` is
    required; everything else piggybacks off environment defaults so the LLM
    can focus on crafting nuanced delivery.
    """

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ELEVENLABS_API_KEY before calling speech().")

    resolved_voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "pqHfZKP75CvOlQylNhV4")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{resolved_voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    payload: Dict[str, Any] = {
        "text": text,
    }

    voice_settings = _build_voice_settings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
    )
    if voice_settings:
        payload["voice_settings"] = voice_settings

    candidate_models: List[str]
    if model_id is not None:
        candidate_models = [model_id]
    else:
        candidate_models = []
        env_model = "eleven_v3"
        candidate_models.append(env_model)


    resolved_model_id = None
    response = None
    tried_models: set[str] = set()
    for candidate in candidate_models:
        if candidate in tried_models or not candidate:
            continue
        tried_models.add(candidate)

        payload["model_id"] = candidate
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        if response.ok:
            resolved_model_id = candidate
            break

        status = response.status_code
        # Gracefully fall back to older models if the candidate is unsupported.
        if status in {400, 404, 422}:
            response.close()
            continue

        response.raise_for_status()

    if response is None:
        raise RuntimeError("Failed to contact ElevenLabs text-to-speech service.")

    if not response.ok:
        response.raise_for_status()

    if resolved_model_id is None:
        resolved_model_id = payload.get("model_id", "")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    audio_path = _AUDIO_OUTPUT_DIR / f"speech_{timestamp}.mp3"

    with audio_path.open("wb") as audio_file:
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            audio_file.write(chunk)

    if playback:
        _play_audio(audio_path)

    return {
        "type": "speech",
        "text": text,
        "audio_path": str(audio_path),
        "voice_id": resolved_voice_id,
        "model_id": resolved_model_id,
    }


# Register speech() so the LLM can call it.
register_function(
    RegisteredFunction(
        name="speech",
        schema={
            "name": "speech",
            "description": (
                "Convert Herdora's expressive script into voiced audio via ElevenLabs. "
                "Supports plain text or SSML with <break> and <phoneme> tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The script to vocalize. Use bracketed audio tags like [sighs] "
                            "for expressions, plus optional SSML <break> tags and ARPAbet phonemes."
                        ),
                    },
                    "voice_id": {
                        "type": "string",
                        "description": "Override the default ElevenLabs voice ID (optional).",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Override the ElevenLabs model (optional).",
                    },
                    "stability": {
                        "type": "number",
                        "description": "0.0-1.0 stability control for delivery (optional).",
                    },
                    "similarity_boost": {
                        "type": "number",
                        "description": "0.0-1.0 timbre similarity to the reference voice (optional).",
                    },
                    "style": {
                        "type": "number",
                        "description": "0.0-1.0 style intensity for voices that support it (optional).",
                    },
                    "playback": {
                        "type": "boolean",
                        "description": "Whether to play the audio immediately on the server (default true).",
                    },
                },
                "required": ["text"],
            },
        },
        implementation=speech,
    )
)
