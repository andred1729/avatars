"""Herdora speech pipeline.

Generates a sarcastic response via OpenAI and vocalizes it with ElevenLabs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import requests

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[0] / "prompts" / "avatar_system_prompt.txt"
DEFAULT_USER_PROMPT = (
    "Share your bleakest assessment of the current code, then begrudgingly offer to help."
)

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


def _request_herdora_completion(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("HERDORA_API_KEY")
    if not api_key:
        raise EnvironmentError("Set HERDORA_API_KEY before calling talk().")

    base_url = os.getenv("HERDORA_BASE_URL", "https://pygmalion.herdora.com/v1")
    model = os.getenv("HERDORA_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(model=model, messages=messages)
        return (response.choices[0].message.content or "").strip()
    except ImportError:
        import openai  # type: ignore

        openai.api_key = api_key
        openai.api_base = base_url
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return (response["choices"][0]["message"]["content"] or "").strip()


def talk(
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> Dict[str, object]:
    """Generate Herdora's sarcastic take."""

    messages = _build_messages(user_prompt)
    script = _request_herdora_completion(messages)
    if not script:
        raise RuntimeError("Herdora returned an empty reply for Herdora's script.")

    return {
        "text": script,
    }