"""Herdora speech pipeline with function-calling support."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from llm_functions import call_registered_function, function_schemas

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


def _build_messages(user_prompt: str) -> List[Dict[str, Any]]:
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


_STAGE_DIRECTION_PATTERN = re.compile(r"\*(.*?)\*")
_CODE_BLOCK_PATTERN = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?\s*(.*?)```", re.DOTALL)


_AUDIO_TAG_ALIASES: Dict[str, str] = {
    "sigh": "[sighs]",
    "sighs": "[sighs]",
    "sighs wearily": "[sighs]",
    "sigh heavily": "[sighs]",
    "long sigh": "[sighs]",
    "exhale": "[exhales]",
    "exhales": "[exhales]",
    "laugh": "[laughs]",
    "laughs": "[laughs]",
    "laughing": "[laughs]",
    "chuckles": "[laughs]",
    "giggles": "[laughs]",
    "laughs harder": "[laughs harder]",
    "starts laughing": "[starts laughing]",
    "whisper": "[whispers]",
    "whispers": "[whispers]",
    "whispering": "[whispers]",
    "mutters": "[sarcastic]",
    "mutters darkly": "[sarcastic]",
    "mutters under breath": "[sarcastic]",
    "groans": "[dramatically]",
    "groans theatrically": "[dramatically]",
    "dramatic gasp": "[dramatically]",
    "gasps": "[dramatically]",
    "happy gasp": "[happy gasp]",
    "snorts": "[snorts]",
    "sarcastic": "[sarcastic]",
    "excited": "[excited]",
    "curious": "[curious]",
    "crying": "[crying]",
    "weeps": "[crying]",
    "teary": "[crying]",
    "mischievous": "[mischievously]",
    "mischievously": "[mischievously]",
    "warmly": "[warmly]",
    "delighted": "[delighted]",
    "impressed": "[impressed]",
    "dramatically": "[dramatically]",
}


def _stage_direction_to_audio_tag(direction: str) -> str:
    """Map Herdora's stage directions to ElevenLabs v3 audio tags."""

    collapsed = " ".join(direction.split())
    normalized = collapsed.lower()
    if not collapsed or not any(char.isalpha() for char in collapsed):
        return "[sighs]"

    if normalized in _AUDIO_TAG_ALIASES:
        return _AUDIO_TAG_ALIASES[normalized]

    for alias, tag in _AUDIO_TAG_ALIASES.items():
        if normalized.startswith(alias):
            return tag

    return f"[{collapsed}]"


def _normalize_stage_direction(match: re.Match[str]) -> str:
    """Convert asterisk cues into ElevenLabs-compatible audio tags."""

    inner = match.group(1)
    return _stage_direction_to_audio_tag(inner.strip())


def _normalize_script(script: str) -> str:
    """Normalize Herdora's cues so ElevenLabs v3 picks up expressive tags."""

    normalized = _STAGE_DIRECTION_PATTERN.sub(_normalize_stage_direction, script)
    normalized = re.sub(r"(?<!\S)\*(?!\S)", "[sighs]", normalized)
    normalized = re.sub(r"\s+", lambda m: " " if "\n" not in m.group(0) else m.group(0), normalized)
    return normalized.strip()


def _extract_executable_code(raw_text: str) -> tuple[Optional[str], bool]:
    """Pull the first executable code block (or treat entire text as code)."""

    candidate: Optional[str] = None
    block_match = _CODE_BLOCK_PATTERN.search(raw_text)
    found_code_block = bool(block_match)

    if block_match:
        candidate = block_match.group(1).strip()
    else:
        candidate = raw_text.strip()

    if not candidate:
        return None, found_code_block

    try:
        compile(candidate, "<herdora_generated>", "exec")
    except SyntaxError:
        return None, found_code_block

    return candidate, found_code_block


def _format_speech_invocation(arguments: Mapping[str, Any]) -> str:
    """Render a deterministic speech(...) invocation as executable code."""

    parameter_order = (
        "text",
        "voice_id",
        "model_id",
        "stability",
        "similarity_boost",
        "style",
        "playback",
    )

    lines: List[str] = ["speech("]
    for name in parameter_order:
        if name not in arguments:
            continue
        value = arguments[name]
        if isinstance(value, str):
            value_repr = json.dumps(value)
        else:
            value_repr = repr(value)
        lines.append(f"    {name}={value_repr},")
    lines.append(")")
    return "\n".join(lines)


def _execute_generated_code(code: str, actions: List[Dict[str, Any]]) -> bool:
    """Execute Herdora's emitted script while tracking speech actions.

    Returns True when the generated code invoked speech(); otherwise False.
    """

    import llm_functions  # Imported lazily to avoid circular dependencies

    def speech_wrapper(
        text: str,
        *,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        style: Optional[float] = None,
        playback: bool = True,
    ) -> Dict[str, Any]:
        raw_arguments: Dict[str, Any] = {
            "text": text,
            "voice_id": voice_id,
            "model_id": model_id,
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "playback": playback,
        }

        sanitized_arguments = {
            key: value for key, value in raw_arguments.items() if value is not None or key == "playback"
        }
        sanitized_arguments["text"] = _normalize_script(str(sanitized_arguments["text"]))

        result = call_registered_function("speech", sanitized_arguments)
        actions.append(
            {
                "name": "speech",
                "arguments": sanitized_arguments,
                "result": result,
            }
        )
        return result

    original_speech = llm_functions.speech
    llm_functions.speech = speech_wrapper  # type: ignore[assignment]

    execution_globals: Dict[str, Any] = {
        "__name__": "__main__",
        "speech": speech_wrapper,
    }

    starting_len = len(actions)
    invoked_speech = False

    try:
        exec(code, execution_globals, {})
        new_actions = actions[starting_len:]
        invoked_speech = any(action.get("name") == "speech" for action in new_actions)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("Failed to execute Herdora's generated script.") from exc
    finally:
        llm_functions.speech = original_speech  # type: ignore[assignment]

    return invoked_speech


def _request_herdora_completion(
    messages: List[Dict[str, Any]],
    *,
    allow_functions: bool,
    allowed_functions: Optional[Iterable[str]] = None,
) -> Any:
    api_key = os.getenv("HERDORA_API_KEY")
    if not api_key:
        raise EnvironmentError("Set HERDORA_API_KEY before calling talk().")

    base_url = os.getenv("HERDORA_BASE_URL", "https://pygmalion.herdora.com/v1")
    model = os.getenv("HERDORA_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if allow_functions:
            schemas = list(function_schemas())
            if allowed_functions is not None:
                allowed_set = {name for name in allowed_functions}
                schemas = [schema for schema in schemas if schema.get("name") in allowed_set]
            if schemas:
                kwargs["functions"] = schemas
                kwargs["function_call"] = "auto"
            else:
                kwargs["function_call"] = "none"
        else:
            kwargs["function_call"] = "none"
        return client.chat.completions.create(**kwargs)
    except ImportError:
        import openai  # type: ignore

        openai.api_key = api_key
        openai.api_base = base_url
        kwargs = {"model": model, "messages": messages}
        if allow_functions:
            schemas = list(function_schemas())
            if allowed_functions is not None:
                allowed_set = {name for name in allowed_functions}
                schemas = [schema for schema in schemas if schema.get("name") in allowed_set]
            if schemas:
                kwargs["functions"] = schemas
                kwargs["function_call"] = "auto"
            else:
                kwargs["function_call"] = "none"
        else:
            kwargs["function_call"] = "none"
        return openai.ChatCompletion.create(**kwargs)


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """Normalize OpenAI SDK message objects into plain dictionaries."""

    if isinstance(message, dict):
        return message

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):  # OpenAI 1.x (pydantic models)
        return model_dump()

    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _extract_content(message_dict: Dict[str, Any]) -> str:
    """Pull text content from the message payload."""

    content = message_dict.get("content", "")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(text_parts)
    return str(content or "")


def _maybe_get_function_call(message_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    call = message_dict.get("function_call") or message_dict.get("tool_calls")
    if isinstance(call, list):
        return call[0] if call else None
    return call


def _first_choice(response: Any) -> Any:
    if isinstance(response, dict):
        return response["choices"][0]
    return response.choices[0]


def talk(
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> Dict[str, Any]:
    """Generate Herdora's sarcastic take using function-calling tools."""

    messages: List[Dict[str, Any]] = _build_messages(user_prompt)
    actions: List[Dict[str, Any]] = []

    first_response = _request_herdora_completion(
        messages,
        allow_functions=True,
        allowed_functions=("speech",),
    )
    first_choice = _first_choice(first_response)
    first_message_payload = first_choice["message"] if isinstance(first_choice, dict) else first_choice.message
    first_message = _message_to_dict(first_message_payload)

    fn_call = _maybe_get_function_call(first_message)
    if fn_call:
        function_name = fn_call.get("name")
        raw_arguments = fn_call.get("arguments") or "{}"
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed_arguments = {}

        if function_name:
            sanitized_arguments = dict(parsed_arguments)
            if "text" in sanitized_arguments:
                sanitized_arguments["text"] = _normalize_script(str(sanitized_arguments["text"]))

            function_result = call_registered_function(function_name, sanitized_arguments)
            actions.append(
                {
                    "name": function_name,
                    "arguments": sanitized_arguments,
                    "result": function_result,
                }
            )

            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": json.dumps(sanitized_arguments),
                        },
                    },
                    {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_result),
                    },
                ]
            )

            second_response = _request_herdora_completion(messages, allow_functions=False)
            second_choice = _first_choice(second_response)
            second_message_payload = (
                second_choice["message"] if isinstance(second_choice, dict) else second_choice.message
            )
            second_message = _message_to_dict(second_message_payload)
            final_text = _extract_content(second_message)
        else:
            final_text = _extract_content(first_message)
    else:
        final_text = _extract_content(first_message)

    final_text = final_text.strip()

    code_snippet, had_code_block = _extract_executable_code(final_text)
    fallback_reason: Optional[str] = None

    if code_snippet:
        if _execute_generated_code(code_snippet, actions):
            return {
                "text": code_snippet,
                "actions": actions,
            }
        fallback_reason = "Generated script executed without invoking speech."

    if had_code_block and not code_snippet:
        raise RuntimeError("Generated code block was not runnable.")

    normalized_text = _normalize_script(final_text)
    if not normalized_text:
        normalized_text = "Herdora had nothing to add beyond snarling into the void."

    existing_speech = next((action for action in actions if action.get("name") == "speech"), None)
    if existing_speech:
        rendered_script = _format_speech_invocation(existing_speech["arguments"])
        return {
            "text": rendered_script,
            "actions": actions,
        }

    fallback_text = normalized_text
    if fallback_reason:
        summary = normalized_text or "I wound up speechless."
        fallback_text = (
            f"*sighs wearily* {fallback_reason} So here's a begrudging recap: {summary}"
        )
    if not fallback_text.startswith("*"):
        fallback_text = f"*sighs wearily* {fallback_text}"
    fallback_text = _normalize_script(fallback_text)
    if not fallback_text:
        fallback_text = "*sighs wearily* I wound up speechless."

    fallback_arguments: Dict[str, Any] = {
        "text": fallback_text,
        "stability": 0.25,
        "similarity_boost": 0.6,
    }
    fallback_script = _format_speech_invocation(fallback_arguments)
    _execute_generated_code(fallback_script, actions)

    return {
        "text": fallback_script,
        "actions": actions,
    }
