"""Herdora speech pipeline with simplified function calling and lightweight memory."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
import os
import re
import textwrap
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from llm_functions import call_registered_function, function_schemas
from write import write as rewrite_code

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "avatar_system_prompt.txt"
DEFAULT_USER_PROMPT = (
    "Share your bleakest assessment of the current code, then begrudgingly offer to help."
)
_MAX_TOOL_ITERATIONS = 8

_LOCALHOST_IMAGE_DIR = Path(
    os.getenv("LOCALHOST_IMAGE_DIR", str(BASE_DIR / "runtime" / "views"))
)
_LOCALHOST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

_SUPPORTED_IMAGE_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing system prompt. Ensure prompts/avatar_system_prompt.txt exists."
        ) from exc


def _describe_registered_functions() -> str:
    """Render a human-friendly summary of the function registry."""

    descriptions: List[str] = []
    for schema in function_schemas():
        name = str(schema.get("name", "unknown"))
        description = str(schema.get("description", "")).strip()
        parameters = schema.get("parameters", {}) if isinstance(schema, Mapping) else {}
        properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
        param_list = ", ".join(properties.keys()) if properties else ""
        if param_list:
            descriptions.append(f"- {name}({param_list}) — {description}")
        else:
            descriptions.append(f"- {name} — {description}")
    return "\n".join(descriptions)


_STAGE_DIRECTION_PATTERN = re.compile(r"\*(.*?)\*")
_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


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


class ConversationMemory:
    """Tracks recent dialogue and a rolling summary for the local session."""

    def __init__(self, *, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._turns: List[tuple[str, str]] = []
        self._summary: Optional[str] = None
        self._lock = threading.Lock()
        self._last_image_hash: Optional[str] = None
        self._last_improved_code: Optional[str] = None

    def conversation_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            messages: List[Dict[str, str]] = []
            if self._summary:
                messages.append({
                    "role": "system",
                    "content": f"Conversation summary for context:\n{self._summary}",
                })
            for user_text, assistant_text in self._turns:
                if user_text:
                    messages.append({"role": "user", "content": user_text})
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
            return messages

    def summary_source_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            messages: List[Dict[str, str]] = []
            if self._summary:
                messages.append({
                    "role": "system",
                    "content": f"Existing summary context:\n{self._summary}",
                })
            for user_text, assistant_text in self._turns:
                if user_text:
                    messages.append({"role": "user", "content": user_text})
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
            return messages

    def should_summarize(self) -> bool:
        with self._lock:
            return len(self._turns) >= self._max_turns

    def has_pending_history(self) -> bool:
        with self._lock:
            return bool(self._turns)

    def last_image_hash(self) -> Optional[str]:
        with self._lock:
            return self._last_image_hash

    def last_improved_code(self) -> Optional[str]:
        with self._lock:
            return self._last_improved_code

    def record_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        improved_code: Optional[str] = None,
        image_hash: Optional[str] = None,
    ) -> None:
        sanitized_user = user_text.strip()
        sanitized_assistant = assistant_text.strip()
        with self._lock:
            self._turns.append((sanitized_user, sanitized_assistant))
            if improved_code:
                normalized_code = improved_code.strip()
                if normalized_code:
                    self._last_improved_code = normalized_code
            if image_hash:
                self._last_image_hash = image_hash

    def apply_summary(self, summary_text: str) -> None:
        cleaned_summary = summary_text.strip()
        with self._lock:
            if cleaned_summary:
                self._summary = cleaned_summary
            self._turns.clear()


_CONVERSATION_MEMORY = ConversationMemory()


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
    """Normalize Herdora's cues so ElevenLabs picks up expressive tags."""

    normalized = _STAGE_DIRECTION_PATTERN.sub(_normalize_stage_direction, script)
    normalized = re.sub(r"(?<!\S)\*(?!\S)", "[sighs]", normalized)
    normalized = re.sub(r"\s+", lambda m: " " if "\n" not in m.group(0) else m.group(0), normalized)
    return normalized.strip()


def _latest_localhost_image() -> Optional[Path]:
    try:
        candidates: List[Path] = []
        for suffix in _SUPPORTED_IMAGE_MIME:
            candidates.extend(_LOCALHOST_IMAGE_DIR.glob(f"*{suffix}"))
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    except FileNotFoundError:
        return None


def _load_localhost_snapshot() -> Optional[Tuple[Dict[str, Any], str, str]]:
    path = _latest_localhost_image()
    if not path:
        return None

    mime = _SUPPORTED_IMAGE_MIME.get(path.suffix.lower())
    if not mime:
        guessed_mime, _ = mimetypes.guess_type(path.name)
        mime = guessed_mime or "image/png"

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None

    data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    image_hash = hashlib.sha256(data).hexdigest()
    payload: Dict[str, Any] = {
        "type": "image_url",
        "image_url": {
            "url": data_url,
            "detail": "high",
        },
    }
    return payload, image_hash, path.name


def _prepare_submission_text(
    user_prompt: str,
    memory: ConversationMemory,
    snapshot_descriptor: Optional[str],
) -> str:
    submission = user_prompt.strip()
    last_output = memory.last_improved_code()
    note_lines: List[str] = []

    if last_output and submission and submission == last_output.strip():
        note_lines.append(
            "Note: Submission matches your previous Output. Treat it as reference unless new requests appear."
        )
        submission = ""

    if snapshot_descriptor:
        note_lines.append(snapshot_descriptor)

    if submission:
        if note_lines:
            note_lines.insert(0, f"Submission:\n{submission}")
        else:
            note_lines.append(f"Submission:\n{submission}")
    elif note_lines:
        pass
    else:
        note_lines.append("Submission is empty; relying on visual context only.")

    return "\n\n".join(note_lines).strip()


def _extract_python_script(response_text: str) -> Optional[str]:
    """Pull the first Python code block (or inline script) from the model reply."""

    match = _CODE_BLOCK_PATTERN.search(response_text)
    if match:
        return textwrap.dedent(match.group(1)).strip()

    if "speech(" in response_text:
        return response_text.strip()

    return None


def _sanitize_script_lines(script: str) -> str:
    """Strip leading comment markers so we can parse commented-out speech calls."""

    sanitized_lines: List[str] = []
    for raw_line in script.splitlines():
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            sanitized_lines.append(stripped.lstrip("#").lstrip())
        else:
            sanitized_lines.append(raw_line)
    return textwrap.dedent("\n".join(sanitized_lines)).strip()


def _extract_speech_kwargs(script: str) -> Optional[Dict[str, Any]]:
    """Parse a script and return kwargs for speech(...) if present."""

    sanitized_script = _sanitize_script_lines(script)
    if not sanitized_script:
        return None

    try:
        tree = ast.parse(sanitized_script, mode="exec")
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        else:
            continue

        if func_name != "speech":
            continue

        call_kwargs: Dict[str, Any] = {}

        if node.args:
            try:
                call_kwargs.setdefault("text", ast.literal_eval(node.args[0]))
            except Exception:
                return None

        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                call_kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            except Exception:
                return None

        text_value = call_kwargs.get("text")
        if isinstance(text_value, str):
            return call_kwargs

    return None


def _normalize_improved_code(candidate: str) -> str:
    """Strip markdown fences, surrounding quotes, and escaped newlines."""

    text = (candidate or "").strip()
    if not text:
        return ""

    block_match = _CODE_BLOCK_PATTERN.search(text)
    if block_match:
        text = block_match.group(1).strip()

    if text[:3] in {'"""', "'''"} and text[-3:] == text[:3]:
        text = text[3:-3]
    elif text.startswith(('"', "'")) and text.endswith(('"', "'")) and len(text) >= 2:
        try:
            text = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            text = text[1:-1]

    text = text.replace("\\r\n", "\n").replace("\\n", "\n").replace("\r\n", "\n")
    return text.strip()


def _extract_improved_from_actions(actions: Optional[List[Dict[str, Any]]]) -> str:
    if not actions:
        return ""
    for action in reversed(actions):
        if action.get("name") != "write":
            continue
        for container_key in ("arguments", "result"):
            container = action.get(container_key)
            if isinstance(container, Mapping):
                raw_code = container.get("improved_code") or container.get("text")
                if isinstance(raw_code, str):
                    normalized = _normalize_improved_code(raw_code)
                    if normalized:
                        return normalized
    return ""


def _build_messages(
    submission_text: str,
    memory_messages: List[Dict[str, str]],
    image_payload: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    persona = _load_system_prompt()
    system_lines = [
        persona,
        "Stay focused on code critique and deliver a single spoken reply Herdora can voice.",
        (
            "Always respond directly to the user's Submission field. The Output panel shows"
            " your prior work—treat matching snippets as your own and avoid repeating them."
        ),
        "Keep it under three sentences, weave in vocal cues like [sighs] or [mutters], and end with a begrudging offer to assist.",
        "Use the attached localhost visual only as supplemental context; rely on the Submission text for precise code changes.",
    ]

    function_list = _describe_registered_functions()
    if function_list:
        system_lines.append("Available tools you can call directly:")
        system_lines.append(function_list)
        system_lines.append("Call a tool when you need it instead of writing manual boilerplate.")

    messages: List[Dict[str, Any]] = [{"role": "system", "content": "\n\n".join(system_lines)}]
    if memory_messages:
        messages.extend(memory_messages)
    user_content_parts: List[Dict[str, Any]] = []
    if submission_text:
        user_content_parts.append({"type": "text", "text": submission_text})
    if image_payload:
        user_content_parts.append(image_payload)

    if not user_content_parts:
        user_content: Any = "Submission was empty."
    elif len(user_content_parts) == 1 and user_content_parts[0].get("type") == "text":
        user_content = user_content_parts[0]["text"]
    else:
        user_content = user_content_parts

    messages.append({"role": "user", "content": user_content})
    return messages


def _request_herdora_completion(
    messages: List[Dict[str, Any]],
    *,
    allow_functions: bool = True,
) -> Any:
    api_key = os.getenv("HERDORA_API_KEY")
    if not api_key:
        raise EnvironmentError("Set HERDORA_API_KEY before calling talk().")

    base_url = os.getenv("HERDORA_BASE_URL", "https://pygmalion.herdora.com/v1")
    model = os.getenv("HERDORA_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")

    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if allow_functions:
        schemas = list(function_schemas())
        if schemas:
            kwargs["functions"] = schemas
            kwargs["function_call"] = "auto"
        else:
            kwargs["function_call"] = "none"
    else:
        kwargs["function_call"] = "none"

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        return client.chat.completions.create(**kwargs)
    except ImportError:
        import openai  # type: ignore

        openai.api_key = api_key
        openai.api_base = base_url
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


def _extract_tool_calls(message_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls_field = message_dict.get("tool_calls")
    if isinstance(tool_calls_field, list) and tool_calls_field:
        normalized_calls: List[Dict[str, Any]] = []
        for call in tool_calls_field:
            if not isinstance(call, Mapping):
                continue
            function_payload = call.get("function", {})
            if not isinstance(function_payload, Mapping):
                continue
            normalized_calls.append(
                {
                    "id": call.get("id"),
                    "name": function_payload.get("name"),
                    "arguments": function_payload.get("arguments", {}),
                }
            )
        return normalized_calls

    function_call = message_dict.get("function_call")
    if isinstance(function_call, Mapping):
        return [
            {
                "id": None,
                "name": function_call.get("name"),
                "arguments": function_call.get("arguments", {}),
            }
        ]
    return []


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return {}


def _invoke_tool(
    name: Optional[str],
    arguments: Dict[str, Any],
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not name:
        return {}

    sanitized_arguments = dict(arguments)
    if name == "speech" and "text" in sanitized_arguments:
        sanitized_arguments["text"] = _normalize_script(str(sanitized_arguments["text"]))

    result = call_registered_function(name, sanitized_arguments)
    actions.append({"name": name, "arguments": sanitized_arguments, "result": result})
    return result


def _chat_with_tools(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Iteratively query the model, executing tools until a final reply arrives."""

    working_messages = list(messages)
    actions: List[Dict[str, Any]] = []

    for _ in range(_MAX_TOOL_ITERATIONS):
        response = _request_herdora_completion(working_messages, allow_functions=True)
        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        message_payload = choice["message"] if isinstance(choice, dict) else choice.message
        message_dict = _message_to_dict(message_payload)

        tool_calls = _extract_tool_calls(message_dict)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name")
                parsed_arguments = _parse_arguments(call.get("arguments"))
                result = _invoke_tool(name, parsed_arguments, actions)

                assistant_tool_message: Dict[str, Any] = {"role": "assistant", "content": None}
                serialized_arguments = json.dumps(actions[-1]["arguments"])
                if call.get("id"):
                    assistant_tool_message["tool_calls"] = [
                        {
                            "id": call.get("id"),
                            "type": "function",
                            "function": {"name": name, "arguments": serialized_arguments},
                        }
                    ]
                else:
                    assistant_tool_message["function_call"] = {
                        "name": name,
                        "arguments": serialized_arguments,
                    }
                working_messages.append(assistant_tool_message)
                working_messages.append(
                    {
                        "role": "function",
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            continue

        final_text = _extract_content(message_dict).strip()
        return final_text, actions, message_dict

    raise RuntimeError("Exceeded tool iteration limit without final response.")


def _summarize_conversation(memory: ConversationMemory) -> None:
    summary_messages = memory.summary_source_messages()
    if not summary_messages:
        memory.apply_summary("")
        return

    persona = _load_system_prompt()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": persona},
        {
            "role": "system",
            "content": (
                "You are refreshing your working memory. Summarize the conversation so far "
                "in under 150 words, focusing on the user's goals, code issues, tone, and "
                "any open follow-up work."
            ),
        },
    ]
    messages.extend(summary_messages)
    messages.append(
        {
            "role": "user",
            "content": "Summarize the conversation so you can reference it in future replies.",
        }
    )

    response = _request_herdora_completion(messages, allow_functions=False)
    choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
    message_payload = choice["message"] if isinstance(choice, dict) else choice.message
    message_dict = _message_to_dict(message_payload)
    summary_text = _extract_content(message_dict).strip()
    memory.apply_summary(summary_text)


def talk(user_prompt: str = DEFAULT_USER_PROMPT) -> Dict[str, Any]:
    """Generate Herdora's sarcastic take using function-calling tools with memory."""

    if _CONVERSATION_MEMORY.should_summarize() and _CONVERSATION_MEMORY.has_pending_history():
        _summarize_conversation(_CONVERSATION_MEMORY)

    memory_messages = _CONVERSATION_MEMORY.conversation_messages()

    snapshot_info = _load_localhost_snapshot()
    image_payload: Optional[Dict[str, Any]] = None
    image_hash_for_record: Optional[str] = None
    snapshot_descriptor: Optional[str] = None

    if snapshot_info:
        payload, image_hash, image_name = snapshot_info
        if image_hash != _CONVERSATION_MEMORY.last_image_hash():
            image_payload = payload
            image_hash_for_record = image_hash
            snapshot_descriptor = (
                f"Visual context: Attached localhost capture '{image_name}' for reference."
            )
        else:
            snapshot_descriptor = (
                f"Visual context unchanged: previous localhost capture '{image_name}' remains current."
            )

    submission_text = _prepare_submission_text(
        user_prompt,
        _CONVERSATION_MEMORY,
        snapshot_descriptor,
    )
    messages = _build_messages(submission_text, memory_messages, image_payload)

    final_text, actions, assistant_message = _chat_with_tools(messages)

    has_speech_action = any(action.get("name") == "speech" for action in actions)

    if not has_speech_action:
        script = _extract_python_script(final_text)
        if script:
            speech_kwargs = _extract_speech_kwargs(script)
            if speech_kwargs and isinstance(speech_kwargs.get("text"), str):
                speech_kwargs["text"] = _normalize_script(str(speech_kwargs["text"]))
                call_arguments = dict(speech_kwargs)
                result = call_registered_function("speech", call_arguments)
                actions.append({"name": "speech", "arguments": call_arguments, "result": result})
                final_text = call_arguments["text"]
                has_speech_action = True

    if not final_text:
        speech_action = next((action for action in reversed(actions) if action.get("name") == "speech"), None)
        if speech_action:
            final_text = str(speech_action["arguments"].get("text", ""))

    final_text = final_text.strip()

    if final_text and not has_speech_action:
        normalized_text = _normalize_script(final_text)
        call_arguments = {"text": normalized_text}
        result = call_registered_function("speech", call_arguments)
        actions.append({"name": "speech", "arguments": call_arguments, "result": result})
        final_text = normalized_text
        has_speech_action = True

    if not final_text:
        final_text = "*sighs* Herdora handled the response via audio only."

    improved_code: str = ""
    rewrite_payload: Optional[Dict[str, Any]] = None
    rewrite_error: Optional[str] = None
    try:
        rewrite_payload = rewrite_code(user_prompt)
        if isinstance(rewrite_payload, Mapping):
            primary_candidate = rewrite_payload.get("text")
            if isinstance(primary_candidate, str):
                improved_code = _normalize_improved_code(primary_candidate)
            if not improved_code:
                improved_code = _extract_improved_from_actions(rewrite_payload.get("actions"))
    except Exception as exc:  # pragma: no cover - defensive guard
        rewrite_error = str(exc)

    _CONVERSATION_MEMORY.record_turn(
        user_prompt,
        final_text,
        improved_code=improved_code,
        image_hash=image_hash_for_record,
    )

    response: Dict[str, Any] = {
        "text": final_text,
        "speech": final_text,
        "actions": actions,
        "assistant_message": assistant_message,
        "improved_code": improved_code,
    }

    if rewrite_payload is not None:
        response["rewrite"] = rewrite_payload
    if rewrite_error is not None:
        response["rewrite_error"] = rewrite_error

    return response
