"""Code rewriting pipeline leveraging Herdora's LLM tooling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from llm_functions import call_registered_function, function_schemas

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "avatar_write_prompt.txt"


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing system prompt. Ensure prompts/avatar_write_prompt.txt exists."
        ) from exc


def _build_messages(user_code: str) -> List[Dict[str, Any]]:
    persona = _load_system_prompt()
    return [
        {"role": "system", "content": persona},
        {"role": "user", "content": user_code},
    ]


def _request_herdora_completion(
    messages: List[Dict[str, Any]],
    *,
    allow_functions: bool,
    allowed_functions: Optional[Iterable[str]] = None,
) -> Any:
    api_key = os.getenv("HERDORA_API_KEY")
    if not api_key:
        raise EnvironmentError("Set HERDORA_API_KEY before calling write().")

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
    if isinstance(message, dict):
        return message

    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _maybe_get_function_call(message_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    call = message_dict.get("function_call") or message_dict.get("tool_calls")
    if isinstance(call, list):
        return call[0] if call else None
    return call


def _first_choice(response: Any) -> Any:
    if isinstance(response, dict):
        return response["choices"][0]
    return response.choices[0]


def _extract_content(message_dict: Dict[str, Any]) -> str:
    content = message_dict.get("content", "")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(text_parts)
    return str(content or "")


def write(user_code: str) -> Dict[str, Any]:
    """Request an improved version of ``user_code`` and surface the rewrite."""

    messages = _build_messages(user_code)
    actions: List[Dict[str, Any]] = []

    response = _request_herdora_completion(
        messages,
        allow_functions=True,
        allowed_functions=("write",),
    )
    choice = _first_choice(response)
    message_payload = choice["message"] if isinstance(choice, dict) else choice.message
    assistant_message = _message_to_dict(message_payload)

    function_call = _maybe_get_function_call(assistant_message)
    if function_call:
        function_name = function_call.get("name")
        raw_arguments = function_call.get("arguments") or "{}"
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed_arguments = {}

        if function_name:
            result = call_registered_function(function_name, parsed_arguments)
            actions.append(
                {
                    "name": function_name,
                    "arguments": parsed_arguments,
                    "result": result,
                }
            )
            improved_code = str(result.get("improved_code", "")).strip()
            if improved_code:
                response: Dict[str, Any] = {"text": improved_code, "actions": actions}
                analysis_text = str(result.get("analysis") or "").strip()
                if analysis_text:
                    response["analysis"] = analysis_text
                artifact_path = result.get("artifact_path")
                if isinstance(artifact_path, str) and artifact_path:
                    response["artifact_path"] = artifact_path
                language = result.get("language")
                if isinstance(language, str) and language:
                    response["language"] = language
                return response

    fallback_text = _extract_content(assistant_message).strip()
    if fallback_text:
        return {"text": fallback_text, "actions": actions}

    return {
        "text": "No improved code was produced.",
        "actions": actions,
    }
