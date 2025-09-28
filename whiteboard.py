"""In-memory plus persisted stroke registry for the collaborative whiteboard."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

_BASE_DIR = Path(__file__).resolve().parent
_RUNTIME_DIR = _BASE_DIR / "runtime"
_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
_STATE_PATH = _RUNTIME_DIR / "whiteboard_state.json"

_STATE_LOCK = threading.Lock()

_DEFAULT_STATE: Dict[str, Any] = {"strokes": [], "version": 0}


def _load_state_unlocked() -> Dict[str, Any]:
    if not _STATE_PATH.exists():
        return {"strokes": [], "version": 0}
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"strokes": [], "version": 0}
    if not isinstance(data, dict):
        return {"strokes": [], "version": 0}
    data.setdefault("strokes", [])
    data.setdefault("version", 0)
    return data


def _save_state_unlocked(state: Dict[str, Any]) -> None:
    _STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def get_state() -> Dict[str, Any]:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        return json.loads(json.dumps(state))


def append_strokes(strokes: List[Dict[str, Any]]) -> Dict[str, Any]:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        state_strokes = state.setdefault("strokes", [])
        state_strokes.extend(strokes)
        state["version"] = int(state.get("version", 0)) + 1
        _save_state_unlocked(state)
        return json.loads(json.dumps(state))


def clear_board() -> Dict[str, Any]:
    with _STATE_LOCK:
        state = {"strokes": [], "version": int(_load_state_unlocked().get("version", 0)) + 1}
        _save_state_unlocked(state)
        return json.loads(json.dumps(state))
