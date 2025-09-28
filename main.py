from pathlib import Path
import os
import json
import tempfile
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional, Mapping, Tuple

from fastapi import Body, FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from talk import talk
from dotenv import load_dotenv

import whiteboard

load_dotenv()

app = FastAPI(title="Code Submission App")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

LOCALHOST_IMAGE_DIR = Path(
    os.getenv("LOCALHOST_IMAGE_DIR", str(BASE_DIR / "runtime" / "views"))
)
LOCALHOST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


EXPLAIN_KEYWORDS = (
    "explain",
    "explanation",
    "teach",
    "teaching",
    "tutorial",
    "walk me through",
    "help me understand",
    "understand",
    "break down",
    "step by step",
    "diagram",
    "whiteboard",
    "visualize",
    "draw",
    "illustrate",
    "show me",
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def should_open_whiteboard(prompt: str) -> bool:
    normalized = (prompt or "").lower()
    return any(keyword in normalized for keyword in EXPLAIN_KEYWORDS)


def _transcribe_audio_file(audio_path: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if elevenlabs_key:
        model_id = os.getenv("ELEVENLABS_TRANSCRIBE_MODEL", "scribe_v1") or "scribe_v1"
        language = os.getenv("ELEVENLABS_TRANSCRIBE_LANGUAGE")
        tag_events = _env_flag("ELEVENLABS_TRANSCRIBE_TAG_EVENTS")
        diarize = _env_flag("ELEVENLABS_TRANSCRIBE_DIARIZE")

        try:
            from elevenlabs.client import ElevenLabs  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install the 'elevenlabs' package to enable speech-to-text transcription."
            ) from exc

        client = ElevenLabs(api_key=elevenlabs_key)

        audio_bytes = BytesIO(audio_path.read_bytes())
        audio_bytes.seek(0)

        kwargs: Dict[str, Any] = {
            "file": audio_bytes,
            "model_id": model_id,
        }
        if language:
            kwargs["language_code"] = language
        if tag_events:
            kwargs["tag_audio_events"] = True
        if diarize:
            kwargs["diarize"] = True

        try:
            transcription = client.speech_to_text.convert(**kwargs)
        except Exception as exc:  # pragma: no cover - external dependency
            status_code = getattr(exc, "status_code", None)
            if status_code == 401:
                raise PermissionError("Invalid transcription API key") from exc
            raise RuntimeError(f"ElevenLabs transcription failed: {exc}") from exc

        if isinstance(transcription, Mapping):
            payload = dict(transcription)
            text = payload.get("text") or payload.get("transcription") or payload.get("transcript")
        else:
            payload = getattr(transcription, "model_dump", lambda: None)()
            text = getattr(transcription, "text", None)
            if text is None and isinstance(payload, dict):
                text = payload.get("text") or payload.get("transcription") or payload.get("transcript")

        if not text:
            raise RuntimeError("Transcription response did not include text.")
        return str(text).strip(), payload if isinstance(payload, dict) else None

    # Fallback to OpenAI transcription if ElevenLabs key is not configured.
    api_key = os.getenv("HERDORA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set ELEVENLABS_API_KEY or (HERDORA_API_KEY / OPENAI_API_KEY) for audio transcription."
        )

    base_url = os.getenv("HERDORA_BASE_URL")
    model = os.getenv("HERDORA_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(model=model, file=audio_file)
        text = getattr(response, "text", None)
        if text is None and isinstance(response, Mapping):
            text = response.get("text")
        if not text:
            raise RuntimeError("Transcription response did not include text.")
        payload_dict: Optional[Dict[str, Any]] = None
        if isinstance(response, Mapping):
            payload_dict = dict(response)
        else:
            model_dump = getattr(response, "model_dump", None)
            if callable(model_dump):
                payload_dict = model_dump()

        return str(text).strip(), payload_dict
    except ImportError:
        import openai  # type: ignore

        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        with audio_path.open("rb") as audio_file:
            response = openai.Audio.transcribe(model=model, file=audio_file)  # type: ignore[attr-defined]
        text = response.get("text") if isinstance(response, dict) else getattr(response, "text", None)
        if not text:
            raise RuntimeError("Transcription response did not include text.")
        payload_dict: Optional[Dict[str, Any]] = None
        if isinstance(response, dict):
            payload_dict = dict(response)
        else:
            model_dump = getattr(response, "model_dump", None)
            if callable(model_dump):
                payload_dict = model_dump()

        return str(text).strip(), payload_dict
    except Exception as exc:  # pragma: no cover - external dependency
        if exc.__class__.__name__ == "AuthenticationError":
            raise PermissionError("Invalid transcription API key") from exc
        raise


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request) -> HTMLResponse:
    """Render the code submission form."""
    whiteboard_state = whiteboard.get_state()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "submitted": False,
            "code": "",
            "result": "",
            "whiteboard_state": json.dumps(whiteboard_state),
            "auto_open_whiteboard": False,
        },
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_code(request: Request, code: str = Form(...)) -> HTMLResponse:
    """Trigger Herdora's critique and return the improved code."""
    explanation_mode = should_open_whiteboard(code)

    if not os.getenv("HERDORA_API_KEY"):
        whiteboard_state = whiteboard.get_state()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "submitted": True,
                "code": code,
                "error": "API key for Herdora must be set.",
                "whiteboard_state": json.dumps(whiteboard_state),
                "auto_open_whiteboard": explanation_mode,
                "result": "" if explanation_mode else "",
            },
        )
    talk_result = talk(code)

    improved_code = str(talk_result.get("improved_code", "")).strip()
    if not improved_code:
        improved_code = "No improved code was produced."
    if explanation_mode:
        improved_code = ""
    elif improved_code == "No improved code was produced.":
        improved_code = ""
    whiteboard_state = whiteboard.get_state()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "submitted": True,
            "code": code,
            "result": improved_code,
            "whiteboard_state": json.dumps(whiteboard_state),
            "auto_open_whiteboard": explanation_mode,
        },
    )


@app.get("/whiteboard/state")
async def whiteboard_state_endpoint() -> JSONResponse:
    return JSONResponse(whiteboard.get_state())


@app.post("/whiteboard/strokes")
async def whiteboard_strokes_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    strokes = payload.get("strokes", [])
    state = whiteboard.append_strokes(strokes if isinstance(strokes, list) else [])
    return JSONResponse(state)


@app.post("/whiteboard/clear")
async def whiteboard_clear_endpoint() -> JSONResponse:
    state = whiteboard.clear_board()
    return JSONResponse(state)


@app.post("/whiteboard/send")
async def whiteboard_send_endpoint(payload: Dict[str, Any] = Body(default=None)) -> JSONResponse:
    payload = payload or {}
    prompt = str(payload.get("prompt") or "Explain the current whiteboard.").strip()
    board_state = whiteboard.get_state()
    request_text = (
        "Whiteboard context (normalized coordinates JSON follows).\n"
        f"{json.dumps(board_state)}\n\n"
        f"User request: {prompt}"
    )
    talk_result = talk(request_text)
    return JSONResponse({"talk": talk_result, "whiteboard": board_state})


@app.post("/submit/voice")
async def submit_voice(
    request: Request,
    audio: UploadFile = File(...),
    screenshot: Optional[UploadFile] = File(None),
    code: str = Form(""),
) -> JSONResponse:
    if not os.getenv("HERDORA_API_KEY"):
        raise HTTPException(status_code=400, detail="API key for Herdora must be set.")

    if audio.content_type and not audio.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Uploaded file is not recognized as audio.")

    audio_suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp_audio:
        audio_path = Path(tmp_audio.name)
        audio_bytes = await audio.read()
        tmp_audio.write(audio_bytes)

    if screenshot is not None:
        screenshot_bytes = await screenshot.read()
        if screenshot_bytes:
            shot_suffix = Path(screenshot.filename or "capture.png").suffix or ".png"
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            capture_path = LOCALHOST_IMAGE_DIR / f"voice_capture_{timestamp}{shot_suffix}"
            capture_path.write_bytes(screenshot_bytes)

    try:
        transcript_text, transcript_payload = _transcribe_audio_file(audio_path)
    except PermissionError as exc:
        raise HTTPException(
            status_code=400,
            detail="Audio transcription failed: invalid API key for transcription provider.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {exc}") from exc
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass

    transcript_text = transcript_text.strip()
    if not transcript_text:
        raise HTTPException(status_code=400, detail="Transcription failed; empty transcript.")

    code_text = (code or "").strip()

    prompt_parts = []
    if code_text:
        prompt_parts.append(code_text)
    if transcript_text:
        prompt_parts.append(f"Voice notes:\n{transcript_text}")

    combined_prompt = "\n\n".join(prompt_parts) if prompt_parts else transcript_text
    explanation_mode = should_open_whiteboard(combined_prompt)

    talk_result = talk(combined_prompt)
    improved_code = str(talk_result.get("improved_code", "")).strip()
    if not improved_code or improved_code == "No improved code was produced.":
        improved_code = ""
    if explanation_mode:
        improved_code = ""

    whiteboard_state = whiteboard.get_state()

    suppress_output = explanation_mode or not improved_code

    response_payload: Dict[str, Any] = {
        "transcript": transcript_text,
        "talk": talk_result,
        "whiteboard": whiteboard_state,
        "result": improved_code,
        "suppress_output": suppress_output,
    }
    response_payload["whiteboard_action"] = "open" if explanation_mode else "close"
    if transcript_payload:
        response_payload["transcription_details"] = transcript_payload

    if transcript_payload:
        try:
            transcripts_dir = BASE_DIR / "runtime" / "transcripts"
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
            transcript_path = transcripts_dir / f"recording_{timestamp}.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "transcript": transcript_text,
                        "metadata": transcript_payload,
                        "code": code_text,
                        "combined_prompt": combined_prompt,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            response_payload["transcription_file"] = str(transcript_path)
        except Exception as exc:  # pragma: no cover - best effort logging
            response_payload["transcription_file_error"] = str(exc)

    return JSONResponse(response_payload)
