from pathlib import Path
import os
import json
from typing import Any, Dict
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from talk import talk
from dotenv import load_dotenv

import whiteboard

load_dotenv()

app = FastAPI(title="Code Submission App")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


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
        },
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_code(request: Request, code: str = Form(...)) -> HTMLResponse:
    """Trigger Herdora's critique and return the improved code."""
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
            },
        )
    talk_result = talk(code)

    improved_code = str(talk_result.get("improved_code", "")).strip()
    if not improved_code:
        improved_code = "No improved code was produced."
    whiteboard_state = whiteboard.get_state()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "submitted": True,
            "code": code,
            "result": improved_code,
            "whiteboard_state": json.dumps(whiteboard_state),
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
