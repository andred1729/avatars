from pathlib import Path
import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from talk import talk
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Code Submission App")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request) -> HTMLResponse:
    """Render the code submission form."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "submitted": False, "code": "", "result": ""},
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_code(request: Request, code: str = Form(...)) -> HTMLResponse:
    """Trigger Herdora's critique and return the improved code."""
    if not os.getenv("HERDORA_API_KEY"):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "submitted": True,
                "code": code,
                "error": "API key for Herdora must be set.",
            },
        )
    talk_result = talk(code)

    improved_code = str(talk_result.get("improved_code", "")).strip()
    if not improved_code:
        improved_code = "No improved code was produced."
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "submitted": True,
            "code": code,
            "result": improved_code,
        },
    )
