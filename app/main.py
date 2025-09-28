from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Code Submission App")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request) -> HTMLResponse:
    """Render the code submission form."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "submitted": False, "code": ""},
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_code(request: Request, code: str = Form(...)) -> HTMLResponse:
    """Display the submitted code back to the user."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "submitted": True, "code": code},
    )
