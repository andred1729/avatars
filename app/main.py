from pathlib import Path

import argparse

from talk import talk, DEFAULT_USER_PROMPT

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Code Submission App")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def llmprompt(code: str) -> None:
     parser = argparse.ArgumentParser(
         description="Run Herdora and print the generated script without playback."
     )
     parser.add_argument(
         "prompt",
         nargs="?",
         default=DEFAULT_USER_PROMPT,
         help="Optional prompt to feed Herdora; defaults to the built-in prompt.",
     )
     args = parser.parse_args()

     result = talk(args.prompt)
     print(result["text"])


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
    #call the llm here
    llmprompt(code)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "submitted": True, "code": code},
    )
