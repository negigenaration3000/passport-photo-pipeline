from pathlib import Path
import shutil
import uuid
import requests

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse, Response

from config import settings
from orchestrator import process

app = FastAPI(title="Passport Photo Pipeline")

TEMPLATES_DIR = Path(__file__).parent / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="Index template not found")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/fetch-image")
async def fetch_image(url: str = Query(..., description="External image URL to fetch")):
    """Bypass browser CORS when loading an image from a URL pasted by the user."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg")
        return Response(content=resp.content, media_type=content_type)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {err}")


@app.post("/process")
async def process_photo(
    file: UploadFile = File(...),
    bg: str = Form("blue"),
):
    bg_choice = bg.lower().strip()
    if bg_choice not in ("blue", "white"):
        bg_choice = "blue"

    # Ensure uploads and outputs directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine file extension safely
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        suffix = ".jpg"

    unique_id = uuid.uuid4().hex
    upload_path = settings.UPLOAD_DIR / f"{unique_id}{suffix}"
    output_path = settings.OUTPUT_DIR / f"passport_{unique_id}.jpg"

    # Save uploaded file
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {err}")

    # Run 12-step passport pipeline
    try:
        process(
            image_path=upload_path,
            output_path=output_path,
            bg=bg_choice,
        )
        return FileResponse(
            path=output_path,
            media_type="image/jpeg",
            filename=f"passport_{bg_choice}.jpg",
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {err}")
