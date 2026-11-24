from pathlib import Path
import json
import queue
import shutil
import threading
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from config import settings
from orchestrator import process

app = FastAPI(title="Passport Photo Pipeline")

TEMPLATES_DIR = Path(__file__).parent / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"
JOBS: dict[str, dict] = {}


def _prepare_job_paths(file: UploadFile, bg: str) -> tuple[str, str, Path, Path]:
    bg_choice = bg.lower().strip()
    if bg_choice not in ("blue", "white"):
        bg_choice = "blue"

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        suffix = ".jpg"

    job_id = uuid.uuid4().hex
    upload_path = settings.UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = settings.OUTPUT_DIR / f"passport_{job_id}.jpg"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {err}")

    return job_id, bg_choice, upload_path, output_path


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


@app.post("/process/start")
def start_process_photo(
    file: UploadFile = File(...),
    bg: str = Form("blue"),
):
    job_id, bg_choice, upload_path, output_path = _prepare_job_paths(file, bg)
    event_queue: queue.Queue = queue.Queue()
    JOBS[job_id] = {
        "queue": event_queue,
        "output_path": output_path,
        "bg": bg_choice,
        "status": "running",
    }

    def worker() -> None:
        try:
            process(
                image_path=upload_path,
                output_path=output_path,
                bg=bg_choice,
                on_progress=lambda pct, msg: event_queue.put(
                    {"pct": pct, "msg": msg, "status": "running"}
                ),
            )
            JOBS[job_id]["status"] = "done"
            event_queue.put({"pct": 100, "msg": "Complete!", "status": "done"})
        except Exception as err:
            JOBS[job_id]["status"] = "error"
            event_queue.put(
                {"pct": 0, "msg": f"Pipeline processing failed: {err}", "status": "error"}
            )

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/process/stream/{job_id}")
def stream_process_progress(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    event_queue: queue.Queue = job["queue"]

    def event_stream():
        while True:
            try:
                event = event_queue.get(timeout=15.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(event)}\n\n"
            if event.get("status") in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/process/result/{job_id}")
def get_process_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    output_path: Path = job["output_path"]
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not ready")
    bg_choice = job["bg"]
    return FileResponse(
        path=output_path,
        media_type="image/jpeg",
        filename=f"passport_{bg_choice}.jpg",
    )


@app.post("/process")
def process_photo(
    file: UploadFile = File(...),
    bg: str = Form("blue"),
):
    _, bg_choice, upload_path, output_path = _prepare_job_paths(file, bg)

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
