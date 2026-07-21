from __future__ import annotations

import asyncio
import io
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.cerebras_client import CEREBRAS_API_KEY, MODEL
from app.pipeline import MAX_CHARS, run_pipeline
from app.supabase_client import get_job, list_jobs, save_job, storage_mode

app = FastAPI(title="Proofreading Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 2_000_000


class ProofreadRequest(BaseModel):
    text: str


def _validate_text(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds the {MAX_CHARS:,} character limit.",
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "storage": storage_mode(),
        "cerebras_key_configured": bool(CEREBRAS_API_KEY),
        "max_chars": MAX_CHARS,
    }


@app.post("/proofread")
async def proofread_endpoint(req: ProofreadRequest):
    _validate_text(req.text)
    try:
        result = await run_pipeline(req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    saved = save_job(req.text, result)
    return {"id": saved.get("id"), **result}


@app.post("/proofread/stream")
async def proofread_stream(req: ProofreadRequest):
    """Same pipeline, but streams progress events as Server-Sent Events:
    {"type":"progress","done":n,"total":n} ... {"type":"result","payload":{...}}
    """
    _validate_text(req.text)
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(done: int, total: int):
        await queue.put({"type": "progress", "done": done, "total": total})

    async def run():
        try:
            result = await run_pipeline(req.text, progress)
            saved = save_job(req.text, result)
            await queue.put(
                {"type": "result", "payload": {"id": saved.get("id"), **result}}
            )
        except Exception as e:
            await queue.put({"type": "error", "detail": str(e)})
        finally:
            await queue.put(None)

    async def gen():
        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            await task

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    """Extract plain text from an uploaded .txt, .md, or .docx file."""
    name = (file.filename or "").lower()
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 MB).")
    if name.endswith(".docx"):
        try:
            from docx import Document

            doc = Document(io.BytesIO(data))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            raise HTTPException(
                status_code=400, detail="Could not read this .docx file."
            )
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .txt, .md, or .docx file.",
        )
    if not text.strip():
        raise HTTPException(status_code=400, detail="The file contains no text.")
    return {"filename": file.filename, "text": text, "chars": len(text)}


@app.get("/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/jobs")
def list_jobs_endpoint(limit: int = 20):
    return list_jobs(min(max(limit, 1), 100))


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
