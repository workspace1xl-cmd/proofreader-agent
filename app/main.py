from __future__ import annotations

import asyncio
import io
import json
import logging
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.cerebras_client import CEREBRAS_API_KEY, MODEL
from app.config import (
    MAX_CHARS,
    MAX_DOCX_ENTRIES,
    MAX_DOCX_UNCOMPRESSED_BYTES,
    MAX_UPLOAD_BYTES,
    PIPELINE_TIMEOUT_SECONDS,
    SSE_HEARTBEAT_SECONDS,
)
from app.pipeline import run_pipeline
from app.supabase_client import get_job, list_jobs, save_job, storage_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Proofreading Agent", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ProofreadRequest(BaseModel):
    text: str


class ExportRequest(BaseModel):
    text: str
    title: str = "Proofread document"


def _validate_text(text: str) -> None:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds the {MAX_CHARS:,} character limit.",
        )


async def _run_with_timeout(
    text: str,
    progress: Any = None,
    agent_status: Any = None,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(PIPELINE_TIMEOUT_SECONDS):
            return await run_pipeline(text, progress, agent_status)
    except TimeoutError as exc:
        raise RuntimeError(
            "The review timed out. Please retry or split this document into sections."
        ) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "model": MODEL,
        "storage": storage_mode(),
        "cerebras_key_configured": bool(CEREBRAS_API_KEY),
        "max_chars": MAX_CHARS,
    }


@app.post("/proofread")
async def proofread_endpoint(req: ProofreadRequest) -> dict[str, Any]:
    _validate_text(req.text)
    try:
        result = await _run_with_timeout(req.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    saved = await asyncio.to_thread(save_job, req.text, result)
    return {"id": saved.get("id"), **result}


@app.post("/proofread/stream")
async def proofread_stream(
    req: ProofreadRequest, request: Request
) -> StreamingResponse:
    _validate_text(req.text)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=64)

    async def progress(done: int, total: int) -> None:
        await queue.put({"type": "progress", "done": done, "total": total})

    async def agent_status(key: str, label: str, state: str) -> None:
        await queue.put(
            {
                "type": "agent",
                "key": key,
                "label": label,
                "state": state,
            }
        )

    async def runner() -> None:
        try:
            result = await _run_with_timeout(req.text, progress, agent_status)
            saved = await asyncio.to_thread(save_job, req.text, result)
            await queue.put(
                {"type": "result", "payload": {"id": saved.get("id"), **result}}
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Streaming review failed")
            await queue.put(
                {
                    "type": "error",
                    "detail": (
                        "The review could not be completed. Please retry in a moment."
                    ),
                }
            )
        finally:
            await queue.put(None)

    async def events() -> Any:
        task = asyncio.create_task(runner(), name="proofread-pipeline")
        event_id = 0
        try:
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=SSE_HEARTBEAT_SECONDS
                    )
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                event_id += 1
                payload = json.dumps(item, ensure_ascii=False)
                yield f"id: {event_id}\ndata: {payload}\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _validate_docx_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_ENTRIES:
                raise ValueError("too many archive entries")
            total = sum(entry.file_size for entry in entries)
            if total > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError("expanded document is too large")
            if not any(entry.filename == "word/document.xml" for entry in entries):
                raise ValueError("missing Word document content")
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise ValueError("invalid DOCX archive") from exc


def _extract_docx(data: bytes) -> str:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    _validate_docx_archive(data)
    document = Document(io.BytesIO(data))
    parts: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document).text
            if paragraph.strip():
                parts.append(paragraph)
        elif child.tag == qn("w:tbl"):
            rows: list[str] = []
            for row in Table(child, document).rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                parts.append("\n".join(rows))
    return "\n\n".join(parts)


async def _read_limited(file: UploadFile) -> bytes:
    data = bytearray()
    while chunk := await file.read(64 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 2 MB).")
    return bytes(data)


@app.post("/extract")
async def extract_endpoint(
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    name = (file.filename or "").lower()
    data = await _read_limited(file)
    if name.endswith(".docx"):
        try:
            text = await asyncio.to_thread(_extract_docx, data)
        except Exception as exc:
            logger.info("Rejected malformed DOCX %r: %s", file.filename, exc)
            raise HTTPException(
                status_code=400, detail="Could not read this .docx file."
            ) from exc
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8-sig", errors="replace")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .txt, .md, or .docx file.",
        )
    _validate_text(text)
    return {"filename": file.filename, "text": text, "chars": len(text)}


@app.post("/export/docx")
async def export_docx(req: ExportRequest) -> StreamingResponse:
    _validate_text(req.text)
    from docx import Document

    def build() -> io.BytesIO:
        document = Document()
        document.add_heading(req.title[:120], level=1)
        for paragraph in req.text.split("\n\n"):
            document.add_paragraph(paragraph)
        buffer = io.BytesIO()
        document.save(buffer)
        buffer.seek(0)
        return buffer

    buffer = await asyncio.to_thread(build)
    safe_name = (
        "".join(char for char in req.title if char.isalnum() or char in " -_")[:60]
        or "document"
    )
    return StreamingResponse(
        buffer,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'},
    )


@app.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> dict[str, Any]:
    job = await asyncio.to_thread(get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/jobs")
async def list_jobs_endpoint(limit: int = 20) -> list[dict[str, Any]]:
    return await asyncio.to_thread(list_jobs, min(max(limit, 1), 100))


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
