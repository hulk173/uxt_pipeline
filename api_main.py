# api_main.py
from __future__ import annotations

import io
import os
import sqlite3
from uuid import uuid4
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from uxt_pipeline.core import extract_docx, extract_html, extract_pdf, chunk
from uxt_pipeline.db import DB_PATH, SessionLocal, Run, ChunkRow, init_db, rebuild_fts

API_TITLE = "UXT API"
API_DESC = "Upload → Extract → Chunk → Store → Search"
API_VERSION = "0.1.0"

UXT_USER = os.getenv("UXT_USER", "admin")
UXT_PASS = os.getenv("UXT_PASS", "admin")

STORAGE_ROOT = os.path.join("uxt_pipeline", "storage")

app = FastAPI(title=API_TITLE, description=API_DESC, version=API_VERSION)
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def auth(creds: HTTPBasicCredentials = Depends(security)):
    if not (creds.username == UXT_USER and creds.password == UXT_PASS):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad credentials")
    return True

@app.on_event("startup")
def _startup():
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    init_db()

@app.get("/health")
def health():
    return {"ok": True, "db": os.path.abspath(DB_PATH)}

# ----------------------------- Runs ---------------------------------

FMT = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".html": extract_html,
    ".htm": extract_html,
}

@app.get("/runs", dependencies=[Depends(auth)])
def list_runs(db: Session = Depends(get_db)):
    rows = db.query(Run).order_by(Run.id.desc()).all()
    out = []
    for r in rows:
        n = db.query(ChunkRow).filter(ChunkRow.run_id == r.id).count()
        out.append({
            "id": r.id,
            "created_at": r.created_at,
            "chunk_size": r.chunk_size,
            "overlap": r.overlap,
            "comment": r.comment,
            "n_chunks": n,
        })
    return out

@app.post("/runs", dependencies=[Depends(auth)])
async def create_run(
    chunk_size: int = 800,
    overlap: int = 100,
    comment: str = "",
    files: List[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
):
    # створюємо запис про прогін
    run = Run(chunk_size=int(chunk_size), overlap=int(overlap), comment=comment or "")
    db.add(run)
    db.commit()
    db.refresh(run)

    # директорія для оригіналів
    run_dir = os.path.join(STORAGE_ROOT, str(run.id))
    os.makedirs(run_dir, exist_ok=True)

    for up in files:
        # UploadFile.filename може бути None => гарантуємо str
        raw_name = up.filename
        filename: str = raw_name if isinstance(raw_name, str) and raw_name else f"uploaded_{uuid4().hex}"
        stored_path = os.path.join(run_dir, filename)

        # читаємо тіло і зберігаємо оригінал
        body = await up.read()
        with open(stored_path, "wb") as out:
            out.write(body)

        # парсинг і чанкінг
        suf = os.path.splitext(filename)[1].lower()
        handler = FMT.get(suf)
        if not handler:
            # пропускаємо невідомі типи
            continue

        try:
            elements = handler(stored_path)
            chunks = chunk(elements, size=int(chunk_size), overlap=int(overlap))
            for c in chunks:
                db.add(ChunkRow(
                    run_id=run.id,
                    source=filename,
                    text=c.text,
                    len_words=len(c.text.split()),
                ))
            db.commit()
        except Exception as e:
            # не валимо весь прогін
            print(f"[WARN] failed to process {filename}: {e}")

    n = db.query(ChunkRow).filter(ChunkRow.run_id == run.id).count()
    return {"id": run.id, "chunks": n}

@app.get("/runs/{run_id}/chunks", dependencies=[Depends(auth)])
def get_chunks(run_id: int, db: Session = Depends(get_db)):
    rows = db.query(ChunkRow).filter(ChunkRow.run_id == run_id).order_by(ChunkRow.id.asc()).all()
    return [{"id": c.id, "source": c.source, "len_words": c.len_words, "text": c.text} for c in rows]

@app.delete("/runs/{run_id}", dependencies=[Depends(auth)])
def delete_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    db.delete(run)
    db.commit()
    # видаляємо директорію з оригіналами (опційно)
    try:
        import shutil
        shutil.rmtree(os.path.join(STORAGE_ROOT, str(run_id)), ignore_errors=True)
    except Exception:
        pass
    return {"ok": True}

# ----------------------------- Export ---------------------------------

@app.get("/runs/{run_id}/export.csv", dependencies=[Depends(auth)])
def export_csv(run_id: int, db: Session = Depends(get_db)):
    rows = db.query(ChunkRow).filter(ChunkRow.run_id == run_id).all()
    df = pd.DataFrame([{"source": r.source, "len_words": r.len_words, "text": r.text} for r in rows])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}.csv"'},
    )

@app.get("/runs/{run_id}/export.parquet", dependencies=[Depends(auth)])
def export_parquet(run_id: int, db: Session = Depends(get_db)):
    rows = db.query(ChunkRow).filter(ChunkRow.run_id == run_id).all()
    df = pd.DataFrame([{"source": r.source, "len_words": r.len_words, "text": r.text} for r in rows])
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
    except Exception as e:
        raise HTTPException(500, f"Parquet export failed (install pyarrow): {e}")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}.parquet"'},
    )

# ----------------------------- FTS5 search -----------------------------

@app.get("/search", dependencies=[Depends(auth)])
def search(
    q: str = Query(..., min_length=2, description='FTS5: "exact phrase", word*, term1 AND term2'),
    run_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if run_id is None:
        sql = """
        SELECT c.id, c.run_id, c.source, c.len_words,
               snippet(chunks_fts, 0, '[', ']', ' … ', 10) AS snippet
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY c.id DESC
        LIMIT ?
        """
        rows = cur.execute(sql, (q, limit)).fetchall()
    else:
        sql = """
        SELECT c.id, c.run_id, c.source, c.len_words,
               snippet(chunks_fts, 0, '[', ']', ' … ', 10) AS snippet
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ? AND c.run_id = ?
        ORDER BY c.id DESC
        LIMIT ?
        """
        rows = cur.execute(sql, (q, run_id, limit)).fetchall()
    con.close()
    return [
        {"chunk_id": r[0], "run_id": r[1], "source": r[2], "len_words": r[3], "snippet": r[4]}
        for r in rows
    ]

@app.post("/rebuild-fts", dependencies=[Depends(auth)])
def api_rebuild_fts():
    rebuild_fts()
    return {"ok": True}

# ----------------------------- Local run -------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)
