# api_main.py
from __future__ import annotations

import json
import os
import shutil
import string
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import KMeans

# локальні модулі
from uxt_pipeline.index.build_index import build_index, load_index, search as index_search

# ------------------------- Константи та шляхи -------------------------
DATA_DIR = Path("data")
IN_DIR = DATA_DIR / "in"
OUT_DIR = DATA_DIR / "out"
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
HISTORY_PATH = OUT_DIR / "history.jsonl"

# ------------------------- Утиліти -------------------------
def json_default(o: Any):
    from datetime import date, datetime
    from pathlib import Path as _Path
    import numpy as _np

    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, _Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    if isinstance(o, _np.generic):
        return o.item()
    return str(o)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=json_default) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def add_history(event: str, payload: Dict[str, Any]) -> None:
    rec = {"time": datetime.now(), "event": event, "payload": payload}
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")


def split_by_sentences(text: str) -> List[str]:
    # простий розбивач на "речення"
    parts: List[str] = []
    buf: List[str] = []
    stops = set(".!?…")
    for ch in text:
        buf.append(ch)
        if ch in stops:
            parts.append("".join(buf).strip())
            buf = []
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 120, min_text_chars: int = 30) -> List[str]:
    sents = split_by_sentences(text)
    if not sents:
        sents = [text]
    chunks: List[str] = []
    cur = ""
    for sent in sents:
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = f"{cur} {sent}".strip()
        else:
            if len(cur) >= min_text_chars:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = f"{tail} {sent}".strip()
    if len(cur) >= min_text_chars:
        chunks.append(cur)
    return chunks


def extract_text_from_bytes(filename: str, raw: bytes, strategy: str = "fast", ocr_languages: str = "eng+ukr") -> str:
    """
    Спрощений екстрактор тексту з (filename, bytes)
    """
    suffix = (Path(filename).suffix or "").lower()

    if suffix in (".txt", ""):
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if suffix in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(raw, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            return raw.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import PyPDF2  # type: ignore
            reader = PyPDF2.PdfReader(BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if suffix == ".docx":
        try:
            import docx  # type: ignore
            doc = docx.Document(BytesIO(raw))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return raw.decode("utf-8", errors="ignore")

    return raw.decode("utf-8", errors="ignore")


def sanitize_doc_id(name: str) -> str:
    valid = string.ascii_letters + string.digits + "-_"
    base = "".join(ch if ch in valid else "_" for ch in name)
    return base.strip("_") or "document"


# ------------------------- FastAPI app -------------------------
app = FastAPI(title="UXT Pipeline API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------- Моделі -------------------------
class EvaluateRequest(BaseModel):
    query: str
    expected_text: str = ""
    k: int = 5


# ------------------------- Менеджер задач -------------------------
JOBS: Dict[str, Dict[str, Any]] = {}


def set_job(job_id: str, **kw):
    JOBS.setdefault(job_id, {})
    JOBS[job_id].update(kw)


# ------------------------- Ендпоїнти -------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}


@app.get("/history")
def history():
    return read_jsonl(HISTORY_PATH)


@app.delete("/outputs")
def clear_outputs():
    try:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        add_history("clear_outputs", {"ok": True})
        return {"ok": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/ingest")
def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: str = Form("fast"),
    ocr_languages: str = Form("eng+ukr"),
    max_chars: int = Form(1000),
    overlap: int = Form(120),
    join_same_type: str = Form("true"),
    min_text_chars: int = Form(30),
    strip_whitespace: str = Form("true"),
    index_backend: str = Form("faiss"),
    model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    top_k: int = Form(5),
    normalize: str = Form("true"),
):
    doc_id = sanitize_doc_id(file.filename or f"upload_{uuid.uuid4().hex[:6]}")
    job_id = uuid.uuid4().hex
    set_job(job_id, status="queued", file=file.filename, doc_id=doc_id, created=datetime.now().isoformat())

    dest = IN_DIR / f"{doc_id}{Path(file.filename or '').suffix}"
    dest.write_bytes(file.file.read())

    norm = str(normalize).lower() in {"1", "true", "yes", "y"}

    background_tasks.add_task(
        _process_one,
        job_id=job_id,
        upload_path=dest,
        params={
            "strategy": strategy,
            "ocr_languages": ocr_languages,
            "max_chars": int(max_chars),
            "overlap": int(overlap),
            "min_text_chars": int(min_text_chars),
            "index_backend": index_backend,
            "model_name": model_name,
            "normalize": bool(norm),
            "doc_id": doc_id,
        },
    )
    add_history("ingest_queued", {"job_id": job_id, "doc_id": doc_id, "file": file.filename})
    return {"job_id": job_id}


def _process_one(job_id: str, upload_path: Path, params: Dict[str, Any]):
    try:
        set_job(job_id, status="running", started=datetime.now().isoformat())

        # 1) Текст
        raw = upload_path.read_bytes()
        text = extract_text_from_bytes(upload_path.name, raw, strategy=params["strategy"], ocr_languages=params["ocr_languages"])
        if not text.strip():
            raise RuntimeError("Екстракція тексту повернула порожній результат")

        # 2) Чанкiнг
        chunks_text = chunk_text(
            text,
            max_chars=params["max_chars"],
            overlap=params["overlap"],
            min_text_chars=params["min_text_chars"],
        )

        # 3) Формуємо записи чанків
        chunks: List[Dict[str, Any]] = []
        for i, t in enumerate(chunks_text):
            chunks.append({
                "id": f"{params['doc_id']}#{i+1}",
                "doc_id": params["doc_id"],
                "text": t,
                "type": "text",
                "meta": {"page": None, "order": i + 1, "ingested_at": datetime.now().isoformat()},
            })

        # 4) Пишемо chunks.jsonl
        write_jsonl(CHUNKS_PATH, chunks)

        # 5) Будуємо індекс
        info = build_index(
            chunks=chunks,
            out_dir=str(OUT_DIR),
            backend=params.get("index_backend", "faiss"),
            model_name=params.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            normalize=params.get("normalize", True),
        )

        set_job(job_id, status="finished", finished=datetime.now().isoformat(), index=info)
        add_history("ingest_done", {"job_id": job_id, "doc_id": params["doc_id"], "chunks": len(chunks)})
    except Exception as ex:
        set_job(job_id, status="error", error=str(ex), finished=datetime.now().isoformat())
        add_history("ingest_error", {"job_id": job_id, "error": str(ex)})


@app.get("/job/{job_id}")
def job_status(job_id: str):
    state = JOBS.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return state


@app.get("/search")
def search(q: str, k: int = 5):
    try:
        results = index_search(str(OUT_DIR), q, k=k)
        id2text = {}
        for row in read_jsonl(CHUNKS_PATH):
            id2text[row.get("id")] = row.get("text", "")
        payload = []
        for r in results:
            payload.append({
                "score": float(r["score"]),
                "chunk": {
                    "id": r.get("id"),
                    "doc_id": r.get("doc_id"),
                    "text": id2text.get(r.get("id"), "")
                }
            })
        add_history("search", {"q": q, "k": k, "results": len(payload)})
        return payload
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/ask")
def ask(q: str, k: int = 5):
    try:
        res = index_search(str(OUT_DIR), q, k=k)
        id2row = {row.get("id"): row for row in read_jsonl(CHUNKS_PATH)}
        sources = []
        ctx_parts = []
        for r in res:
            row = id2row.get(r["id"], {})
            txt = (row.get("text") or "").strip()
            if not txt:
                continue
            sources.append({
                "id": r["id"],
                "doc_id": r.get("doc_id"),
                "score": float(r["score"]),
                "preview": txt[:300]
            })
            ctx_parts.append(txt)

        answer = f"Ключові фрагменти за запитом: «{q}»:\n\n" + "\n\n---\n\n".join(ctx_parts[:k])

        add_history("ask", {"q": q, "k": k, "sources": len(sources)})
        return {"answer": answer, "sources": sources}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        results = index_search(str(OUT_DIR), req.query, k=req.k)
        id2text = {row.get("id"): row.get("text", "") for row in read_jsonl(CHUNKS_PATH)}
        hits = 0
        rows = []
        needle = (req.expected_text or "").lower().strip()
        for r in results:
            txt = id2text.get(r["id"], "")
            ok = False
            if needle:
                ok = needle in txt.lower()
            rows.append({"id": r["id"], "doc_id": r["doc_id"], "score": float(r["score"]), "match": ok})
            if ok:
                hits += 1
        recall = hits / max(1, req.k)
        prec = hits / max(1, len([x for x in rows if x["match"]])) if any(x["match"] for x in rows) else 0.0
        add_history("evaluate", {"q": req.query, "k": req.k, "hits": hits})
        return {"recall@k": float(recall), "precision@k": float(prec), "k": int(req.k), "results": rows}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/clusters")
def clusters(n_clusters: int = 8):
    try:
        backend, index_obj, meta = load_index(str(OUT_DIR))
        model_name = meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

        if backend == "sklearn":
            X = index_obj["emb"]
        else:
            rows = read_jsonl(CHUNKS_PATH)
            texts = [r.get("text", "") for r in rows]
            if not texts:
                raise RuntimeError("no chunks to cluster")
            from sentence_transformers import SentenceTransformer
            mdl = SentenceTransformer(model_name)
            X = mdl.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            if meta.get("normalize", True):
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                X = X / norms

        km = KMeans(n_init="auto", random_state=42, n_clusters=int(n_clusters))
        labels = km.fit_predict(X)

        rows = read_jsonl(CHUNKS_PATH)
        out = []
        for i, r in enumerate(rows[: len(labels)]):
            out.append({
                "id": r.get("id"),
                "doc_id": r.get("doc_id"),
                "cluster": int(labels[i]),
                "text_preview": (r.get("text","")[:160] or "").replace("\n", " ")
            })

        add_history("clusters", {"n_clusters": n_clusters, "count": len(out)})
        return {"clusters": out}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
