from __future__ import annotations

import json
import os
import re
import shutil
import string
import tempfile
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- UXT modules ----
from uxt_pipeline.ingest.partitioner import _partition_one
from uxt_pipeline.transform.chunker import semantic_chunk
from uxt_pipeline.index.build_index import build_index, load_index, search as index_search

# ------------------------- Consts -------------------------
DATA_DIR = Path("data")
IN_DIR = DATA_DIR / "in"
OUT_DIR = DATA_DIR / "out"
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
HISTORY_PATH = OUT_DIR / "history.jsonl"
META_PATH = OUT_DIR / "meta.json"
INDEX_FAISS = OUT_DIR / "index.faiss"

API_VERSION = "1.3.0"

# ------------------------- Utils -------------------------
def json_default(o: Any):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    if isinstance(o, np.generic):
        return o.item()
    return str(o)

def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
        tf.write(text)
        tmp_name = tf.name
    Path(tmp_name).replace(path)

def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append rows atomically to JSONL."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=json_default) + "\n")
        payload = tmp.read_text(encoding="utf-8")
        if path.exists():
            with path.open("a", encoding="utf-8") as out:
                out.write(payload)
        else:
            _atomic_write_text(path, payload)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

def read_jsonl_soft(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Return (rows, bad_count), ignoring broken lines safely."""
    if not path.exists():
        return [], 0
    rows: List[Dict[str, Any]] = []
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = (line or "").strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            bad += 1
            continue
    return rows, bad

def add_history(event: str, payload: Dict[str, Any]) -> None:
    rec = {"time": datetime.now(), "event": event, "payload": payload}
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")

def sanitize_doc_id(name: str) -> str:
    valid = string.ascii_letters + string.digits + "-_"
    base = "".join(ch if ch in valid else "_" for ch in (name or "document"))
    return base.strip("_") or "document"

_WS_RE = re.compile(r"[ \t\r\f\v]+")

def normalize_whitespace(text: str) -> str:
    if not text:
        return text
    return "\n".join(_WS_RE.sub(" ", ln).strip() for ln in (text.replace("\x00", "")).splitlines())

def looks_binary(text: str) -> bool:
    if not text:
        return True
    sample = text[:2000]
    bad = sum((ord(c) < 9 or (13 < ord(c) < 32)) for c in sample)
    return bad > 20

def detect_lang_safe(text: str) -> str:
    """Very safe language detection with graceful fallback."""
    try:
        # prefer langid (fast & offline), but fall back to langdetect if not present
        try:
            import langid  # type: ignore
            lang, _ = langid.classify(text[:2000])
            return lang or "und"
        except Exception:
            from langdetect import detect  # type: ignore
            return detect(text[:2000]) or "und"
    except Exception:
        return "und"

def html_to_markdown_fallback(html: str) -> str:
    # ultra-simple fallback: strip tags -> pipe table if possible
    try:
        import bs4  # type: ignore
        soup = bs4.BeautifulSoup(html, "html.parser")
        return soup.get_text(" ").strip()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html or "").strip()

def table_to_markdown(element: Dict[str, Any]) -> Optional[str]:
    """Try to convert unstructured Table element to markdown text."""
    # unstructured may give 'text' already; sometimes there is 'metadata' with 'text_as_html'
    txt = (element.get("text") or "").strip()
    if txt:
        return txt
    meta = element.get("metadata") or {}
    html = meta.get("text_as_html") or ""
    if html:
        return html_to_markdown_fallback(html)
    return None

# ------------------------- FastAPI -------------------------
app = FastAPI(title="UXT Pipeline API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ------------------------- Models -------------------------
class EvaluateRequest(BaseModel):
    query: str
    expected_text: str = ""
    k: int = 5
    score_min: float = 0.0
    lang_filter: str = ""  # "uk,en" etc.

# ------------------------- Jobs -------------------------
JOBS: Dict[str, Dict[str, Any]] = {}
def set_job(job_id: str, **kw):
    JOBS.setdefault(job_id, {})
    JOBS[job_id].update(kw)

# ------------------------- Endpoints -------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now().isoformat(), "version": API_VERSION}

# ---------------------------------------------------------
#   JOBS ENDPOINTS
# ---------------------------------------------------------

@app.get("/jobs")
def jobs():
    # Сортуємо джоби за created (нові зверху)
    return [
        JOBS[k] | {"job_id": k}
        for k in sorted(
            JOBS.keys(),
            key=lambda x: JOBS[x].get("created", ""),
            reverse=True
        )
    ]


@app.get("/job/{job_id}")
def job_status(job_id: str):
    state = JOBS.get(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="job not found")
    return state | {"job_id": job_id}


# ---------------------------------------------------------
#   HISTORY ENDPOINTS
# ---------------------------------------------------------

@app.get("/history")
def history():
    rows, bad = read_jsonl_soft(HISTORY_PATH)
    return {"items": rows, "bad_lines": bad}


@app.delete("/history/clear")
def clear_history():
    try:
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        add_history("history_cleared", {"ok": True})
        return {"ok": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

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

@app.get("/chunks")
def chunks_info(sample: int = Query(0, ge=0, le=50)):
    rows, bad = read_jsonl_soft(CHUNKS_PATH)
    info = {
        "count": len(rows),
        "bad_lines": bad,
        "path": str(CHUNKS_PATH),
    }
    if sample > 0 and rows:
        info["sample"] = rows[:sample]
    return info

@app.post("/chunks/repair")
def chunks_repair():
    rows, _ = read_jsonl_soft(CHUNKS_PATH)
    try:
        payload = "".join(json.dumps(r, ensure_ascii=False, default=json_default) + "\n" for r in rows)
        _atomic_write_text(CHUNKS_PATH, payload)
        add_history("chunks_repair", {"kept": len(rows)})
        return {"ok": True, "kept": len(rows)}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/index/info")
def index_info():
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    faiss_ok = INDEX_FAISS.exists()
    chunks_cnt = read_jsonl_soft(CHUNKS_PATH)[0].__len__()
    return {"meta": meta, "faiss": faiss_ok, "chunks": chunks_cnt, "out_dir": str(OUT_DIR)}

@app.post("/ingest")
def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # ---- Ingest/Unstructured controls ----
    strategy: str = Form("fast"),                 # fast | hi_res
    ocr_languages: str = Form("eng+ukr"),
    skip_elements: str = Form(""),                # CSV: Figure,PageBreak,...
    include_metadata: str = Form("true"),
    drop_headers_footers: str = Form("true"),
    # ---- Chunking controls ----
    max_chars: int = Form(1000),
    overlap: int = Form(120),
    join_same_type: str = Form("true"),
    min_text_chars: int = Form(30),
    strip_whitespace: str = Form("true"),
    # ---- Index controls ----
    index_backend: str = Form("faiss"),           # faiss | sklearn
    model_name: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    normalize: str = Form("true"),
    # ---- New features ----
    lang_detect: str = Form("true"),              # detect language per chunk
    prefer_langs: str = Form("uk,en"),            # used only for UI hints, logging
    table_as_markdown: str = Form("true"),
):
    doc_id = sanitize_doc_id(file.filename or f"upload_{uuid.uuid4().hex[:6]}")
    job_id = uuid.uuid4().hex
    set_job(job_id, status="queued", file=file.filename, doc_id=doc_id, created=datetime.now().isoformat())

    suffix = Path(file.filename or "").suffix
    dest = IN_DIR / f"{doc_id}{suffix}"
    dest.write_bytes(file.file.read())

    params = {
        "strategy": strategy,
        "ocr_languages": ocr_languages,
        "skip_elements": [s.strip() for s in (skip_elements or "").split(",") if s.strip()],
        "include_metadata": str(include_metadata).lower() in {"1", "true", "yes", "y"},
        "drop_headers_footers": str(drop_headers_footers).lower() in {"1", "true", "yes", "y"},
        "max_chars": int(max_chars),
        "overlap": int(overlap),
        "join_same_type": str(join_same_type).lower() in {"1", "true", "yes", "y"},
        "min_text_chars": int(min_text_chars),
        "strip_whitespace": str(strip_whitespace).lower() in {"1", "true", "yes", "y"},
        "index_backend": index_backend,
        "model_name": model_name,
        "normalize": str(normalize).lower() in {"1", "true", "yes", "y"},
        "doc_id": doc_id,
        "upload_path": str(dest),
        "lang_detect": str(lang_detect).lower() in {"1", "true", "yes", "y"},
        "prefer_langs": prefer_langs,
        "table_as_markdown": str(table_as_markdown).lower() in {"1", "true", "yes", "y"},
    }

    background_tasks.add_task(_process_one, job_id=job_id, params=params)
    add_history("ingest_queued", {"job_id": job_id, "doc_id": doc_id, "file": file.filename})
    return {"job_id": job_id}

def _partition_safe(path: str, strategy: str, ocr_languages: str, skip_elements: List[str], include_metadata: bool):
    try:
        return _partition_one(
            path,
            strategy=strategy,
            ocr_languages=ocr_languages,
            skip_elements=skip_elements or None,
            include_metadata=include_metadata,
        )
    except Exception:
        alt = "hi_res" if strategy != "hi_res" else "fast"
        return _partition_one(
            path,
            strategy=alt,
            ocr_languages=ocr_languages,
            skip_elements=skip_elements or None,
            include_metadata=include_metadata,
        )

def _process_one(job_id: str, params: Dict[str, Any]):
    try:
        set_job(job_id, status="running", started=datetime.now().isoformat())
        upload_path = params["upload_path"]

        # 1) Parse
        parsed = _partition_safe(
            upload_path,
            strategy=params["strategy"],
            ocr_languages=params["ocr_languages"],
            skip_elements=params["skip_elements"],
            include_metadata=params["include_metadata"],
        )

        elements = parsed.get("elements", [])

        # Filters
        if params["drop_headers_footers"]:
            elements = [e for e in elements if e.get("type") not in {"Header", "Footer"}]

        cleaned = []
        for e in elements:
            t = (e.get("type") or "").strip()
            txt = (e.get("text") or "").strip()

            # Convert tables if asked
            if params["table_as_markdown"] and t == "Table":
                md = table_to_markdown(e)
                txt = (md or "").strip()

            txt = normalize_whitespace(txt)
            if not txt or looks_binary(txt):
                continue

            ee = dict(e)
            ee["text"] = txt
            cleaned.append(ee)
        elements = cleaned

        # 2) Chunking
        doc = {"doc_id": params["doc_id"], "elements": elements}
        chunks = semantic_chunk(
            doc,
            max_chars=params["max_chars"],
            overlap=params["overlap"],
            join_same_type=params["join_same_type"],
            min_text_chars=params["min_text_chars"],
            strip_whitespace=params["strip_whitespace"],
        )

        # 3) Persist chunks with meta.lang
        rows: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            text_i = ch.text or ""
            lang = "und"
            if params["lang_detect"]:
                try:
                    lang = detect_lang_safe(text_i)
                except Exception:
                    lang = "und"

            rows.append({
                "id": f"{params['doc_id']}#{i+1}",
                "doc_id": params["doc_id"],
                "text": text_i,
                "type": ch.type,
                "meta": {
                    "ingested_at": datetime.now().isoformat(),
                    "order": i+1,
                    "lang": lang
                }
            })
        append_jsonl(CHUNKS_PATH, rows)

        # 4) Build index over ALL chunks
        all_rows, _bad = read_jsonl_soft(CHUNKS_PATH)
        info = build_index(
            chunks=all_rows,               # expect dicts with id/doc_id/text
            out_dir=str(OUT_DIR),
            backend=params["index_backend"],
            model_name=params["model_name"],
            normalize=params["normalize"],
        )

        META_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

        set_job(job_id, status="finished", finished=datetime.now().isoformat(), index=info, file=params["upload_path"], doc_id=params["doc_id"])
        add_history("ingest_done", {"job_id": job_id, "doc_id": params["doc_id"], "chunks_added": len(rows), "total_chunks": len(all_rows)})
    except Exception as ex:
        set_job(job_id, status="error", error=str(ex), finished=datetime.now().isoformat())
        add_history("ingest_error", {"job_id": job_id, "error": str(ex)})

def _filter_rows_by_lang(rows: List[Dict[str, Any]], lang_filter: str) -> List[Dict[str, Any]]:
    if not lang_filter:
        return rows
    want = {x.strip().lower() for x in lang_filter.split(",") if x.strip()}
    if not want:
        return rows
    out = []
    for r in rows:
        meta = r.get("meta") or {}
        lang = str(meta.get("lang", "und")).lower()
        if lang in want:
            out.append(r)
    return out

@app.get("/search")
def search(q: str, k: int = 5, score_min: float = 0.0, lang_filter: str = ""):
    try:
        res = index_search(str(OUT_DIR), q, k=k)
        all_rows, _ = read_jsonl_soft(CHUNKS_PATH)
        id2row = {row.get("id"): row for row in _filter_rows_by_lang(all_rows, lang_filter)}
        payload = []
        for r in res:
            sc = float(r.get("score", 0.0))
            if sc < float(score_min):
                continue
            row = id2row.get(r.get("id"))
            if not row:
                continue
            payload.append({
                "score": sc,
                "id": r.get("id"),
                "doc_id": r.get("doc_id"),
                "text": row.get("text", ""),
                "meta": row.get("meta", {})
            })
        add_history("search", {"q": q, "k": k, "score_min": score_min, "results": len(payload), "lang_filter": lang_filter})
        return payload
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/ask")
def ask(q: str, k: int = 5, score_min: float = 0.0, lang_filter: str = ""):
    try:
        res = index_search(str(OUT_DIR), q, k=k)
        all_rows, _ = read_jsonl_soft(CHUNKS_PATH)
        id2row = {row.get("id"): row for row in _filter_rows_by_lang(all_rows, lang_filter)}

        sources, ctx = [], []
        for r in res:
            sc = float(r.get("score", 0.0))
            if sc < float(score_min):
                continue
            row = id2row.get(r["id"])
            if not row:
                continue
            txt = (row.get("text") or "").strip()
            if not txt:
                continue
            sources.append({
                "id": r["id"],
                "doc_id": r.get("doc_id"),
                "score": sc,
                "preview": txt[:300],
                "meta": row.get("meta", {})
            })
            ctx.append(txt)
        answer = "Ключові фрагменти за запитом: «{}»:\n\n{}".format(q, "\n\n---\n\n".join(ctx[:k]) if ctx else "(порожньо)")
        add_history("ask", {"q": q, "k": k, "sources": len(sources), "score_min": score_min, "lang_filter": lang_filter})
        return {"answer": answer, "sources": sources}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        res = index_search(str(OUT_DIR), req.query, k=req.k)
        all_rows, _ = read_jsonl_soft(CHUNKS_PATH)
        id2text = {row.get("id"): row.get("text","") for row in _filter_rows_by_lang(all_rows, req.lang_filter)}
        rows, hits = [], 0
        needle = (req.expected_text or "").lower().strip()
        for r in res:
            sc = float(r.get("score", 0.0))
            if sc < float(req.score_min):
                continue
            txt = id2text.get(r["id"], "")
            ok = bool(needle and needle in txt.lower())
            rows.append({"id": r["id"], "doc_id": r["doc_id"], "score": sc, "match": ok})
            if ok:
                hits += 1
        recall = hits / max(1, req.k)
        positives = sum(1 for x in rows if x["match"])
        prec = (hits / max(1, positives)) if positives else 0.0
        add_history("evaluate", {"q": req.query, "k": req.k, "hits": hits, "score_min": req.score_min, "lang_filter": req.lang_filter})
        return {"recall@k": float(recall), "precision@k": float(prec), "k": int(req.k), "results": rows}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/clusters")
def clusters(n_clusters: int = 8, lang_filter: str = ""):
    try:
        backend, index_obj, meta = load_index(str(OUT_DIR))
        model_name = meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

        if backend == "sklearn":
            X = index_obj["emb"]
            rows, _ = read_jsonl_soft(CHUNKS_PATH)
            rows = _filter_rows_by_lang(rows, lang_filter)
            texts = [r.get("text","") for r in rows][:len(X)]
        else:
            rows, _ = read_jsonl_soft(CHUNKS_PATH)
            rows = _filter_rows_by_lang(rows, lang_filter)
            texts = [r.get("text","") for r in rows if r.get("text")]
            if not texts:
                raise RuntimeError("no chunks to cluster")
            from sentence_transformers import SentenceTransformer
            mdl = SentenceTransformer(model_name)
            enc = mdl.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            X = np.asarray(enc, dtype=np.float32)
            if meta.get("normalize", True):
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                X = X / norms

        from sklearn.cluster import KMeans
        km = KMeans(n_init="auto", random_state=42, n_clusters=int(n_clusters))
        labels = km.fit_predict(X)

        out = []
        for i, r in enumerate(rows[: len(labels)]):
            out.append({
                "id": r.get("id"),
                "doc_id": r.get("doc_id"),
                "cluster": int(labels[i]),
                "text_preview": (r.get("text","")[:160] or "").replace("\n", " "),
                "meta": r.get("meta", {})
            })
        add_history("clusters", {"n_clusters": n_clusters, "count": len(out), "lang_filter": lang_filter})
        return {"clusters": out}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
