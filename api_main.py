# api_main.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil, threading, traceback, json, os, re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# pipeline pieces (твої модулі)
from uxt_pipeline.utils import load_config, uuid4, ensure_dir
from uxt_pipeline.models import IngestJob, Chunk, SearchResult
from uxt_pipeline.ingest.partitioner import _partition_one
from uxt_pipeline.transform.chunker import semantic_chunk
from uxt_pipeline.storage.export_jsonl import export_jsonl
from uxt_pipeline.storage.export_parquet import export_parquet
from uxt_pipeline.storage.export_sqlite import export_sqlite
from uxt_pipeline.index.build_index import build_index
from uxt_pipeline.index.search import search as search_index
from uxt_pipeline.analysis.nlp import sentiment_score, summarize_text

# Легка LLM для RAG-QA (з фолбеком)
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    _qa_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    _qa_mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
except Exception:
    _qa_tok = _qa_mod = None  # працюємо з фолбеком

# ---------------- defaults ----------------
DEFAULTS: Dict[str, Any] = {
    "ingest": {
        "strategy": "fast",
        "ocr_languages": "eng+ukr",
        "skip_elements": ["Footer", "Header"],
        "output_dir": "data/out",
        "include_metadata": True,
    },
    "chunking": {
        "max_chars": 1200,
        "overlap": 150,
        "join_same_type": True,
        "min_text_chars": 30,
        "strip_whitespace": True,
    },
    "export": {
        "jsonl_path": "data/out/chunks.jsonl",
        "parquet_path": "data/out/chunks.parquet",
        "sqlite_path": "data/out/chunks.sqlite",
        "table_name": "chunks",
    },
    "index": {
        "backend": "faiss",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "index_path": "data/out/index.faiss",
        "meta_path": "data/out/index_meta.json",
        "normalize": True,
    },
    "api": {"cors_origins": ["*"]},
}

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    if not isinstance(b, dict):
        return out
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

CFG_PATH = "configs/default.yaml"
try:
    user_cfg = load_config(CFG_PATH)
except Exception:
    user_cfg = {}
CONFIG = deep_merge(DEFAULTS, user_cfg)
ING, CHN, EXP, IDX = CONFIG["ingest"], CONFIG["chunking"], CONFIG["export"], CONFIG["index"]

# ---------------- state ----------------
JOBS: Dict[str, IngestJob] = {}
DOCS: Dict[str, List[Chunk]] = {}
LOCK = threading.Lock()

HISTORY_PATH = Path("data/history.jsonl")
ensure_dir(str(HISTORY_PATH))

def _append_history(row: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------------- app ----------------
app = FastAPI(title="UXT Pipeline API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "version": "1.1.0"}

# ---------------- helpers ----------------
def _effective_cfg(
    strategy: Optional[str], ocr_languages: Optional[str],
    max_chars: Optional[int], overlap: Optional[int],
    join_same_type: Optional[bool], min_text_chars: Optional[int],
    strip_whitespace: Optional[bool], index_backend: Optional[str],
    model_name: Optional[str], top_k: Optional[int], normalize: Optional[bool] = None,
):
    eff_ing = dict(ING)
    eff_chn = dict(CHN)
    eff_idx = dict(IDX)
    if strategy: eff_ing["strategy"] = strategy
    if ocr_languages: eff_ing["ocr_languages"] = ocr_languages
    if max_chars is not None: eff_chn["max_chars"] = int(max_chars)
    if overlap is not None: eff_chn["overlap"] = int(overlap)
    if join_same_type is not None: eff_chn["join_same_type"] = bool(join_same_type)
    if min_text_chars is not None: eff_chn["min_text_chars"] = int(min_text_chars)
    if strip_whitespace is not None: eff_chn["strip_whitespace"] = bool(strip_whitespace)
    if index_backend: eff_idx["backend"] = index_backend
    if model_name: eff_idx["model_name"] = model_name
    if top_k is not None: eff_idx["top_k"] = int(top_k)
    if normalize is not None: eff_idx["normalize"] = bool(normalize)
    return eff_ing, eff_chn, eff_idx

def _process_one(job_id: str, tmp_path: Path, eff_ing: Dict[str, Any], eff_chn: Dict[str, Any], eff_idx: Dict[str, Any]) -> None:
    try:
        parsed = _partition_one(
            str(tmp_path),
            strategy=eff_ing["strategy"],
            ocr_languages=eff_ing["ocr_languages"],
            skip_elements=eff_ing["skip_elements"],
            include_metadata=eff_ing["include_metadata"],
        )

        chunks: List[Chunk] = semantic_chunk(
            parsed,
            max_chars=eff_chn["max_chars"],
            overlap=eff_chn["overlap"],
            join_same_type=eff_chn["join_same_type"],
            min_text_chars=eff_chn["min_text_chars"],
            strip_whitespace=eff_chn["strip_whitespace"],
        )

        with LOCK:
            DOCS[parsed["doc_id"]] = chunks

        export_jsonl(chunks, EXP["jsonl_path"])
        if chunks:
            try:
                export_parquet(chunks, EXP["parquet_path"])
            except Exception:
                pass
        export_sqlite(chunks, EXP["sqlite_path"], EXP["table_name"])

        build_index(
            chunks,
            backend=eff_idx["backend"],
            model_name=eff_idx["model_name"],
            index_path=IDX["index_path"],
            meta_path=IDX["meta_path"],
            normalize=eff_idx.get("normalize", True),
        )

        with LOCK:
            JOBS[job_id].status = "finished"
        _append_history({"event": "ingest_finished", "doc_id": parsed.get("doc_id"), "chunks": len(chunks)})

    except Exception as exc:
        with LOCK:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"{exc}\n{traceback.format_exc()}"

# ---------------- ingest ----------------
@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    strategy: Optional[str] = Form(None),
    ocr_languages: Optional[str] = Form(None),
    max_chars: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    join_same_type: Optional[bool] = Form(None),
    min_text_chars: Optional[int] = Form(None),
    strip_whitespace: Optional[bool] = Form(None),
    index_backend: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    top_k: Optional[int] = Form(None),
    normalize: Optional[bool] = Form(None),
) -> JSONResponse:
    job_id = uuid4()
    safe_name = file.filename or "uploaded.bin"
    with LOCK:
        JOBS[job_id] = IngestJob(id=job_id, status="queued", input_paths=[safe_name])

    eff_ing, eff_chn, eff_idx = _effective_cfg(
        strategy, ocr_languages, max_chars, overlap, join_same_type, min_text_chars,
        strip_whitespace, index_backend, model_name, top_k, normalize
    )

    uploads_dir = Path(ING["output_dir"]) / "uploads"
    ensure_dir(str(uploads_dir))
    tmp_path = uploads_dir / safe_name
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t = threading.Thread(target=_process_one, args=(job_id, tmp_path, eff_ing, eff_chn, eff_idx), daemon=True)
    t.start()
    with LOCK:
        JOBS[job_id].status = "processing"

    payload = {"job_id": job_id, "doc_hint": safe_name, "effective_config": {"ingest": eff_ing, "chunking": eff_chn, "index": eff_idx}}
    return JSONResponse(content=jsonable_encoder(payload), status_code=202)

# ---------------- batch ingest ----------------
@app.post("/ingest_batch")
async def ingest_batch(
    files: List[UploadFile] = File(...),
    strategy: Optional[str] = Form(None),
    ocr_languages: Optional[str] = Form(None),
    max_chars: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    join_same_type: Optional[bool] = Form(None),
    min_text_chars: Optional[int] = Form(None),
    strip_whitespace: Optional[bool] = Form(None),
    index_backend: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    top_k: Optional[int] = Form(None),
    normalize: Optional[bool] = Form(None),
) -> JSONResponse:
    eff_ing, eff_chn, eff_idx = _effective_cfg(
        strategy, ocr_languages, max_chars, overlap, join_same_type, min_text_chars,
        strip_whitespace, index_backend, model_name, top_k, normalize
    )
    uploads_dir = Path(ING["output_dir"]) / "uploads"
    ensure_dir(str(uploads_dir))

    jobs = []
    for f in files:
        job_id = uuid4()
        safe_name = f.filename or "uploaded.bin"
        with LOCK:
            JOBS[job_id] = IngestJob(id=job_id, status="queued", input_paths=[safe_name])
        tmp_path = uploads_dir / safe_name
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        t = threading.Thread(target=_process_one, args=(job_id, tmp_path, eff_ing, eff_chn, eff_idx), daemon=True)
        t.start()
        with LOCK:
            JOBS[job_id].status = "processing"
        jobs.append({"job_id": job_id, "doc_hint": safe_name})

    return JSONResponse(content=jsonable_encoder({"jobs": jobs}), status_code=202)

# ---------------- status / doc / search ----------------
@app.get("/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(content=jsonable_encoder(job))

@app.get("/doc/{doc_id}")
def get_doc(doc_id: str):
    chunks = DOCS.get(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="doc not found")
    return JSONResponse(content=jsonable_encoder(chunks))

@app.get("/search")
def search(q: str):
    res: List[SearchResult] = search_index(q, CONFIG)
    _append_history({"event": "search", "q": q, "results": len(res)})
    return JSONResponse(content=jsonable_encoder(res))

# ---------------- evaluation ----------------
@app.post("/evaluate")
def evaluate(payload: Dict[str, Any] = Body(...)):
    query = payload.get("query", "")
    expected = payload.get("expected_text", "")
    k = int(payload.get("k", IDX["top_k"]))
    results: List[SearchResult] = search_index(query, CONFIG)[:k]

    hits = 0
    rows = []
    for r in results:
        txt = r.chunk.text or ""
        ok = (expected.lower() in txt.lower()) if expected else False
        hits += 1 if ok else 0
        rows.append({"score": r.score, "match": ok, "text": txt[:200]})

    recall_at_k = 1.0 if hits > 0 else 0.0
    precision_at_k = hits / max(1, len(results))
    out = {"recall@k": recall_at_k, "precision@k": precision_at_k, "k": k, "results": rows}
    _append_history({"event": "evaluate", "q": query, "recall@k": recall_at_k, "precision@k": precision_at_k})
    return JSONResponse(content=jsonable_encoder(out))

# ---------------- NLP ----------------
@app.post("/analyze/sentiment")
def analyze_sentiment(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    s = sentiment_score(text)
    return JSONResponse(content=jsonable_encoder(s))

@app.post("/analyze/summarize")
def analyze_summarize(payload: Dict[str, Any] = Body(...)):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    summary = summarize_text(text)
    return JSONResponse(content=jsonable_encoder({"summary": summary}))

# ---------------- RAG-QA ----------------
def _answer_flant5(prompt: str, max_new_tokens: int = 160) -> str:
    if _qa_tok is None or _qa_mod is None:
        # фолбек: повернемо перший абзац контексту
        body = prompt.split("Context:", 1)[-1].strip()
        return re.split(r"\n{2,}", body)[0][:400]
    ids = _qa_tok(prompt, return_tensors="pt").input_ids
    out = _qa_mod.generate(ids, max_new_tokens=max_new_tokens)
    return _qa_tok.decode(out[0], skip_special_tokens=True)

@app.get("/ask")
def ask(q: str, k: int = 5, max_context_chars: int = 3500):
    hits: List[SearchResult] = search_index(q, CONFIG)[:k]
    ctx_parts, src = [], []
    for i, h in enumerate(hits):
        t = (h.chunk.text or "").strip().replace("\n", " ")
        if not t:
            continue
        ctx_parts.append(f"[{i+1}] {t}")
        src.append({"rank": i + 1, "score": float(h.score), "doc_id": h.chunk.doc_id, "text": t[:300]})
    ctx = "\n\n".join(ctx_parts)
    if len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars]
    prompt = f"Answer concisely and factually using ONLY the context.\nQuestion: {q}\nContext:\n{ctx}\nAnswer:"
    answer = _answer_flant5(prompt)
    return {"answer": answer, "sources": src}

# ---------------- clusters ----------------
@app.get("/clusters")
def clusters(n_clusters: int = 8):
    if not os.path.exists(IDX["meta_path"]):
        return {"clusters": []}
    with open(IDX["meta_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = [Chunk(**c) for c in meta.get("chunks", [])]
    if not chunks:
        return {"clusters": []}

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.cluster import KMeans
    except Exception as exc:
        return {"detail": f"install sentence-transformers & scikit-learn: {exc}"}

    model = SentenceTransformer(IDX["model_name"])
    X = model.encode([c.text for c in chunks], convert_to_numpy=True, show_progress_bar=False)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    out = [{"id": c.id, "doc_id": c.doc_id, "label": int(lbl)} for c, lbl in zip(chunks, labels)]
    return {"clusters": out, "n_clusters": n_clusters}

# ---------------- utils ----------------
@app.delete("/outputs")
def clear_outputs():
    out_dir = Path(ING["output_dir"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "uploads").mkdir(parents=True, exist_ok=True)
    _append_history({"event": "clear_outputs"})
    return {"cleared": True}

@app.get("/history")
def history(limit: int = 200):
    if not HISTORY_PATH.exists():
        return []
    rows = [json.loads(x) for x in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[-limit:]
