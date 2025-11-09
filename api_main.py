# api_main.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil, copy, threading, traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from uxt_pipeline.utils import load_config, uuid4, ensure_dir
from uxt_pipeline.models import IngestJob, Chunk
from uxt_pipeline.ingest.partitioner import _partition_one
from uxt_pipeline.transform.chunker import semantic_chunk
from uxt_pipeline.storage.export_jsonl import export_jsonl
from uxt_pipeline.storage.export_parquet import export_parquet
from uxt_pipeline.storage.export_sqlite import export_sqlite
from uxt_pipeline.index.build_index import build_index
from uxt_pipeline.index.search import search as search_index

# ---------------- defaults + safe merge ----------------
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
        "min_text_chars": 20,
        "strip_whitespace": True,
    },
    "export": {
        "jsonl_path": "data/out/chunks.jsonl",
        "parquet_path": "data/out/chunks.parquet",
        "sqlite_path": "data/out/chunks.sqlite",
        "table_name": "chunks",
    },
    "index": {
        "backend": "faiss",  # faiss | sklearn
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "index_path": "data/out/index.faiss",
        "meta_path": "data/out/index_meta.json",
    },
    "api": {"cors_origins": ["*"]},
}
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

CONFIG_PATH = "configs/default.yaml"
try:
    user_cfg = load_config(CONFIG_PATH)
except Exception:
    user_cfg = {}
CONFIG = deep_merge(DEFAULTS, user_cfg)
ING, CHN, EXP, IDX = CONFIG["ingest"], CONFIG["chunking"], CONFIG["export"], CONFIG["index"]

# ---------------- in-memory state ----------------
JOBS: Dict[str, IngestJob] = {}
DOCS: Dict[str, List[Chunk]] = {}
LOCK = threading.Lock()

# ---------------- app ----------------
app = FastAPI(title="UXT Pipeline API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(): return {"ok": True, "version": "0.2.0"}

# ---------------- bg worker ----------------
def _process_job(job_id: str, tmp_path: Path, eff_ing: Dict[str, Any], eff_chn: Dict[str, Any], eff_idx: Dict[str, Any]) -> None:
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
            try: export_parquet(chunks, EXP["parquet_path"])
            except Exception: pass
        export_sqlite(chunks, EXP["sqlite_path"], EXP["table_name"])

        build_index(
            chunks,
            backend=eff_idx["backend"],
            model_name=eff_idx["model_name"],
            index_path=IDX["index_path"],
            meta_path=IDX["meta_path"],
        )

        with LOCK:
            JOBS[job_id].status = "finished"
    except Exception as e:
        with LOCK:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"{e}\n{traceback.format_exc()}"

# ---------------- ingest (async) ----------------
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
) -> JSONResponse:
    job_id = uuid4()
    safe_name = file.filename or "uploaded.bin"
    job = IngestJob(id=job_id, status="queued", input_paths=[safe_name])
    with LOCK:
        JOBS[job_id] = job

    # effective config
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

    # save upload
    uploads_dir = Path(ING["output_dir"]) / "uploads"
    ensure_dir(str(uploads_dir))
    tmp_path = uploads_dir / safe_name
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # start background thread
    t = threading.Thread(target=_process_job, args=(job_id, tmp_path, eff_ing, eff_chn, eff_idx), daemon=True)
    t.start()
    with LOCK:
        JOBS[job_id].status = "processing"

    payload = {
        "job_id": job_id,
        "doc_hint": safe_name,
        "effective_config": {"ingest": eff_ing, "chunking": eff_chn, "index": eff_idx},
    }
    return JSONResponse(content=jsonable_encoder(payload), status_code=202)

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
    res = search_index(q, CONFIG)
    return JSONResponse(content=jsonable_encoder(res))
