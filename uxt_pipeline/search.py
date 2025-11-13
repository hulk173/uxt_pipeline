from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json, os, numpy as np
from uxt_pipeline.models import SearchResult, Chunk

def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _search_faiss(index_path: str, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    import faiss
    faiss.normalize_L2(q)
    index = faiss.read_index(index_path)
    D, I = index.search(q.astype(np.float32), top_k)
    return D[0], I[0]

def _search_sklearn(index_path: str, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity
    nn = joblib.load(index_path)
    X = nn._fit_X
    sims = cosine_similarity(q, X)[0]
    idx = np.argsort(-sims)[:top_k]
    scores = sims[idx]
    return scores, idx

def search(query: str, config: Dict[str, Any]) -> List[SearchResult]:
    backend   = config["index"]["backend"]
    model_name= config["index"]["model_name"]
    top_k     = int(config["index"]["top_k"])
    index_path= config["index"]["index_path"]
    meta_path = config["index"]["meta_path"]

    if not (os.path.exists(meta_path) and os.path.exists(index_path)):
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("chunks"):
        return []

    enc = _load_encoder(model_name)
    q = enc.encode([query], convert_to_numpy=True)

    if backend == "faiss":
        scores, idxs = _search_faiss(index_path, q, top_k)
    else:
        scores, idxs = _search_sklearn(index_path, q, top_k)

    chunks = [Chunk(**c) for c in meta["chunks"]]
    results: List[SearchResult] = []
    for s, i in zip(scores, idxs):
        results.append(SearchResult(score=float(s), chunk=chunks[int(i)]))
    return results
