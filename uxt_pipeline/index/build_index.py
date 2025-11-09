# uxt_pipeline/index/build_index.py
from __future__ import annotations
from typing import List
import json
import os
import numpy as np
from uxt_pipeline.models import Chunk
from uxt_pipeline.utils import ensure_dir

def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def _build_faiss_index(emb: np.ndarray, index_path: str) -> None:
    import faiss  # <--- Додано імпорт всередині функції
    ensure_dir(index_path)
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb.astype(np.float32))
    faiss.write_index(index, index_path)


def _build_sklearn_index(emb: np.ndarray, index_path: str) -> None:
    ensure_dir(index_path)
    from sklearn.neighbors import NearestNeighbors
    import joblib
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(emb)
    joblib.dump(nn, index_path)

def build_index(
    chunks: List[Chunk],
    backend: str,
    model_name: str,
    index_path: str,
    meta_path: str,
) -> None:
    # Порожній набір — пишемо «порожню» мету і прибираємо індекс
    if not chunks:
        ensure_dir(meta_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"backend": backend, "model": model_name, "ids": [], "chunks": [], "shape": [0, 0]},
                f,
                ensure_ascii=False,
            )
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
        except Exception:
            pass
        return

    texts = [c.text for c in chunks]
    enc = _load_encoder(model_name)
    emb = enc.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    if backend == "faiss":
        _build_faiss_index(emb, index_path)
    else:
        _build_sklearn_index(emb, index_path)

    # 🔒 JSON-safe метадані: використовуємо mode="json"
    meta = {
        "backend": backend,
        "model": model_name,
        "ids": [c.id for c in chunks],
        "chunks": [c.model_dump(mode="json") for c in chunks],
        "shape": list(emb.shape),
    }
    ensure_dir(meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
