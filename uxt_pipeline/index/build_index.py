# uxt_pipeline/index/build_index.py
from __future__ import annotations

from typing import List, Tuple, Any
from pathlib import Path
import json
import numpy as np

from uxt_pipeline.models import Chunk
from uxt_pipeline.utils import ensure_dir
from sentence_transformers import SentenceTransformer  # type: ignore

# FAISS може бути відсутній; оголошуємо як Any, щоб Pylance не лаявся
try:
    import faiss  # type: ignore
    faiss = faiss  # type: Any
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore[assignment]
    HAS_FAISS = False

# sklearn як альтернатива
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    import joblib  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def _embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    if HAS_FAISS and faiss is not None and hasattr(faiss, "normalize_L2"):
        faiss.normalize_L2(x)  # type: ignore[attr-defined]
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _store_meta(chunks: List[Chunk], meta_path: str, model_name: str) -> None:
    meta = {
        "model_name": model_name,
        "chunks": [c.model_dump() if hasattr(c, "model_dump") else c.__dict__ for c in chunks],
    }
    ensure_dir(Path(meta_path).parent.as_posix())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def _build_faiss_index(emb: np.ndarray, index_path: str) -> None:
    if not HAS_FAISS or faiss is None:
        raise RuntimeError("FAISS is not installed. Install faiss-cpu to use the FAISS backend.")
    index = faiss.IndexFlatIP(emb.shape[1])  # type: ignore[attr-defined]
    index.add(emb.astype(np.float32))
    ensure_dir(Path(index_path).parent.as_posix())
    faiss.write_index(index, index_path)  # type: ignore[attr-defined]


def _build_sklearn_index(emb: np.ndarray, index_path: str) -> None:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not installed.")
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(emb)
    ensure_dir(Path(index_path).parent.as_posix())
    joblib.dump(nn, index_path)


def build_index(
    chunks: List[Chunk],
    *,
    backend: str = "faiss",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: str = "data/out/index.faiss",
    meta_path: str = "data/out/index_meta.json",
    normalize: bool = True,
) -> Tuple[str, str]:
    texts = [c.text for c in chunks if (getattr(c, "text", "") or "").strip()]
    if not texts:
        _store_meta(chunks, meta_path, model_name)
        ensure_dir(Path(index_path).parent.as_posix())
        Path(index_path).write_bytes(b"")
        return index_path, meta_path

    emb = _embed_texts(texts, model_name)
    if normalize:
        emb = _l2_normalize(emb)

    if backend.lower() == "faiss":
        _build_faiss_index(emb, index_path)
    elif backend.lower() == "sklearn":
        _build_sklearn_index(emb, index_path)
    else:
        raise ValueError(f"Unknown index backend: {backend}")

    _store_meta(chunks, meta_path, model_name)
    return index_path, meta_path
