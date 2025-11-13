from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # faiss optional
    faiss = None  # type: ignore

# -------------------------- utils --------------------------

def _to_numpy(x: Any) -> np.ndarray:
    """
    Robust conversion to np.ndarray[float32] from:
    - torch.Tensor or list[torch.Tensor]
    - list[list[float]] / list[np.ndarray] / np.ndarray
    """
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    if torch is not None and isinstance(x, torch.Tensor):  # single tensor
        return x.detach().cpu().numpy().astype(np.float32, copy=False)

    # list[Tensor] → stack
    if torch is not None and isinstance(x, list) and x and isinstance(x[0], torch.Tensor):
        arr = torch.stack(x).detach().cpu().numpy()
        return arr.astype(np.float32, copy=False)

    # list[...] or ndarray → asarray
    return np.asarray(x, dtype=np.float32)

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms

# -------------------------- I/O ----------------------------

def _save_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

# -------------------------- API ----------------------------

def build_index(
    chunks: List[Dict[str, Any]],
    out_dir: str,
    backend: str = "faiss",  # "faiss" | "sklearn"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Build vector index from chunks (each row must have fields: id, doc_id, text).
    Writes:
      - out_dir/index.faiss (if faiss)
      - out_dir/meta.json   (mapping + settings)
      - out_dir/emb.npy     (if sklearn)
    Returns meta dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # texts & ids
    texts: List[str] = []
    ids: List[str] = []
    doc_ids: List[str] = []
    for r in chunks:
        t = (r.get("text") or "").strip()
        if not t:
            continue
        texts.append(t)
        ids.append(str(r.get("id")))
        doc_ids.append(str(r.get("doc_id")))

    if not texts:
        raise RuntimeError("No texts to index")

    # embeddings
    from sentence_transformers import SentenceTransformer
    mdl = SentenceTransformer(model_name)
    embs_raw = mdl.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    X = _to_numpy(embs_raw)  # (N, D) float32
    if normalize:
        X = _l2_normalize(X)

    dim = int(X.shape[1])

    meta: Dict[str, Any] = {
        "backend": backend,
        "dim": dim,
        "count": int(X.shape[0]),
        "ids": ids,          # position ↔ id
        "doc_ids": doc_ids,  # position ↔ doc_id
        "model_name": model_name,
        "normalize": bool(normalize),
    }

    if backend == "faiss":
        if faiss is None:
            raise RuntimeError("faiss is not installed. Use backend='sklearn' or install faiss-cpu.")
        index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
        index.add(X)  # type: ignore
        faiss.write_index(index, str(out / "index.faiss"))  # type: ignore
    elif backend == "sklearn":
        # save matrix for manual dot-product search
        np.save(out / "emb.npy", X)
    else:
        raise ValueError("backend must be 'faiss' or 'sklearn'")

    _save_meta(out / "meta.json", meta)
    return meta

def load_index(out_dir: str) -> Tuple[str, Any, Dict[str, Any]]:
    out = Path(out_dir)
    meta = _load_meta(out / "meta.json")
    backend = meta.get("backend", "faiss")
    if backend == "faiss":
        if faiss is None:
            raise RuntimeError("faiss is not installed but meta expects faiss backend.")
        index = faiss.read_index(str(out / "index.faiss"))  # type: ignore
        return "faiss", index, meta
    else:
        X = np.load(out / "emb.npy")
        return "sklearn", {"emb": _to_numpy(X)}, meta

def search(out_dir: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    backend, index, meta = load_index(out_dir)
    from sentence_transformers import SentenceTransformer

    model_name = meta.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    mdl = SentenceTransformer(model_name)
    qv_raw = mdl.encode([query], convert_to_numpy=True, show_progress_bar=False)
    qv = _to_numpy(qv_raw)  # (1, D)
    if meta.get("normalize", True):
        qv = _l2_normalize(qv)

    ids = meta.get("ids", [])
    doc_ids = meta.get("doc_ids", [])

    if backend == "faiss":
        D, I = index.search(qv, k)  # type: ignore
        scores = D[0].tolist()
        idxs = I[0].tolist()
    else:
        X = index["emb"]  # (N, D)
        scores = (X @ qv.T).ravel().tolist()  # cosine/IP
        idxs = np.argsort(scores)[::-1][:k].tolist()
        scores = [scores[i] for i in idxs]

    out: List[Dict[str, Any]] = []
    for rank, (i, s) in enumerate(zip(idxs, scores)):
        if i < 0 or i >= len(ids):
            continue
        out.append({"rank": rank + 1, "id": ids[i], "doc_id": doc_ids[i], "score": float(s)})
    return out
