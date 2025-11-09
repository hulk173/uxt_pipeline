# uxt_pipeline/index/build_index.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# FAISS опційний
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors  # type: ignore


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _json_serial(obj: Any) -> str:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Построчне L2-нормування. Безпечне до нульових векторів.
    """
    if arr.ndim != 2:
        raise ValueError("Expected 2D embeddings array")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


@dataclass
class BuiltIndexInfo:
    backend: str
    dim: int
    count: int
    index_path: str
    meta_path: str
    model_name: str


def build_index(
    chunks: List[Dict[str, Any]],
    out_dir: str,
    backend: str = "faiss",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Створює індекс у каталозі out_dir.
    Очікується, що кожен chunk має {"id": str, "doc_id": str, "text": str}
    """
    ensure_dir(out_dir)
    out = Path(out_dir)

    texts: List[str] = []
    ids: List[str] = []
    doc_ids: List[str] = []
    for ch in chunks:
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        texts.append(t)
        ids.append(str(ch.get("id", "")))
        doc_ids.append(str(ch.get("doc_id", "")))

    if not texts:
        raise ValueError("No non-empty texts in chunks to build index")

    # 1) Ембеддинги
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    # 2) L2-нормування (для cosine через IP у FAISS і для sklearn cosine)
    if normalize:
        emb = _l2_normalize(emb)

    dim = int(emb.shape[1])

    # 3) Побудова індексу
    backend = (backend or "faiss").lower()
    if backend not in {"faiss", "sklearn"}:
        backend = "faiss"

    index_path = str(out / ("index.faiss" if backend == "faiss" else "index.pkl"))
    meta_path = str(out / "meta.json")

    if backend == "faiss":
        if not HAS_FAISS:
            backend = "sklearn"  # fallback
        else:
            # cosine similarity через IP на нормованих векторах
            index = faiss.IndexFlatIP(dim)
            index.add(emb)  # type: ignore
            faiss.write_index(index, index_path)

    if backend == "sklearn":
        nn = NearestNeighbors(n_neighbors=10, metric="cosine")
        nn.fit((emb))
        with open(index_path, "wb") as f:
            pickle.dump({"nn": nn, "emb": emb}, f)

    # 4) Метадані
    meta: Dict[str, Any] = {
        "backend": backend,
        "model_name": model_name,
        "dim": dim,
        "count": int(emb.shape[0]),
        "ids": ids,
        "doc_ids": doc_ids,
        "created_at": datetime.now(),
        "normalize": bool(normalize),
        "index_path": index_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, default=_json_serial, indent=2)

    return BuiltIndexInfo(
        backend=backend,
        dim=dim,
        count=len(texts),
        index_path=index_path,
        meta_path=meta_path,
        model_name=model_name,
    ).__dict__


def load_index(out_dir: str) -> Tuple[str, Any, Dict[str, Any]]:
    out = Path(out_dir)
    meta_path = out / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {out_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    backend = meta.get("backend", "faiss")
    index_path = meta.get("index_path")

    if backend == "faiss":
        if not HAS_FAISS:
            raise RuntimeError("FAISS backend requested but faiss is not available")
        index = faiss.read_index(index_path)
        return backend, index, meta

    with open(index_path, "rb") as f:
        payload = pickle.load(f)
    return backend, payload, meta


def search(out_dir: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    backend, index_obj, meta = load_index(out_dir)

    # encode query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(meta["model_name"])
    q = model.encode([query], convert_to_numpy=True).astype(np.float32)
    if meta.get("normalize", True):
        q = _l2_normalize(q)

    ids = meta.get("ids", [])
    doc_ids = meta.get("doc_ids", [])

    if backend == "faiss":
        D, I = index_obj.search(q, k)  # (1, k)
        idxs = I[0].tolist()
        scores = D[0].tolist()
    else:
        nn = index_obj["nn"]
        dist, idxs_arr = nn.kneighbors(q, n_neighbors=k)
        idxs = idxs_arr[0].tolist()
        # cosine distance -> similarity
        scores = (1.0 - dist[0]).tolist()

    results: List[Dict[str, Any]] = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        rid = ids[i] if i < len(ids) else ""
        rdoc = doc_ids[i] if i < len(doc_ids) else ""
        results.append({"rank": rank, "score": float(s), "id": rid, "doc_id": rdoc})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or query vector index")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    b.add_argument("--chunks", type=str, required=True)
    b.add_argument("--out", type=str, required=True)
    b.add_argument("--backend", type=str, default="faiss", choices=["faiss", "sklearn"])
    b.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--no-normalize", action="store_true")

    q = sub.add_parser("search")
    q.add_argument("--out", type=str, required=True)
    q.add_argument("--query", type=str, required=True)
    q.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "build":
        rows = [json.loads(s) for s in Path(args.chunks).read_text(encoding="utf-8").splitlines() if s.strip()]
        info = build_index(
            rows,
            out_dir=args.out,
            backend=args.backend,
            model_name=args.model,
            normalize=not args.no_normalize,
        )
        print(json.dumps(info, indent=2, ensure_ascii=False, default=_json_serial))
    elif args.cmd == "search":
        print(json.dumps(search(args.out, args.query, k=args.k), indent=2, ensure_ascii=False))
    else:
        parser.print_help()
