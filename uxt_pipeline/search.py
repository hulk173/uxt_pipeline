from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os, json, numpy as np

# --- опційні імпорти ---
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover
    hnswlib = None  # type: ignore


@dataclass
class IndexConfig:
    engine: str = "faiss"  # faiss | hnsw | numpy
    dim: int = 384
    store_dir: str = ".index"
    ef_construction: int = 200
    M: int = 32
    nprobe: int = 10


class VectorIndex:
    """Диск-персистентний векторний індекс із метаданими чанків."""

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        os.makedirs(cfg.store_dir, exist_ok=True)
        self.meta_path = os.path.join(cfg.store_dir, "meta.json")
        self.dim = int(cfg.dim)
        self.meta: Dict[str, Dict[str, Any]] = {}

    # ----------------- побудова -----------------
    def build(self, embeddings: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be (N,{self.dim})")

        if self.cfg.engine == "faiss":
            if faiss is None:
                raise RuntimeError("faiss-cpu не встановлено. pip install faiss-cpu")
            index = faiss.IndexFlatIP(self.dim)  # cosine, якщо вектори L2-нормовані
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, os.path.join(self.cfg.store_dir, "faiss.index"))

        elif self.cfg.engine == "hnsw":
            if hnswlib is None:
                raise RuntimeError("hnswlib не встановлено. pip install hnswlib")
            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.init_index(
                max_elements=int(embeddings.shape[0]),
                ef_construction=int(self.cfg.ef_construction),
                M=int(self.cfg.M),
            )
            labels = np.arange(embeddings.shape[0])
            index.add_items(embeddings.astype(np.float32), labels)
            index.save_index(os.path.join(self.cfg.store_dir, "hnsw.index"))

        else:  # numpy fallback
            # Явно називаємо аргументи — це заспокоює Pylance
            np.savez_compressed(
                file=os.path.join(self.cfg.store_dir, "numpy_index.npz"),
                X=embeddings.astype(np.float32),
            )

        # метадані
        self.meta = {str(i): metas[i] for i in range(len(metas))}
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    # ----------------- завантаження -----------------
    def _load_backend(self) -> Any:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if self.cfg.engine == "faiss":
            if faiss is None:
                raise RuntimeError("faiss-cpu не встановлено.")
            return faiss.read_index(os.path.join(self.cfg.store_dir, "faiss.index"))

        if self.cfg.engine == "hnsw":
            if hnswlib is None:
                raise RuntimeError("hnswlib не встановлено.")
            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.load_index(os.path.join(self.cfg.store_dir, "hnsw.index"))
            return index

        data = np.load(os.path.join(self.cfg.store_dir, "numpy_index.npz"))
        X: np.ndarray = data["X"]
        return X

    # ----------------- пошук -----------------
    def search(self, q_emb: np.ndarray, top_k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        if q_emb.ndim != 2 or q_emb.shape[0] != 1 or q_emb.shape[1] != self.dim:
            raise ValueError(f"Query embedding must be (1,{self.dim})")

        backend = self._load_backend()

        if self.cfg.engine == "faiss":
            sims, ids = backend.search(q_emb.astype(np.float32), int(top_k))  # type: ignore[attr-defined]
            hits: List[Tuple[int, float, Dict[str, Any]]] = []
            for i, s in zip(ids[0], sims[0]):
                if int(i) < 0:
                    continue
                md = self.meta.get(str(int(i)), {})
                hits.append((int(i), float(s), md))
            return hits

        if self.cfg.engine == "hnsw":
            labels, dists = backend.knn_query(q_emb.astype(np.float32), k=int(top_k))  # type: ignore[attr-defined]
            hits: List[Tuple[int, float, Dict[str, Any]]] = []
            for i, d in zip(labels[0], dists[0]):
                md = self.meta.get(str(int(i)), {})
                hits.append((int(i), float(1.0 - d), md))  # cosine -> similarity
            return hits

        # numpy fallback
        X: np.ndarray = backend  # (N, d)
        v: np.ndarray = q_emb[0]  # (d,)
        x_norms = np.linalg.norm(X, axis=1)  # <-- тут явно передаємо x
        v_norm = float(np.linalg.norm(v))
        denom = (x_norms * (v_norm + 1e-9)) + 1e-9
        sims = (X @ v) / denom
        order = np.argsort(-sims)[:int(top_k)]
        return [(int(i), float(sims[i]), self.meta.get(str(int(i)), {})) for i in order]
