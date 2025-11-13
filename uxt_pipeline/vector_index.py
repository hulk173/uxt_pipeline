from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os, json, numpy as np

# --- опційні бекенди ---
try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    import hnswlib  # type: ignore
except Exception:
    hnswlib = None  # type: ignore


@dataclass
class IndexConfig:
    """Конфігурація для VectorIndex"""
    engine: str = "faiss"      # faiss | hnsw | numpy
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
        """Створює індекс і зберігає його на диск."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be (N,{self.dim})")

        engine = self.cfg.engine.lower()

        # --- FAISS ---
        if engine == "faiss":
            if faiss is None:
                raise RuntimeError("faiss-cpu не встановлено. Використай pip install faiss-cpu")
            index = faiss.IndexFlatIP(self.dim)  # cosine на L2-нормованих векторах
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, os.path.join(self.cfg.store_dir, "faiss.index"))

        # --- HNSW ---
        elif engine == "hnsw":
            if hnswlib is None:
                raise RuntimeError("hnswlib не встановлено. Використай pip install hnswlib")
            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.init_index(
                max_elements=int(embeddings.shape[0]),
                ef_construction=int(self.cfg.ef_construction),
                M=int(self.cfg.M),
            )
            labels = np.arange(embeddings.shape[0])
            index.add_items(embeddings.astype(np.float32), labels)
            index.save_index(os.path.join(self.cfg.store_dir, "hnsw.index"))

        # --- NUMPY fallback ---
        else:
            np.savez_compressed(
                os.path.join(self.cfg.store_dir, "numpy_index.npz"),
                X=embeddings.astype(np.float32),
            )

        # --- метадані ---
        self.meta = {str(i): metas[i] for i in range(len(metas))}
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    # ----------------- завантаження -----------------
    def _load_backend(self) -> Any:
        """Завантаження індексу з диску."""
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"meta.json не знайдено у {self.cfg.store_dir}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        engine = self.cfg.engine.lower()

        # --- FAISS ---
        if engine == "faiss":
            if faiss is None:
                raise RuntimeError("faiss-cpu не встановлено.")
            return faiss.read_index(os.path.join(self.cfg.store_dir, "faiss.index"))

        # --- HNSW ---
        if engine == "hnsw":
            if hnswlib is None:
                raise RuntimeError("hnswlib не встановлено.")
            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.load_index(os.path.join(self.cfg.store_dir, "hnsw.index"))
            return index

        # --- NUMPY fallback ---
        data = np.load(os.path.join(self.cfg.store_dir, "numpy_index.npz"))
        return data["X"]

    # ----------------- пошук -----------------
    def search(self, q_emb: np.ndarray, top_k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Пошук найближчих векторів.
        Повертає список кортежів (індекс, схожість, метадані).
        """
        if q_emb.ndim != 2 or q_emb.shape[0] != 1 or q_emb.shape[1] != self.dim:
            raise ValueError(f"Query embedding must be (1,{self.dim})")

        backend = self._load_backend()
        engine = self.cfg.engine.lower()

        # --- FAISS ---
        if engine == "faiss":
            sims, ids = backend.search(q_emb.astype(np.float32), int(top_k))  # type: ignore[attr-defined]
            hits: List[Tuple[int, float, Dict[str, Any]]] = []
            for i, s in zip(ids[0], sims[0]):
                if int(i) < 0:
                    continue
                md = self.meta.get(str(int(i)), {})
                hits.append((int(i), float(s), md))
            return hits

        # --- HNSW ---
        if engine == "hnsw":
            labels, dists = backend.knn_query(q_emb.astype(np.float32), k=int(top_k))  # type: ignore[attr-defined]
            hits: List[Tuple[int, float, Dict[str, Any]]] = []
            for i, d in zip(labels[0], dists[0]):
                md = self.meta.get(str(int(i)), {})
                hits.append((int(i), float(1.0 - d), md))  # cosine -> similarity
            return hits

        # --- NUMPY fallback ---
        if isinstance(backend, np.ndarray):
            X = backend  # (N, d)
            v = q_emb[0]
            x_norms = np.linalg.norm(X, axis=1)
            v_norm = float(np.linalg.norm(v))
            sims = (X @ v) / (x_norms * (v_norm + 1e-9) + 1e-9)
            order = np.argsort(-sims)[:int(top_k)]
            return [(int(i), float(sims[i]), self.meta.get(str(int(i)), {})) for i in order]

        # --- fallback ---
        return []
