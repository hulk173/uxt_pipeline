from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, cast
import numpy as np

# --- опційні імпорти (не валимо проект, якщо не встановлено) ---
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore


@dataclass
class EmbeddingConfig:
    backend: str = "sentence_transformers"  # "sentence_transformers" | "e5" | "tfidf"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"


class Embedder:
    """Уніфікований інтерфейс для бекендів ембеддингів."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.model: Any = None
        self.vec: Any = None
        self.dim: int = 0

        if cfg.backend in {"sentence_transformers", "e5"}:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers не встановлено. Встанови: pip install sentence-transformers"
                )
            self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
            self.dim = int(self.model.get_sentence_embedding_dimension())
        elif cfg.backend == "tfidf":
            if TfidfVectorizer is None:
                raise RuntimeError("scikit-learn не встановлено. pip install scikit-learn")
            self.vec = TfidfVectorizer(max_features=4096)
            self.dim = 4096
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")

    def fit(self, texts: List[str]) -> None:
        if self.cfg.backend == "tfidf":
            assert self.vec is not None
            self.vec.fit(texts)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if self.cfg.backend in {"sentence_transformers", "e5"}:
            assert self.model is not None
            embs = self.model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            arr = cast(np.ndarray, embs).astype("float32")
        elif self.cfg.backend == "tfidf":
            assert self.vec is not None
            # sparse -> float32 dense
            arr = self.vec.transform(texts)  # type: ignore[no-untyped-call]
            arr = arr.astype("float32").toarray()
        else:  # pragma: no cover
            raise ValueError(self.cfg.backend)

        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            arr = arr / norms
        return arr
