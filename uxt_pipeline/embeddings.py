from dataclasses import dataclass
from typing import List, Any, cast
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:
    TfidfVectorizer = None  # type: ignore

@dataclass
class EmbeddingConfig:
    backend: str = "sentence_transformers"  # "sentence_transformers" | "tfidf"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"

class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.model: Any = None
        self.vec: Any = None
        self.dim: int = 0

        if cfg.backend == "sentence_transformers":
            if SentenceTransformer is None:
                raise RuntimeError("Install sentence-transformers")
            self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
            self.dim = int(self.model.get_sentence_embedding_dimension())
        elif cfg.backend == "tfidf":
            if TfidfVectorizer is None:
                raise RuntimeError("Install scikit-learn")
            self.vec = TfidfVectorizer(max_features=4096)
            self.dim = 4096
        else:
            raise ValueError(cfg.backend)

    def fit(self, texts: List[str]) -> None:
        if self.cfg.backend == "tfidf":
            self.vec.fit(texts)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if self.cfg.backend == "sentence_transformers":
            arr = cast(np.ndarray, self.model.encode(
                texts, batch_size=self.cfg.batch_size,
                convert_to_numpy=True, show_progress_bar=False
            )).astype("float32")
        else:
            arr = self.vec.transform(texts).astype("float32").toarray()
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            arr = arr / norms
        return arr
