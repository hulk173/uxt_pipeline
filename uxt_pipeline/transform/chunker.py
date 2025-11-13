from typing import Dict, Any, List
from itertools import groupby
from uxt_pipeline.utils import make_id
from uxt_pipeline.models import Chunk

def _clean_text(s: str, strip_ws: bool = True) -> str:
    s = s.replace("\u00a0", " ")
    return s.strip() if strip_ws else s

def semantic_chunk(
    doc: Dict[str, Any],
    max_chars: int = 1200,
    overlap: int = 150,
    join_same_type: bool = True,
    min_text_chars: int = 20,
    strip_whitespace: bool = True,
) -> List[Chunk]:
    doc_id = doc["doc_id"]
    elements = doc["elements"]

    if join_same_type:
        merged = []
        for typ, grp in groupby(elements, key=lambda e: e["type"]):
            buf_texts, buf_metas = [], []
            for el in grp:
                t = _clean_text(el["text"], strip_whitespace)
                if t:
                    buf_texts.append(t)
                    buf_metas.append(el.get("meta", {}))
            if buf_texts:
                merged.append({"type": typ, "text": "\n".join(buf_texts), "meta": {"sources": buf_metas}})
        elements = merged

    chunks: List[Chunk] = []
    cidx = 0
    for el in elements:
        text = _clean_text(el["text"], strip_whitespace)
        if not text or len(text) < min_text_chars:
            continue
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            piece = text[start:end]
            if len(piece) >= min_text_chars:
                chunk = Chunk(
                    id=make_id(f"{doc_id}:{cidx}:{piece[:32]}"),
                    doc_id=doc_id,
                    chunk_id=cidx,
                    type=el["type"],
                    text=piece,
                    meta=el.get("meta", {}),
                )
                chunks.append(chunk)
                cidx += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks
