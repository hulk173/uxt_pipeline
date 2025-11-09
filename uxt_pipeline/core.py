from __future__ import annotations
import re, uuid
from typing import List

from langdetect import detect
from uxt_pipeline.types import Element, Chunk

# парсери
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html

def _norm(text: str) -> str:
    t = re.sub(r"\s+", " ", text.replace("\u00A0", " ")).strip()
    try:
        detect(t)  # просто ініціюємо детекцію (можеш використати lang у майбутньому)
    except Exception:
        pass
    return t

def _wrap(raw) -> List[Element]:
    out: List[Element] = []
    for e in raw:
        txt = (getattr(e, "text", "") or "").strip()
        if not txt:
            continue
        out.append(Element(
            type=getattr(e, "category", getattr(e, "type", "Text")),
            text=_norm(txt),
            page_number=getattr(getattr(e, "metadata", None), "page_number", None),
            level=getattr(e, "level", None),
            section_path=getattr(getattr(e, "metadata", None), "section_path", None),
        ))
    return out

def extract_pdf(path: str) -> List[Element]:
    return _wrap(partition_pdf(filename=path, infer_table_structure=True))

def extract_docx(path: str) -> List[Element]:
    return _wrap(partition_docx(filename=path))

def extract_html(path: str) -> List[Element]:
    return _wrap(partition_html(filename=path))

def chunk(elements: List[Element], size: int = 800, overlap: int = 100) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf: list[str] = []
    section: list[str] = []

    def flush():
        if not buf:
            return
        txt = " ".join(buf).strip()
        if not txt:
            return
        chunks.append(Chunk(id=str(uuid.uuid4()), text=txt, section_path=section.copy(), meta={}))

    tokens = 0
    for e in elements:
        if e.type.lower().startswith("title"):
            flush()
            buf = []
            section = (e.section_path or []) + [e.text[:80]]
            tokens = 0
            continue
        words = e.text.split()
        if tokens + len(words) > size and buf:
            keep = max(0, len(buf) - overlap)
            buf = buf[keep:]
            tokens = len(buf)
        buf.extend(words)
        tokens += len(words)
    flush()
    return chunks
