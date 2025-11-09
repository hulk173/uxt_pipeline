# uxt_pipeline/storage/export_jsonl.py
from typing import List
from uxt_pipeline.models import Chunk
from uxt_pipeline.utils import write_jsonl

def export_jsonl(chunks: List[Chunk], path: str) -> None:
    # JSON-safe дамп: datetime/UUID -> str
    rows = [c.model_dump(mode="json") for c in chunks]
    write_jsonl(rows, path)
