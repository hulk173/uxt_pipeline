from __future__ import annotations
from typing import Sequence, Union, Dict, Any
from pathlib import Path
import json
from ._coerce import to_dict_like
from uxt_pipeline.models import Chunk  # лише для тип-підказок

def export_jsonl(chunks: Sequence[Union[Chunk, Dict[str, Any]]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(to_dict_like(ch), ensure_ascii=False) + "\n")
