from __future__ import annotations
from typing import Sequence, Union, Dict, Any
from pathlib import Path
import pandas as pd
from ._coerce import to_dict_like
from uxt_pipeline.models import Chunk  # type hints

def export_parquet(chunks: Sequence[Union[Chunk, Dict[str, Any]]], path: str) -> None:
    rows = [to_dict_like(ch) for ch in chunks]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(p, index=False)
