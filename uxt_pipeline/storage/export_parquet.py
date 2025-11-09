# uxt_pipeline/storage/export_parquet.py
from typing import List
from uxt_pipeline.models import Chunk
from uxt_pipeline.utils import ensure_dir
import pandas as pd

def export_parquet(chunks: List[Chunk], path: str) -> None:
    ensure_dir(path)
    # теж використовуємо JSON-safe дамп, щоб у parquet були вже готові типи
    rows = [c.model_dump(mode="json") for c in chunks]
    df = pd.DataFrame(rows)
    # meta залишаємо як колонку з dict (parquet це вміє), або можна .astype(str) за потреби
    df.to_parquet(path, index=False)
