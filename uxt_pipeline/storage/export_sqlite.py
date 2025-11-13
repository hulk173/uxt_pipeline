from __future__ import annotations
from typing import Sequence, Union, Dict, Any
import sqlite3
from pathlib import Path
from ._coerce import to_dict_like
from uxt_pipeline.models import Chunk  # type hints

def export_sqlite(chunks: Sequence[Union[Chunk, Dict[str, Any]]], path: str, table: str) -> None:
    rows = [to_dict_like(ch) for ch in chunks]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    try:
        if not rows:
            # все одно створимо пусту таблицю
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (id TEXT)')
            conn.commit()
            return
        cols = sorted({k for r in rows for k in r.keys()})
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs});')
        placeholders = ", ".join("?" for _ in cols)
        sql = f'INSERT INTO "{table}" ({", ".join(cols)}) VALUES ({placeholders});'
        conn.executemany(sql, [[str(r.get(c, "")) for c in cols] for r in rows])
        conn.commit()
    finally:
        conn.close()
