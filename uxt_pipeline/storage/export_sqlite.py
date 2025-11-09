# uxt_pipeline/storage/export_sqlite.py
from typing import List
from uxt_pipeline.models import Chunk
from uxt_pipeline.utils import ensure_dir
import sqlite3, json, os

def export_sqlite(chunks: List[Chunk], sqlite_path: str, table: str = "chunks") -> None:
    ensure_dir(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table}(
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_id INTEGER,
            type TEXT,
            text TEXT,
            meta TEXT,
            created_at TEXT
        )
    """)
    cur.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?,?,?)",
        [
            (
                c.id, c.doc_id, c.chunk_id, c.type, c.text,
                json.dumps(c.meta, ensure_ascii=False), c.created_at.isoformat()
            )
            for c in chunks
        ],
    )
    conn.commit()
    conn.close()
