from typing import List
from uxt_pipeline.models import Chunk
from uxt_pipeline.storage.export_jsonl import export_jsonl
from uxt_pipeline.storage.export_parquet import export_parquet
from uxt_pipeline.storage.export_sqlite import export_sqlite

def _sample_chunks() -> List[Chunk]:
    return [
        Chunk(id="1", doc_id="d", chunk_id=0, type="Text", text="hello", meta={}),
        Chunk(id="2", doc_id="d", chunk_id=1, type="Text", text="world", meta={}),
    ]

def test_exporters(tmp_path):
    js = tmp_path / "a.jsonl"
    pq = tmp_path / "a.parquet"
    sq = tmp_path / "a.sqlite"
    export_jsonl(_sample_chunks(), str(js))
    export_parquet(_sample_chunks(), str(pq))
    export_sqlite(_sample_chunks(), str(sq), "chunks")
    assert js.exists() and pq.exists() and sq.exists()
