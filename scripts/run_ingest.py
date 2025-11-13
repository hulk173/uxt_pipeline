from pathlib import Path
from uxt_pipeline.utils import load_config
from uxt_pipeline.ingest.partitioner import partition_dir
from uxt_pipeline.transform.chunker import semantic_chunk
from uxt_pipeline.storage.export_jsonl import export_jsonl
from uxt_pipeline.storage.export_parquet import export_parquet
from uxt_pipeline.storage.export_sqlite import export_sqlite
from uxt_pipeline.index.build_index import build_index

def main():
    cfg = load_config("configs/default.yaml")
    results = partition_dir(
        input_dir=cfg["ingest"]["input_dir"],
        glob=cfg["ingest"]["glob"],
        strategy=cfg["ingest"]["strategy"],
        ocr_languages=cfg["ingest"]["ocr_languages"],
        skip_elements=cfg["ingest"]["skip_elements"],
        include_metadata=cfg["ingest"]["include_metadata"],
        max_concurrency=cfg["ingest"]["max_concurrency"],
    )
    all_chunks = []
    for doc in results:
        chunks = semantic_chunk(
            doc,
            max_chars=cfg["chunking"]["max_chars"],
            overlap=cfg["chunking"]["overlap"],
            join_same_type=cfg["chunking"]["join_same_type"],
            min_text_chars=cfg["chunking"]["min_text_chars"],
            strip_whitespace=cfg["chunking"]["strip_whitespace"],
        )
        all_chunks.extend([c.model_dump() for c in chunks])

    export_jsonl(all_chunks, cfg["export"]["jsonl_path"])
    try:
        export_parquet(all_chunks, cfg["export"]["parquet_path"])
    except Exception:
        pass
    export_sqlite(all_chunks, cfg["export"]["sqlite_path"], cfg["export"]["table_name"])

    build_index(
        all_chunks,
        out_dir=cfg["index"]["out_dir"],
        backend=cfg["index"]["backend"],
        model_name=cfg["index"]["model_name"],
        normalize=cfg["index"]["normalize"],
    )

if __name__ == "__main__":
    main()
