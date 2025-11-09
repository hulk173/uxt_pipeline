# scripts/evaluate.py
"""
Дуже простий каркас оцінки:
- вимірює швидкість/сторінку та відсоток витягнутого тексту як len(parsed_text)/len(raw_text_ocr) (якщо еталон є).
- Placeholder для додавання своїх метрик (Rouge-L etc.).
"""
import time, json, statistics
from pathlib import Path
from uxt_pipeline.utils import load_config
from uxt_pipeline.ingest.partitioner import _partition_one

def main():
    cfg = load_config("configs/default.yaml")
    bench_dir = Path("data/bench")
    files = sorted(p for p in bench_dir.glob("**/*.*") if p.is_file())
    speeds = []
    lengths = []
    for p in files:
        t0 = time.time()
        res = _partition_one(
            str(p),
            strategy=cfg["ingest"]["strategy"],
            ocr_languages=cfg["ingest"]["ocr_languages"],
            skip_elements=cfg["ingest"]["skip_elements"],
            include_metadata=cfg["ingest"]["include_metadata"],
        )
        dt = time.time() - t0
        txt = "\n".join([e["text"] for e in res["elements"] if e["text"]])
        speeds.append(dt)
        lengths.append(len(txt))

    print("Files:", len(files))
    if speeds:
        print("Avg seconds per file:", statistics.mean(speeds))
    if lengths:
        print("Avg extracted chars:", statistics.mean(lengths))

if __name__ == "__main__":
    main()
