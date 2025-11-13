from pathlib import Path
from uxt_pipeline.io_jsonl import safe_read_jsonl, write_jsonl_atomic

def main():
    src = Path("data/out/chunks.jsonl")
    if not src.exists():
        print("Файл data/out/chunks.jsonl не знайдено"); return
    rows, bad = safe_read_jsonl(src)
    if not rows:
        print("Немає валідних рядків. Зробіть Ingest заново."); return
    dst = src.with_suffix(".fixed.jsonl")
    write_jsonl_atomic(dst, rows)
    print(f"OK: {len(rows)} валідних рядків записано в {dst} (пропущено битих: {bad})")
    # dst.replace(src)

if __name__ == "__main__":
    main()
