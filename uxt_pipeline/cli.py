import argparse, json
from pathlib import Path
from rich import print
from uxt_pipeline.core import extract_pdf, extract_docx, extract_html, chunk

FMT = {".pdf": extract_pdf, ".docx": extract_docx, ".html": extract_html, ".htm": extract_html}

def main():
    p = argparse.ArgumentParser("uxt")
    p.add_argument("path")
    p.add_argument("--out", default="uxt_pipeline/results/out.jsonl")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=100)
    args = p.parse_args()

    src = Path(args.path)
    files = [src] if src.is_file() else [f for f in src.rglob("*") if f.suffix.lower() in FMT]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    items = []
    for f in files:
        handler = FMT[f.suffix.lower()]
        els = handler(str(f))
        chs = chunk(els, size=args.chunk_size, overlap=args.overlap)
        items.extend([{**c.model_dump(), "source": str(f)} for c in chs])

    with open(args.out, "w", encoding="utf-8") as fo:
        for it in items: fo.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[green]✅ Готово:[/green] {args.out} | {len(items)} чанків")
