# uxt_pipeline/utils.py
import os, json, uuid, yaml, hashlib
from pathlib import Path
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True) if Path(p).suffix else Path(p).mkdir(parents=True, exist_ok=True)

def make_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def uuid4() -> str:
    return str(uuid.uuid4())

def write_jsonl(rows, path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
