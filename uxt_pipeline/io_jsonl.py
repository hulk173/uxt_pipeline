from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path
import json, os

def _json_default(o: Any):
    from datetime import date, datetime
    from pathlib import Path as _Path
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, _Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    if _np is not None and isinstance(o, _np.generic):
        return o.item()
    return str(o)

def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def safe_read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    rows: List[Dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad
