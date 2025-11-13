from __future__ import annotations
from typing import Any, Dict

def to_dict_like(x: Any) -> Dict[str, Any]:
    # Pydantic v2
    if hasattr(x, "model_dump"):
        return x.model_dump()  # type: ignore[no-any-return]
    # Pydantic v1
    if hasattr(x, "dict"):
        return x.dict()  # type: ignore[no-any-return]
    if isinstance(x, dict):
        return x
    return dict(x)  # best-effort
