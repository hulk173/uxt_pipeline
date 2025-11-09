# uxt_pipeline/ingest/partitioner.py
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from unstructured.partition.auto import partition
from unstructured.documents.elements import Element


def _element_meta_to_dict(el: Element, keep: set[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    md = getattr(el, "metadata", None)
    if md is None:
        return meta
    # уніфікуємо до dict на різних версіях unstructured
    if hasattr(md, "to_dict"):
        meta = md.to_dict()  # type: ignore[call-arg]
    elif isinstance(md, dict):
        meta = md
    else:
        meta = {k: v for k, v in getattr(md, "__dict__", {}).items() if not k.startswith("_")}
    # відфільтруємо корисні ключі
    return {k: v for k, v in meta.items() if k in keep}


def _partition_one(
    file_path: str,
    strategy: str = "fast",
    ocr_languages: str = "eng",
    skip_elements: List[str] | None = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Повертає {"doc_id": <basename>, "elements": [{"type","text","meta"}, ...]}
    Гарантовано повертає dict навіть при часткових збоях.
    """
    keep_keys = {"filename", "filetype", "page_number", "coordinates", "last_modified", "languages", "text_as_html"}
    out: List[Dict[str, Any]] = []
    doc_id = Path(file_path).name

    try:
        elements: List[Element] = partition(
            filename=file_path,
            strategy=strategy,
            ocr_languages=ocr_languages,
            include_metadata=include_metadata,
        )
    except Exception as e:
        # у разі помилки парсингу — повертаємо «порожній» документ із помилкою в метаданих
        return {
            "doc_id": doc_id,
            "elements": [
                {
                    "type": "Error",
                    "text": "",
                    "meta": {"filename": doc_id, "error": str(e)},
                }
            ],
        }

    if skip_elements:
        to_skip = set(skip_elements)
        elements = [el for el in elements if getattr(el, "category", None) not in to_skip]

    for el in elements:
        meta: Dict[str, Any] = {}
        if include_metadata:
            meta = _element_meta_to_dict(el, keep_keys)
        out.append(
            {
                "type": getattr(el, "category", "Unknown"),
                "text": str(el),
                "meta": meta,
            }
        )

    # завжди повертаємо коректний словник
    return {"doc_id": doc_id, "elements": out}


def partition_dir(
    input_dir: str,
    glob: str = "**/*.*",
    *,
    strategy: str = "fast",
    ocr_languages: str = "eng",
    skip_elements: List[str] | None = None,
    include_metadata: bool = True,
    max_concurrency: int = 4,
) -> List[Dict[str, Any]]:
    paths = [str(p) for p in Path(input_dir).rglob(glob) if p.is_file()]
    results: List[Dict[str, Any]] = []

    if not paths:
        return results

    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futs = {
            ex.submit(
                _partition_one,
                p,
                strategy=strategy,
                ocr_languages=ocr_languages,
                skip_elements=skip_elements,
                include_metadata=include_metadata,
            ): p
            for p in paths
        }
        for fut in as_completed(futs):
            results.append(fut.result())

    return results
