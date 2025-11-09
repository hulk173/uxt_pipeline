# uxt_pipeline/analysis/nlp.py
from __future__ import annotations
from typing import Dict, Any

# alias щоб не сварився type checker
from transformers import pipeline as hf_pipeline  # type: ignore

__all__ = ["sentiment_score", "summarize_text"]

_SENTIMENT: Any = None
_SUMMARY: Any = None

def _sentiment():
    global _SENTIMENT
    if _SENTIMENT is None:
        _SENTIMENT = hf_pipeline("sentiment-analysis")  # type: ignore[arg-type]
    return _SENTIMENT

def _summary():
    global _SUMMARY
    if _SUMMARY is None:
        _SUMMARY = hf_pipeline("summarization")  # type: ignore[arg-type]
    return _SUMMARY

def sentiment_score(text: str) -> Dict[str, Any]:
    """Повертає label/score і зручний показник 'positivity' 0..1."""
    clf = _sentiment()
    res = clf(text)[0]  # type: ignore[index]
    label = str(res.get("label", "NEUTRAL")).upper()
    score = float(res.get("score", 0.5))
    positivity = score if "POS" in label else (1.0 - score if "NEG" in label else 0.5)
    return {"label": label, "score": score, "positivity": positivity}

def summarize_text(text: str, max_len: int = 150, min_len: int = 40) -> str:
    sm = _summary()
    out = sm(text, max_length=max_len, min_length=min_len, do_sample=False)[0]  # type: ignore[index]
    return str(out.get("summary_text", ""))
