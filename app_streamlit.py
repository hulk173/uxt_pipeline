from __future__ import annotations
from uxt_pipeline.io_jsonl import safe_read_jsonl

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yaml

# ============================ SETUP =============================
st.set_page_config(
    page_title="UXT Pipeline • Pro Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# session_state
if "report_items" not in st.session_state:
    st.session_state["report_items"] = []
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

# API / config
API_URL = os.environ.get("UXT_API_URL", "http://localhost:8000")
CFG_DIR = Path("configs"); CFG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CFG_PATH = CFG_DIR / "default.yaml"
LOCAL_CFG_PATH = CFG_DIR / "local.yaml"

BASE_DEFAULTS: Dict[str, Any] = {
    "ingest": {"strategy": "fast", "ocr_languages": "eng+ukr", "skip_elements": ["Figure","PageBreak"]},
    "chunking": {"max_chars": 1000, "overlap": 120, "join_same_type": True, "min_text_chars": 30, "strip_whitespace": True},
    "index": {"backend": "faiss", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "top_k": 5, "normalize": True},
    "lang": {"detect": True, "prefer": "uk,en", "table_as_markdown": True}
}

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    if not isinstance(b, dict):
        return out
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_defaults() -> Dict[str, Any]:
    base = dict(BASE_DEFAULTS)
    if DEFAULT_CFG_PATH.exists():
        base = deep_merge(base, yaml.safe_load(DEFAULT_CFG_PATH.read_text()) or {})
    if LOCAL_CFG_PATH.exists():
        base = deep_merge(base, yaml.safe_load(LOCAL_CFG_PATH.read_text()) or {})
    return base

def save_defaults(cfg: Dict[str, Any]) -> None:
    LOCAL_CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

def highlight(text: str, query: str) -> str:
    if not query:
        return text
    try:
        patt = re.compile(re.escape(query), re.IGNORECASE)
        return patt.sub(lambda m: f"**{m.group(0)}**", text)
    except Exception:
        return text

def add_to_report(item: Dict[str, Any]):
    st.session_state["report_items"].append(item)

def df_to_csv_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def render_pdf(html_src: str) -> Optional[bytes]:
    try:
        from weasyprint import HTML  # type: ignore
        return HTML(string=html_src).write_pdf()
    except Exception:
        return None

# ============================ THEME =============================
THEME_VARS = {
    "Light": {
        "--bg": "#ffffff","--fg": "#0f172a","--muted": "#5e6b7a",
        "--card": "rgba(127,127,127,0.04)","--border": "rgba(16,24,40,.12)",
        "--ok-bg": "#E9FFF2","--ok-fg": "#067647","--ok-br": "#BBF7D0",
        "--warn-bg": "#FFF7E6","--warn-fg": "#A15C06","--warn-br": "#FDE68A",
        "--err-bg": "#FFECEC","--err-fg": "#B42318","--err-br": "#FCA5A5",
        "--info-bg": "#EEF6FF","--info-fg": "#1D4ED8","--info-br": "#BFDBFE",
        "--pill-br": "rgba(125,125,125,.2)","--chip-br": "rgba(125,125,125,.15)",
    },
    "Dark": {
        "--bg": "#0b1020","--fg": "#e5e7eb","--muted": "#9aa5b1",
        "--card": "rgba(255,255,255,0.04)","--border": "rgba(255,255,255,.12)",
        "--ok-bg": "#072d19","--ok-fg": "#7CFFB2","--ok-br": "#0b5b3a",
        "--warn-bg": "#2f2304","--warn-fg": "#FFD07A","--warn-br": "#5a470b",
        "--err-bg": "#2b0a0a","--err-fg": "#FF9B9B","--err-br": "#681010",
        "--info-bg": "#0b1b34","--info-fg": "#9ac1ff","--info-br": "#173061",
        "--pill-br": "rgba(255,255,255,.25)","--chip-br": "rgba(255,255,255,.2)",
    },
}

def inject_theme_css(mode: str):
    v = THEME_VARS[mode]
    st.markdown(
        f"""
        <style>
        .block-container {{padding-top: 1.2rem; padding-bottom: 2rem;}}
        :root {{
            --bg:{v['--bg']}; --fg:{v['--fg']}; --muted:{v['--muted']};
            --card:{v['--card']}; --border:{v['--border']};
            --ok-bg:{v['--ok-bg']}; --ok-fg:{v['--ok-fg']}; --ok-br:{v['--ok-br']};
            --warn-bg:{v['--warn-bg']}; --warn-fg:{v['--warn-fg']}; --warn-br:{v['--warn-br']};
            --err-bg:{v['--err-bg']}; --err-fg:{v['--err-fg']}; --err-br:{v['--err-br']};
            --info-bg:{v['--info-bg']}; --info-fg:{v['--info-fg']}; --info-br:{v['--info-br']};
            --pill-br:{v['--pill-br']}; --chip-br:{v['--chip-br']};
        }}
        html, body {{ background: var(--bg) !important; color: var(--fg) !important; }}
        .badge {{display:inline-block; padding: 0.2rem .55rem; border-radius: 999px; font-size: .78rem; font-weight:600; border:1px solid;}}
        .b-ok  {{background:var(--ok-bg); color:var(--ok-fg); border-color:var(--ok-br);}}
        .b-warn{{background:var(--warn-bg); color:var(--warn-fg); border-color:var(--warn-br);}}
        .b-err {{background:var(--err-bg); color:var(--err-fg); border-color:var(--err-br);}}
        .b-info{{background:var(--info-bg); color:var(--info-fg); border-color:var(--info-br);}}
        .card{{border:1px solid var(--border); border-radius:14px; padding:1rem 1.1rem; background:var(--card);}}
        .muted{{color: var(--muted); font-size: .9rem;}}
        .pill{{font-weight:600; padding:.15rem .5rem; border-radius:999px; border:1px solid var(--pill-br)}}
        .chip{{display:inline-flex; gap:.35rem; align-items:center; padding:.15rem .55rem; border-radius:999px; font-size:.78rem; border:1px solid var(--chip-br)}}
        .footer{{opacity:.75; font-size:.85rem; padding-top: 1.2rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================ SIDEBAR =============================
st.sidebar.subheader("🎨 Theme")
st.session_state["theme"] = st.sidebar.radio("Mode", ["Light", "Dark"], index=(0 if st.session_state["theme"] == "Light" else 1))
inject_theme_css(st.session_state["theme"])

cfg = load_defaults()

st.sidebar.subheader("⚙️ Settings")
strategy = st.sidebar.selectbox("OCR strategy", ["fast", "hi_res"], index=(0 if cfg["ingest"]["strategy"]=="fast" else 1))
ocr_languages = st.sidebar.text_input("OCR languages", value=cfg["ingest"]["ocr_languages"], key="sb_ocr_langs")
skip_elements = st.sidebar.text_input("Skip elements (CSV)", value=",".join(cfg["ingest"].get("skip_elements", [])), key="sb_skip_elems")
st.sidebar.markdown("---")
st.sidebar.caption("Chunking")
max_chars = st.sidebar.slider("max_chars", 300, 2000, int(cfg["chunking"]["max_chars"]), 50, key="sb_maxchars")
overlap = st.sidebar.slider("overlap", 0, 400, int(cfg["chunking"]["overlap"]), 10, key="sb_overlap")
join_same_type = st.sidebar.checkbox("join_same_type", value=bool(cfg["chunking"]["join_same_type"]), key="sb_join_same_type")
min_text_chars = st.sidebar.slider("min_text_chars", 0, 200, int(cfg["chunking"]["min_text_chars"]), 5, key="sb_min_text_chars")
strip_whitespace = st.sidebar.checkbox("strip_whitespace", value=bool(cfg["chunking"]["strip_whitespace"]), key="sb_strip_ws")
st.sidebar.markdown("---")
st.sidebar.caption("Language & Tables")
lang_detect = st.sidebar.checkbox("Detect language", value=bool(cfg["lang"]["detect"]), key="sb_lang_detect")
prefer_langs = st.sidebar.text_input("Preferred langs (hint)", value=cfg["lang"]["prefer"], key="sb_prefer_langs")
table_as_markdown = st.sidebar.checkbox("Convert tables to Markdown", value=bool(cfg["lang"]["table_as_markdown"]), key="sb_table_md")
st.sidebar.markdown("---")
st.sidebar.caption("Index")
index_backend = st.sidebar.selectbox("backend", ["faiss", "sklearn"], index=(0 if cfg["index"]["backend"]=="faiss" else 1))
model_name = st.sidebar.selectbox(
    "model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ],
    index=0 if cfg["index"]["model_name"].endswith("L6-v2") else 1
)
top_k = st.sidebar.slider("top_k", 1, 20, int(cfg["index"]["top_k"]), 1, key="sb_topk")
normalize = st.sidebar.checkbox("normalize (L2)", value=bool(cfg["index"]["normalize"]), key="sb_norm")
st.sidebar.markdown("---")
csa, csb = st.sidebar.columns(2)
with csa:
    if st.button("💾 Save defaults"):
        save_defaults(
            {
                "ingest": {"strategy": strategy, "ocr_languages": ocr_languages, "skip_elements": [s.strip() for s in skip_elements.split(",") if s.strip()]},
                "chunking": {"max_chars": int(max_chars), "overlap": int(overlap), "join_same_type": bool(join_same_type),
                             "min_text_chars": int(min_text_chars), "strip_whitespace": bool(strip_whitespace)},
                "index": {"backend": index_backend, "model_name": model_name, "top_k": int(top_k), "normalize": bool(normalize)},
                "lang": {"detect": bool(lang_detect), "prefer": prefer_langs, "table_as_markdown": bool(table_as_markdown)},
            }
        )
        st.sidebar.success("Saved configs/local.yaml")
with csb:
    if st.button("🧹 Clear outputs"):
        try:
            r = requests.delete(f"{API_URL}/outputs")
            st.sidebar.success(r.json())
        except Exception as ex:
            st.sidebar.error(ex)

# ============================ HEADER =============================
hl, hr = st.columns([0.75, 0.25], vertical_alignment="center")
with hl:
    st.markdown("### 🔎 UXT Pipeline • **Pro Dashboard**")
    st.caption("Завантаження → Чанкінг → Індекс → Пошук/RAG → Аналітика → Report Builder.")
with hr:
    try:
        ok = requests.get(f"{API_URL}/health", timeout=3).ok
    except Exception:
        ok = False
    st.markdown(
        f"<div style='text-align:right'><span class='badge {'b-ok' if ok else 'b-err'}'>API {'ONLINE' if ok else 'OFFLINE'}</span>&nbsp;&nbsp;<span class='muted tiny'>{API_URL}</span></div>",
        unsafe_allow_html=True,
    )

with st.expander("ℹ️ Коротка інструкція", expanded=False):
    st.markdown(
        """
1. **Ingest** документ(и) у вкладці *Ingest* (OCR, skip elements, мови).
2. Перевірте *Preview* — фільтруйте за `doc_id`/`type`/текстом.
3. Шукайте у *Search* (можна обмежити мовою), або ставте запитання у *Ask*.
4. Дивіться *Metrics*, *Clusters* і *Visualization*.
5. У *Report Builder* зберіть фрагменти → експортуйте **HTML/PDF**.
        """
    )

# ============================ TABS =============================
tabs = st.tabs(["📥 Ingest","👁 Preview","🔎 Search","💬 Ask","📊 Metrics","🗂 Clusters","🗺 Visualization","📖 Report Builder","🕰 History"])
(tab_ing, tab_prev, tab_search, tab_ask, tab_metrics, tab_clusters, tab_viz, tab_report, tab_hist) = tabs

# --------------------------- INGEST -----------------------------
with tab_ing:
    st.markdown("#### 1) Ingest")
    l, r = st.columns([0.62, 0.38], vertical_alignment="bottom")
    with l:
        f = st.file_uploader("Обрати документ (PDF/DOCX/HTML/TXT)", type=["pdf", "docx", "html", "txt"])
        st.caption("Ліміт 200MB per file")
    with r:
        st.markdown(
            "<div class='card'><b>Поради</b><div class='muted'>• Для сканів — <b>hi_res</b> і коректні OCR мови.<br>"
            "• Не ставте дуже малий <b>min_text_chars</b> — буде шум.<br>"
            "• Для cosine лишіть <b>normalize</b> увімкненим.<br>"
            "• Можна <b>Skip elements</b> (Figure, PageBreak) у сайдбарі.</div></div>",
            unsafe_allow_html=True,
        )

    if st.button("🚀 Ingest", disabled=f is None, type="primary"):
        if f is None:
            st.warning("Оберіть файл.")
            st.stop()
        files = {"file": (f.name or "uploaded.bin", f.getvalue())}
        data = {
            "strategy": strategy,
            "ocr_languages": ocr_languages,
            "skip_elements": skip_elements,
            "include_metadata": json.dumps(True),
            "drop_headers_footers": json.dumps(True),
            "max_chars": int(max_chars),
            "overlap": int(overlap),
            "join_same_type": json.dumps(bool(join_same_type)),
            "min_text_chars": int(min_text_chars),
            "strip_whitespace": json.dumps(bool(strip_whitespace)),
            "index_backend": index_backend,
            "model_name": model_name,
            "normalize": json.dumps(bool(normalize)),
            "lang_detect": json.dumps(bool(lang_detect)),
            "prefer_langs": prefer_langs,
            "table_as_markdown": json.dumps(bool(table_as_markdown)),
        }
        try:
            r = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=60)
            if not r.ok:
                st.error(r.text)
            else:
                # ---- Безпечне отримання job_id ----
                try:
                    payload = r.json()
                except Exception:
                    payload = None
                job_id = None
                if isinstance(payload, dict) and "job_id" in payload:
                    job_id = str(payload["job_id"])
                elif isinstance(payload, (str, int)):
                    job_id = str(payload)
                if not job_id:
                    st.error(f"Unexpected /ingest response: {payload if payload is not None else r.text[:300]}")
                    st.stop()

                st.info(f"job: {job_id}")

                # ---- Опитування стану job з перевіркою типів ----
                with st.spinner("Обробка…"):
                    for _ in range(180):
                        rr = requests.get(f"{API_URL}/job/{job_id}", timeout=10)
                        try:
                            s = rr.json()
                        except Exception:
                            st.error(f"/job returned non-JSON:\n{rr.text[:400]}")
                            break

                        if not isinstance(s, dict):
                            st.error(f"/job returned {type(s).__name__}: {s}")
                            break

                        status = s.get("status")
                        if status == "finished":
                            st.success("Готово!")
                            st.json(s)
                            break
                        if status == "error":
                            st.error(s.get("error", "unknown error"))
                            break
                        time.sleep(5)
        except Exception as ex:
            st.error(ex)

# --------------------------- PREVIEW -----------------------------
with tab_prev:
    st.markdown("#### 2) Preview")
    p = Path("data/out/chunks.jsonl")
    if not p.exists():
        st.info("Немає `data/out/chunks.jsonl` — зробіть Ingest.")
    else:
        rows, bad = safe_read_jsonl(p)
        if bad:
            st.warning(f"Пропущено пошкоджених рядків у JSONL: {bad}. "
                       "Можна виконати ремонт (див. scripts/repair_jsonl.py).")
        if not rows:
            st.info("Файл порожній. Зробіть Ingest ще раз.")
            st.stop()
        df = pd.DataFrame(rows)
        st.caption(f"Чанків: {len(df)}")
        c1, c2, c3 = st.columns(3)
        with c1: f_doc = st.text_input("Фільтр doc_id (regex)", "", key="pv_doc")
        with c2: f_type = st.text_input("Фільтр type (regex)", "", key="pv_type")
        with c3: f_text = st.text_input("Пошук у тексті (regex)", "", key="pv_text")
        fdf = df.copy()
        if f_doc:  fdf = fdf[fdf["doc_id"].astype(str).str.contains(f_doc, regex=True, na=False)]
        if f_type: fdf = fdf[fdf["type"].astype(str).str.contains(f_type, regex=True, na=False)]
        if f_text: fdf = fdf[fdf["text"].astype(str).str.contains(f_text, regex=True, na=False)]
        st.dataframe(fdf, width="stretch", height=420)

        st.markdown("**Додати у Report** (за індексом рядка)")
        irow = st.number_input("row index", min_value=0, max_value=max(0, len(fdf)-1), value=0, step=1, key="pv_row")
        if st.button("➕ Add row to report", disabled=len(fdf)==0, key="pv_add"):
            row = fdf.iloc[int(irow)]
            add_to_report({"type":"chunk","doc_id": row.get("doc_id"), "text": row.get("text",""), "meta": row.get("meta",{})})
            st.success("Додано у звіт")

# --------------------------- SEARCH -----------------------------
with tab_search:
    st.markdown("#### 3) Search")
    q = st.text_input("Запит", key="se_query")
    colS1, colS2, colS3 = st.columns([0.5, 0.25, 0.25])
    with colS2:
        lang_filter = st.text_input("Lang filter (CSV)", value="", key="se_lang_filter")
    with colS3:
        score_min = st.number_input("score_min", value=0.0, min_value=0.0, max_value=1.0, step=0.01, key="se_score_min")
    if st.button("🔎 Search", disabled=not q, key="se_btn"):
        try:
            res = requests.get(f"{API_URL}/search", params={"q": q, "k": top_k, "lang_filter": lang_filter, "score_min": score_min}, timeout=60).json()
            if not res:
                st.info("Нічого не знайдено.")
            for i, it in enumerate(res, 1):
                score = float(it["score"])
                badge = "b-ok" if score >= 0.65 else ("b-warn" if score >= 0.45 else "b-info")
                lang = (it.get("meta") or {}).get("lang", "und")
                st.markdown(
                    f"<span class='badge {badge}'>score={score:.3f}</span> &nbsp; "
                    f"<span class='pill'>{it['doc_id']}</span> &nbsp; <span class='chip'>lang: {lang}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(highlight(it["text"], q))
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"➕ Add result #{i} to report", key=f"se_add_{i}"):
                        add_to_report({"type": "chunk","doc_id": it["doc_id"],"text": it["text"],"meta": {"score": score, "lang": lang}})
                        st.success("Додано у звіт")
                with c2:
                    with st.expander("📎 Copy citation"):
                        st.text_area("Copy", value=it["text"], height=80, key=f"se_copy_{i}")
                st.divider()
        except Exception as ex:
            st.error(ex)

# --------------------------- ASK (RAG) --------------------------
with tab_ask:
    st.markdown("#### 4) Ask (RAG-QA)")
    cq1, cq2, cq3 = st.columns([0.55, 0.2, 0.25])
    with cq1: qq = st.text_input("Запитання", key="ask_query")
    with cq2: kk = st.slider("к (контекст)", 1, 10, int(top_k), 1, key="ask_k")
    with cq3: lang_filter_ask = st.text_input("Lang filter (CSV)", value="", key="ask_lang_filter")
    if st.button("💬 Ask", disabled=not qq, key="ask_btn"):
        data = requests.get(f"{API_URL}/ask", params={"q": qq, "k": kk, "lang_filter": lang_filter_ask}, timeout=180).json()
        st.markdown("##### Відповідь")
        st.markdown(data.get("answer", ""))
        st.markdown("##### Джерела")
        src = pd.DataFrame(data.get("sources", []))
        if len(src):
            st.dataframe(src, width="stretch", height=240)
            df_to_csv_download(src, "⬇️ Export sources", "ask_sources.csv")
            if st.button("➕ Add answer to report", key="ask_add"):
                add_to_report({"type": "title", "text": "Відповідь на запитання"})
                add_to_report({"type": "note", "text": qq})
                add_to_report({"type": "chunk","doc_id": "[RAG Answer]","text": data.get("answer", ""),"meta": {"k": kk}})
                st.success("Додано у звіт")
        else:
            st.info("Цитати відсутні")

# --------------------------- METRICS ----------------------------
with tab_metrics:
    st.markdown("#### 5) Metrics")
    m1, m2 = st.columns(2)
    with m1: mq = st.text_area("Query", "", key="met_query")
    with m2: expected = st.text_area("Expected snippet", "", key="met_expected")
    colM1, colM2 = st.columns([0.5, 0.5])
    with colM1: k = st.slider("k", 1, 20, int(top_k), 1, key="met_k")
    with colM2: lang_filter_eval = st.text_input("Lang filter (CSV)", value="", key="met_lang_filter")
    if st.button("🏁 Evaluate", disabled=not mq, key="met_btn"):
        out = requests.post(f"{API_URL}/evaluate", json={"query": mq, "expected_text": expected, "k": k, "lang_filter": lang_filter_eval}, timeout=60).json()
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("recall@k", f"{out['recall@k']:.3f}")
        with c2: st.metric("precision@k", f"{out['precision@k']:.3f}")
        with c3: st.metric("k", out["k"])
        st.dataframe(pd.DataFrame(out["results"]), width="stretch", height=280)

# --------------------------- CLUSTERS ---------------------------
with tab_clusters:
    st.markdown("#### 6) Clusters")
    nc1, nc2 = st.columns([0.5, 0.5])
    with nc1: nc = st.slider("Кількість кластерів", 2, 30, 8, 1, key="cl_nc")
    with nc2: lang_filter_cl = st.text_input("Lang filter (CSV)", value="", key="cl_lang_filter")
    if st.button("🗂 Compute clusters", key="cl_btn"):
        data = requests.get(f"{API_URL}/clusters", params={"n_clusters": nc, "lang_filter": lang_filter_cl}, timeout=180).json()
        if "clusters" in data:
            df = pd.DataFrame(data["clusters"])
            st.dataframe(df, width="stretch", height=400)
            df_to_csv_download(df, "⬇️ Export clusters", "clusters.csv")
        else:
            st.info(data)

# --------------------------- VISUALIZATION ----------------------
with tab_viz:
    st.markdown("#### 7) Visualization (UMAP)")
    p = Path("data/out/chunks.jsonl")
    if not p.exists():
        st.info("Немає `data/out/chunks.jsonl`. Зробіть Ingest.")
    else:
        rows, bad = safe_read_jsonl(p)
        texts = [r.get("text", "") for r in rows if r.get("text")]
        ids = [r.get("id") for r in rows if r.get("text")]
        if not texts:
            st.warning("Немає придатних чанків.")
        else:
            vis_model = st.selectbox(
                "Модель",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-MiniLM-L12-v2",
                    "sentence-transformers/paraphrase-MiniLM-L6-v2",
                ],
                index=0,
                key="viz_model"
            )
            nmax = st.slider("Максимум точок", 200, 5000, min(1200, len(texts)), 100, key="viz_nmax")
            if st.button("🗺 Compute & Plot", type="primary", key="viz_btn"):
                try:
                    from sentence_transformers import SentenceTransformer
                    import plotly.express as px
                    import umap
                except Exception:
                    st.error("Потрібно: sentence-transformers, umap-learn, plotly")
                    st.stop()
                model = SentenceTransformer(vis_model)
                X = model.encode(texts[:nmax], convert_to_numpy=True, show_progress_bar=True)
                X = np.asarray(X, dtype=np.float32)
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
                emb2d = np.asarray(reducer.fit_transform(X), dtype=float)
                dfv = pd.DataFrame({"x": emb2d[:, 0], "y": emb2d[:, 1], "id": ids[:nmax], "text": texts[:nmax]})
                import plotly.express as px
                fig = px.scatter(dfv, x="x", y="y", hover_data=["id", "text"], title="UMAP Embeddings")
                st.plotly_chart(fig, use_container_width=True)

# --------------------------- REPORT BUILDER ---------------------
with tab_report:
    st.markdown("#### 8) Report Builder")
    st.caption("Збирайте фрагменти, додавайте заголовки й нотатки, експортуйте HTML/PDF з титульною сторінкою.")

    st.markdown("##### Обкладинка (титульна сторінка)")
    cv1, cv2 = st.columns(2)
    with cv1:
        uni_name = st.text_input("Заклад освіти", value="Національний університет", key="rb_uni")
        faculty = st.text_input("Факультет / Інститут", value="Факультет комп’ютерних наук", key="rb_fac")
        dept = st.text_input("Кафедра", value="Кафедра інформаційних систем", key="rb_dept")
        work_type = st.selectbox("Тип роботи", ["Курсова робота", "Кваліфікаційна робота", "Дипломна робота"], index=2, key="rb_worktype")
        topic = st.text_area("Тема роботи", value="Метод обробки неструктурованих даних для вдосконалення систем аналізу тексту", height=80, key="rb_topic")
    with cv2:
        student = st.text_input("Студент", value="Оліфіренко Кирило", key="rb_student")
        group = st.text_input("Група", value="6М", key="rb_group")
        specialty = st.text_input("Спеціальність", value="Комп’ютерні науки", key="rb_spec")
        supervisor = st.text_input("Керівник", value="доц. І. І. Прізвище", key="rb_super")
        city = st.text_input("Місто", value="Київ", key="rb_city")
        year = st.text_input("Рік", value="2025", key="rb_year")
    add_logo = st.checkbox("Додати логотип на обкладинку (вбудований SVG)", value=True, key="rb_logo")

    st.markdown("---")

    with st.expander("➕ Додати елемент вручну"):
        tt = st.selectbox("Тип", ["Заголовок", "Нотатка", "Порожній розділ"], key="rb_add_type")
        val = st.text_area("Текст", placeholder="Введіть текст (для Заголовка/Нотатки)", key="rb_add_text")
        if st.button("Додати", key="rb_add_btn"):
            if tt == "Заголовок":
                add_to_report({"type": "title", "text": val.strip() or "Новий розділ"})
            elif tt == "Нотатка":
                add_to_report({"type": "note", "text": val.strip()})
            else:
                add_to_report({"type": "title", "text": "—"})
            st.success("Додано")

    report_items = st.session_state["report_items"]
    if not report_items:
        st.info("Поки що порожньо. Додайте фрагменти зі сторінок Preview/Search/Ask або через форму вище.")
    else:
        for idx, it in enumerate(report_items):
            c1, c2, c3, c4 = st.columns([0.06, 0.06, 0.78, 0.10])
            with c1:
                if st.button("⬆️", key=f"rb_up_{idx}", disabled=(idx == 0)):
                    report_items[idx - 1], report_items[idx] = report_items[idx], report_items[idx - 1]
            with c2:
                if st.button("⬇️", key=f"rb_dn_{idx}", disabled=(idx == len(report_items) - 1)):
                    report_items[idx + 1], report_items[idx] = report_items[idx], report_items[idx + 1]
            with c3:
                if it["type"] == "title":
                    st.markdown(f"**[Заголовок]** {it.get('text','')}")
                elif it["type"] == "note":
                    st.markdown(f"<span class='muted'>[Нотатка]</span> {it.get('text','')}", unsafe_allow_html=True)
                else:
                    meta = it.get("meta", {})
                    score_txt = f" • score={meta.get('score'):.3f}" if "score" in meta else ""
                    lang_txt = f" • lang={meta.get('lang')}" if "lang" in meta else ""
                    st.markdown(f"<span class='pill'>{it.get('doc_id','')}</span>{score_txt}{lang_txt}", unsafe_allow_html=True)
                    st.write(it.get("text", ""))
            with c4:
                if st.button("🗑", key=f"rb_rm_{idx}"):
                    report_items.pop(idx)
                    st.rerun()

        st.divider()
        title = st.text_input("Назва звіту (для шапки HTML)", value="Звіт UXT Pipeline", key="rb_title")
        add_toc = st.checkbox("Згенерувати зміст (TOC)", value=True, key="rb_toc")

        def build_html() -> str:
            palette = THEME_VARS[st.session_state["theme"]]
            logo_svg = (
                """
                <svg width="72" height="72" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
                    <defs><linearGradient id="g" x1="0" x2="1"><stop stop-color="#6366f1"/><stop offset="1" stop-color="#22c55e"/></linearGradient></defs>
                    <circle cx="64" cy="64" r="60" fill="url(#g)"/>
                    <path d="M36 70l18 18 38-50" fill="none" stroke="white" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>"""
                if add_logo
                else ""
            )

            css = f"""
            @page {{ size: A4; margin: 24mm 18mm 24mm 18mm; }}
            body {{ font-family: -apple-system, Segoe UI, Inter, system-ui, sans-serif; background:{palette['--bg']}; color:{palette['--fg']}; }}
            .container {{ max-width: 880px; margin: 1.5rem auto; }}
            .muted {{ color:{palette['--muted']}; }}
            .section {{ margin: 1.1rem 0; padding: .8rem 1rem; border:1px solid {palette['--border']}; border-radius:12px; background:{palette['--card']}; }}
            .pill {{ display:inline-block; padding:.12rem .5rem; border-radius:999px; border:1px solid {palette['--pill-br']}; font-weight:600; }}
            .toc {{ margin: 1rem 0 1.4rem; padding:.8rem 1rem; border-left:4px solid {palette['--info-fg']}; background:{palette['--info-bg']}; }}
            .center {{ text-align:center; }}
            .titlepage {{ height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; }}
            .titlebox {{ border:1px solid {palette['--border']}; background:{palette['--card']}; border-radius: 16px; padding: 28px 34px; max-width: 720px; }}
            .meta {{ margin-top: 10px; }}
            .footer-meta {{ margin-top: 18px; }}
            .pagebreak {{ page-break-after: always; }}
            """

            cover = f"""
            <div class="titlepage">
              <div class="center">{logo_svg}</div>
              <div class="titlebox">
                <div class="center">
                    <div class="muted">{uni_name}</div>
                    <div class="muted">{faculty}</div>
                    <div class="muted">{dept}</div>
                </div>
                <h1 class="center" style="margin: 18px 0 4px 0;">{work_type}</h1>
                <h2 class="center" style="margin: 0 0 16px 0;">{topic}</h2>
                <div class="meta">
                    <div><b>Студент:</b> {student}</div>
                    <div><b>Група:</b> {group} • <b>Спеціальність:</b> {specialty}</div>
                    <div><b>Керівник:</b> {supervisor}</div>
                </div>
                <div class="footer-meta center muted">{city} • {year}</div>
              </div>
            </div>
            <div class="pagebreak"></div>"""

            buf = [f"<html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>"]
            buf.append(cover)
            buf.append("<div class='container'>")

            if add_toc:
                buf.append("<div class='toc'><b>Зміст</b><ol>")
                num = 1
                for it in report_items:
                    if it["type"] == "title":
                        t = it.get("text", "").strip() or f"Розділ {num}"
                        buf.append(f"<li><a href='#sec{num}'>{t}</a></li>")
                        num += 1
                buf.append("</ol></div>")

            sec = 1
            for it in report_items:
                if it["type"] == "title":
                    t = it.get("text", "").strip() or f"Розділ {sec}"
                    buf.append(f"<h2 id='sec{sec}'>{t}</h2>")
                    sec += 1
                elif it["type"] == "note":
                    buf.append(f"<p class='muted'>{it.get('text','')}</p>")
                else:
                    doc = it.get("doc_id", "")
                    txt = (it.get("text", "") or "").replace("\n", "<br>")
                    meta = it.get("meta", {})
                    score_txt = f"<span class='muted'> (score={meta.get('score'):.3f})</span>" if "score" in meta else ""
                    lang_txt = f"<span class='muted'> [lang={meta.get('lang')}]</span>" if "lang" in meta else ""
                    buf.append(f"<div class='section'><div class='muted'>Витяг з документа <span class='pill'>{doc}</span>{score_txt}{lang_txt}</div><div>{txt}</div></div>")

            buf.append("</div></body></html>")
            return "".join(buf)

        html = build_html()
        st.markdown("**Попередній перегляд HTML**")
        components.html(html, height=520, scrolling=True)

        st.download_button("⬇️ Завантажити HTML", data=html.encode("utf-8"), file_name="uxt_report.html", mime="text/html")

        pdf_bytes: Optional[bytes] = render_pdf(html)
        if pdf_bytes is not None:
            st.download_button("⬇️ Завантажити PDF", data=pdf_bytes, file_name="uxt_report.pdf", mime="application/pdf")
        else:
            st.info("Для експорту PDF встановіть `weasyprint` (та системні залежності Cairo/Pango). Поки що доступний експорт HTML.")

        if st.button("🧹 Очистити звіт", key="rb_clear"):
            st.session_state["report_items"] = []
            st.success("Очищено")

# --------------------------- HISTORY ----------------------------
with tab_hist:
    st.markdown("#### 9) History")
    try:
        hist = requests.get(f"{API_URL}/history", timeout=10).json()
        items = hist.get("items", hist) if isinstance(hist, dict) else hist
    except Exception:
        items = []
    if not items:
        st.info("Історія порожня.")
    else:
        dfh = pd.DataFrame(items)
        st.dataframe(dfh, width="stretch", height=350)
        df_to_csv_download(dfh, "⬇️ Export history", "history.csv")

# ============================ FOOTER =============================
st.markdown(
    "<div class='footer'>© 2025 UXT Pipeline — програмний комплекс для обробки неструктурованих даних та вдосконалення систем аналізу тексту. Розроблено в рамках дипломного дослідження.</div>",
    unsafe_allow_html=True,
)
