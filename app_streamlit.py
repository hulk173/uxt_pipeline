# app_streamlit.py
from __future__ import annotations
import os, json, time, re, io, base64
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

# =========================
# ---------- THEME --------
# =========================
st.set_page_config(
    page_title="UXT Pipeline • Pro Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Small CSS polish ---
st.markdown(
    """
    <style>
    /* tighten default paddings */
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    /* badge styles */
    .badge {display:inline-block; padding: 0.2rem .55rem; border-radius: 999px; font-size: 0.78rem; font-weight:600; vertical-align: middle;}
    .b-ok{background:#E9FFF2; color:#067647; border:1px solid #BBF7D0;}
    .b-warn{background:#FFF7E6; color:#A15C06; border:1px solid #FDE68A;}
    .b-err{background:#FFECEC; color:#B42318; border:1px solid #FCA5A5;}
    .b-info{background:#EEF6FF; color:#1D4ED8; border:1px solid #BFDBFE;}
    /* metric cards */
    .card{border:1px solid var(--secondary-background-color); border-radius:14px; padding:1rem 1.1rem; background:rgba(127,127,127,0.03);}
    .card h4{margin:0 0 .3rem 0;}
    .muted{color: var(--text-color-secondary, #5e6b7a); font-size: .9rem;}
    .chip{display:inline-flex; gap:.35rem; align-items:center; padding:.15rem .55rem; border-radius:999px; font-size:.78rem; border:1px solid rgba(125,125,125,.15)}
    .chip i{opacity:.7}
    .pill{font-weight:600; padding:.15rem .5rem; border-radius:999px; border:1px solid rgba(125,125,125,.2)}
    .tiny{font-size:.78rem;}
    .kbd{padding:.05rem .35rem; border:1px solid rgba(125,125,125,.4); border-radius:4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.82em;}
    /* tables */
    .stDataFrame {border-radius: 12px; overflow: hidden;}
    /* footer */
    .footer{opacity:.7; font-size:.85rem; padding-top: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# --------- CONFIG --------
# =========================
API_URL = os.environ.get("UXT_API_URL", "http://localhost:8000")
CFG_DIR = Path("configs"); CFG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CFG_PATH = CFG_DIR / "default.yaml"
LOCAL_CFG_PATH   = CFG_DIR / "local.yaml"

BASE_DEFAULTS: Dict[str, Any] = {
    "ingest": {"strategy": "fast", "ocr_languages": "eng+ukr"},
    "chunking": {
        "max_chars": 1000, "overlap": 120, "join_same_type": True,
        "min_text_chars": 30, "strip_whitespace": True
    },
    "index": {
        "backend": "faiss",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": 5,
        "normalize": True
    },
}

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    if not isinstance(b, dict): return out
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

defaults = load_defaults()

# =========================
# --------- HEADER --------
# =========================
left, right = st.columns([0.75, 0.25], vertical_alignment="center")
with left:
    st.markdown("### 🔎 UXT Pipeline • **Pro Dashboard**")
    st.caption("Метод обробки неструктурованих даних: завантаження → чанкінг → індексація → пошук / RAG-QA → аналітика.")
with right:
    ok = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        ok = r.ok
    except Exception:
        ok = False
    st.markdown(
        f"""
        <div style="text-align:right">
            <span class="badge {'b-ok' if ok else 'b-err'}">
                API { 'ONLINE' if ok else 'OFFLINE'}
            </span>
            &nbsp;&nbsp;<span class="tiny">{API_URL}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# HELP / STEPS
with st.expander("ℹ️ Як користуватись (покроково) — натисніть для інструкції", expanded=False):
    st.markdown(
        """
1. **Ingest**: завантажте документ (PDF/DOCX/HTML). За потреби змініть параметри в лівій панелі.
2. **Preview**: перегляньте нарізані чанки. За бажанням — відфільтруйте по doc_id або типу.
3. **Search**: виконайте семантичний пошук — збіги у тексті буде підсвічено.
4. **Ask**: поставте запитання (RAG-QA). Відповідь буде з **цитатами** джерел.
5. **Metrics**: заміряйте `recall@k` / `precision@k` на невеликому золотому наборі.
6. **Clusters / Visualization**: групуйте та дивіться UMAP-проекцію ембеддингів.
        """
    )

# =========================
# -------- SIDEBAR --------
# =========================
st.sidebar.subheader("⚙️ Налаштування")
strategy = st.sidebar.selectbox("Стратегія OCR", ["fast", "hi_res"], index=0, help="Швидкість/якість парсингу (unstructured).")
ocr_languages = st.sidebar.text_input("OCR languages", value=defaults["ingest"]["ocr_languages"], help="Напр. eng+ukr, eng, deu…")

st.sidebar.markdown("---")
st.sidebar.caption("Чанкінг")
max_chars       = st.sidebar.slider("max_chars", 300, 2000, int(defaults["chunking"]["max_chars"]), 50)
overlap         = st.sidebar.slider("overlap", 0, 400, int(defaults["chunking"]["overlap"]), 10)
join_same_type  = st.sidebar.checkbox("join_same_type", value=bool(defaults["chunking"]["join_same_type"]))
min_text_chars  = st.sidebar.slider("min_text_chars", 0, 200, int(defaults["chunking"]["min_text_chars"]), 5)
strip_whitespace= st.sidebar.checkbox("strip_whitespace", value=bool(defaults["chunking"]["strip_whitespace"]))

st.sidebar.markdown("---")
st.sidebar.caption("Індекс")
index_backend = st.sidebar.selectbox("backend", ["faiss", "sklearn"], index=0, help="Під капотом: FAISS/Sklearn.")
model_name = st.sidebar.selectbox(
    "Sentence model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ], index=0
)
top_k = st.sidebar.slider("top_k", 1, 20, int(defaults["index"]["top_k"]), 1)
normalize = st.sidebar.checkbox("normalize (L2)", value=bool(defaults["index"]["normalize"]), help="Рекомендується для cosine/IP.")

st.sidebar.markdown("---")
colsa, colsb = st.sidebar.columns(2)
with colsa:
    if st.button("💾 Save defaults"):
        cfg = {
            "ingest": {"strategy": strategy, "ocr_languages": ocr_languages},
            "chunking": {
                "max_chars": int(max_chars), "overlap": int(overlap),
                "join_same_type": bool(join_same_type),
                "min_text_chars": int(min_text_chars),
                "strip_whitespace": bool(strip_whitespace),
            },
            "index": {
                "backend": index_backend, "model_name": model_name,
                "top_k": int(top_k), "normalize": bool(normalize)
            },
        }
        save_defaults(cfg)
        st.sidebar.success("Збережено в configs/local.yaml")
with colsb:
    if st.button("🧹 Clear outputs"):
        try:
            r = requests.delete(f"{API_URL}/outputs"); st.sidebar.success(r.json())
        except Exception as ex:
            st.sidebar.error(ex)

# =========================
# ---------- TABS ---------
# =========================
tabs = st.tabs(["📥 Ingest", "👁 Preview", "🔎 Search", "💬 Ask", "📊 Metrics", "🗂 Clusters", "🗺 Visualization", "🕰 History"])
tab_ingest, tab_preview, tab_search, tab_ask, tab_metrics, tab_clusters, tab_viz, tab_hist = tabs

# ---------- helpers ----------
def highlight(text: str, query: str) -> str:
    if not query: return text
    try:
        patt = re.compile(re.escape(query), re.IGNORECASE)
        return patt.sub(lambda m: f"**{m.group(0)}**", text)
    except Exception:
        return text

def df_to_csv_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def code_copy_box(text: str, label: str="Copy"):
    st.text_area(label, value=text, height=80)

# =========================
# -------- INGEST ----------
# =========================
with tab_ingest:
    st.markdown("#### 1) Завантаження та нарізка документів")
    left, right = st.columns([0.6, 0.4], vertical_alignment="bottom")

    with left:
        f = st.file_uploader("Обрати документ", type=["pdf", "docx", "html", "txt"])
        st.caption("Ліміт 200MB/файл • Підтримка PDF/DOCX/HTML/TXT")
    with right:
        st.markdown("<div class='card'><h4>Поради</h4><div class='muted'>• Для сканів використовуйте <b>hi_res</b> та правильну мову OCR.<br>• Не занижуйте <b>min_text_chars</b> — це зменшить шум.<br>• Для cosine схожості залиште <b>normalize</b> увімкненим.</div></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.18, 0.18, 0.64])
    with c1:
        run = st.button("🚀 Ingest", type="primary", disabled=f is None)
    with c2:
        batch = st.file_uploader("Batch (multi)", accept_multiple_files=True, label_visibility="collapsed")

    if run and f is None:
        st.warning("Оберіть файл.")
    elif run and f is not None:
        files = {"file": (f.name or "uploaded.bin", f.getvalue())}
        data = {
            "strategy": strategy, "ocr_languages": ocr_languages,
            "max_chars": int(max_chars), "overlap": int(overlap),
            "join_same_type": json.dumps(bool(join_same_type)),
            "min_text_chars": int(min_text_chars),
            "strip_whitespace": json.dumps(bool(strip_whitespace)),
            "index_backend": index_backend, "model_name": model_name,
            "top_k": int(top_k), "normalize": json.dumps(bool(normalize)),
        }
        try:
            r = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=60)
            if not r.ok:
                st.error(r.text)
            else:
                job_id = r.json()["job_id"]
                st.info(f"job: {job_id}")
                with st.spinner("Обробка…"):
                    for _ in range(180):
                        s = requests.get(f"{API_URL}/job/{job_id}", timeout=10).json()
                        if s.get("status") == "finished": st.success("Готово!"); st.json(s); break
                        if s.get("status") == "error": st.error(s.get("error")); break
                        time.sleep(5)
        except Exception as ex:
            st.error(ex)

    if batch:
        files = [("files", (x.name or "file", x.getvalue())) for x in batch]
        data = {
            "strategy": strategy, "ocr_languages": ocr_languages,
            "max_chars": int(max_chars), "overlap": int(overlap),
            "join_same_type": json.dumps(bool(join_same_type)),
            "min_text_chars": int(min_text_chars),
            "strip_whitespace": json.dumps(bool(strip_whitespace)),
            "index_backend": index_backend, "model_name": model_name,
            "top_k": int(top_k), "normalize": json.dumps(bool(normalize)),
        }
        if st.button("📦 Ingest batch", type="secondary"):
            r = requests.post(f"{API_URL}/ingest_batch", files=files, data=data, timeout=180)
            st.json(r.json())

# =========================
# -------- PREVIEW --------
# =========================
with tab_preview:
    st.markdown("#### 2) Попередній перегляд чанків")
    p = Path("data/out/chunks.jsonl")
    if not p.exists():
        st.info("Немає `data/out/chunks.jsonl`. Зробіть Ingest.")
    else:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        df = pd.DataFrame(rows)
        st.caption(f"Рядків: {len(df)}")
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            doc_filter = st.text_input("Фільтр doc_id (regex)", "")
        with colf2:
            type_filter = st.text_input("Фільтр type (regex)", "")
        with colf3:
            text_filter = st.text_input("Пошук у тексті (regex)", "")

        fdf = df.copy()
        if doc_filter:  fdf = fdf[fdf["doc_id"].astype(str).str.contains(doc_filter, regex=True, na=False)]
        if type_filter: fdf = fdf[fdf["type"].astype(str).str.contains(type_filter, regex=True, na=False)]
        if text_filter: fdf = fdf[fdf["text"].astype(str).str.contains(text_filter, regex=True, na=False)]

        st.dataframe(fdf, use_container_width=True, height=420)
        df_to_csv_download(fdf, "⬇️ Export CSV", "chunks_filtered.csv")

# =========================
# -------- SEARCH ---------
# =========================
with tab_search:
    st.markdown("#### 3) Семантичний пошук")
    q = st.text_input("Запит")
    if st.button("🔎 Search", disabled=not q):
        try:
            res = requests.get(f"{API_URL}/search", params={"q": q}, timeout=60).json()
            if not res:
                st.info("Нічого не знайдено.")
            for it in res:
                score = float(it["score"])
                badge = "b-ok" if score >= 0.65 else ("b-warn" if score >= 0.45 else "b-info")
                st.markdown(f"<span class='badge {badge}'>score={score:.3f}</span> &nbsp; <span class='pill'>{it['chunk']['doc_id']}</span>", unsafe_allow_html=True)
                st.markdown(highlight(it["chunk"]["text"], q))
                with st.expander("📎 Цитата • copy"):
                    code_copy_box(it["chunk"]["text"], label="Під копіювання")
                st.divider()
        except Exception as ex:
            st.error(ex)

# =========================
# ---------- ASK ----------
# =========================
with tab_ask:
    st.markdown("#### 4) RAG-QA — поставте запитання до ваших документів")
    cq1, cq2 = st.columns([0.7, 0.3])
    with cq1:
        qq = st.text_input("Запитання")
    with cq2:
        kk = st.slider("k (контекст)", 1, 10, int(top_k), 1)

    if st.button("💬 Ask", disabled=not qq):
        data = requests.get(f"{API_URL}/ask", params={"q": qq, "k": kk}, timeout=180).json()
        st.markdown(f"##### Відповідь")
        st.markdown(data.get("answer", ""))
        st.markdown("##### Джерела")
        src = pd.DataFrame(data.get("sources", []))
        if len(src):
            st.dataframe(src, use_container_width=True, height=240)
            df_to_csv_download(src, "⬇️ Export sources", "ask_sources.csv")
        else:
            st.info("Цитати відсутні (перевірте індекс/чанки).")

# =========================
# -------- METRICS --------
# =========================
with tab_metrics:
    st.markdown("#### 5) Метрики швидкої оцінки якості")
    m1, m2 = st.columns(2)
    with m1: mq = st.text_area("Query", "")
    with m2: expected = st.text_area("Expected snippet", "")
    k = st.slider("k", 1, 20, int(top_k), 1, help="Скільки результатів враховувати.")
    if st.button("🏁 Evaluate", disabled=not mq):
        data = {"query": mq, "expected_text": expected, "k": k}
        out = requests.post(f"{API_URL}/evaluate", json=data, timeout=60).json()
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("recall@k", f"{out['recall@k']:.3f}")
        with c2: st.metric("precision@k", f"{out['precision@k']:.3f}")
        with c3: st.metric("k", out["k"])
        st.dataframe(pd.DataFrame(out["results"]), use_container_width=True, height=280)

# =========================
# ------- CLUSTERS --------
# =========================
with tab_clusters:
    st.markdown("#### 6) Кластеризація чанків")
    nc = st.slider("Кількість кластерів", 2, 30, 8, 1)
    if st.button("🗂 Compute clusters"):
        data = requests.get(f"{API_URL}/clusters", params={"n_clusters": nc}, timeout=180).json()
        if "clusters" in data:
            df = pd.DataFrame(data["clusters"])
            st.dataframe(df, use_container_width=True, height=400)
            df_to_csv_download(df, "⬇️ Export clusters", "clusters.csv")
        else:
            st.info(data)

# =========================
# ------ VISUALIZATION ----
# =========================
with tab_viz:
    st.markdown("#### 7) Візуалізація ембеддингів (UMAP)")
    p = Path("data/out/chunks.jsonl")
    if not p.exists():
        st.info("Немає `data/out/chunks.jsonl`. Зробіть Ingest.")
    else:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
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
                ], index=0
            )
            nmax = st.slider("Максимум точок", 200, 5000, min(1200, len(texts)), 100)
            if st.button("🗺 Compute & Plot", type="primary"):
                try:
                    from sentence_transformers import SentenceTransformer
                    import umap, plotly.express as px
                except Exception as ex:
                    st.error("Необхідно встановити: sentence-transformers, umap-learn, plotly"); st.stop()
                model = SentenceTransformer(vis_model)
                X = model.encode(texts[:nmax], convert_to_numpy=True, show_progress_bar=True)
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
                emb2d = np.asarray(reducer.fit_transform(X), dtype=float)
                dfv = pd.DataFrame({"x": emb2d[:, 0], "y": emb2d[:, 1], "id": ids[:nmax], "text": texts[:nmax]})
                fig = px.scatter(dfv, x="x", y="y", hover_data=["id", "text"], title="UMAP Embeddings")
                st.plotly_chart(fig, use_container_width=True)

# =========================
# -------- HISTORY --------
# =========================
with tab_hist:
    st.markdown("#### 8) Історія подій")
    try:
        hist = requests.get(f"{API_URL}/history", timeout=10).json()
    except Exception as ex:
        hist = []
    if not hist:
        st.info("Історія порожня. Здійсніть ingest/search/ask/evaluate.")
    else:
        dfh = pd.DataFrame(hist)
        st.dataframe(dfh, use_container_width=True, height=350)
        df_to_csv_download(dfh, "⬇️ Export history", "history.csv")

# =========================
# -------- FOOTER ---------
# =========================
st.markdown(
    """
    <div class="footer">
        <span class="muted">Готово ✅ • Для дипломного звіту скріни: Ingest → Preview → Search → Ask → Metrics → Clusters → Visualization.</span>
        <br><span class="muted">Порада: для україномовних даних пам’ятайте про правильні OCR-мови та можливо більший max_chars (800–1200).</span>
    </div>
    """,
    unsafe_allow_html=True,
)
