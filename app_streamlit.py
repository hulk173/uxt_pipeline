# app_streamlit.py
import time, json
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="UXT Pipeline", layout="wide")
st.title("UXT Pipeline UI")

# ---- sidebar settings ----
st.sidebar.header("Налаштування")
strategy = st.sidebar.selectbox("Стратегія", ["fast", "hi_res"], index=0)
ocr_languages = st.sidebar.text_input("OCR languages", "eng+ukr")
max_chars = st.sidebar.number_input("max_chars", 200, 5000, 1200, 100)
overlap = st.sidebar.number_input("overlap", 0, 1000, 150, 10)
join_same_type = st.sidebar.checkbox("join_same_type", True)
min_text_chars = st.sidebar.number_input("min_text_chars", 0, 1000, 20, 5)
strip_whitespace = st.sidebar.checkbox("strip_whitespace", True)
index_backend = st.sidebar.selectbox("backend", ["faiss", "sklearn"], index=0)
model_name = st.sidebar.selectbox(
    "sentence-transformers model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ],
    index=0,
)
top_k = st.sidebar.number_input("top_k", 1, 50, 5, 1)

tab1, tab2, tab3 = st.tabs(["Ingest", "Preview", "Search"])

with tab1:
    st.subheader("Upload & Ingest")
    f = st.file_uploader("Оберіть документ (PDF/DOCX/HTML...)")
    if st.button("Ingest", disabled=f is None, type="primary", use_container_width=True):
        if f is None:
            st.warning("Спочатку оберіть файл.")
        else:
            files = {"file": (f.name or "uploaded.bin", f.getvalue())}
            data = {
                "strategy": strategy,
                "ocr_languages": ocr_languages,
                "max_chars": int(max_chars),
                "overlap": int(overlap),
                "join_same_type": json.dumps(bool(join_same_type)),
                "min_text_chars": int(min_text_chars),
                "strip_whitespace": json.dumps(bool(strip_whitespace)),
                "index_backend": index_backend,
                "model_name": model_name,
                "top_k": int(top_k),
            }
            # короткий timeout для старту; далі полимо статус
            r = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=30)
            if not r.ok:
                try: st.error(r.json())
                except Exception: st.error(r.text)
            else:
                resp = r.json()
                job_id = resp["job_id"]
                with st.spinner(f"Processing job {job_id}..."):
                    # poll up to 30 minutes
                    for _ in range(180):
                        s = requests.get(f"{API_URL}/job/{job_id}", timeout=10)
                        info = s.json()
                        st.status = info.get("status", "unknown")
                        if info.get("status") == "finished":
                            st.success("Готово ✅")
                            st.json(info)
                            break
                        if info.get("status") == "error":
                            st.error(info.get("error", "unknown error"))
                            break
                        time.sleep(10)
                    else:
                        st.warning("Перевищено час очікування (30 хв). Перевір /job вручну.")

with tab2:
    st.subheader("Preview exported chunks")
    p = Path("data/out/chunks.jsonl")
    if p.exists():
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Ще не експортували. Спочатку виконайте Ingest.")

with tab3:
    st.subheader("Semantic search")
    q = st.text_input("Query")
    if st.button("Search", disabled=not q):
        r = requests.get(f"{API_URL}/search", params={"q": q}, timeout=60)
        if r.ok:
            res = r.json()
            if not res:
                st.info("Нічого не знайдено.")
            for it in res:
                st.markdown(f"**Score:** {it['score']:.4f} • **Doc:** {it['chunk']['doc_id']} • **Type:** {it['chunk']['type']}")
                st.write(it["chunk"]["text"])
                st.divider()
        else:
            try: st.error(r.json())
            except Exception: st.error(r.text)
