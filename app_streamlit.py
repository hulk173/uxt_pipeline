from __future__ import annotations
import io, os, requests
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"
DEFAULT_USER = os.getenv("UXT_USER", "admin")
DEFAULT_PASS = os.getenv("UXT_PASS", "admin")

st.set_page_config(page_title="UXT — Text Analysis System", page_icon="📄", layout="wide")
st.title("📄 UXT — Upload → Chunk → Analyze")

# ——— авторизація до API
st.sidebar.header("🔐 Авторизація")
user = st.sidebar.text_input("Користувач", DEFAULT_USER)
password = st.sidebar.text_input("Пароль", DEFAULT_PASS, type="password")
auth = (user, password)

# ——— параметри чанкінгу
chunk_size = st.sidebar.number_input("Chunk size", 100, 4000, 800, 50)
overlap = st.sidebar.number_input("Overlap", 0, 1000, 100, 10)
comment = st.sidebar.text_input("Коментар (опціонально)", placeholder="напр., Лаби / PDF-скани")

# ——— завантаження файлів → прогін через API
uploaded = st.file_uploader("Upload PDF / DOCX / HTML", type=["pdf","docx","html","htm"], accept_multiple_files=True)
if uploaded and st.button("🚀 Запустити прогін через API"):
    with st.spinner("Обробка..."):
        files = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in uploaded]
        r = requests.post(f"{API_URL}/runs", auth=auth, data={"chunk_size": chunk_size, "overlap": overlap, "comment": comment}, files=files)
    if r.ok:
        st.success(f"✅ Run створено: {r.json()['id']} | чанків: {r.json()['chunks']}")
    else:
        st.error(f"❌ {r.status_code}: {r.text}")

st.write("---")
st.subheader("🗂️ Історія прогонів")

if st.button("🔄 Оновити список"):
    r = requests.get(f"{API_URL}/runs", auth=auth)
    if r.ok:
        st.session_state["runs"] = pd.DataFrame(r.json())
    else:
        st.error(f"Помилка: {r.status_code} — {r.text}")

runs_df = st.session_state.get("runs")
if runs_df is not None:
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    if not runs_df.empty:
        run_id = st.number_input("Run ID", int(runs_df["id"].min()), int(runs_df["id"].max()), int(runs_df["id"].iloc[0]), 1)
        c1, c2, c3 = st.columns(3)
        if c1.button("📊 Показати чанки"):
            rr = requests.get(f"{API_URL}/runs/{run_id}/chunks", auth=auth)
            if rr.ok:
                df = pd.DataFrame(rr.json())
                st.dataframe(df, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(); ax.hist(df["len_words"], bins=20); ax.set_title("Chunk length distribution"); st.pyplot(fig)
            else:
                st.error(f"Помилка: {rr.status_code}")

        if c2.button("🗑️ Видалити Run"):
            requests.delete(f"{API_URL}/runs/{run_id}", auth=auth)
            st.warning("Видалено. Натисни «Оновити список».")
        if c3.button("⬇️ Експорт CSV"):
            exp = requests.get(f"{API_URL}/runs/{run_id}/export.csv", auth=auth)
            if exp.ok:
                st.download_button("⬇️ Download CSV", data=exp.content, file_name=f"run_{run_id}.csv", mime="text/csv")
            else:
                st.error("Не вдалося експортувати.")

st.write("---")
st.subheader("🔎 Пошук по чанках (FTS5)")

q = st.text_input('Запит (FTS5): напр. "neural network", learn*')
colA, colB = st.columns([1,1])
with colA:
    run_filter = st.number_input("Фільтр за Run ID (0 = всі)", min_value=0, value=0, step=1)
with colB:
    limit = st.number_input("Ліміт", min_value=1, max_value=1000, value=50, step=10)

if st.button("Шукати"):
    params = {"q": q, "limit": int(limit)}
    if run_filter > 0:
        params["run_id"] = int(run_filter)
    sr = requests.get(f"{API_URL}/search", params=params, auth=auth)
    if sr.ok:
        res = pd.DataFrame(sr.json())
        if not res.empty:
            st.dataframe(res, use_container_width=True, hide_index=True)
        else:
            st.info("Нічого не знайдено.")
    else:
        st.error(f"Помилка пошуку: {sr.status_code} — {sr.text}")
