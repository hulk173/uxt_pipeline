# UXT Pipeline — Pro

Пайплайн обробки неструктурованих документів з чанкінгом, FAISS/Sklearn індексом, семантичним пошуком і RAG-QA.

## Фічі
- Ingest (PDF/DOCX/HTML) + OCR (unstructured, hi_res/fast)
- Чанкінг (max_chars/overlap/join_same_type/min_text_chars)
- Експорт JSONL/Parquet/SQLite
- Індекс FAISS або Sklearn (+ L2 normalize)
- Semantic Search, Clusters, Metrics (recall@k/precision@k)
- RAG-QA `/ask` (FLAN-T5 з фолбеком)
- Sentiment / Summarize (transformers)
- UMAP + Plotly візуалізація
- Streamlit UI + FastAPI
- Docker-compose

## Запуск (локально)
```bash
uvicorn api_main:app --reload --port 8000
streamlit run app_streamlit.py
