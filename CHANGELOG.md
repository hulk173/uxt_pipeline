# Changelog
Усі значні зміни цього проєкту документуються у цьому файлі.

## [1.0.0] — 2025-11-XX
### Added
- Повноцінний пайплайн обробки неструктурованих документів (PDF/DOCX/HTML).
- Алгоритм семантичного чанкінгу з параметрами `chunk_size`, `overlap`.
- Нормалізація тексту: очистка, уніфікація пробілів, детекція мови.
- Модуль ембеддингів (`uxt_pipeline/embeddings.py`).
- Інвертований індекс та kNN-пошук на FAISS/HNSWlib (`search.py`).
- SQLite + SQLAlchemy + FTS5 текстовий пошук.
- Streamlit UI для інтерактивної обробки файлів.
- FastAPI backend:
  - `/runs` для обробки документів
  - `/search` для семантичних запитів
  - `/runs/{id}` для отримання метаданих.
- CLI-інструменти:
  - `uxt ingest file.pdf --out out.jsonl`
  - `uxt index data/`
  - `uxt query "запит"`.
- Система експорту: JSONL, Parquet, SQLite.
- Юніт-тести для chunking/normalization.
- Конфігурації: `configs/default.yaml`.
- Скрипти для бенчмарків: `scripts/evaluate.py`.
- Архітектурна документація (`docs/architecture.md`).

### Changed
- Покращено структуру каталогу `uxt_pipeline/`.
- Уніфіковано логування та винесено параметри в конфіг.

### Fixed
- Виправлені помилки partition для DOCX.
- Виправлено edge-cases при розбитті таблиць та переліків.
- Виправлений FTS5-індекс (забрано `rowid` як поле).

### Removed
- Старі прототипи CLI та неактуальні модулі.

