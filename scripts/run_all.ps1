# 1) індексація з БД чанків (спростимо: з CSV, який експортуєш із API)
# Приклад підготовки CSV: columns = id,text,meta_json
$CORPUS = "data/corpus.csv"
$QUERIES = "data/queries.csv"

# 2) оцінка пошуку
python scripts/eval_retrieval.py --corpus $CORPUS --queries $QUERIES --backend sentence_transformers --model sentence-transformers/all-MiniLM-L6-v2 --engine faiss --k 10 --out results/retrieval_eval.json

# 3) зведення у звіт (за наявності ноутбука)
jupyter nbconvert --to html --execute notebooks/report.ipynb --output results/report.html
