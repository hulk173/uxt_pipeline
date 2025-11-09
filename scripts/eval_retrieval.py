from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
from typing import List, Dict, Set
from uxt_pipeline.embeddings import Embedder, EmbeddingConfig
from uxt_pipeline.search import VectorIndex, IndexConfig

def recall_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    return 1.0 if any(i in relevant for i in ranked[:k]) else 0.0

def mrr_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    for idx, lab in enumerate(ranked[:k], 1):
        if lab in relevant:
            return 1.0 / idx
    return 0.0

def ndcg_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    dcg = 0.0
    for i, lab in enumerate(ranked[:k], 1):
        rel = 1.0 if lab in relevant else 0.0
        dcg += rel / np.log2(i + 1)
    idcg = 1.0 / np.log2(1 + 1)
    return float(dcg / (idcg + 1e-9))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="CSV: id,text,meta_json")
    ap.add_argument("--queries", required=True, help="CSV: q,positives_json")
    ap.add_argument("--backend", default="sentence_transformers")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--engine", default="faiss")
    ap.add_argument("--out", default="results/retrieval_eval.json")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) ембеддер
    emb = Embedder(EmbeddingConfig(backend=args.backend, model_name=args.model))

    # 2) корпус
    corp = pd.read_csv(args.corpus)
    ids: List[int] = [int(x) for x in corp["id"].tolist()]
    texts: List[str] = [str(x) for x in corp["text"].tolist()]
    meta_col = corp["meta_json"].tolist() if "meta_json" in corp.columns else ["{}"] * len(ids)
    metas: List[Dict] = [json.loads(m) if isinstance(m, str) else {} for m in meta_col]

    X = emb.encode(texts)  # (N, d)

    # 3) індекс
    idx = VectorIndex(IndexConfig(engine=args.engine, dim=X.shape[1]))
    idx.build(X, metas)

    # 4) запити
    qs = pd.read_csv(args.queries)
    Rk, MRR, NDCG = [], [], []
    for _, row in qs.iterrows():
        q = str(row["q"])
        rel_ids: Set[int] = set(int(i) for i in json.loads(row["positives_json"]))
        qv = emb.encode([q])
        hits = idx.search(qv, top_k=args.k)
        ranked = [hid for (hid, _score, _meta) in hits]
        Rk.append(recall_at_k(ranked, rel_ids, args.k))
        MRR.append(mrr_at_k(ranked, rel_ids, args.k))
        NDCG.append(ndcg_at_k(ranked, rel_ids, args.k))

    res = {
        "Recall@k": float(np.mean(Rk)) if len(Rk) else 0.0,
        "MRR@k": float(np.mean(MRR)) if len(MRR) else 0.0,
        "nDCG@k": float(np.mean(NDCG)) if len(NDCG) else 0.0,
        "k": args.k,
        "engine": args.engine,
        "backend": args.backend,
        "model": args.model,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(res)

if __name__ == "__main__":
    main()
