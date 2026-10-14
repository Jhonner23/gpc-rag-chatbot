"""Precision@k, Recall@k y F1@k del recuperador sobre el banco de los protocolos."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from gpc_rag.pipelines.query_pipeline import QueryPipeline

KS = (1, 3, 6)
BANK = Path("eval/golden_protocolos.json")
OUT_CSV = Path("eval/retrieval_metrics_protocolos.csv")


def main() -> None:
    data = json.loads(BANK.read_text(encoding="utf-8"))
    pipe = QueryPipeline()
    print(f"Chunks en el indice: {len(pipe.retriever._bm25_chunks)}")
    pipe.retriever.retrieve("neumonía")  # calentamiento del reranker

    rows = []
    for i, item in enumerate(data, 1):
        t = time.perf_counter()
        retrieved = pipe.retriever.retrieve(item["question"])
        t_rec = time.perf_counter() - t
        ids = [rc.chunk.chunk_id for rc in retrieved]
        rel = set(item["relevant_chunk_ids"])
        row = {"id": i, "should_refuse": item["should_refuse"], "t_recuperacion_s": round(t_rec, 2),
               "n_relevantes": len(rel), "recuperados": " | ".join(ids)}
        if rel:
            for k in KS:
                top = ids[:k]
                hits = sum(cid in rel for cid in top)
                p = hits / len(top) if top else 0.0
                r = hits / len(rel)
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                row[f"P@{k}"], row[f"R@{k}"], row[f"F1@{k}"] = round(p, 3), round(r, 3), round(f1, 3)
            row["hit@6"] = int(any(cid in rel for cid in ids[:6]))
        rows.append(row)
        print(f"[{i}/{len(data)}] {t_rec:.1f} s  " + ("R@6=" + str(row.get('R@6')) if rel else "fuera de alcance"))

    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ev = [r for r in rows if "P@1" in r]
    print(f"\n=== RESUMEN ({len(ev)} preguntas con fragmentos relevantes) ===")
    for k in KS:
        print(f"k={k}: P={statistics.mean(r[f'P@{k}'] for r in ev):.3f}  "
              f"R={statistics.mean(r[f'R@{k}'] for r in ev):.3f}  "
              f"F1={statistics.mean(r[f'F1@{k}'] for r in ev):.3f}")
    print(f"Hit rate@6: {statistics.mean(r['hit@6'] for r in ev):.3f}")
    tr = [r["t_recuperacion_s"] for r in rows]
    print(f"Tiempo recuperacion: media {statistics.mean(tr):.1f} s | mediana {statistics.median(tr):.1f} s")
    print(f"Detalle en {OUT_CSV}")


if __name__ == "__main__":
    main()
