"""Extrae candidatos de pregunta/respuesta desde los chunks indexados.

Los chunks cuya seccion es una pregunta clinica numerada (ej. "13. En
pacientes...") contienen la Recomendacion formal de la GPC -- son la mejor
fuente de ground truth porque pregunta y respuesta vienen del mismo
documento que se va a evaluar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from hydra import compose, initialize

from gpc_rag.vectorstore.qdrant_store import QdrantStore

_NUMBERED = re.compile(r"^\d{1,3}\.\s")


def main() -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config")

    store = QdrantStore(cfg, vector_size=cfg.embeddings_cfg.dimensions)
    chunks = store.scroll_all_chunks()

    candidates = []
    for c in chunks:
        if _NUMBERED.match(c.section):
            candidates.append(
                {
                    "question": c.section,
                    "context_excerpt": c.text[:500],
                    "source_file": c.source_file,
                    "page": c.page,
                    "section": c.section,
                    "ground_truth_answer": "",
                }
            )

    out_path = Path("eval/qa_candidates.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(candidates)} candidatos escritos en {out_path}")


if __name__ == "__main__":
    main()
