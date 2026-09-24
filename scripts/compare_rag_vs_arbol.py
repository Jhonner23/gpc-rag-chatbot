"""Genera el reporte de comparacion lado a lado RAG vs. arbol de decision.

Cierra el pendiente de la seccion 10.6 de la arquitectura (objetivo de
evaluacion planteado por la asesora): para cada caso clinico curado en
eval/casos_clinicos.json, corre el arbol correspondiente (determinista, sin
LLM) y, si hay infraestructura disponible, tambien el pipeline del RAG con
una pregunta en lenguaje natural equivalente -- y deja las dos respuestas
una al lado de la otra en un reporte Markdown para que un humano las
compare. El script no decide cual respuesta es clinicamente mejor.

Uso (con Ollama y Qdrant corriendo, como para el pipeline normal del RAG):
    PYTHONPATH=src uv run python scripts/compare_rag_vs_arbol.py

Uso sin infraestructura del RAG disponible (solo genera el lado del arbol,
util para revisar el formato del reporte o si Ollama/Qdrant no estan a la
mano en este momento):
    PYTHONPATH=src uv run python scripts/compare_rag_vs_arbol.py --sin-rag

El reporte se escribe en eval/comparacion_rag_arbol.md (no se versiona --
se regenera en cada corrida, igual que eval/ragas_results.csv, ver
.gitignore). El dataset de casos (eval/casos_clinicos.json) si se versiona:
es el input curado, no un artefacto de una corrida puntual.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from gpc_rag.eval.compare import (
    CasoClinicoError,
    CasoComparacion,
    comparar_caso,
    generar_reporte_markdown,
)
from gpc_rag.trees.registry import get_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CASOS_PATH = Path(__file__).parents[1] / "eval" / "casos_clinicos.json"
_REPORTE_PATH = Path(__file__).parents[1] / "eval" / "comparacion_rag_arbol.md"


def _cargar_casos(path: Path) -> list[dict[str, Any]]:
    casos: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return casos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara la respuesta del RAG y del arbol de decision para el mismo caso clinico."
    )
    parser.add_argument(
        "--casos",
        type=Path,
        default=_CASOS_PATH,
        help="JSON con los casos clinicos a comparar (ver eval/casos_clinicos.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPORTE_PATH,
        help="Ruta del reporte Markdown a generar.",
    )
    parser.add_argument(
        "--sin-rag",
        action="store_true",
        help="No corre el pipeline del RAG (util si Ollama/Qdrant no estan disponibles ahora).",
    )
    args = parser.parse_args()

    casos = _cargar_casos(args.casos)
    logger.info("Cargados %d casos clinicos desde %s", len(casos), args.casos)

    rag = None
    if not args.sin_rag:
        try:
            # Carga perezosa: evita pedir Ollama/Qdrant/langgraph si se corre con --sin-rag.
            from gpc_rag.pipelines.query_pipeline import get_pipeline  # noqa: PLC0415

            rag = get_pipeline()
        except Exception as exc:
            logger.warning(
                "No se pudo inicializar el pipeline del RAG (%s: %s) -- el reporte solo "
                "tendra el lado del arbol. Corre con Ollama y Qdrant activos para el lado "
                "completo, o pasa --sin-rag para omitir este intento.",
                type(exc).__name__,
                exc,
            )

    resultados: list[CasoComparacion] = []
    for caso in casos:
        tree = get_tree(caso["tree_id"])
        try:
            resultado = comparar_caso(
                caso_id=caso["caso_id"],
                tree=tree,
                pregunta=caso["pregunta"],
                respuestas_arbol=caso["respuestas_arbol"],
                rag=rag,
            )
        except CasoClinicoError:
            logger.exception("Caso %s mal formado, se omite", caso["caso_id"])
            continue
        resultados.append(resultado)
        logger.info(
            "%s -> hoja %s%s",
            caso["caso_id"],
            resultado.resultado_arbol.node_id,
            "" if resultado.respuesta_rag or resultado.rag_error is None else " (RAG no disponible)",
        )

    reporte = generar_reporte_markdown(resultados)
    args.out.write_text(reporte, encoding="utf-8")
    logger.info("Reporte escrito en %s (%d casos)", args.out, len(resultados))


if __name__ == "__main__":
    main()
