"""Harness de comparacion lado a lado: respuesta del RAG vs. respuesta del arbol.

Objetivo de evaluacion planteado explicitamente por la asesora (ver seccion
10.6 de la arquitectura): para la misma pregunta/caso clinico, poner una al
lado de la otra la respuesta libre del RAG y la respuesta determinista del
arbol de decision, con sus citas, para que un humano (no este script) juzgue
si concuerdan clinicamente. Este modulo NO decide cual respuesta es
"correcta" -- coherente con el principio ya establecido en el proyecto de
que la validacion clinica es responsabilidad de la clinica, no del codigo
(ver trees/schema.py: "Validacion = fidelidad de extraccion, no
re-validacion clinica").

Dos piezas:
- `avanzar_arbol_con_respuestas`: recorre un DecisionTree con un diccionario
  fijo {variable: respuesta} (un "caso clinico"), sin necesitar interaccion
  turno a turno como el wizard -- util para automatizar la evaluacion.
- `comparar_caso` / `generar_reporte_markdown`: arman el resultado lado a
  lado para uno o varios casos.

El lado del RAG necesita un `RagResponder` real (`QueryPipeline`, que a su
vez necesita Ollama + Qdrant corriendo) -- por eso se recibe como parametro
en vez de importarse directo, lo que permite probar el armado del reporte
con un responder falso, sin depender de infraestructura viva (mismo patron
que `tests/chat/fake_chainlit.py` para el wizard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gpc_rag.common.models import RagAnswer
from gpc_rag.trees.engine import TreeSession, TreeValidationError
from gpc_rag.trees.schema import DecisionNode, DecisionTree, LeafNode


class RagResponder(Protocol):
    """Lo minimo que necesita este harness del lado del RAG.

    `QueryPipeline.answer` ya cumple esta forma -- no hace falta heredar de
    nada, Python la acepta por duck typing (Protocol es solo para que mypy lo
    verifique). Un doble de pruebas solo necesita implementar este metodo.
    """

    def answer(self, question: str, extra_instruction: str | None = None) -> RagAnswer: ...


class CasoClinicoError(ValueError):
    """El caso clinico no trae respuesta para una variable que el arbol pregunta."""


@dataclass(frozen=True)
class ResultadoArbol:
    """A donde llego el recorrido determinista del arbol para un caso dado."""

    node_id: str
    recommendation: str
    evidence_level: str | None
    guide_file: str
    section: str
    page: int | None
    source_quote: str
    ruta: list[str]  # ids de los nodos visitados, en orden -- trazabilidad


@dataclass(frozen=True)
class CasoComparacion:
    """Un caso clinico y el resultado de correrlo por las dos vias."""

    caso_id: str
    tree_id: str
    pregunta: str
    respuestas_arbol: dict[str, bool | int | float | str]
    resultado_arbol: ResultadoArbol
    respuesta_rag: RagAnswer | None = None
    rag_error: str | None = None  # motivo si no se pudo obtener respuesta del RAG


def avanzar_arbol_con_respuestas(
    tree: DecisionTree, respuestas: dict[str, bool | int | float | str]
) -> ResultadoArbol:
    """Recorre `tree` de principio a fin usando un caso clinico fijo (no interactivo).

    Cada nodo de decision visitado necesita su `variable` presente en
    `respuestas` -- si falta, `CasoClinicoError` dice exactamente cual (esto
    es lo que distingue un caso mal armado de un bug real del motor).
    """
    session = TreeSession(tree)
    while not session.is_finished():
        node = session.current_node()
        assert isinstance(node, DecisionNode)  # is_finished() ya lo garantiza
        if node.variable not in respuestas:
            raise CasoClinicoError(
                f"el caso no trae respuesta para la variable {node.variable!r} "
                f"(nodo {node.node_id!r} del arbol {tree.tree_id!r})"
            )
        try:
            session.answer(respuestas[node.variable])
        except TreeValidationError as exc:
            raise CasoClinicoError(
                f"la respuesta {respuestas[node.variable]!r} para la variable "
                f"{node.variable!r} no matchea ningun branch del nodo {node.node_id!r}: {exc}"
            ) from exc

    leaf = session.current_node()
    assert isinstance(leaf, LeafNode)
    return ResultadoArbol(
        node_id=leaf.node_id,
        recommendation=leaf.recommendation,
        evidence_level=leaf.evidence_level,
        guide_file=leaf.source.guide_file,
        section=leaf.source.section,
        page=leaf.source.page,
        source_quote=leaf.source_quote,
        ruta=list(session.trail),
    )


def comparar_caso(
    *,
    caso_id: str,
    tree: DecisionTree,
    pregunta: str,
    respuestas_arbol: dict[str, bool | int | float | str],
    rag: RagResponder | None,
) -> CasoComparacion:
    """Corre un caso clinico por el arbol y, si hay `rag` disponible, tambien por el RAG.

    `rag=None` permite generar solo el lado del arbol (por ejemplo si no hay
    Ollama/Qdrant corriendo a la mano) -- el reporte lo deja explicito en vez
    de fallar entero.
    """
    resultado_arbol = avanzar_arbol_con_respuestas(tree, respuestas_arbol)

    respuesta_rag: RagAnswer | None = None
    rag_error: str | None = None
    if rag is not None:
        try:
            respuesta_rag = rag.answer(pregunta)
        except Exception as exc:
            rag_error = f"{type(exc).__name__}: {exc}"

    return CasoComparacion(
        caso_id=caso_id,
        tree_id=tree.tree_id,
        pregunta=pregunta,
        respuestas_arbol=respuestas_arbol,
        resultado_arbol=resultado_arbol,
        respuesta_rag=respuesta_rag,
        rag_error=rag_error,
    )


def generar_reporte_markdown(resultados: list[CasoComparacion]) -> str:
    """Arma el reporte lado a lado en Markdown -- para leer, no para que el codigo decida nada.

    Cada caso queda en su propia seccion con dos bloques uno debajo del otro
    (arbol y RAG) en vez de una tabla, porque las respuestas del RAG suelen
    ser largas y una tabla las vuelve ilegibles.
    """
    lineas: list[str] = [
        "# Comparacion RAG vs. arbol de decision",
        "",
        "Generado por `scripts/compare_rag_vs_arbol.py`. Este reporte no emite ",
        "un veredicto de cual respuesta es clinicamente correcta -- eso lo ",
        "decide quien lo revise (ver `src/gpc_rag/eval/compare.py`).",
        "",
    ]

    for r in resultados:
        lineas.append(f"## {r.caso_id} -- {r.tree_id}")
        lineas.append("")
        lineas.append(f"**Pregunta:** {r.pregunta}")
        lineas.append("")
        lineas.append(
            "**Caso (respuestas usadas para el arbol):** "
            + ", ".join(f"{k}={v}" for k, v in r.respuestas_arbol.items())
        )
        lineas.append("")

        lineas.append("### Arbol de decision")
        lineas.append("")
        lineas.append(f"- **Nodo hoja:** `{r.resultado_arbol.node_id}`")
        lineas.append(f"- **Recomendacion:** {r.resultado_arbol.recommendation}")
        if r.resultado_arbol.evidence_level:
            lineas.append(f"- **Nivel de evidencia:** {r.resultado_arbol.evidence_level}")
        pagina = f", pag. {r.resultado_arbol.page}" if r.resultado_arbol.page is not None else ""
        lineas.append(
            f"- **Fuente:** {r.resultado_arbol.guide_file} -- {r.resultado_arbol.section}{pagina}"
        )
        lineas.append(f"  > {r.resultado_arbol.source_quote}")
        lineas.append("")

        lineas.append("### RAG")
        lineas.append("")
        if r.respuesta_rag is not None:
            lineas.append(r.respuesta_rag.answer)
            lineas.append("")
            if r.respuesta_rag.sources:
                lineas.append("**Fuentes citadas por el RAG:**")
                for fuente in r.respuesta_rag.sources_as_dicts():
                    pag = f", pag. {fuente['page']}" if fuente["page"] is not None else ""
                    lineas.append(f"- {fuente['source_file']} -- {fuente['section']}{pag}")
            lineas.append("")
        elif r.rag_error:
            lineas.append(f"*No disponible: {r.rag_error}*")
            lineas.append("")
        else:
            lineas.append("*No se corrio el lado del RAG para este caso.*")
            lineas.append("")

    return "\n".join(lineas)
