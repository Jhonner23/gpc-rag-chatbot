"""Pruebas del harness de comparacion RAG vs. arbol (src/gpc_rag/eval/compare.py).

No dependen de Ollama ni Qdrant: el lado del arbol usa los JSON reales del
bosque (via trees.registry, igual que el wizard), y el lado del RAG usa un
`RagResponder` falso -- basta con que implemente `.answer(pregunta)`, el
mismo patron de doble de pruebas que ya usa tests/chat/fake_chainlit.py para
no depender de un servidor Chainlit real.
"""

from __future__ import annotations

import pytest

from gpc_rag.common.models import Chunk as ChunkModel
from gpc_rag.common.models import RagAnswer, RetrievedChunk
from gpc_rag.eval.compare import (
    CasoClinicoError,
    avanzar_arbol_con_respuestas,
    comparar_caso,
    generar_reporte_markdown,
)
from gpc_rag.trees.registry import get_tree

_TREE_CURB65 = "nac-2026-severidad-hospitalizacion"
_TREE_IDSA = "nac-2026-criterios-idsa-ats-uci"


class _RagFalso:
    """Doble de pruebas para RagResponder: devuelve siempre la misma respuesta fija."""

    def __init__(self, respuesta: str = "Respuesta simulada del RAG.", *, falla: bool = False):
        self._respuesta = respuesta
        self._falla = falla
        self.preguntas_recibidas: list[str] = []

    def answer(self, question: str, extra_instruction: str | None = None) -> RagAnswer:
        self.preguntas_recibidas.append(question)
        if self._falla:
            raise RuntimeError("Ollama no disponible (simulado)")
        chunk = ChunkModel(text="...", source_file="Guia de practica clinica Colombiana NAC 2026.pdf", section="Tabla 3", page=199)
        return RagAnswer(
            question=question,
            answer=self._respuesta,
            sources=[RetrievedChunk(chunk=chunk, score=0.9)],
        )


def test_avanzar_arbol_curb65_llega_a_riesgo_bajo():
    tree = get_tree(_TREE_CURB65)
    resultado = avanzar_arbol_con_respuestas(
        tree,
        {
            "confusion_nueva": False,
            "urea_elevada": False,
            "frecuencia_respiratoria_alta": False,
            "presion_arterial_baja": False,
            "edad_65_o_mas": False,
        },
    )
    assert resultado.node_id == "leaf_score0"
    assert resultado.ruta[0] == tree.root_node_id
    assert resultado.ruta[-1] == "leaf_score0"


def test_avanzar_arbol_idsa_uci_directa_por_criterio_mayor():
    tree = get_tree(_TREE_IDSA)
    resultado = avanzar_arbol_con_respuestas(tree, {"major_vm_invasiva": True})
    assert resultado.node_id == "leaf_uci_directa"


def test_avanzar_arbol_idsa_tres_menores_con_corte_temprano():
    tree = get_tree(_TREE_IDSA)
    resultado = avanzar_arbol_con_respuestas(
        tree,
        {
            "major_vm_invasiva": False,
            "major_choque_septico": False,
            "minor_fr30": True,
            "minor_pafi250": True,
            "minor_infiltrados_multilobares": False,
            "minor_confusion": True,
            # a proposito NO se incluyen los criterios restantes (bun20 en
            # adelante): el corte temprano del arbol no deberia necesitarlos.
        },
    )
    assert resultado.node_id == "leaf_uci_criterios_menores"


def test_avanzar_arbol_falla_claro_si_falta_una_variable():
    tree = get_tree(_TREE_CURB65)
    with pytest.raises(CasoClinicoError, match="confusion_nueva"):
        avanzar_arbol_con_respuestas(tree, {})


def test_comparar_caso_sin_rag_deja_ese_lado_vacio():
    tree = get_tree(_TREE_CURB65)
    resultado = comparar_caso(
        caso_id="caso-1",
        tree=tree,
        pregunta="Paciente sin criterios CURB-65, requiere hospitalizacion?",
        respuestas_arbol={
            "confusion_nueva": False,
            "urea_elevada": False,
            "frecuencia_respiratoria_alta": False,
            "presion_arterial_baja": False,
            "edad_65_o_mas": False,
        },
        rag=None,
    )
    assert resultado.respuesta_rag is None
    assert resultado.rag_error is None
    assert resultado.resultado_arbol.node_id == "leaf_score0"


def test_comparar_caso_con_rag_disponible():
    tree = get_tree(_TREE_CURB65)
    rag = _RagFalso("Segun la guia, no requiere hospitalizacion.")
    resultado = comparar_caso(
        caso_id="caso-1",
        tree=tree,
        pregunta="Paciente sin criterios CURB-65, requiere hospitalizacion?",
        respuestas_arbol={
            "confusion_nueva": False,
            "urea_elevada": False,
            "frecuencia_respiratoria_alta": False,
            "presion_arterial_baja": False,
            "edad_65_o_mas": False,
        },
        rag=rag,
    )
    assert resultado.respuesta_rag is not None
    assert resultado.respuesta_rag.answer == "Segun la guia, no requiere hospitalizacion."
    assert rag.preguntas_recibidas == ["Paciente sin criterios CURB-65, requiere hospitalizacion?"]


def test_comparar_caso_con_rag_que_falla_no_tumba_el_harness():
    tree = get_tree(_TREE_CURB65)
    rag = _RagFalso(falla=True)
    resultado = comparar_caso(
        caso_id="caso-1",
        tree=tree,
        pregunta="cualquier pregunta",
        respuestas_arbol={
            "confusion_nueva": False,
            "urea_elevada": False,
            "frecuencia_respiratoria_alta": False,
            "presion_arterial_baja": False,
            "edad_65_o_mas": False,
        },
        rag=rag,
    )
    assert resultado.respuesta_rag is None
    assert resultado.rag_error is not None
    assert "no disponible" in resultado.rag_error.lower() or "RuntimeError" in resultado.rag_error


def test_generar_reporte_markdown_incluye_ambos_lados():
    tree = get_tree(_TREE_CURB65)
    rag = _RagFalso("Respuesta del RAG de prueba.")
    resultado = comparar_caso(
        caso_id="caso-riesgo-bajo",
        tree=tree,
        pregunta="pregunta de prueba",
        respuestas_arbol={
            "confusion_nueva": False,
            "urea_elevada": False,
            "frecuencia_respiratoria_alta": False,
            "presion_arterial_baja": False,
            "edad_65_o_mas": False,
        },
        rag=rag,
    )

    reporte = generar_reporte_markdown([resultado])

    assert "caso-riesgo-bajo" in reporte
    assert "Respuesta del RAG de prueba." in reporte
    assert "leaf_score0" in reporte
    assert "Manejo ambulatorio" in reporte or "manejo ambulatorio" in reporte


def test_generar_reporte_markdown_deja_explicito_cuando_rag_no_disponible():
    tree = get_tree(_TREE_CURB65)
    resultado = comparar_caso(
        caso_id="caso-1",
        tree=tree,
        pregunta="pregunta de prueba",
        respuestas_arbol={
            "confusion_nueva": False,
            "urea_elevada": False,
            "frecuencia_respiratoria_alta": False,
            "presion_arterial_baja": False,
            "edad_65_o_mas": False,
        },
        rag=_RagFalso(falla=True),
    )

    reporte = generar_reporte_markdown([resultado])
    assert "No disponible" in reporte
