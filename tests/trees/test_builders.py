"""Pruebas de los constructores genericos de arboles trellis (trees/builders.py).

Usan instrumentos sinteticos pequenos (no CURB-65/IDSA-ATS reales) para que
las pruebas sean rapidas de leer y no dependan de los arboles de produccion
-- lo que se prueba es que el ALGORITMO de construccion (enumeracion de
nodos, corte temprano, mapeo de puntaje a hoja) es correcto, no un
instrumento clinico en particular.
"""

from __future__ import annotations

import pytest

from gpc_rag.trees.builders import (
    Criterio,
    Fuente,
    construir_arbol_mayor_o_n_menores,
    construir_arbol_suma_puntaje,
)
from gpc_rag.trees.engine import TreeSession
from gpc_rag.trees.schema import DecisionTree

_FUENTE = Fuente(guide_file="guia-de-prueba.pdf", section="Seccion de prueba", page=1)


def _criterios(n: int) -> list[Criterio]:
    return [Criterio(variable=f"c{i}", pregunta=f"criterio {i}?", source_quote=f"cita {i}") for i in range(n)]


class TestConstruirArbolSumaPuntaje:
    def test_arbol_valido_contra_el_schema(self):
        criterios = _criterios(3)
        interpretacion = {n: (f"recomendacion para puntaje {n}", f"cita interpretacion {n}") for n in range(4)}
        arbol_dict = construir_arbol_suma_puntaje(
            tree_id="prueba-suma",
            title="Prueba suma de puntaje",
            description="...",
            gpc_source="guia-de-prueba.pdf",
            criterios=criterios,
            fuente_criterios=_FUENTE,
            fuente_interpretacion=_FUENTE,
            interpretacion=interpretacion,
        )
        tree = DecisionTree.model_validate(arbol_dict)
        assert tree.tree_id == "prueba-suma"
        nodos_decision_esperados = 1 + 2 + 3  # criterio idx=1,2,3 -> 1,2,3 nodos cada uno
        hojas_esperadas = 4  # una por puntaje posible, 0..3
        assert len(tree.nodes) == nodos_decision_esperados + hojas_esperadas

    @pytest.mark.parametrize(
        ("respuestas", "puntaje_esperado"),
        [
            ((False, False, False), 0),
            ((True, False, False), 1),
            ((False, True, True), 2),
            ((True, True, True), 3),
        ],
    )
    def test_recorrido_llega_al_puntaje_correcto_sin_importar_el_orden(self, respuestas, puntaje_esperado):
        criterios = _criterios(3)
        interpretacion = {n: (f"puntaje {n}", f"cita {n}") for n in range(4)}
        arbol_dict = construir_arbol_suma_puntaje(
            tree_id="prueba-suma",
            title="t",
            description="d",
            gpc_source="guia-de-prueba.pdf",
            criterios=criterios,
            fuente_criterios=_FUENTE,
            fuente_interpretacion=_FUENTE,
            interpretacion=interpretacion,
        )
        tree = DecisionTree.model_validate(arbol_dict)
        session = TreeSession(tree)
        for respuesta in respuestas:
            session.answer(respuesta)
        hoja = session.current_node()
        assert hoja.node_id == f"leaf_score{puntaje_esperado}"

    def test_falla_claro_si_falta_un_puntaje_en_la_interpretacion(self):
        criterios = _criterios(2)
        interpretacion = {0: ("a", "cita"), 1: ("b", "cita")}  # falta el puntaje 2
        with pytest.raises(ValueError, match=r"\[2\]"):
            construir_arbol_suma_puntaje(
                tree_id="x",
                title="t",
                description="d",
                gpc_source="g.pdf",
                criterios=criterios,
                fuente_criterios=_FUENTE,
                fuente_interpretacion=_FUENTE,
                interpretacion=interpretacion,
            )


class TestConstruirArbolMayorONMenores:
    def _build(self, *, n_mayores=2, n_menores=4, umbral=2):
        return construir_arbol_mayor_o_n_menores(
            tree_id="prueba-mayor-menor",
            title="t",
            description="d",
            gpc_source="guia-de-prueba.pdf",
            criterios_mayores=_criterios(n_mayores),
            criterios_menores=[
                Criterio(variable=f"m{i}", pregunta=f"menor {i}?", source_quote=f"cita menor {i}")
                for i in range(n_menores)
            ],
            umbral_menores=umbral,
            fuente_criterios=_FUENTE,
            hoja_directa=("leaf_directa", "indicacion directa", _FUENTE, "cita directa", "Fuerte"),
            hoja_umbral_menores=("leaf_umbral", "indicacion por menores", _FUENTE, "cita umbral", None),
            hoja_sin_indicacion=("leaf_sin", "sin indicacion", _FUENTE, "cita sin indicacion", None),
        )

    def test_arbol_valido_contra_el_schema(self):
        tree = DecisionTree.model_validate(self._build())
        assert tree.tree_id == "prueba-mayor-menor"

    def test_cualquier_criterio_mayor_lleva_a_la_hoja_directa(self):
        tree = DecisionTree.model_validate(self._build())

        # primer mayor en True
        session = TreeSession(tree)
        session.answer(True)
        assert session.current_node().node_id == "leaf_directa"

        # segundo mayor en True (el primero en False)
        session = TreeSession(tree)
        session.answer(False)
        session.answer(True)
        assert session.current_node().node_id == "leaf_directa"

    def test_corte_temprano_al_alcanzar_el_umbral_de_menores(self):
        tree = DecisionTree.model_validate(self._build(n_menores=4, umbral=2))
        session = TreeSession(tree)
        session.answer(False)  # mayor 0
        session.answer(False)  # mayor 1
        session.answer(True)  # menor 0 -> cuenta 1
        session.answer(True)  # menor 1 -> cuenta 2, alcanza el umbral

        assert session.current_node().node_id == "leaf_umbral"
        # el recorrido NO debio preguntar los menores 2 y 3 (corte temprano)
        assert len(session.trail) == 5  # major0, major1, minor0, minor1, leaf

    def test_menos_del_umbral_de_menores_no_da_indicacion(self):
        tree = DecisionTree.model_validate(self._build(n_menores=3, umbral=3))
        session = TreeSession(tree)
        session.answer(False)  # mayor 0
        session.answer(False)  # mayor 1
        session.answer(True)  # menor 0
        session.answer(False)  # menor 1
        session.answer(True)  # menor 2 -> total 2, no alcanza el umbral de 3

        assert session.current_node().node_id == "leaf_sin"

    def test_falla_claro_sin_criterios_mayores(self):
        with pytest.raises(ValueError, match="al menos un criterio mayor"):
            construir_arbol_mayor_o_n_menores(
                tree_id="x",
                title="t",
                description="d",
                gpc_source="g.pdf",
                criterios_mayores=[],
                criterios_menores=_criterios(2),
                umbral_menores=1,
                fuente_criterios=_FUENTE,
                hoja_directa=("a", "r", _FUENTE, "c", None),
                hoja_umbral_menores=("b", "r", _FUENTE, "c", None),
                hoja_sin_indicacion=("c", "r", _FUENTE, "c", None),
            )

    def test_falla_claro_si_el_umbral_de_menores_es_invalido(self):
        with pytest.raises(ValueError, match="umbral_menores"):
            self._build(n_menores=3, umbral=5)
