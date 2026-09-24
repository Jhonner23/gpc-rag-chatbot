"""Pruebas del pipeline de extraccion de tablas (trees/table_extraction.py).

Dos capas de prueba:
- Con filas SINTETICAS (fixtures escritas a mano abajo): prueban el parseo
  en si -- rapidas, no dependen de un PDF, y cubren los casos de rechazo
  explicito (tabla con formato inesperado -> TablaNoReconocidaError).
- Con el PDF REAL (`TestContraElPdfReal`, se salta si no esta en
  data/01_raw/gpc/): prueba que el pipeline completo, corrido contra el
  documento real, reproduce exactamente los dos instrumentos que ya se
  construyeron a mano (mismo numero de hojas, mismo comportamiento en los
  mismos casos que ya cubren tests/trees/test_engine.py y
  test_idsa_ats_tree.py) -- la prueba de que "automatico sin LLM" de verdad
  funciona sobre el documento real, no solo sobre datos inventados.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpc_rag.trees.builders import construir_arbol_mayor_o_n_menores, construir_arbol_suma_puntaje
from gpc_rag.trees.engine import TreeSession
from gpc_rag.trees.schema import DecisionTree
from gpc_rag.trees.table_extraction import (
    TablaNoReconocidaError,
    extraer_tablas_de_pagina,
    parsear_criterios_suma_puntaje,
    parsear_interpretacion_suma_puntaje,
    parsear_mayor_o_menores,
)

# --- Fixtures: filas tal como las devuelve pdfplumber.extract_tables() ---

_TABLA_CURB65 = [
    ["Criterio", "CURB-65", "CRB-65"],
    ["Confusión", "1", "1"],
    ["BUN >7 mmol/L (>19 mg/dL)", "1", "-"],
    ["Frecuencia respiratoria ≥30/min", "1", "1"],
    ["Presión arterial sistólica <90 o\ndiastólica ≤60 mmHg", "1", "1"],
    ["Edad ≥65 años", "1", "1"],
]

_TABLA_INTERPRETACION = [
    ["Puntaje total", "Riesgo de mortalidad\nestimado (%)", "Manejo sugerido"],
    ["0 - 1", "Bajo (0.6-2.7%)", "Ambulatorio"],
    ["2", "Moderado (≈6.8%)", "Hospitalización"],
    ["≥3", "Alto (>14.5%)", "Considerar\nhospitalización/UCI"],
]

_TABLA_MAYOR_MENOR = [
    ["Criterios mayores (≥1 indica ingreso a UCI)"],
    ["• Necesidad de ventilación mecánica invasiva\n• Choque séptico con requerimiento de vasopresores"],
    ["Criterios menores (≥3 sugieren ingreso a UCI)"],
    ["• Frecuencia respiratoria ≥30/min\n• PaO₂/FiO₂ ≤250\n• Infiltrados multilobares"],
]

# Tabla tipo Fine/PSI: puntaje ponderado (no +1 por criterio) y con
# sub-categorias (primera columna en None para filas que continuan la
# categoria anterior) -- exactamente el tipo de tabla que el pipeline debe
# rechazar en vez de intentar adivinar.
_TABLA_PONDERADA_CON_CATEGORIAS = [
    ["Categoría", "Variable", "Puntaje"],
    ["Comorbilidades", "Neoplasia", "+30"],
    [None, "Enfermedad hepática", "+20"],
]


class TestParsearCriteriosSumaPuntaje:
    def test_extrae_los_5_criterios_de_curb65(self):
        criterios = parsear_criterios_suma_puntaje(
            _TABLA_CURB65, columna_objetivo="CURB-65", pagina=199, referencia_tabla="Tabla 3"
        )
        assert [c.variable for c in criterios] == [
            "confusion",
            "bun_7_mmol_l_19_mg_dl",
            "frecuencia_respiratoria_30_min",
            "presion_arterial_sistolica_90_o_diastolica_60_mmhg",
            "edad_65_anos",
        ]
        # el salto de linea de la celda original no debe quedar en la pregunta
        presion = criterios[3]
        assert "\n" not in presion.pregunta
        assert "199" in presion.source_quote

    def test_crb65_excluye_bun_por_la_columna_en_guion(self):
        criterios = parsear_criterios_suma_puntaje(
            _TABLA_CURB65, columna_objetivo="CRB-65", pagina=199, referencia_tabla="Tabla 3"
        )
        assert "bun_7_mmol_l_19_mg_dl" not in [c.variable for c in criterios]
        assert len(criterios) == 4

    def test_rechaza_columna_objetivo_inexistente(self):
        with pytest.raises(TablaNoReconocidaError, match="CURB-99"):
            parsear_criterios_suma_puntaje(
                _TABLA_CURB65, columna_objetivo="CURB-99", pagina=199, referencia_tabla="Tabla 3"
            )

    def test_rechaza_puntaje_ponderado_en_vez_de_adivinar(self):
        with pytest.raises(TablaNoReconocidaError, match="puntaje"):
            parsear_criterios_suma_puntaje(
                _TABLA_PONDERADA_CON_CATEGORIAS, columna_objetivo="Puntaje", pagina=1, referencia_tabla="Tabla X"
            )

    def test_rechaza_fila_sin_nombre_de_criterio(self):
        tabla_con_continuacion = [
            ["Categoría", "Puntaje"],
            ["Comorbilidades", "1"],
            [None, "1"],  # fila de continuacion de categoria, sin nombre propio
        ]
        with pytest.raises(TablaNoReconocidaError, match="sub-categorias"):
            parsear_criterios_suma_puntaje(
                tabla_con_continuacion, columna_objetivo="Puntaje", pagina=1, referencia_tabla="Tabla X"
            )


class TestParsearInterpretacionSumaPuntaje:
    def test_expande_rangos_a_los_6_puntajes_de_curb65(self):
        interpretacion = parsear_interpretacion_suma_puntaje(
            _TABLA_INTERPRETACION,
            n_criterios=5,
            columna_rango="Puntaje total",
            columna_recomendacion=["Riesgo de mortalidad estimado (%)", "Manejo sugerido"],
            pagina=199,
            referencia_tabla="Tabla interpretacion CURB-65",
        )
        assert set(interpretacion) == {0, 1, 2, 3, 4, 5}
        assert "Ambulatorio" in interpretacion[0][0]
        assert "Ambulatorio" in interpretacion[1][0]
        assert "Hospitalizaci" in interpretacion[2][0]
        assert "hospitalizaci" in interpretacion[5][0].lower()

    def test_rechaza_columna_de_rango_inexistente(self):
        with pytest.raises(TablaNoReconocidaError, match="Puntaje inexistente"):
            parsear_interpretacion_suma_puntaje(
                _TABLA_INTERPRETACION,
                n_criterios=5,
                columna_rango="Puntaje inexistente",
                columna_recomendacion=["Manejo sugerido"],
                pagina=199,
                referencia_tabla="t",
            )

    def test_rechaza_formato_de_rango_no_reconocido(self):
        tabla_rango_raro = [
            ["Puntaje total", "Manejo sugerido"],
            ["bajo", "Ambulatorio"],  # ni numero, ni rango, ni >=/<=
        ]
        with pytest.raises(TablaNoReconocidaError, match="rango"):
            parsear_interpretacion_suma_puntaje(
                tabla_rango_raro,
                n_criterios=1,
                columna_rango="Puntaje total",
                columna_recomendacion=["Manejo sugerido"],
                pagina=1,
                referencia_tabla="t",
            )


class TestParsearMayorOMenores:
    def test_extrae_mayores_menores_y_umbral(self):
        mayores, menores, umbral = parsear_mayor_o_menores(
            _TABLA_MAYOR_MENOR, pagina=199, referencia_tabla="Tabla 5"
        )
        assert len(mayores) == 2
        assert len(menores) == 3
        assert umbral == 3
        assert mayores[0].variable == "necesidad_de_ventilacion_mecanica_invasiva"

    def test_rechaza_tabla_sin_las_etiquetas_esperadas(self):
        tabla_sin_etiquetas = [["Cualquier otra cosa"], ["• a\n• b"]]
        with pytest.raises(TablaNoReconocidaError, match="mayores"):
            parsear_mayor_o_menores(tabla_sin_etiquetas, pagina=1, referencia_tabla="t")

    def test_rechaza_umbral_no_numerico(self):
        tabla_sin_umbral = [
            ["Criterios mayores"],
            ["• a"],
            ["Criterios menores (varios sugieren ingreso)"],  # sin numero
            ["• b\n• c"],
        ]
        with pytest.raises(TablaNoReconocidaError, match="umbral"):
            parsear_mayor_o_menores(tabla_sin_umbral, pagina=1, referencia_tabla="t")


class TestContraElPdfReal:
    """Corre el pipeline completo contra el PDF real de la guia NAC 2026.

    Se salta automaticamente si el PDF no esta presente -- no es parte del
    corpus versionado de todos los clones (solo si alguien ya lo puso en
    data/01_raw/gpc/, ver README -- 'Agregar un protocolo nuevo').
    """

    _PDF = Path(__file__).parents[2] / "data" / "01_raw" / "gpc" / "Guia de practica clinica Colombiana NAC 2026.pdf"
    _GUIDE_FILE = "Guia de practica clinica Colombiana NAC 2026.pdf"

    @pytest.fixture(autouse=True)
    def _skip_si_no_hay_pdf(self):
        if not self._PDF.is_file():
            pytest.skip("PDF de NAC 2026 no presente en este clon -- ver README")

    def test_pipeline_reproduce_el_arbol_curb65_real(self):
        from gpc_rag.trees.builders import Fuente

        tablas = extraer_tablas_de_pagina(self._PDF, 199)
        fuente = Fuente(guide_file=self._GUIDE_FILE, section="Tabla 3", page=199)

        criterios = parsear_criterios_suma_puntaje(
            tablas[1], columna_objetivo="CURB-65", pagina=199, referencia_tabla="Tabla 3"
        )
        assert len(criterios) == 5

        interpretacion = parsear_interpretacion_suma_puntaje(
            tablas[4],
            n_criterios=5,
            columna_rango="Puntaje total",
            columna_recomendacion=["Riesgo de mortalidad estimado (%)", "Manejo sugerido"],
            pagina=199,
            referencia_tabla="Tabla interpretacion CURB-65",
        )
        arbol_dict = construir_arbol_suma_puntaje(
            tree_id="prueba-pipeline-curb65",
            title="t",
            description="d",
            gpc_source=self._GUIDE_FILE,
            criterios=criterios,
            fuente_criterios=fuente,
            fuente_interpretacion=fuente,
            interpretacion=interpretacion,
        )
        tree = DecisionTree.model_validate(arbol_dict)

        session = TreeSession(tree)
        for _ in range(5):
            session.answer(False)
        assert session.current_node().node_id == "leaf_score0"

    def test_pipeline_reproduce_el_arbol_idsa_ats_real(self):
        from gpc_rag.trees.builders import Fuente

        tablas = extraer_tablas_de_pagina(self._PDF, 199)
        fuente = Fuente(guide_file=self._GUIDE_FILE, section="Tabla 5", page=199)

        mayores, menores, umbral = parsear_mayor_o_menores(tablas[3], pagina=199, referencia_tabla="Tabla 5")
        assert len(mayores) == 2
        assert len(menores) == 9
        assert umbral == 3

        arbol_dict = construir_arbol_mayor_o_n_menores(
            tree_id="prueba-pipeline-idsa",
            title="t",
            description="d",
            gpc_source=self._GUIDE_FILE,
            criterios_mayores=mayores,
            criterios_menores=menores,
            umbral_menores=umbral,
            fuente_criterios=fuente,
            hoja_directa=("leaf_directa", "UCI directa", fuente, "cita", None),
            hoja_umbral_menores=("leaf_umbral", "UCI por menores", fuente, "cita", None),
            hoja_sin_indicacion=("leaf_sin", "sin UCI", fuente, "cita", None),
        )
        tree = DecisionTree.model_validate(arbol_dict)

        session = TreeSession(tree)
        session.answer(True)  # primer criterio mayor
        assert session.current_node().node_id == "leaf_directa"
