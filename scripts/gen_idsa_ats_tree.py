"""Genera el arbol de decision de criterios IDSA/ATS 2007 para ingreso a UCI (Tabla 5, pag. 199).

Segundo arbol del bosque nac-2026 (el primero fue CURB-65). Se genera por
script (no a mano) para evitar errores de copia en los ~30 nodos del trellis,
pero el resultado final es JSON estatico versionado como los demas arboles --
el generador no se ejecuta en produccion.
"""

from __future__ import annotations

import json
from pathlib import Path

GUIDE_FILE = "Guia de practica clinica Colombiana NAC 2026.pdf"
SECTION_TABLA5 = (
    "Punto de buena practica (tras la recomendacion 10): en pacientes hospitalizados sin "
    "indicacion inmediata de traslado a UCI por vasopresores o ventilacion mecanica, pueden "
    "usarse los criterios de severidad IDSA/ATS 2007 (Tabla 5) junto con el juicio clinico "
    "para decidir el traslado a UCI (suma de criterios menores)."
)

MAJORS = [
    (
        "major_vm_invasiva",
        "El paciente requiere ventilacion mecanica invasiva?",
        "Necesidad de ventilacion mecanica invasiva (Tabla 5, Criterios mayores, pag. 199).",
    ),
    (
        "major_choque_septico",
        "El paciente esta en choque septico con requerimiento de vasopresores?",
        "Choque septico con requerimiento de vasopresores (Tabla 5, Criterios mayores, pag. 199).",
    ),
]

UMBRAL_CRITERIOS_MENORES = 3  # Tabla 5: >=3 criterios menores sugieren ingreso a UCI

MINORS = [
    (
        "minor_fr30",
        "Frecuencia respiratoria >= 30 respiraciones/min?",
        "Frecuencia respiratoria >=30/min (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_pafi250",
        "Relacion PaO2/FiO2 <= 250?",
        "PaO2/FiO2 <=250 (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_infiltrados_multilobares",
        "La radiografia de torax muestra infiltrados multilobares?",
        "Infiltrados multilobares (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_confusion",
        "El paciente presenta confusion o desorientacion?",
        "Confusion o desorientacion (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_bun20",
        "BUN >= 20 mg/dL?",
        "BUN >=20 mg/dL (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_leucopenia",
        "El paciente tiene leucopenia (<4.000 celulas/mm3)?",
        "Leucopenia (<4.000 celulas/mm3) (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_trombocitopenia",
        "El paciente tiene trombocitopenia (<100.000 plaquetas/mm3)?",
        "Trombocitopenia (<100.000 plaquetas/mm3) (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_hipotermia",
        "El paciente tiene hipotermia (temperatura <36 C)?",
        "Hipotermia (<36 C) (Tabla 5, Criterios menores, pag. 199).",
    ),
    (
        "minor_hipotension_liquidos",
        "El paciente presenta hipotension que requiere reanimacion agresiva con liquidos intravenosos?",
        "Hipotension que requiere liquidos intravenosos agresivos (Tabla 5, Criterios menores, pag. 199).",
    ),
]

LEAF_UCI_DIRECTA = "leaf_uci_directa"
LEAF_UCI_CRITERIOS_MENORES = "leaf_uci_criterios_menores"
LEAF_NO_UCI = "leaf_no_uci_por_criterios"


def source(section: str = SECTION_TABLA5, page: int = 199) -> dict:
    return {"guide_file": GUIDE_FILE, "section": section, "page": page}


def step_id(n: int, count: int) -> str:
    return f"step{n}_count{count}"


def build_nodes() -> dict:
    nodes: dict[str, dict] = {}

    # --- Criterios mayores: cualquiera de los dos, por si solo, indica UCI directa ---
    nodes["major_vm_invasiva"] = {
        "type": "decision",
        "node_id": "major_vm_invasiva",
        "question": MAJORS[0][1],
        "variable": MAJORS[0][0],
        "input_type": "boolean",
        "branches": [
            {"operator": "eq", "value": True, "next_node_id": LEAF_UCI_DIRECTA, "label": "Si"},
            {"operator": "eq", "value": False, "next_node_id": "major_choque_septico", "label": "No"},
        ],
        "source": source(),
        "source_quote": MAJORS[0][2],
    }
    nodes["major_choque_septico"] = {
        "type": "decision",
        "node_id": "major_choque_septico",
        "question": MAJORS[1][1],
        "variable": MAJORS[1][0],
        "input_type": "boolean",
        "branches": [
            {"operator": "eq", "value": True, "next_node_id": LEAF_UCI_DIRECTA, "label": "Si"},
            {"operator": "eq", "value": False, "next_node_id": step_id(1, 0), "label": "No"},
        ],
        "source": source(),
        "source_quote": MAJORS[1][2],
    }

    # --- Criterios menores: trellis step{n}_count{c}, n=1..9, c=0,1,2 ---
    n_total = len(MINORS)
    for idx, (var, question, quote) in enumerate(MINORS, start=1):
        for count in (0, 1, 2):
            node_id = step_id(idx, count)
            is_last = idx == n_total

            # rama "No" (criterio ausente): el conteo no cambia
            false_target = LEAF_NO_UCI if is_last else step_id(idx + 1, count)

            # rama "Si" (criterio presente): el conteo sube en 1
            new_count = count + 1
            if new_count >= UMBRAL_CRITERIOS_MENORES:
                true_target = LEAF_UCI_CRITERIOS_MENORES
            elif is_last:
                true_target = LEAF_NO_UCI
            else:
                true_target = step_id(idx + 1, new_count)

            nodes[node_id] = {
                "type": "decision",
                "node_id": node_id,
                "question": question,
                "variable": var,
                "input_type": "boolean",
                "branches": [
                    {"operator": "eq", "value": True, "next_node_id": true_target, "label": "Si"},
                    {"operator": "eq", "value": False, "next_node_id": false_target, "label": "No"},
                ],
                "source": source(),
                "source_quote": quote,
            }

    # --- Hojas ---
    nodes[LEAF_UCI_DIRECTA] = {
        "type": "leaf",
        "node_id": LEAF_UCI_DIRECTA,
        "recommendation": (
            "Cumple >=1 criterio mayor IDSA/ATS 2007 (ventilacion mecanica invasiva o choque "
            "septico con requerimiento de vasopresores): indicacion de ingreso directo a UCI."
        ),
        "evidence_level": "Fuerte, Moderada calidad",
        "source": source(
            "Recomendacion 10.2: Recomendamos admision directa a UCI en pacientes con hipotension "
            "que requiere vasopresores o insuficiencia respiratoria que requiere ventilacion mecanica.",
            page=198,
        ),
        "source_quote": (
            "Recomendacion 10.2 Recomendamos admision directa a UCI en pacientes con hipotension "
            "que requiere vasopresores o insuficiencia respiratoria que requiere ventilacion "
            "mecanica. (Fuerte, Moderada calidad) (pag. 198)."
        ),
    }
    nodes[LEAF_UCI_CRITERIOS_MENORES] = {
        "type": "leaf",
        "node_id": LEAF_UCI_CRITERIOS_MENORES,
        "recommendation": (
            "Cumple >=3 criterios menores IDSA/ATS 2007 (sin criterios mayores): sugiere ingreso "
            "a UCI. Correlacionar con juicio clinico -- punto de buena practica (tras la "
            "recomendacion 10): usar los criterios IDSA/ATS 2007 (Tabla 5) junto con el juicio "
            "clinico para decidir el traslado a UCI (pag. 198)."
        ),
        "evidence_level": "Punto de buena practica",
        "source": source(),
        "source_quote": "Criterios menores (>=3 sugieren ingreso a UCI) (Tabla 5, pag. 199).",
    }
    nodes[LEAF_NO_UCI] = {
        "type": "leaf",
        "node_id": LEAF_NO_UCI,
        "recommendation": (
            "No cumple criterios mayores ni >=3 criterios menores IDSA/ATS 2007: sin indicacion "
            "de UCI por estos criterios. Continuar manejo hospitalario general con reevaluacion "
            "clinica periodica."
        ),
        "evidence_level": "Punto de buena practica",
        "source": source(),
        "source_quote": (
            "Criterios menores (<3): no se alcanza el umbral que sugiere ingreso a UCI "
            "(Tabla 5, pag. 199)."
        ),
    }

    return nodes


def build_tree() -> dict:
    return {
        "tree_id": "nac-2026-criterios-idsa-ats-uci",
        "title": "Criterios IDSA/ATS 2007 para ingreso a UCI",
        "description": (
            "Determina si un paciente hospitalizado con NAC tiene indicacion de ingreso a UCI "
            "segun los criterios IDSA/ATS 2007 (Tabla 5): 1 criterio mayor, o >=3 de 9 criterios "
            "menores."
        ),
        "gpc_source": GUIDE_FILE,
        "root_node_id": "major_vm_invasiva",
        "nodes": build_nodes(),
    }


if __name__ == "__main__":
    out_path = Path(__file__).parents[1] / "src" / "gpc_rag" / "trees" / "data" / "nac-2026" / "criterios-idsa-ats-uci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_tree(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"escrito: {out_path}")
