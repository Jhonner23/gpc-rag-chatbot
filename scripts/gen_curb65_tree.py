"""Genera el arbol de decision CURB-65/CRB-65 (Tabla 3, pag. 199).

Primer arbol del bosque nac-2026. Se reescribe como generador (igual que
gen_idsa_ats_tree.py) para que TODO el bosque sea reproducible desde el PDF
fuente con un script versionado, no solo el segundo arbol -- requisito de
replicabilidad de la tesis: cualquiera debe poder reconstruir exactamente
los mismos arboles corriendo estos scripts, sin depender de una sesion de
chat que ya no existe.

Verificado (ver README del bosque) que el JSON que produce este script es
identico al que ya esta versionado en trees/data/nac-2026/ -- este generador
documenta como se llego a ese archivo, no lo cambia.
"""

from __future__ import annotations

import json
from pathlib import Path

GUIDE_FILE = "Guia de practica clinica Colombiana NAC 2026.pdf"
SECTION_TABLA3 = (
    "10. En pacientes adultos con diagnostico de NAC, debe utilizarse una regla de prediccion "
    "pronostica junto con el juicio clinico, en lugar del juicio clinico por si solo, para "
    "determinar el sitio de atencion y orientar la necesidad de soporte? (Tabla 3: CURB-65 y CRB-65)"
)
SECTION_INTERPRETACION = (
    "10. En pacientes adultos con diagnostico de NAC, debe utilizarse una regla de prediccion "
    "pronostica junto con el juicio clinico, en lugar del juicio clinico por si solo, para "
    "determinar el sitio de atencion y orientar la necesidad de soporte? (Tabla Interpretacion "
    "CURB-65, pag. 199)"
)

# Orden exacto en el que la Tabla 3 de la guia lista los 5 criterios.
CRITERIOS = [
    (
        "confusion_nueva",
        "El paciente presenta confusion de aparicion nueva?",
        "Confusion | CURB-65: 1 | CRB-65: 1 (Tabla 3, pag. 199).",
    ),
    (
        "urea_elevada",
        "Nitrogeno ureico/urea elevado (BUN > 19 mg/dL o urea > 7 mmol/L)?",
        "BUN >7 mmol/L (>19 mg/dL) | CURB-65: 1 | CRB-65: - (Tabla 3, pag. 199). "
        "Nota: CRB-65 no incluye este criterio (no requiere laboratorio).",
    ),
    (
        "frecuencia_respiratoria_alta",
        "Frecuencia respiratoria >= 30 respiraciones por minuto?",
        "Frecuencia respiratoria >=30/min | CURB-65: 1 | CRB-65: 1 (Tabla 3, pag. 199).",
    ),
    (
        "presion_arterial_baja",
        "Presion arterial baja (sistolica < 90 mmHg o diastolica <= 60 mmHg)?",
        "Presion arterial sistolica <90 o diastolica <=60 mmHg | CURB-65: 1 | CRB-65: 1 "
        "(Tabla 3, pag. 199).",
    ),
    (
        "edad_65_o_mas",
        "Edad >= 65 anos?",
        "Edad >=65 anos | CURB-65: 1 | CRB-65: 1 (Tabla 3, pag. 199).",
    ),
]

# Clasificacion del riesgo (Tabla de interpretacion CURB-65, misma pag. 199).
INTERPRETACION = {
    0: ("Bajo", "manejo sugerido Ambulatorio", "Puntaje total 0-1: riesgo de mortalidad Bajo (0.6-2.7%), manejo sugerido Ambulatorio (Tabla de interpretacion, pag. 199)."),
    1: ("Bajo", "manejo sugerido Ambulatorio", "Puntaje total 0-1: riesgo de mortalidad Bajo (0.6-2.7%), manejo sugerido Ambulatorio (Tabla de interpretacion, pag. 199)."),
    2: ("Moderado", "manejo sugerido Hospitalizacion", "Puntaje total 2: riesgo de mortalidad Moderado (~6.8%), manejo sugerido Hospitalizacion (Tabla de interpretacion, pag. 199)."),
    3: ("Alto", "manejo sugerido Considerar hospitalizacion/UCI", "Puntaje total >=3: riesgo de mortalidad Alto (>14.5%), manejo sugerido Considerar hospitalizacion/UCI (Tabla de interpretacion, pag. 199)."),
    4: ("Alto", "manejo sugerido Considerar hospitalizacion/UCI", "Puntaje total >=3: riesgo de mortalidad Alto (>14.5%), manejo sugerido Considerar hospitalizacion/UCI (Tabla de interpretacion, pag. 199)."),
    5: ("Alto", "manejo sugerido Considerar hospitalizacion/UCI", "Puntaje total >=3: riesgo de mortalidad Alto (>14.5%), manejo sugerido Considerar hospitalizacion/UCI (Tabla de interpretacion, pag. 199)."),
}

RIESGO_A_TEXTO = {
    0: "CURB-65 = 0. Riesgo bajo: manejo ambulatorio.",
    1: "CURB-65 = 1. Riesgo bajo: manejo ambulatorio.",
    2: "CURB-65 = 2. Riesgo intermedio: considerar hospitalizacion.",
    3: "CURB-65 = 3. Riesgo alto: hospitalizacion, considerar manejo en UCI segun criterio clinico adicional.",
    4: "CURB-65 = 4. Riesgo alto: hospitalizacion, considerar manejo en UCI segun criterio clinico adicional.",
    5: "CURB-65 = 5. Riesgo alto: hospitalizacion, considerar manejo en UCI segun criterio clinico adicional.",
}


def source(section: str, page: int = 199) -> dict:
    return {"guide_file": GUIDE_FILE, "section": section, "page": page}


def step_id(n: int, score: int) -> str:
    return f"step{n}_score{score}"


def build_nodes() -> dict:
    nodes: dict[str, dict] = {}
    n_total = len(CRITERIOS)  # 5

    for idx, (var, question, quote) in enumerate(CRITERIOS, start=1):
        max_score_antes = idx - 1  # cuantos criterios ya se preguntaron
        for score in range(0, max_score_antes + 1):
            node_id = step_id(idx, score)
            is_last = idx == n_total

            false_target = f"leaf_score{score}" if is_last else step_id(idx + 1, score)
            true_target = f"leaf_score{score + 1}" if is_last else step_id(idx + 1, score + 1)

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
                "source": source(SECTION_TABLA3),
                "source_quote": quote,
            }

    for score in range(0, n_total + 1):
        leaf_id = f"leaf_score{score}"
        _, _, quote = INTERPRETACION[score]
        nodes[leaf_id] = {
            "type": "leaf",
            "node_id": leaf_id,
            "recommendation": RIESGO_A_TEXTO[score],
            "evidence_level": None,
            "source": source(SECTION_INTERPRETACION),
            "source_quote": quote,
        }

    return nodes


def build_tree() -> dict:
    return {
        "tree_id": "nac-2026-severidad-hospitalizacion",
        "title": "Severidad y decision de hospitalizacion (CURB-65)",
        "description": (
            "Estima la severidad de la NAC y sugiere el nivel de manejo (ambulatorio, "
            "hospitalizacion, considerar UCI) segun el puntaje CURB-65."
        ),
        "gpc_source": GUIDE_FILE,
        "root_node_id": "step1_score0",
        "nodes": build_nodes(),
    }


if __name__ == "__main__":
    out_path = (
        Path(__file__).parents[1]
        / "src"
        / "gpc_rag"
        / "trees"
        / "data"
        / "nac-2026"
        / "severidad-hospitalizacion.json"
    )
    out_path.write_text(json.dumps(build_tree(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"escrito: {out_path}")
