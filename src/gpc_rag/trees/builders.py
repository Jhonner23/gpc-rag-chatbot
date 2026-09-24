"""Constructores genericos de arboles tipo "trellis", para no repetir la logica.

`scripts/gen_curb65_tree.py` y `scripts/gen_idsa_ats_tree.py` implementan
cada uno, por separado, el mismo patron de bajo nivel (nodos `step{n}_...`
que acumulan un puntaje/conteo segun la respuesta) -- funciono para dos
arboles, pero un tercero hecho a mano volveria a copiar esa logica por
tercera vez, y es exactamente el tipo de codigo donde un error de indice en
un `next_node_id` es facil de cometer y dificil de notar a simple vista.

Este modulo generaliza los dos patrones ya usados en el bosque `nac-2026/`
para que un generador nuevo (escrito a mano o propuesto por
`trees/extraction.py`) solo tenga que dar los DATOS del instrumento
(criterios, pesos, interpretacion) y no reescribir el algoritmo de
enumeracion de nodos:

- `construir_arbol_suma_puntaje`: instrumentos tipo CURB-65 -- N preguntas
  booleanas, cada "Si" suma 1 al puntaje, y el puntaje final decide la hoja
  (via una tabla de interpretacion).
- `construir_arbol_mayor_o_n_menores`: instrumentos tipo IDSA/ATS -- M
  "criterios mayores" (cualquiera, por si solo, decide la hoja "directa") y
  N "criterios menores" (con corte temprano: en cuanto se alcanza el umbral,
  el recorrido termina sin preguntar los criterios menores restantes).

Los dos generadores existentes NO se reescribieron para usar este modulo --
ya estaban verificados como reproducibles (`git diff --stat` vacio) y
tocarlos sin necesidad es el tipo de riesgo que no vale la pena correr sobre
codigo que ya funciona. Un generador nuevo si deberia usar este modulo en
vez de copiar el patron de los dos anteriores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Criterio:
    """Una pregunta booleana del instrumento, con su cita a la GPC."""

    variable: str
    pregunta: str
    source_quote: str


@dataclass(frozen=True)
class Fuente:
    """De donde sale un nodo -- mismo shape que `trees/schema.py:Source`."""

    guide_file: str
    section: str
    page: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"guide_file": self.guide_file, "section": self.section, "page": self.page}


def _nodo_decision(
    node_id: str, criterio: Criterio, fuente: Fuente, true_target: str, false_target: str
) -> dict[str, Any]:
    return {
        "type": "decision",
        "node_id": node_id,
        "question": criterio.pregunta,
        "variable": criterio.variable,
        "input_type": "boolean",
        "branches": [
            {"operator": "eq", "value": True, "next_node_id": true_target, "label": "Si"},
            {"operator": "eq", "value": False, "next_node_id": false_target, "label": "No"},
        ],
        "source": fuente.as_dict(),
        "source_quote": criterio.source_quote,
    }


def _nodo_hoja(
    node_id: str, recomendacion: str, fuente: Fuente, source_quote: str, evidence_level: str | None = None
) -> dict[str, Any]:
    return {
        "type": "leaf",
        "node_id": node_id,
        "recommendation": recomendacion,
        "evidence_level": evidence_level,
        "source": fuente.as_dict(),
        "source_quote": source_quote,
    }


def construir_arbol_suma_puntaje(  # noqa: PLR0913 -- todos keyword-only y con nombre propio; dividirlos en un dataclass de config seria mas indireccion, no menos, para un builder de uso interno
    *,
    tree_id: str,
    title: str,
    description: str,
    gpc_source: str,
    criterios: list[Criterio],
    fuente_criterios: Fuente,
    fuente_interpretacion: Fuente,
    interpretacion: dict[int, tuple[str, str]],
) -> dict[str, Any]:
    """Arbol tipo CURB-65: N preguntas booleanas, el puntaje final decide la hoja.

    `interpretacion` mapea cada puntaje posible (0..len(criterios)) a
    (recomendacion, source_quote_de_la_hoja) -- debe tener una entrada para
    cada puntaje, si falta alguna se lanza KeyError explicito en vez de
    generar un arbol con una hoja faltante.
    """
    n_total = len(criterios)
    faltantes = set(range(n_total + 1)) - set(interpretacion)
    if faltantes:
        raise ValueError(
            f"'interpretacion' no cubre los puntajes {sorted(faltantes)} "
            f"(el arbol tiene {n_total} criterios, puntajes posibles 0..{n_total})"
        )

    def step_id(n: int, score: int) -> str:
        return f"step{n}_score{score}"

    nodes: dict[str, Any] = {}
    for idx, criterio in enumerate(criterios, start=1):
        is_last = idx == n_total
        max_score_antes = idx - 1
        for score in range(0, max_score_antes + 1):
            node_id = step_id(idx, score)
            false_target = f"leaf_score{score}" if is_last else step_id(idx + 1, score)
            true_target = f"leaf_score{score + 1}" if is_last else step_id(idx + 1, score + 1)
            nodes[node_id] = _nodo_decision(node_id, criterio, fuente_criterios, true_target, false_target)

    for score in range(0, n_total + 1):
        leaf_id = f"leaf_score{score}"
        recomendacion, quote = interpretacion[score]
        nodes[leaf_id] = _nodo_hoja(leaf_id, recomendacion, fuente_interpretacion, quote)

    return {
        "tree_id": tree_id,
        "title": title,
        "description": description,
        "gpc_source": gpc_source,
        "root_node_id": step_id(1, 0),
        "nodes": nodes,
    }


def construir_arbol_mayor_o_n_menores(  # noqa: PLR0913 -- ver nota en construir_arbol_suma_puntaje
    *,
    tree_id: str,
    title: str,
    description: str,
    gpc_source: str,
    criterios_mayores: list[Criterio],
    criterios_menores: list[Criterio],
    umbral_menores: int,
    fuente_criterios: Fuente,
    hoja_directa: tuple[str, str, Fuente, str, str | None],
    hoja_umbral_menores: tuple[str, str, Fuente, str, str | None],
    hoja_sin_indicacion: tuple[str, str, Fuente, str, str | None],
) -> dict[str, Any]:
    """Arbol tipo IDSA/ATS: M criterios mayores (cualquiera -> hoja directa) +
    N criterios menores con corte temprano al alcanzar `umbral_menores`.

    Cada `hoja_*` es (node_id, recomendacion, fuente, source_quote,
    evidence_level) -- se arma asi en vez de pedir el dict completo porque
    son solo 3 hojas fijas (directa / por umbral de menores / sin
    indicacion), a diferencia del numero variable de hojas de
    `construir_arbol_suma_puntaje`.
    """
    if not criterios_mayores:
        raise ValueError("un instrumento 'mayor o N menores' necesita al menos un criterio mayor")
    if umbral_menores < 1 or umbral_menores > len(criterios_menores):
        raise ValueError(
            f"umbral_menores={umbral_menores} debe estar entre 1 y {len(criterios_menores)} "
            "(la cantidad de criterios menores)"
        )

    id_directa, rec_directa, fuente_directa, quote_directa, nivel_directa = hoja_directa
    id_umbral, rec_umbral, fuente_umbral, quote_umbral, nivel_umbral = hoja_umbral_menores
    id_sin, rec_sin, fuente_sin, quote_sin, nivel_sin = hoja_sin_indicacion

    def step_id(n: int, count: int) -> str:
        return f"minor_step{n}_count{count}"

    nodes: dict[str, Any] = {}

    # --- Mayores: cualquiera, por si solo, lleva a la hoja directa. ---
    n_mayores = len(criterios_mayores)
    primer_menor = step_id(1, 0)
    for idx, criterio in enumerate(criterios_mayores):
        node_id = f"major_{criterio.variable}"
        es_ultimo_mayor = idx == n_mayores - 1
        false_target = primer_menor if es_ultimo_mayor else f"major_{criterios_mayores[idx + 1].variable}"
        nodes[node_id] = _nodo_decision(node_id, criterio, fuente_criterios, id_directa, false_target)

    # --- Menores: trellis con corte temprano en cuanto se llega al umbral. ---
    n_menores = len(criterios_menores)
    for idx, criterio in enumerate(criterios_menores, start=1):
        is_last = idx == n_menores
        for count in range(0, min(idx - 1, umbral_menores - 1) + 1):
            node_id = step_id(idx, count)
            false_target = id_sin if is_last else step_id(idx + 1, count)

            new_count = count + 1
            if new_count >= umbral_menores:
                true_target = id_umbral
            elif is_last:
                true_target = id_sin
            else:
                true_target = step_id(idx + 1, new_count)

            nodes[node_id] = _nodo_decision(node_id, criterio, fuente_criterios, true_target, false_target)

    nodes[id_directa] = _nodo_hoja(id_directa, rec_directa, fuente_directa, quote_directa, nivel_directa)
    nodes[id_umbral] = _nodo_hoja(id_umbral, rec_umbral, fuente_umbral, quote_umbral, nivel_umbral)
    nodes[id_sin] = _nodo_hoja(id_sin, rec_sin, fuente_sin, quote_sin, nivel_sin)

    return {
        "tree_id": tree_id,
        "title": title,
        "description": description,
        "gpc_source": gpc_source,
        "root_node_id": f"major_{criterios_mayores[0].variable}",
        "nodes": nodes,
    }
