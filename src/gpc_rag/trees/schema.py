"""Esquema (Pydantic) del arbol de decision por protocolo clinico.

Una GPC produce un BOSQUE de arboles -- uno por cada punto de decision
clinica identificable (ej. severidad/hospitalizacion, eleccion de
antibiotico, duracion del tratamiento), no un unico arbol por documento.
Cada arbol se navega en tiempo de consulta sin invocar ningun LLM: es una
maquina de estados determinista definida enteramente por este JSON.

Los arboles se generan offline -- a mano (ver scripts/gen_*_tree.py) o de
forma automatica y determinista a partir de tablas de la GPC (ver
trees/table_extraction.py + trees/builders.py + scripts/extract_tree_from_table.py,
sin LLM: deteccion geometrica de tablas + parseo por reglas fijas) -- y se
versionan en git bajo
`trees/data/<gpc_slug>/<arbol_slug>.json` -- son artefactos curados y
pequenos (solo fragmentos citados de la guia, no el documento completo),
igual que `eval/golden_dataset.json`.

Cada nodo hoja y de decision guarda `source_quote`: la frase textual exacta
de la GPC de la que se derivo. No es documentacion opcional -- es lo que
permite verificar automaticamente que la extraccion es fiel al documento
(ver trees/fidelity.py), dado que el contenido clinico de la GPC en si ya
esta validado por la clinica que la emite; lo que puede fallar es la
extraccion automatica, no la guia.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

Operator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "between"]
InputType = Literal["boolean", "numeric", "enum"]


class Source(BaseModel):
    """Cita exacta a la GPC de origen (mismo formato que RetrievedChunk)."""

    guide_file: str
    section: str
    page: int | None = None


class EnumOption(BaseModel):
    """Una opcion valida para un nodo de decision de tipo `enum`."""

    value: str
    label: str


class Branch(BaseModel):
    """Una condicion sobre la variable del nodo padre -> a que nodo saltar.

    `value` se interpreta segun `operator`:
    - eq/ne/gt/gte/lt/lte: valor unico a comparar (bool, int, float o str).
    - in: lista de valores validos.
    - between: lista [minimo, maximo] (inclusive).
    """

    operator: Operator
    value: bool | int | float | str | list[float | int | str]
    next_node_id: str
    label: str | None = None  # texto para mostrar como boton en el wizard


class DecisionNode(BaseModel):
    """Nodo de decision: le pregunta algo al usuario y navega segun la respuesta.

    `variable` es el id normalizado de la pregunta (ej. "curb65_confusion"),
    usado solo para trazabilidad -- el motor de ejecucion no necesita
    resolver expresiones, solo evaluar los `branches` en orden y tomar el
    primero cuya condicion matchee la respuesta del usuario.
    """

    type: Literal["decision"] = "decision"
    node_id: str
    question: str
    variable: str
    input_type: InputType
    options: list[EnumOption] | None = None  # requerido si input_type == "enum"
    branches: list[Branch]
    source: Source | None = None  # de donde sale el criterio, si aplica
    source_quote: str | None = None

    @model_validator(mode="after")
    def _validar_enum_options(self) -> DecisionNode:
        if self.input_type == "enum" and not self.options:
            raise ValueError(f"nodo {self.node_id}: input_type=enum requiere 'options'")
        if not self.branches:
            raise ValueError(f"nodo {self.node_id}: un nodo de decision necesita al menos un branch")
        return self


class LeafNode(BaseModel):
    """Nodo hoja: el resultado final del recorrido -- una recomendacion citada."""

    type: Literal["leaf"] = "leaf"
    node_id: str
    recommendation: str
    evidence_level: str | None = None
    source: Source
    source_quote: str


Node = Annotated[Union[DecisionNode, LeafNode], Field(discriminator="type")]


class DecisionTree(BaseModel):
    """Un arbol completo: un punto de decision de una GPC."""

    tree_id: str
    title: str
    description: str
    gpc_source: str  # nombre del archivo de la guia (sin ruta)
    root_node_id: str
    nodes: dict[str, Node]

    @model_validator(mode="after")
    def _validar_integridad(self) -> DecisionTree:
        if self.root_node_id not in self.nodes:
            raise ValueError(f"root_node_id {self.root_node_id!r} no existe en 'nodes'")
        for node_id, node in self.nodes.items():
            if node.node_id != node_id:
                raise ValueError(f"clave {node_id!r} no coincide con node.node_id {node.node_id!r}")
            if isinstance(node, DecisionNode):
                for branch in node.branches:
                    if branch.next_node_id not in self.nodes:
                        raise ValueError(
                            f"nodo {node_id!r}: branch apunta a next_node_id "
                            f"{branch.next_node_id!r} que no existe"
                        )
        return self
