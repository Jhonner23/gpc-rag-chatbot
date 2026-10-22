"""Arboles clinicos aprobados por el comite de la clinica, como diagramas de
flujo (nodos y flechas), y su recorrido.

Los YAML de cada protocolo (agents/protocols/<id>/<version>/arbol.yaml) se
usan tal como los aprobo el comite: este modulo solo normaliza su formato a
un grafo comun, sin cambiar el contenido clinico. Se aceptan los dos
formatos entregados:

- ITU (CT-PL-197): `metadata`, `nodes` (id, label, type, src) y `edges`.
- NAC (CT-PL-193): `document` y `diagram` con `sections`, `nodes` (id, kind,
  text, section) y `edges` (con `reference` e `independent`).

Semantica del recorrido:

- Un nodo de decision (rombo) es una pregunta; sus flechas son las opciones.
  Si una opcion tiene un calificador ("Si + ITU baja"), solo se ofrece cuando
  ese calificador coincide con un nodo ya recorrido (la clasificacion previa
  del caso). Si ninguna opcion aplica, el recorrido se detiene sin ruta: el
  agente se abstiene (no infiere rutas que el diagrama no describe).
- Las flechas `reference` (recuadros de criterios) y la flecha "contexto" de
  un rombo no son ramas: son apoyo para responder esa pregunta.
- Una flecha con etiqueta que sale de un nodo que no es decision ("si
  recurrencia", "si esta hospitalizada") es una continuacion condicional: se
  informa como nota, sin suponer que la condicion se cumple.
- Las flechas `independent` (p. ej. evaluaciones paralelas) se recorren una
  tras otra hasta el nodo donde vuelven a unirse.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

UNKNOWN = "__desconocido__"

_CONTEXT_LABELS = {"contexto"}


@dataclass(frozen=True)
class FlowNode:
    id: str
    kind: str
    text: str
    section: str = ""
    source_ref: str = (
        ""  # referencia al protocolo indicada en el diagrama (p. ej. "p.4")
    )

    @property
    def is_decision(self) -> bool:
        return self.kind == "decision"


@dataclass(frozen=True)
class FlowEdge:
    source: str
    target: str
    label: str = ""
    reference: bool = False
    independent: bool = False

    @property
    def qualifier(self) -> str:
        """Parte de la etiqueta despues de "+", p. ej. "ITU baja" en "Si + ITU baja"."""
        return self.label.split("+", 1)[1].strip() if "+" in self.label else ""


@dataclass
class Flowchart:
    protocol_id: str
    version: str  # "v06", "v09"
    name: str
    description: str
    nodes: dict[str, FlowNode]
    edges: list[FlowEdge]
    start_id: str
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def source_file(self) -> str:
        return f"Árbol clínico {self.protocol_id} {self.version}"

    def out_edges(self, node_id: str) -> list[FlowEdge]:
        """Ramas que se pueden recorrer (sin referencias ni contexto)."""
        return [
            e
            for e in self.edges
            if e.source == node_id
            and not e.reference
            and e.label.lower() not in _CONTEXT_LABELS
        ]

    def support_nodes(self, decision_id: str) -> list[FlowNode]:
        """Recuadros de criterios o de contexto asociados a una pregunta."""
        support = [
            self.nodes[e.source]
            for e in self.edges
            if e.target == decision_id and e.reference
        ]
        support += [
            self.nodes[e.target]
            for e in self.edges
            if e.source == decision_id and e.label.lower() in _CONTEXT_LABELS
        ]
        return support

    def join_node(self, starts: list[str]) -> str | None:
        """Primer nodo alcanzable desde todos los `starts` (donde se unen)."""
        reach = [self._reachable(s) for s in starts]
        for node_id in self._bfs_order(starts[0]):
            if all(node_id in r for r in reach):
                return node_id
        return None

    def _bfs_order(self, start: str) -> list[str]:
        order, seen, queue = [], {start}, deque([start])
        while queue:
            current = queue.popleft()
            order.append(current)
            for e in self.out_edges(current):
                if e.target not in seen:
                    seen.add(e.target)
                    queue.append(e.target)
        return order

    def _reachable(self, start: str) -> set[str]:
        return set(self._bfs_order(start))

    def conditions_to(self, node_id: str) -> list[tuple[FlowNode, str]]:
        """Decisiones que determinan llegar a `node_id`: las del camino mas
        corto desde el inicio en las que alguna opcion ya no permite llegar a
        el. Se omiten las decisiones cuyas opciones llevan todas al nodo,
        porque no lo condicionan."""
        conditions = []
        for decision, label in self.route_to(node_id):
            options = self.out_edges(decision.id)
            if not all(node_id in self._reachable(e.target) for e in options):
                conditions.append((decision, label))
        return conditions

    def route_to(self, node_id: str) -> list[tuple[FlowNode, str]]:
        """Camino mas corto desde el inicio: [(decision, opcion tomada), ...]."""
        parents: dict[str, tuple[str, FlowEdge] | None] = {self.start_id: None}
        queue = deque([self.start_id])
        while queue and node_id not in parents:
            current = queue.popleft()
            for e in self.out_edges(current):
                if e.target not in parents:
                    parents[e.target] = (current, e)
                    queue.append(e.target)
        if node_id not in parents:
            return []
        route: list[tuple[FlowNode, str]] = []
        cursor = node_id
        while parents[cursor] is not None:
            previous, edge = parents[cursor]  # type: ignore[misc]
            if self.nodes[previous].is_decision:
                route.append((self.nodes[previous], edge.label))
            cursor = previous
        return list(reversed(route))


# Carga ----------------------------------------------------------------------


def _clean(text: Any) -> str:
    lines = [line.strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _version_label(raw: Any) -> str:
    digits = re.sub(r"\D", "", str(raw or ""))
    return f"v{int(digits):02d}" if digits else ""


def _parse_itu_style(data: dict[str, Any]) -> Flowchart:
    meta = data.get("metadata") or {}
    nodes = {
        n["id"]: FlowNode(
            id=n["id"],
            kind=str(n.get("type", "")).lower(),
            text=_clean(n.get("label")),
            source_ref=str(n.get("src") or ""),
        )
        for n in data["nodes"]
    }
    edges = [
        FlowEdge(
            source=e["from"], target=e["to"], label=str(e.get("label") or "").strip()
        )
        for e in data["edges"]
    ]
    return Flowchart(
        protocol_id=str(meta.get("protocol", "")).strip(),
        version=_version_label(meta.get("version")),
        name=_clean(meta.get("title")),
        description=_clean(meta.get("scope")),
        nodes=nodes,
        edges=edges,
        start_id=_start_id(nodes),
        policy=data.get("agent_policy") or {},
    )


def _parse_nac_style(data: dict[str, Any]) -> Flowchart:
    doc = data.get("document") or {}
    diagram = data["diagram"]
    sections = {str(k): str(v) for k, v in (diagram.get("sections") or {}).items()}
    nodes = {
        n["id"]: FlowNode(
            id=n["id"],
            kind=str(n.get("kind", "")).lower(),
            text=_clean(n.get("text")),
            section=sections.get(str(n.get("section")), ""),
        )
        for n in diagram["nodes"]
    }
    edges = [
        FlowEdge(
            source=e["from"],
            target=e["to"],
            label=str(e.get("label") or "").strip(),
            reference=bool(e.get("reference")),
            independent=bool(e.get("independent")),
        )
        for e in diagram["edges"]
    ]
    return Flowchart(
        protocol_id=str(doc.get("code", "")).strip(),
        version=_version_label(doc.get("version")),
        name=_clean(doc.get("title")),
        description=_clean(doc.get("scope")),
        nodes=nodes,
        edges=edges,
        start_id=_start_id(nodes),
        policy=data.get("agent_policy") or {},
    )


def _start_id(nodes: dict[str, FlowNode]) -> str:
    starts = [n.id for n in nodes.values() if n.kind == "start"]
    if len(starts) != 1:
        raise ValueError(
            f"El arbol debe tener exactamente un nodo de inicio (tiene {len(starts)})."
        )
    return starts[0]


def parse_flowchart(text: str) -> Flowchart:
    data = yaml.safe_load(text)
    if "diagram" in data:
        chart = _parse_nac_style(data)
    elif "nodes" in data and "edges" in data:
        chart = _parse_itu_style(data)
    else:
        raise ValueError(
            "Formato de arbol no reconocido (se esperaba 'nodes/edges' o 'diagram')."
        )
    validate_flowchart(chart)
    return chart


def validate_flowchart(chart: Flowchart) -> None:
    """Errores de forma que impedirian recorrer el arbol."""
    if not chart.protocol_id or not chart.version:
        raise ValueError("El arbol no declara codigo de protocolo y version.")
    for e in chart.edges:
        for node_id in (e.source, e.target):
            if node_id not in chart.nodes:
                raise ValueError(
                    f"La flecha {e.source} -> {e.target} apunta a un nodo inexistente."
                )
    for node in chart.nodes.values():
        if node.is_decision and len(chart.out_edges(node.id)) < 2:
            raise ValueError(f"La decision {node.id} tiene menos de dos opciones.")


def load_flowchart(path: Path) -> Flowchart:
    return parse_flowchart(path.read_text(encoding="utf-8"))


# Recorrido ------------------------------------------------------------------


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


@dataclass
class Step:
    node: FlowNode
    choice: str | None = None


@dataclass
class WalkResult:
    steps: list[Step]
    status: str  # "completo" | "pendiente" | "sin_ruta" | "dato_desconocido"
    stopped_at: FlowNode | None = None
    options: list[str] = field(default_factory=list)
    notes: list[tuple[str, FlowNode]] = field(default_factory=list)


def decision_options(
    chart: Flowchart, node: FlowNode, steps: list[Step]
) -> list[FlowEdge]:
    """Opciones validas de una decision, dado lo ya recorrido."""
    walked_text = [normalize(s.node.text) for s in steps if not s.node.is_decision]
    options = []
    for edge in chart.out_edges(node.id):
        qualifier = normalize(edge.qualifier)
        if qualifier and not any(qualifier in t for t in walked_text):
            continue
        options.append(edge)
    return options


def walk(chart: Flowchart, answers: dict[str, str]) -> WalkResult:
    """Recorre el arbol desde el inicio con las respuestas disponibles.

    `answers` mapea id de decision -> etiqueta elegida (o UNKNOWN). El
    recorrido es determinista: con las mismas respuestas produce siempre el
    mismo resultado (necesario para reanudar el wizard con interrupt()).
    """
    steps: list[Step] = []
    notes: list[tuple[str, FlowNode]] = []
    visited: set[str] = set()
    agenda: list[tuple[str | None, str | None]] = [(chart.start_id, None)]

    while agenda:
        current, stop_at = agenda.pop(0)
        while current is not None and current != stop_at and current not in visited:
            visited.add(current)
            node = chart.nodes[current]

            if node.is_decision:
                options = decision_options(chart, node, steps)
                if not options:
                    return WalkResult(steps, "sin_ruta", node, [], notes)
                if len(options) == 1:
                    chosen = options[0]
                else:
                    answer = answers.get(node.id)
                    if answer == UNKNOWN:
                        return WalkResult(
                            steps,
                            "dato_desconocido",
                            node,
                            [o.label for o in options],
                            notes,
                        )
                    chosen = next((o for o in options if o.label == answer), None)
                    if chosen is None:
                        return WalkResult(
                            steps, "pendiente", node, [o.label for o in options], notes
                        )
                steps.append(Step(node, chosen.label))
                current = chosen.target
                continue

            steps.append(Step(node))
            edges = chart.out_edges(current)
            independent = [e.target for e in edges if e.independent]
            if independent:
                join = chart.join_node(independent)
                agenda = [(t, join) for t in independent] + [(join, stop_at)] + agenda
                break
            notes += [(e.label, chart.nodes[e.target]) for e in edges if e.label]
            plain = [e for e in edges if not e.label]
            current = plain[0].target if len(plain) == 1 else None

    return WalkResult(steps, "completo", None, [], notes)


# Modo consulta --------------------------------------------------------------

_STOPWORDS = {
    "que",
    "los",
    "las",
    "del",
    "con",
    "por",
    "para",
    "una",
    "uno",
    "unos",
    "unas",
    "como",
    "mas",
    "sus",
    "son",
    "hay",
    "este",
    "esta",
    "estos",
    "estas",
    "entre",
    "sobre",
    "cual",
    "cuales",
    "debe",
    "deben",
    "cuando",
    "donde",
    "paciente",
    "pacientes",
    "manejo",
    "guia",
    "guias",
    "protocolo",
    "recomienda",
    "recomendaciones",
    "recomendado",
    "presentan",
    "disponibles",
    "diferencias",
    "existen",
    "fuentes",
    "corpus",
    "contienen",
    "informacion",
    "relacionada",
    # Terminos que nombran el tema del protocolo y aparecen en casi cualquier
    # pregunta: no distinguen un nodo de otro.
    "neumonia",
    "adquirida",
    "comunidad",
    "enfoque",
    "inicial",
    "considerarse",
    "decidir",
    "condicion",
    "encontradas",
    "existe",
    "situaciones",
    "adultos",
    "adulto",
}


# Sinonimos de la pregunta hacia el vocabulario de los diagramas.
_SYNONYMS = {"pseudomonas": "aeruginosa"}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [_SYNONYMS.get(w, w) for w in words if len(w) > 2 and w not in _STOPWORDS]


Embedder = Callable[[list[str]], list[list[float]]]

# Vectores de los nodos por (protocolo, version): se calculan una vez por
# proceso (unos segundos con bge-m3) y se reutilizan en cada consulta.
_NODE_VECTORS: dict[tuple[str, str], dict[str, list[float]]] = {}


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def _searchable_nodes(chart: Flowchart) -> list[FlowNode]:
    return [n for n in chart.nodes.values() if n.kind != "start"]


def _lexical_ranking(chart: Flowchart, question: str) -> list[tuple[str, float]]:
    candidates = _searchable_nodes(chart)
    query = _tokens(question)
    if not candidates or not query:
        return []
    bm25 = BM25Okapi([_tokens(n.text) for n in candidates])
    scored = [(n.id, s) for n, s in zip(candidates, bm25.get_scores(query), strict=True) if s > 0]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def _dense_ranking(chart: Flowchart, question: str, embed: Embedder) -> list[str]:
    key = (chart.protocol_id, chart.version)
    if key not in _NODE_VECTORS:
        nodes = _searchable_nodes(chart)
        _NODE_VECTORS[key] = dict(zip([n.id for n in nodes], embed([n.text for n in nodes]), strict=True))
    vectors = _NODE_VECTORS[key]
    query = embed([question])[0]
    return sorted(vectors, key=lambda node_id: _cosine(query, vectors[node_id]), reverse=True)


def rank_nodes_for_query(
    chart: Flowchart,
    question: str,
    *,
    top_k: int = 4,
    min_ratio: float = 0.4,
    embed: Embedder | None = None,
) -> list[str]:
    """Nodos del arbol mas relacionados con una pregunta informativa.

    Sin `embed`, solo coincidencia lexica (BM25 sobre el texto de cada nodo).
    Con `embed` (el mismo modelo de embeddings del RAG), se fusiona con la
    similitud semantica (Reciprocal Rank Fusion): las preguntas conceptuales
    ("¿que factores considerar al decidir el manejo?") casi no comparten
    palabras con los nodos que las responden. La pertinencia del resultado la
    juzga despues el verificador.
    """
    lexical = _lexical_ranking(chart, question)
    if embed is None:
        if not lexical:
            return []
        best = lexical[0][1]
        return [node_id for node_id, score in lexical[:top_k] if score >= best * min_ratio]

    dense = _dense_ranking(chart, question, embed)
    fused: dict[str, float] = {}
    for ranking in ([node_id for node_id, _ in lexical], dense):
        for rank, node_id in enumerate(ranking):
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (60 + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)[:top_k]


def initial_section(chart: Flowchart, max_nodes: int = 10) -> tuple[str, list[FlowNode]]:
    """Primera parte del arbol, en el orden del recorrido desde el inicio: la
    primera seccion del diagrama si las tiene; si no, los primeros nodos.
    Responde preguntas generales como "¿cual es el enfoque inicial?"."""
    order = [chart.nodes[i] for i in chart._bfs_order(chart.start_id)]
    first_section = chart.nodes[chart.start_id].section
    if first_section:
        return first_section, [n for n in order if n.section == first_section and n.kind != "start"]
    return "", [n for n in order if n.kind != "start"][:max_nodes]


def decision_overview(chart: Flowchart) -> list[FlowNode]:
    """Decisiones del arbol en el orden en que aparecen desde el inicio
    (para preguntas generales que no apuntan a un nodo concreto)."""
    return [
        chart.nodes[i]
        for i in chart._bfs_order(chart.start_id)
        if chart.nodes[i].is_decision
    ]
