"""Respuesta del RAG generada por tipo de fuente y compuesta en una sola.

Con las guias y revisiones externas, el contexto paso a mezclar fragmentos de
varios documentos y dos idiomas. En una prueba real ("¿que recomiendan las
guias sobre la bacteriuria asintomatica y en que difieren?") el modelo local
de 7B atribuyo al protocolo de la CUB recomendaciones que no tiene, invento
un grupo ("mayores de 50 anos"), invirtio el sentido de una frase de
ClinicalKey y llamo "cubano" al protocolo. Pedirle traducir, atribuir y
comparar en una sola pasada supera lo que hace con fiabilidad.

Por eso la respuesta se genera en dos pasadas, cada una solo con los
fragmentos de su grupo (el protocolo institucional; las guias y revisiones
externas), y se une sin modelo en un unico texto: primero lo del protocolo y
a continuacion lo de las fuentes externas, sin encabezados (lo pidio la
asesora: una sola respuesta). La cita de cada afirmacion dice de donde sale.
Ninguna pasada puede atribuir al protocolo lo que dice una guia externa,
porque no la ve; la contradiccion con el arbol la sigue senalando el
verificador.
"""

from __future__ import annotations

import re

from gpc_rag.common.models import RetrievedChunk
from gpc_rag.common.refusal import is_refusal
from gpc_rag.ingestion.sources import INSTITUTIONAL
from gpc_rag.retrieval.hybrid import retrieval_query

# Cada pasada ve un solo grupo de fuentes: si respondiera "¿que diferencias
# hay entre las guias?" con lo que ve, el protocolo diria "no hay diferencias,
# todo viene del mismo documento" justo antes de que las guias externas digan
# otra cosa. Cada pasada responde solo la parte clinica de la pregunta.
_NO_COMPARISON = (
    "Responde solo el tema clinico de la pregunta. No compares fuentes, no digas si "
    "hay o no diferencias entre guias y no hagas una lista de fuentes: eso se arma "
    "aparte."
)
# Llamadas numericas de los articulos ("[1]", "[12, 13]"), que no son citas
# del corpus.
_NUMERIC_CALLOUT_RE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")

_GROUP_INSTRUCTIONS = {
    INSTITUTIONAL: (
        "Todos los fragmentos de este CONTEXTO son del protocolo institucional de la "
        "Clinica Universitaria Bolivariana (CUB). Responde solo con lo que dice ese "
        "protocolo, sin mencionar otras guias. Si no trae lo que se pregunta, responde "
        "unicamente la frase de que no hay informacion suficiente."
    ),
    "externas": (
        "Todos los fragmentos de este CONTEXTO son guias externas o articulos de "
        "revision, NO el protocolo institucional de la CUB. Responde solo con lo que "
        "dicen, nombrando la fuente de cada dato por su titulo. No digas que dice o "
        "recomienda el protocolo institucional: no esta en este CONTEXTO. Traduce "
        "fielmente al español lo que este en ingles, conservando exactamente "
        "farmacos, dosis, intervalos, duraciones y condiciones, y sin invertir el "
        "sentido de ninguna frase."
    ),
}


def split_by_source(
    sources: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    institutional = [rc for rc in sources if rc.chunk.source_type == INSTITUTIONAL]
    external = [rc for rc in sources if rc.chunk.source_type != INSTITUTIONAL]
    return institutional, external


def _instruction(group: str, extra: str | None) -> str:
    return "\n".join(
        part for part in (_GROUP_INSTRUCTIONS[group], _NO_COMPARISON, extra) if part
    )


def compose(institutional_text: str | None, external_text: str | None) -> str:
    """Une las partes que si responden en un solo texto, sin redactar nada
    nuevo: primero el protocolo, despues las fuentes externas. La parte que
    se abstiene se omite; si ninguna responde, queda el rechazo (corto) para
    que el sistema lo trate como abstencion."""
    parts = [
        _NUMERIC_CALLOUT_RE.sub("", text).strip()
        for text in (institutional_text, external_text)
        if text and not is_refusal(text)
    ]
    if not parts:
        return institutional_text or external_text or ""
    return "\n\n".join(parts)


def generate_by_source(
    generator, question: str, sources: list[RetrievedChunk], extra_instruction: str | None = None
) -> str:
    """Genera una pasada por grupo con fragmentos y compone la respuesta."""
    institutional, external = split_by_source(sources)
    if not external:
        return generator.generate(question, sources, extra_instruction=extra_instruction)
    clinical_question = retrieval_query(question)
    inst_text = (
        generator.generate(
            clinical_question,
            institutional,
            extra_instruction=_instruction(INSTITUTIONAL, extra_instruction),
        )
        if institutional
        else None
    )
    ext_text = generator.generate(
        clinical_question, external, extra_instruction=_instruction("externas", extra_instruction)
    )
    return compose(inst_text, ext_text)
