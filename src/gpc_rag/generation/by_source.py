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
externas), y se compone sin modelo en un unico texto con un bloque por
grupo. Asi el RAG sigue dando una sola respuesta, pero ninguna pasada puede
atribuir al protocolo lo que dice una guia externa, porque no la ve. Las
diferencias quedan a la vista en los dos bloques; la contradiccion con el
arbol la sigue senalando el verificador.
"""

from __future__ import annotations

from gpc_rag.common.models import RetrievedChunk
from gpc_rag.common.refusal import is_refusal
from gpc_rag.ingestion.sources import INSTITUTIONAL

INSTITUTIONAL_HEADING = "Según el protocolo institucional de la Clínica Universitaria Bolivariana (CUB):"
EXTERNAL_HEADING = "Según las guías y revisiones externas:"
NO_INSTITUTIONAL = (
    "El protocolo institucional de la Clínica Universitaria Bolivariana (CUB) "
    "no trae información sobre esto entre los fragmentos recuperados."
)

_GROUP_INSTRUCTIONS = {
    INSTITUTIONAL: (
        "Todos los fragmentos de este CONTEXTO son del protocolo institucional de la "
        "Clinica Universitaria Bolivariana (CUB). Responde solo con lo que dice ese "
        "protocolo, sin mencionar otras guias."
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
    return "\n".join(part for part in (_GROUP_INSTRUCTIONS[group], extra) if part)


def compose(institutional_text: str | None, external_text: str | None) -> str:
    """Une las dos partes en una sola respuesta, sin redactar nada nuevo
    salvo los encabezados y el aviso de que el protocolo no trae el dato."""
    has_inst = bool(institutional_text) and not is_refusal(institutional_text)
    has_ext = bool(external_text) and not is_refusal(external_text)
    if not has_inst and not has_ext:
        # Ningun grupo responde: se devuelve el rechazo tal cual (corto), para
        # que el sistema lo trate como abstencion.
        return institutional_text or external_text or ""
    if has_inst and not external_text:
        return institutional_text.strip()
    blocks = [
        f"**{INSTITUTIONAL_HEADING}**\n"
        + (institutional_text.strip() if has_inst else NO_INSTITUTIONAL)
    ]
    if has_ext:
        blocks.append(f"**{EXTERNAL_HEADING}**\n{external_text.strip()}")
    return "\n\n".join(blocks)


def generate_by_source(
    generator, question: str, sources: list[RetrievedChunk], extra_instruction: str | None = None
) -> str:
    """Genera una pasada por grupo con fragmentos y compone la respuesta."""
    institutional, external = split_by_source(sources)
    if not external:
        return generator.generate(question, sources, extra_instruction=extra_instruction)
    inst_text = (
        generator.generate(
            question, institutional, extra_instruction=_instruction(INSTITUTIONAL, extra_instruction)
        )
        if institutional
        else None
    )
    ext_text = generator.generate(
        question, external, extra_instruction=_instruction("externas", extra_instruction)
    )
    return compose(inst_text, ext_text)
