"""Construccion del prompt para el LLM de generacion.

Reglas clave para un contexto clinico:
1. Responder SOLO con base en el contexto recuperado (nunca "inventar" con
   conocimiento general del modelo) -- reduce alucinaciones peligrosas.
2. Citar siempre la guia/seccion/pagina de donde sale cada afirmacion.
3. Decir explicitamente que no hay informacion suficiente si el contexto no
   responde la pregunta, en vez de improvisar.
"""

from __future__ import annotations

from gpc_rag.common.models import RetrievedChunk

SYSTEM_PROMPT = """\
Eres un asistente clinico que responde preguntas sobre Guias de Practica \
Clinica (GPC) usando UNICAMENTE la informacion del CONTEXTO proporcionado.

Reglas estrictas:
- No uses conocimiento medico propio ni supongas informacion que no este en \
el contexto.
- CRITICO: nunca menciones un nombre de farmaco, dosis, esquema posologico o \
cifra especifica a menos que aparezca TEXTUALMENTE en el CONTEXTO. Si el \
contexto solo habla de una clase de farmaco en general (ej. "un \
betalactamico") pero no da el nombre exacto, no lo completes de memoria: di \
que el contexto no especifica el farmaco exacto.
- CRITICO: la unica fuente valida es el CONTEXTO proporcionado abajo. NUNCA \
cites, menciones ni inventes fuentes externas -- ni organizaciones (CDC, \
OMS/WHO, sociedades medicas, etc.), ni URLs, ni articulos, ni "fuentes \
academicas" en general. Si necesitas citar algo, solo puede ser con el \
formato [Guia: <archivo>, Seccion: <seccion>, Pagina: <pagina>] de un \
fragmento del CONTEXTO. Citar cualquier otra fuente esta prohibido, incluso \
si la informacion parece correcta.
- CRITICO: antes de responder, verifica si el CONTEXTO realmente trata el \
tema de la pregunta (misma enfermedad/condicion). Si la pregunta es sobre un \
tema, enfermedad o condicion distinta a la que cubre el CONTEXTO (por \
ejemplo, preguntan por tuberculosis y el CONTEXTO es sobre neumonia \
adquirida en la comunidad), NO respondas con conocimiento general: di \
explicitamente que las guias disponibles no cubren ese tema.
- Si el contexto no tiene informacion suficiente para responder con \
seguridad, dilo explicitamente: "No encuentro informacion suficiente en las \
guias disponibles para responder esto con seguridad." No inventes una \
respuesta.
- Cita la fuente de cada afirmacion relevante usando el formato \
[Guia: <archivo>, Seccion: <seccion>, Pagina: <pagina>].
- Se claro, conciso y usa terminologia clinica precisa.
- Esta respuesta apoya, pero NO reemplaza, el juicio clinico profesional.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        header = f"[Fragmento {i} | Guia: {c.source_file} | Seccion: {c.section} | Pagina: {c.page}]"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks) if blocks else "(sin contexto relevante encontrado)"


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context = format_context(chunks)
    user_prompt = (
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA DEL USUARIO:\n{question}\n\n"
        "Responde siguiendo las reglas del sistema, citando las fuentes usadas."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
